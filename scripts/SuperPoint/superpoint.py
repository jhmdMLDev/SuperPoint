import argparse
import glob
import os
import time
import warnings

warnings.filterwarnings("ignore")

import cv2
import numpy as np
import torch

# Stub to warn about opencv version.
if int(cv2.__version__[0]) < 3:  # pragma: no cover
    print("Warning: OpenCV 3 is not installed")

# Jet colormap for visualization.
myjet = np.array(
    [
        [0.0, 0.0, 0.5],
        [0.0, 0.0, 0.99910873],
        [0.0, 0.37843137, 1.0],
        [0.0, 0.83333333, 1.0],
        [0.30044276, 1.0, 0.66729918],
        [0.66729918, 1.0, 0.30044276],
        [1.0, 0.90123457, 0.0],
        [1.0, 0.48002905, 0.0],
        [0.99910873, 0.07334786, 0.0],
        [0.5, 0.0, 0.0],
    ]
)


def Preview_Registration_Overlay(
    img_base, img_transformed, intensity, path, invert_transformed=False
):
    """
    Generate a preview of registered images by displaying the base (refernce image) in the R channel and the transformed
    image in the BG channel
    :param img_base: reference image
    :param img_transformed: target image
    :param intensity: intensity of target image (0->1)
    :param path: save path of preview image [*.png]
    :param invert_transformed: 'True' if the target image is to be inverted (255-image)
    :return: (none)
    """
    if invert_transformed:
        img_transformed = 255.0 - img_transformed

    img_reg_preview = np.zeros((img_base.shape[0], img_base.shape[1], 3))
    if len(img_base.shape) == 3:
        img_base = np.mean(np.array(img_base), 2)
    if len(img_transformed.shape) == 3:
        img_transformed = np.mean(np.array(img_transformed), 2)
    img_reg_preview[:, :, 0] = img_base / np.max(img_base)
    img_reg_preview[:, :, 1] = intensity * (img_transformed / np.max(img_transformed))
    img_reg_preview[:, :, 2] = intensity * (img_transformed / np.max(img_transformed))
    img_reg_preview = (img_reg_preview * 255.0).astype("uint8")
    if path is None:
        pass
        # plt.imshow(img_reg_preview)
        # plt.show()
    else:
        cv2.imwrite(path, cv2.cvtColor(img_reg_preview, cv2.COLOR_RGB2BGR))
    return cv2.cvtColor(img_reg_preview, cv2.COLOR_RGB2BGR)


class SuperPointNet(torch.nn.Module):
    """Pytorch definition of SuperPoint Network."""

    def __init__(self):
        super(SuperPointNet, self).__init__()
        self.relu = torch.nn.ReLU(inplace=True)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256
        # Shared Encoder.
        self.conv1a = torch.nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = torch.nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = torch.nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = torch.nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = torch.nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = torch.nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = torch.nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = torch.nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)
        # Detector Head.
        self.convPa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convPb = torch.nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)
        # Descriptor Head.
        self.convDa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = torch.nn.Conv2d(c5, d1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """Forward pass that jointly computes unprocessed point and descriptor
        tensors.
        Input
          x: Image pytorch tensor shaped N x 1 x H x W.
        Output
          semi: Output point pytorch tensor shaped N x 65 x H/8 x W/8.
          desc: Output descriptor pytorch tensor shaped N x 256 x H/8 x W/8.
        """
        # Shared Encoder.
        x = self.relu(self.conv1a(x))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        # Detector Head.
        cPa = self.relu(self.convPa(x))
        semi = self.convPb(cPa)
        # Descriptor Head.
        cDa = self.relu(self.convDa(x))
        desc = self.convDb(cDa)
        dn = torch.norm(desc, p=2, dim=1)  # Compute the norm.
        desc = desc.div(torch.unsqueeze(dn, 1))  # Divide by norm to normalize.
        return semi, desc


class SuperPointFrontend(object):
    """Wrapper around pytorch net to help with pre and post image processing."""

    def __init__(self, weights_path, nms_dist, conf_thresh, nn_thresh, cuda=False):
        self.name = "SuperPoint"
        self.cuda = cuda
        self.nms_dist = nms_dist
        self.conf_thresh = conf_thresh
        self.nn_thresh = nn_thresh  # L2 descriptor distance for good match.
        self.cell = 8  # Size of each output cell. Keep this fixed.
        self.border_remove = 4  # Remove points this close to the border.
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Load the network in inference mode.
        self.net = SuperPointNet()
        if cuda:
            # Train on GPU, deploy on GPU.
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.net.load_state_dict(torch.load(weights_path, map_location=self.device))
            # self.net = self.net.cuda()
        else:
            # Train on GPU, deploy on CPU.
            self.net.load_state_dict(
                torch.load(weights_path, map_location=lambda storage, loc: storage)
            )
        self.net.eval()
        self.net.to(self.device)

    def nms_fast(self, in_corners, H, W, dist_thresh):
        """
        Run a faster approximate Non-Max-Suppression on numpy corners shaped:
          3xN [x_i,y_i,conf_i]^T

        Algo summary: Create a grid sized HxW. Assign each corner location a 1, rest
        are zeros. Iterate through all the 1's and convert them either to -1 or 0.
        Suppress points by setting nearby values to 0.

        Grid Value Legend:
        -1 : Kept.
         0 : Empty or suppressed.
         1 : To be processed (converted to either kept or supressed).

        NOTE: The NMS first rounds points to integers, so NMS distance might not
        be exactly dist_thresh. It also assumes points are within image boundaries.

        Inputs
          in_corners - 3xN numpy array with corners [x_i, y_i, confidence_i]^T.
          H - Image height.
          W - Image width.
          dist_thresh - Distance to suppress, measured as an infinty norm distance.
        Returns
          nmsed_corners - 3xN numpy matrix with surviving corners.
          nmsed_inds - N length numpy vector with surviving corner indices.
        """
        grid = np.zeros((H, W)).astype(int)  # Track NMS data.
        inds = np.zeros((H, W)).astype(int)  # Store indices of points.
        # Sort by confidence and round to nearest int.
        inds1 = np.argsort(-in_corners[2, :])
        corners = in_corners[:, inds1]
        rcorners = corners[:2, :].round().astype(int)  # Rounded corners.
        # Check for edge case of 0 or 1 corners.
        if rcorners.shape[1] == 0:
            return np.zeros((3, 0)).astype(int), np.zeros(0).astype(int)
        if rcorners.shape[1] == 1:
            out = np.vstack((rcorners, in_corners[2])).reshape(3, 1)
            return out, np.zeros((1)).astype(int)
        # Initialize the grid.
        for i, rc in enumerate(rcorners.T):
            grid[rcorners[1, i], rcorners[0, i]] = 1
            inds[rcorners[1, i], rcorners[0, i]] = i
        # Pad the border of the grid, so that we can NMS points near the border.
        pad = dist_thresh
        grid = np.pad(grid, ((pad, pad), (pad, pad)), mode="constant")
        # Iterate through points, highest to lowest conf, suppress neighborhood.
        count = 0
        for i, rc in enumerate(rcorners.T):
            # Account for top and left padding.
            pt = (rc[0] + pad, rc[1] + pad)
            if grid[pt[1], pt[0]] == 1:  # If not yet suppressed.
                grid[pt[1] - pad : pt[1] + pad + 1, pt[0] - pad : pt[0] + pad + 1] = 0
                grid[pt[1], pt[0]] = -1
                count += 1
        # Get all surviving -1's and return sorted array of remaining corners.
        keepy, keepx = np.where(grid == -1)
        keepy, keepx = keepy - pad, keepx - pad
        inds_keep = inds[keepy, keepx]
        out = corners[:, inds_keep]
        values = out[-1, :]
        inds2 = np.argsort(-values)
        out = out[:, inds2]
        out_inds = inds1[inds_keep[inds2]]
        return out, out_inds

    def run(self, img):
        """Process a numpy image to extract points and descriptors.
        Input
          img - HxW numpy float32 input image in range [0,1].
        Output
          corners - 3xN numpy array with corners [x_i, y_i, confidence_i]^T.
          desc - 256xN numpy array of corresponding unit normalized descriptors.
          heatmap - HxW numpy heatmap in range [0,1] of point confidences.
        """
        assert img.ndim == 2, "Image must be grayscale."
        assert img.dtype == np.float32, "Image must be float32."
        H, W = img.shape[0], img.shape[1]
        inp = img.copy()
        inp = inp.reshape(1, H, W)
        inp = torch.from_numpy(inp)
        inp = torch.autograd.Variable(inp).view(1, 1, H, W)
        if self.cuda:
            inp = inp.to(self.device)
        # Forward pass of network.
        outs = self.net.forward(inp)
        semi, coarse_desc = outs[0], outs[1]
        # Convert pytorch -> numpy.
        semi = semi.data.cpu().numpy().squeeze()
        # --- Process points.
        dense = np.exp(semi)  # Softmax.
        dense = dense / (np.sum(dense, axis=0) + 0.00001)  # Should sum to 1.
        # Remove dustbin.
        nodust = dense[:-1, :, :]
        # Reshape to get full resolution heatmap.
        Hc = int(H / self.cell)
        Wc = int(W / self.cell)
        nodust = nodust.transpose(1, 2, 0)
        heatmap = np.reshape(nodust, [Hc, Wc, self.cell, self.cell])
        heatmap = np.transpose(heatmap, [0, 2, 1, 3])
        heatmap = np.reshape(heatmap, [Hc * self.cell, Wc * self.cell])
        xs, ys = np.where(heatmap >= self.conf_thresh)  # Confidence threshold.
        if len(xs) == 0:
            return np.zeros((3, 0)), None, None
        pts = np.zeros((3, len(xs)))  # Populate point data sized 3xN.
        pts[0, :] = ys
        pts[1, :] = xs
        pts[2, :] = heatmap[xs, ys]
        pts, _ = self.nms_fast(pts, H, W, dist_thresh=self.nms_dist)  # Apply NMS.
        inds = np.argsort(pts[2, :])
        pts = pts[:, inds[::-1]]  # Sort by confidence.
        # Remove points along border.
        bord = self.border_remove
        toremoveW = np.logical_or(pts[0, :] < bord, pts[0, :] >= (W - bord))
        toremoveH = np.logical_or(pts[1, :] < bord, pts[1, :] >= (H - bord))
        toremove = np.logical_or(toremoveW, toremoveH)
        pts = pts[:, ~toremove]
        # --- Process descriptor.
        D = coarse_desc.shape[1]
        if pts.shape[1] == 0:
            desc = np.zeros((D, 0))
        else:
            # Interpolate into descriptor map using 2D point locations.
            samp_pts = torch.from_numpy(pts[:2, :].copy())
            samp_pts[0, :] = (samp_pts[0, :] / (float(W) / 2.0)) - 1.0
            samp_pts[1, :] = (samp_pts[1, :] / (float(H) / 2.0)) - 1.0
            samp_pts = samp_pts.transpose(0, 1).contiguous()
            samp_pts = samp_pts.view(1, 1, -1, 2)
            samp_pts = samp_pts.float()
            if self.cuda:
                samp_pts = samp_pts.to(self.device)
            desc = torch.nn.functional.grid_sample(coarse_desc, samp_pts)
            desc = desc.data.cpu().numpy().reshape(D, -1)
            desc /= np.linalg.norm(desc, axis=0)[np.newaxis, :]

        return pts, desc, heatmap


class PointTracker(object):
    """Class to manage a fixed memory of points and descriptors that enables
    sparse optical flow point tracking.

    Internally, the tracker stores a 'tracks' matrix sized M x (2+L), of M
    tracks with maximum length L, where each row corresponds to:
    row_m = [track_id_m, avg_desc_score_m, point_id_0_m, ..., point_id_L-1_m].
    """

    def __init__(self, max_length, nn_thresh):
        if max_length < 2:
            raise ValueError("max_length must be greater than or equal to 2.")
        self.maxl = max_length
        self.nn_thresh = nn_thresh
        self.all_pts = []
        for n in range(self.maxl):
            self.all_pts.append(np.zeros((2, 0)))
        self.last_desc = None
        self.tracks = np.zeros((0, self.maxl + 2))
        self.track_count = 0
        self.max_score = 9999

    def nn_match_two_way(self, desc1, desc2, nn_thresh):
        """
        Performs two-way nearest neighbor matching of two sets of descriptors, such
        that the NN match from descriptor A->B must equal the NN match from B->A.

        Inputs:
          desc1 - NxM numpy matrix of N corresponding M-dimensional descriptors.
          desc2 - NxM numpy matrix of N corresponding M-dimensional descriptors.
          nn_thresh - Optional descriptor distance below which is a good match.

        Returns:
          matches - 3xL numpy array, of L matches, where L <= N and each column i is
                    a match of two descriptors, d_i in image 1 and d_j' in image 2:
                    [d_i index, d_j' index, match_score]^T
        """
        assert desc1.shape[0] == desc2.shape[0]
        if desc1.shape[1] == 0 or desc2.shape[1] == 0:
            return np.zeros((3, 0))
        if nn_thresh < 0.0:
            raise ValueError("'nn_thresh' should be non-negative")
        # Compute L2 distance. Easy since vectors are unit normalized.
        dmat = np.dot(desc1.T, desc2)
        dmat = np.sqrt(2 - 2 * np.clip(dmat, -1, 1))
        # Get NN indices and scores.
        idx = np.argmin(dmat, axis=1)
        scores = dmat[np.arange(dmat.shape[0]), idx]
        # Threshold the NN matches.
        keep = scores < nn_thresh
        # Check if nearest neighbor goes both directions and keep those.
        idx2 = np.argmin(dmat, axis=0)
        keep_bi = np.arange(len(idx)) == idx2[idx]
        keep = np.logical_and(keep, keep_bi)
        idx = idx[keep]
        scores = scores[keep]
        # Get the surviving point indices.
        m_idx1 = np.arange(desc1.shape[1])[keep]
        m_idx2 = idx
        # Populate the final 3xN match data structure.
        matches = np.zeros((3, int(keep.sum())))
        matches[0, :] = m_idx1
        matches[1, :] = m_idx2
        matches[2, :] = scores
        return matches

    def get_offsets(self):
        """Iterate through list of points and accumulate an offset value. Used to
        index the global point IDs into the list of points.

        Returns
          offsets - N length array with integer offset locations.
        """
        # Compute id offsets.
        offsets = []
        offsets.append(0)
        for i in range(len(self.all_pts) - 1):  # Skip last camera size, not needed.
            offsets.append(self.all_pts[i].shape[1])
        offsets = np.array(offsets)
        offsets = np.cumsum(offsets)
        return offsets

    def update(self, pts, desc):
        """Add a new set of point and descriptor observations to the tracker.

        Inputs
          pts - 3xN numpy array of 2D point observations.
          desc - DxN numpy array of corresponding D dimensional descriptors.
        """
        if pts is None or desc is None:
            print("PointTracker: Warning, no points were added to tracker.")
            return
        assert pts.shape[1] == desc.shape[1]
        # Initialize last_desc.
        if self.last_desc is None:
            self.last_desc = np.zeros((desc.shape[0], 0))
        # Remove oldest points, store its size to update ids later.
        remove_size = self.all_pts[0].shape[1]
        self.all_pts.pop(0)
        self.all_pts.append(pts)
        # Remove oldest point in track.
        self.tracks = np.delete(self.tracks, 2, axis=1)
        # Update track offsets.
        for i in range(2, self.tracks.shape[1]):
            self.tracks[:, i] -= remove_size
        self.tracks[:, 2:][self.tracks[:, 2:] < -1] = -1
        offsets = self.get_offsets()
        # Add a new -1 column.
        self.tracks = np.hstack((self.tracks, -1 * np.ones((self.tracks.shape[0], 1))))
        # Try to append to existing tracks.
        matched = np.zeros((pts.shape[1])).astype(bool)
        matches = self.nn_match_two_way(self.last_desc, desc, self.nn_thresh)
        for match in matches.T:
            # Add a new point to it's matched track.
            id1 = int(match[0]) + offsets[-2]
            id2 = int(match[1]) + offsets[-1]
            found = np.argwhere(self.tracks[:, -2] == id1)
            if found.shape[0] > 0:
                matched[int(match[1])] = True
                row = int(found)
                self.tracks[row, -1] = id2
                if self.tracks[row, 1] == self.max_score:
                    # Initialize track score.
                    self.tracks[row, 1] = match[2]
                else:
                    # Update track score with running average.
                    # NOTE(dd): this running average can contain scores from old matches
                    #           not contained in last max_length track points.
                    track_len = (self.tracks[row, 2:] != -1).sum() - 1.0
                    frac = 1.0 / float(track_len)
                    self.tracks[row, 1] = (1.0 - frac) * self.tracks[
                        row, 1
                    ] + frac * match[2]
        # Add unmatched tracks.
        new_ids = np.arange(pts.shape[1]) + offsets[-1]
        new_ids = new_ids[~matched]
        new_tracks = -1 * np.ones((new_ids.shape[0], self.maxl + 2))
        new_tracks[:, -1] = new_ids
        new_num = new_ids.shape[0]
        new_trackids = self.track_count + np.arange(new_num)
        new_tracks[:, 0] = new_trackids
        new_tracks[:, 1] = self.max_score * np.ones(new_ids.shape[0])
        self.tracks = np.vstack((self.tracks, new_tracks))
        self.track_count += new_num  # Update the track count.
        # Remove empty tracks.
        keep_rows = np.any(self.tracks[:, 2:] >= 0, axis=1)
        self.tracks = self.tracks[keep_rows, :]
        # Store the last descriptors.
        self.last_desc = desc.copy()
        return

    def get_tracks(self, min_length):
        """Retrieve point tracks of a given minimum length.
        Input
          min_length - integer >= 1 with minimum track length
        Output
          returned_tracks - M x (2+L) sized matrix storing track indices, where
            M is the number of tracks and L is the maximum track length.
        """
        if min_length < 1:
            raise ValueError("'min_length' too small.")
        valid = np.ones((self.tracks.shape[0])).astype(bool)
        good_len = np.sum(self.tracks[:, 2:] != -1, axis=1) >= min_length
        # Remove tracks which do not have an observation in most recent frame.
        not_headless = self.tracks[:, -1] != -1
        keepers = np.logical_and.reduce((valid, good_len, not_headless))
        returned_tracks = self.tracks[keepers, :].copy()
        return returned_tracks

    def draw_tracks(self, out, tracks):
        """Visualize tracks all overlayed on a single image.
        Inputs
          out - numpy uint8 image sized HxWx3 upon which tracks are overlayed.
          tracks - M x (2+L) sized matrix storing track info.
        """
        # Store the number of points per camera.
        pts_mem = self.all_pts
        N = len(pts_mem)  # Number of cameras/images.
        # Get offset ids needed to reference into pts_mem.
        offsets = self.get_offsets()
        # Width of track and point circles to be drawn.
        stroke = 1
        # Iterate through each track and draw it.
        for track in tracks:
            clr = myjet[int(np.clip(np.floor(track[1] * 10), 0, 9)), :] * 255
            for i in range(N - 1):
                if track[i + 2] == -1 or track[i + 3] == -1:
                    continue
                offset1 = offsets[i]
                offset2 = offsets[i + 1]
                idx1 = int(track[i + 2] - offset1)
                idx2 = int(track[i + 3] - offset2)
                pt1 = pts_mem[i][:2, idx1]
                pt2 = pts_mem[i + 1][:2, idx2]
                p1 = (int(round(pt1[0])), int(round(pt1[1])))
                p2 = (int(round(pt2[0])), int(round(pt2[1])))
                cv2.line(out, p1, p2, clr, thickness=stroke, lineType=16)
                # Draw end points of each track.
                if i == N - 2:
                    clr2 = (255, 0, 0)
                    cv2.circle(out, p2, stroke, clr2, -1, lineType=16)


class VideoStreamer(object):
    """Class to help process image streams. Three types of possible inputs:"
    1.) USB Webcam.
    2.) A directory of images (files in directory matching 'img_glob').
    3.) A video file, such as an .mp4 or .avi file.
    """

    def __init__(self, basedir, camid, height, width, skip, img_glob):
        self.cap = []
        self.camera = False
        self.video_file = False
        self.listing = []
        self.sizer = [height, width]
        self.i = 0
        self.skip = skip
        self.maxlen = 1000000
        # If the "basedir" string is the word camera, then use a webcam.
        if basedir == "camera/" or basedir == "camera":
            print("==> Processing Webcam Input.")
            self.cap = cv2.VideoCapture(camid)
            self.listing = range(0, self.maxlen)
            self.camera = True
        else:
            # Try to open as a video.
            self.cap = cv2.VideoCapture(basedir)
            lastbit = basedir[-4 : len(basedir)]
            if (type(self.cap) == list or not self.cap.isOpened()) and (
                lastbit == ".mp4"
            ):
                raise IOError("Cannot open movie file")
            elif type(self.cap) != list and self.cap.isOpened() and (lastbit != ".txt"):
                print("==> Processing Video Input.")
                num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.listing = range(0, num_frames)
                self.listing = self.listing[:: self.skip]
                self.camera = True
                self.video_file = True
                self.maxlen = len(self.listing)
            else:
                print("==> Processing Image Directory Input.")
                search = os.path.join(basedir, img_glob)
                self.listing = glob.glob(search)
                self.listing.sort()
                self.listing = self.listing[:: self.skip]
                self.maxlen = len(self.listing)
                if self.maxlen == 0:
                    raise IOError(
                        "No images were found (maybe bad '--img_glob' parameter?)"
                    )

    def read_image(self, impath, img_size):
        """Read image as grayscale and resize to img_size.
        Inputs
          impath: Path to input image.
          img_size: (W, H) tuple specifying resize size.
        Returns
          grayim: float32 numpy array sized H x W with values in range [0, 1].
        """
        grayim = cv2.imread(impath, 0)
        if grayim is None:
            raise Exception("Error reading image %s" % impath)
        # Image is resized via opencv.
        interp = cv2.INTER_AREA
        grayim = cv2.resize(grayim, (img_size[1], img_size[0]), interpolation=interp)
        grayim = grayim.astype("float32") / 255.0
        return grayim

    def next_frame(self):
        """Return the next frame, and increment internal counter.
        Returns
           image: Next H x W image.
           status: True or False depending whether image was loaded.
        """
        if self.i == self.maxlen:
            return (None, False)
        if self.camera:
            ret, input_image = self.cap.read()
            if ret is False:
                print(
                    "VideoStreamer: Cannot get image from camera (maybe bad --camid?)"
                )
                return (None, False)
            if self.video_file:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.listing[self.i])
            input_image = cv2.resize(
                input_image,
                (self.sizer[1], self.sizer[0]),
                interpolation=cv2.INTER_AREA,
            )
            input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
            input_image = input_image.astype("float") / 255.0
        else:
            image_file = self.listing[self.i]
            input_image = self.read_image(image_file, self.sizer)
        # Increment internal counter.
        self.i = self.i + 1
        input_image = input_image.astype("float32")
        return (input_image, True)


if __name__ == "__main__":
    input_path = r"your/input/path"
    path_save = (
        r"your/output/path"
    )

    weights_path = "superpoint_v1.pth"
    img_glob = "*.jpg"
    skip = 1
    show_extra = True
    H = 120
    W = 160
    display_scale = 2
    min_length = 2
    max_length = 5
    nms_dist = 4
    conf_thresh = 0.015
    nn_thresh = 0.7
    camid = 0
    waitkey = 1
    cuda = True
    no_display = False
    write = False
    write_dir = (
        r"your/output/path"
    )

    all_desc = []
    all_pts = []
    all_imgs = []

    # This class helps load input images from different sources.
    vs = VideoStreamer(input_path, camid, H, W, skip, img_glob)

    print("==> Loading pre-trained network.")
    # This class runs the SuperPoint network and processes its outputs.
    fe = SuperPointFrontend(
        weights_path=weights_path,
        nms_dist=nms_dist,
        conf_thresh=conf_thresh,
        nn_thresh=nn_thresh,
        cuda=cuda,
    )
    print("==> Successfully loaded pre-trained network.")

    # This class helps merge consecutive point matches into tracks.
    tracker = PointTracker(max_length, nn_thresh=fe.nn_thresh)

    # Create a window to display the demo.
    if not no_display:
        win = "SuperPoint Tracker"
        cv2.namedWindow(win)
    else:
        print("Skipping visualization, will not show a GUI.")

    # Font parameters for visualizaton.
    font = cv2.FONT_HERSHEY_DUPLEX
    font_clr = (255, 255, 255)
    font_pt = (4, 12)
    font_sc = 0.4

    # Create output directory if desired.
    if write:
        print("==> Will write outputs to %s" % write_dir)
        if not os.path.exists(write_dir):
            os.makedirs(write_dir)

    print("==> Running Demo.")
    while True:
        start = time.time()

        # Get a new image.
        img, status = vs.next_frame()
        if status is False:
            break

        # Get points and descriptors.
        start1 = time.time()
        pts, desc, heatmap = fe.run(img)
        pts_useful = pts[:2, :]
        desc_useful = desc[:, :]
        end1 = time.time()

        all_desc.append(desc_useful.transpose())
        all_pts.append(pts_useful.transpose())
        all_imgs.append(img)

        # Add points and descriptors to the tracker.
        tracker.update(pts, desc)

        # Get tracks for points which were match successfully across all frames.
        tracks = tracker.get_tracks(min_length)

        # Primary output - Show point tracks overlayed on top of input image.
        out1 = (np.dstack((img, img, img)) * 255.0).astype("uint8")
        tracks[:, 1] /= float(fe.nn_thresh)  # Normalize track scores to [0,1].
        tracker.draw_tracks(out1, tracks)
        if show_extra:
            cv2.putText(
                out1, "Point Tracks", font_pt, font, font_sc, font_clr, lineType=16
            )

        # Extra output -- Show current point detections.
        out2 = (np.dstack((img, img, img)) * 255.0).astype("uint8")
        for pt in pts.T:
            pt1 = (int(round(pt[0])), int(round(pt[1])))
            cv2.circle(out2, pt1, 1, (0, 255, 0), -1, lineType=16)
        cv2.putText(
            out2, "Raw Point Detections", font_pt, font, font_sc, font_clr, lineType=16
        )

        # Extra output -- Show the point confidence heatmap.
        if heatmap is not None:
            min_conf = 0.001
            heatmap[heatmap < min_conf] = min_conf
            heatmap = -np.log(heatmap)
            heatmap = (heatmap - heatmap.min()) / (
                heatmap.max() - heatmap.min() + 0.00001
            )
            out3 = myjet[np.round(np.clip(heatmap * 10, 0, 9)).astype("int"), :]
            out3 = (out3 * 255).astype("uint8")
        else:
            out3 = np.zeros_like(out2)
        cv2.putText(
            out3, "Raw Point Confidences", font_pt, font, font_sc, font_clr, lineType=16
        )

        # Resize final output.
        if show_extra:
            out = np.hstack((out1, out2, out3))
            out = cv2.resize(out, (3 * display_scale * W, display_scale * H))
        else:
            out = cv2.resize(out1, (display_scale * W, display_scale * H))

        # Display visualization image to screen.
        if not no_display:
            cv2.imshow(win, out)
            key = cv2.waitKey(waitkey) & 0xFF
            if key == ord("q"):
                print("Quitting, 'q' pressed.")
                break

        # Optionally write images to disk.
        if write:
            out_file = os.path.join(write_dir, "frame_%05d.png" % vs.i)
            print("Writing image to %s" % out_file)
            cv2.imwrite(out_file, out)

        end = time.time()
        net_t = 1.0 / float(end1 - start)
        total_t = 1.0 / float(end - start)
        if show_extra:
            print(
                "Processed image %d (net+post_process: %.2f FPS, total: %.2f FPS)."
                % (vs.i, net_t, total_t)
            )

    # Close any remaining windows.
    cv2.destroyAllWindows()

    matcher = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
    matches = matcher.match(all_desc[0], all_desc[1])
    matches = sorted(matches, key=lambda x: x.distance)
    matches = matches[:]
    p1 = np.array([all_pts[0][matches[i].queryIdx] for i in range(len(matches))])
    p2 = np.array([all_pts[1][matches[i].trainIdx] for i in range(len(matches))])

    H_register1, mask = cv2.findHomography(p2, p1, cv2.RANSAC, 15)

    img_target_transformed = cv2.warpPerspective(
        all_imgs[1], H_register1, (all_imgs[1].shape[1], all_imgs[1].shape[0])
    )
    img_target_transformed = (img_target_transformed * 255).astype("uint8")
    img_reference = (all_imgs[0] * 255).astype("uint8")

    cv2.imwrite(path_save + "/Registered.png", img_target_transformed)
    cv2.imwrite(path_save + "/Reference.png", img_reference)

    prv_rigid = Preview_Registration_Overlay(
        img_base=img_reference,
        img_transformed=img_target_transformed,
        intensity=1,
        path=path_save + "/Rigid_registration_preview.png",
    )

    print("==> Finshed Demo.")
