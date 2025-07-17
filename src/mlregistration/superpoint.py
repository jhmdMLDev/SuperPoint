import argparse
import glob
import os
import time
import warnings
import math

warnings.filterwarnings("ignore")

import cv2
import numpy as np
import torch
from scipy import interpolate

from mlregistration.tps import thin_plate_spline_registration
from mlregistration.demons import demons_registration

# Stub to warn about opencv version.
if int(cv2.__version__[0]) < 3:  # pragma: no cover
    print("Warning: OpenCV 3 is not installed")


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
            desc = torch.nn.functional.grid_sample(
                coarse_desc, samp_pts, align_corners=True
            )
            desc = desc.data.cpu().numpy().reshape(D, -1)
            desc /= np.linalg.norm(desc, axis=0)[np.newaxis, :]

        return pts, desc, heatmap


class ApplySuperPointReg:
    def __init__(
        self,
        ref,
        weights_path=r"C:\Users\Javad\Desktop\Model_Check\Superpoint\superpoint_v1.pth",
        conf_thresh=0.015,
    ):
        self.skip = 1
        self.show_extra = True
        self.H = 512
        self.W = 512
        self.display_scale = 2
        self.min_length = 2
        self.max_length = 5
        self.nms_dist = 4
        self.conf_thresh = conf_thresh
        self.nn_thresh = 0.7
        self.camid = 0
        self.waitkey = 1
        self.cuda = True
        self.no_display = False
        self.write = False
        self.tps = False
        self.postprocessing_tps = False
        self.demons = False
        # self.weights_path = "/efs/model_check2023/Superpoint/superpoint_v1.pth"
        self.weights_path = weights_path
        self.interp = cv2.INTER_AREA

        self.fe = SuperPointFrontend(
            weights_path=self.weights_path,
            nms_dist=self.nms_dist,
            conf_thresh=self.conf_thresh,
            nn_thresh=self.nn_thresh,
            cuda=self.cuda,
        )
        self.reference_image = ref
        self.matcher = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

        self.ref = ref
        # self.ref = cv2.resize(self.ref, (self.W, self.H), interpolation=self.interp)
        self.ref = self.ref.astype("float32") / 255.0
        # write a function to do only registration
        self.ref_pts, self.ref_desc, _ = self.fe.run(self.ref)
        self.ref_pts_useful = (
            self.ref_pts[:2, :].transpose() if self.ref_pts is not None else None
        )
        self.ref_desc_useful = (
            self.ref_desc[:, :].transpose() if self.ref_desc is not None else None
        )

    def set_reference(self, ref):
        self.ref = ref
        self.ref = self.ref.astype("float32") / 255.0
        # write a function to do only registration
        self.ref_pts, self.ref_desc, _ = self.fe.run(self.ref)
        self.ref_pts_useful = (
            self.ref_pts[:2, :].transpose() if self.ref_pts is not None else None
        )
        self.ref_desc_useful = (
            self.ref_desc[:, :].transpose() if self.ref_desc is not None else None
        )

    def homography_check(self, matrix):
        if matrix is None:
            return False
        tx = matrix[0, 2]
        ty = matrix[1, 2]

        # Extract rotation component
        theta = -math.atan2(matrix[0, 1], matrix[0, 0]) * 180 / math.pi

        # Extract scaling/shearing component
        sx = matrix[0, 0] / math.cos(theta * math.pi / 180)
        sy = matrix[1, 1] / math.cos(theta * math.pi / 180)

        condition = (
            (max(abs(tx), abs(ty)) < 512 * math.sqrt(2))
            and (0.5 < max(abs(sx), abs(sy)) < 2)
            and (-30 < theta < 30)
        )

        return True if condition else False

    def add_tps(self):
        self.tps = True
        return self

    def add_postprocessing_tps(self):
        self.postprocessing_tps = True
        return self

    def add_demons(self):
        self.demons = True
        return self

    def __call__(self, trg):
        # Normalize target image to [0, 1]
        trg_original = trg.copy()
        trg = trg.astype("float32") / 255.0
        height, width = trg.shape

        # Extract features from the target image
        trg_pts, trg_desc, _ = self.fe.run(trg)

        # Extract points and descriptors
        trg_pts_useful = trg_pts[:2, :].transpose()
        trg_desc_useful = trg_desc.transpose()

        # Match descriptors
        matches = self.matcher.match(self.ref_desc_useful, trg_desc_useful)
        matches = sorted(matches, key=lambda x: x.distance)[:]

        # Extract matched points
        self.p1 = np.array([self.ref_pts_useful[m.queryIdx] for m in matches])
        self.p2 = np.array([trg_pts_useful[m.trainIdx] for m in matches])

        # Check if there are enough matches
        self.success = len(self.p2) >= 4

        if not self.success:
            self.H_transform = None
            return (trg * 255).astype("uint8")

        if self.tps:
            # Apply TPS transformation
            img_target_transformed = thin_plate_spline_registration(
                self.p1, self.p2, trg_original
            )

        else:
            # Calculate homography
            self.H_transform, self.mask = cv2.findHomography(
                self.p2, self.p1, cv2.RANSAC, 15
            )

            if self.H_transform is None or not self.homography_check(self.H_transform):
                self.success = False
                return (trg * 255).astype("uint8")

            # Apply homography transformation
            img_target_transformed = cv2.warpPerspective(
                trg, self.H_transform, (trg.shape[1], trg.shape[0])
            )

        # Convert back to [0, 255]
        if self.postprocessing_tps:
            points = self.p2.reshape(-1, 1, 2)
            p2_transformed = cv2.transform(points, self.H_transform)
            p2_transformed = p2_transformed.reshape(-1, 3)
            p2_transformed = p2_transformed[:, :2] / p2_transformed[:, 2, np.newaxis]
            p1_filtered, p2_transformed_filtered = filter_outliers(
                self.p1, p2_transformed, threshold_percent=40
            )
            img_target_transformed = thin_plate_spline_registration(
                p1_filtered, p2_transformed_filtered, img_target_transformed
            )

        img_target_transformed = (img_target_transformed * 255).astype("uint8")

        if self.demons:
            img_target_transformed = demons_registration(
                fixed_image_np=trg_original, moving_image_np=img_target_transformed
            )

        return img_target_transformed


def filter_outliers(p1, p2_transformed, threshold_percent=20):
    # Compute Euclidean distances
    distances = np.linalg.norm(p1 - p2_transformed, axis=1)

    # Determine the distance threshold
    threshold = np.percentile(distances, 100 - threshold_percent)

    # Filter points where the distance is less than or equal to the threshold
    mask = distances <= threshold
    p1_filtered = p1[mask]
    p2_transformed_filtered = p2_transformed[mask]

    return p1_filtered, p2_transformed_filtered
