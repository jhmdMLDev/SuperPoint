import math
import os
import time
from collections import OrderedDict

import cv2
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import torch
from skimage.morphology import skeletonize
from skimage.transform import PiecewiseAffineTransform, warp
from torch import nn

# AKAZE parameters
weight_AKAZE = 0  # 0
nOctaveLayers = 32

# HOG parameters
"""weight_HOG = 1e-4  # 1e-4
HOG_window_size = (64,64)
winStride = (32, 32)
padding = (48, 48)
HOG_block_size = (32, 32)
HOG_block_stride = (32, 32)
HOG_cell_size = (32, 32)"""

weight_HOG = 1e-4  # 1e-4
HOG_window_size = (32, 32)
winStride = (32, 32)
padding = (48, 48)
HOG_block_size = (16, 16)
HOG_block_stride = (16, 16)
HOG_cell_size = (8, 8)

# Matched keypoint maximum distance (relative to image scale)
match_distance_filter_factor = 0.18  # 0.18

# SIFT parameters
weight_SIFT = 10e-1  # 10e-1
SIFT_contrastThreshold = -0.005
SIFT_sigma = 3
SIFT_edgeThreshold = -0.015

# ORB parameters
weight_ORB = 2e-1  # 2e-1
ORB_count = 2000
ORB_patchSize = 100
ORB_nlevels = 1
ORB_edgeThreshold = 8


HOG_params = {
    "HOG_window_size": HOG_window_size,
    "HOG_block_size": HOG_block_size,
    "HOG_block_stride": HOG_block_stride,
    "HOG_cell_size": HOG_cell_size,
}
SIFT_params = {
    "SIFT_contrastThreshold": SIFT_contrastThreshold,
    "SIFT_sigma": SIFT_sigma,
    "SIFT_edgeThreshold": SIFT_edgeThreshold,
}
ORB_params = {
    "ORB_count": ORB_count,
    "ORB_patchSize": ORB_patchSize,
    "ORB_nlevels": ORB_nlevels,
    "ORB_edgeThreshold": ORB_edgeThreshold,
}


def Crop_Image_Homography(window_size, window_center):
    """
    Crop an image to a specified window size and center
    :param window_size: (W2,H2) window size in pixels
    :type window_size: numpy array [2]
    :param window_center: (X,Y) coordinate of window center
    :type window_center: numpy array [2]
    :return: cropped image
    :rtype: numpy array [W2,H2]
    """

    img_start_idx = [
        int(window_center[0] - window_size[1] / 2),
        int(window_center[1] - window_size[0] / 2),
    ]
    img_end_idx = [
        int(window_center[0] + window_size[1] / 2),
        int(window_center[1] + window_size[0] / 2),
    ]

    # bottom left corner coordinates before and after transform
    bot_left_corner = [(img_start_idx[0], img_start_idx[1]), (0, 0)]
    # top right corner coordinates before and after transform
    top_right_corner = [
        (img_end_idx[0], img_end_idx[1]),
        (window_size[1], window_size[0]),
    ]

    # generate source destination points list
    src_pts = []
    dst_pts = []
    src_pts.append((bot_left_corner[0][0], bot_left_corner[0][1]))  # bottom left
    src_pts.append((top_right_corner[0][0], top_right_corner[0][1]))  # top right
    src_pts.append((bot_left_corner[0][0], top_right_corner[0][1]))  # top left
    src_pts.append((top_right_corner[0][0], bot_left_corner[0][1]))  # bottom right

    dst_pts.append((bot_left_corner[1][0], bot_left_corner[1][1]))  # bottom left
    dst_pts.append((top_right_corner[1][0], top_right_corner[1][1]))  # top right
    dst_pts.append((bot_left_corner[1][0], top_right_corner[1][1]))  # top left
    dst_pts.append((top_right_corner[1][0], bot_left_corner[1][1]))  # bottom right

    src_pts = np.float32(src_pts)
    dst_pts = np.float32(dst_pts)
    # calculate homography matrix
    H, mask = cv2.findHomography(src_pts, dst_pts)
    return H


def gradient_summer(G):
    kernel = np.ones((3, 3))
    return cv2.filter2D(src=G, ddepth=-1, kernel=kernel, borderType=cv2.BORDER_CONSTANT)


def Position_Correlation_Score(
    img_1,
    img_2,
    offset_sweep_range=100,
    offset_sweep_steps=8,
    start_pos=None,
    SLO_frame=False,
):
    """
    Sweep img_2 across img_1, calculating the correlation at each offset step. Find the 2D gradient of the correlations
    at the maximum score offset location to calculate an overall correlation score at the optimal offset.
    :param img_1: base image
    :type img_1: numpy array [W1, H1]
    :param img_2: image to be stepped across img_1
    :type img_2: numpy array [W2, H2]
    :param offset_sweep_range: range (in pixels) of alignment offset sweep (default: 100)
    :type offset_sweep_range: int
    :param offset_sweep_steps: number of alignment offset sweep steps (default: 8)
    :type offset_sweep_steps: int
    :param start_pos: (X,Y) initial position guess (default=None, use center of reference image).
    :type start_pos: numpy array [2]
    :return: score: correlation score (float)
        best_offset: (x,y) position of best offset from the start position
    :rtype:
    """
    if start_pos is None:
        start_pos = (img_1.shape[1] / 2, img_1.shape[0] / 2)
    if offset_sweep_range == 0:
        offset_sweep_range = 0.001
    x_sweep = range(
        -offset_sweep_range, offset_sweep_range, offset_sweep_steps
    )  # range of x-offsets to try
    y_sweep = range(
        -offset_sweep_range, offset_sweep_range, offset_sweep_steps
    )  # range of y-offsets to try

    conv = np.zeros(
        (len(x_sweep), len(y_sweep))
    )  # matrix for storing convolution values
    for ix in range(len(x_sweep)):
        for iy in range(len(y_sweep)):
            # Crop img_1 down to the same window as the where img_b would be after scaling and offsetting
            offset = (start_pos[0] + x_sweep[ix], start_pos[1] + y_sweep[iy])
            img_1_cropped = Crop_Image_Window(
                img=img_1, window_size=img_2.shape, window_center=offset
            )
            # multiply the cropped img_1 with scaled img_b and then sum the result to obtain the correlation
            overlap = np.sum(np.multiply(img_1_cropped, img_2))
            # calculate score by normalizing overlap with scaled img_b size
            conv[ix, iy] = (
                overlap / np.power(img_2.shape[0] * img_2.shape[1], 0.7)
                if not SLO_frame
                else overlap / np.power(img_2.shape[0] * img_2.shape[1], 0.33)
            )

    # calculate the 2D spatial gradient
    gx, gy = np.gradient(conv)
    G = np.abs(gx) + np.abs(gy)
    # Calculate the gradient value (averaged around the max convolution pixel)
    # G_score_multiplier = np.sum(G[(offset_xm - 1):(offset_xm + 1), (offset_ym - 1):(offset_ym + 1)])
    G_score_multiplier = gradient_summer(G)
    # score value for this scaling factor weighted by the gradient magnitude
    scores = conv * G_score_multiplier if not SLO_frame else conv
    score = np.max(scores)
    # find maximum convolution value across all candidate offsets for this scaling factor
    offset_xm, offset_ym = np.unravel_index(scores.argmax(), conv.shape)
    best_offset = (start_pos[0] + x_sweep[offset_xm], start_pos[1] + y_sweep[offset_ym])
    return score, best_offset


def Initial_Align_Images(
    img_FA_vein,
    img_SLO_vein,
    subscaling_factor=2,
    scale_sweep_range=0.2,
    scale_sweep_steps=32,
    scale_AR_sweep_range=0.30,
    scale_AR_sweep_steps=30,
    offset_sweep_range=120,
    offset_sweep_steps=6,
    start_scale=0.4,
    start_pos=None,
    threads=0,
    SLO_frame=False,
):
    """
    Determing optimal alignment scale and offset for overlapping SLO onto FA based on vein overlap correlation.
    :param img_FA_vein: binary image containing segmented veins from FA image (larger image)
    :type img_FA_vein: numpy array [W1, H1]
    :param img_SLO_vein: binary image containing segmented veins from SLO image (smaller image)
    :type img_SLO_vein: numpy array [W2, H2]
    :param subscaling_factor: factor to downscale both image arrays for faster processing.
     Higher number = more downscaling
    :type subscaling_factor: integer
    :param scale_sweep_range: range of scale multipliers to try (default: 0.2)
    :type scale_sweep_range: float
    :param scale_sweep_steps: number of scale steps (default: 10)
    :type scale_sweep_steps: int
    :param scale_AR_sweep_range: range of scale aspect ratio multipliers to try (default: 0.2)
    :type scale_AR_sweep_range: float
    :param scale_AR_sweep_steps: number of scale aspect ratio steps (default: 5)
    :type scale_AR_sweep_steps: int
    :param offset_sweep_range: range (in pixels) of alignment offset sweep (default: 100)
    :type offset_sweep_range: int
    :param offset_sweep_steps: number of alignment offset sweep steps (default: 8)
    :type offset_sweep_steps: int
    :param start_scale: initial scale guess (default=0.5) assumed to be square, relative to original image scale
    :type start_scale: float
    :param start_pos: (X,Y) initial position guess (default=None).
     If 'None', use the center of the img_FA frame as initial
    :type start_pos: numpy array [2]
    :return: alignment_scale: scaling factor (0->1) for SLO alignment
        alignment_offset: center coordinates (in pixels) for aligning scaled SLO onto FA frame
        best_alignment_score: correlation score for alignment output
    :rtype:
    """

    t_start = time.time()

    # Apply downscaling
    img_a = cv2.resize(
        img_FA_vein,
        (
            int(img_FA_vein.shape[1] / subscaling_factor),
            int(img_FA_vein.shape[0] / subscaling_factor),
        ),
    )
    img_b = cv2.resize(
        img_SLO_vein,
        (
            int(img_SLO_vein.shape[1] / subscaling_factor),
            int(img_SLO_vein.shape[0] / subscaling_factor),
        ),
    )
    img_a = img_a / np.max(img_a)
    img_b = img_b / np.max(img_b)

    if start_pos is None:
        start_pos = (img_a.shape[1] / 2, img_a.shape[0] / 2)

    ## Manual scale sweep:
    scale_score = []
    scale_track = []
    pos_track = []
    scale_sweep = start_scale + np.linspace(
        -scale_sweep_range, scale_sweep_range, scale_sweep_steps
    )  # range of scales to try

    # iterate over possible square scale values

    # If enabled, use multithreading
    if threads > 0:
        import threading

        def worker(scales, results_temp):
            for i in range(len(scales)):
                img_b_scale = scales[i]  # candidate scale

                # generate scaled img_b
                img_b_scaled = cv2.resize(
                    img_b,
                    (
                        int(img_b.shape[1] * img_b_scale[1]),
                        int(img_b.shape[0] * img_b_scale[0]),
                    ),
                )
                if not SLO_frame:
                    img_b_scaled = (img_b_scaled > 0.01) * 1

                # Sweep offset to find the correlation score and highest scoring offset
                corr_score, best_offset = Position_Correlation_Score(
                    img_1=img_a,
                    img_2=img_b_scaled,
                    offset_sweep_range=offset_sweep_range,
                    offset_sweep_steps=offset_sweep_steps,
                    start_pos=start_pos,
                    SLO_frame=SLO_frame,
                )

                # store the score value for this scaling factor weighted by the gradient magnitude
                # Larger gradient = faster dropoff of overlap when shifting img_b = more likely that this is a good vein match
                # scale_score.append(corr_score)
                # scale_track.append(img_b_scale)
                # pos_track.append(best_offset)
                results_temp.append([corr_score, img_b_scale, best_offset])

        t = []
        t_res = []
        test_scales = []
        for i_scale in range(len(scale_sweep)):
            scale_ar_sweep = np.linspace(
                scale_sweep[i_scale] * (1 - scale_AR_sweep_range),
                scale_sweep[i_scale] * (1 + scale_AR_sweep_range),
                scale_AR_sweep_steps,
            )
            for i_ar in range(len(scale_ar_sweep)):
                test_scales.append((scale_sweep[i_scale], scale_ar_sweep[i_ar]))
        test_scales_split = np.array_split(test_scales, threads)

        # Measure timing and total loop for the worker
        t_worker = 0
        k = 0
        for i in range(threads):
            t_res.append([])

            # Measuring time
            t_worker_start = time.time()
            t_new = threading.Thread(
                target=worker, args=(test_scales_split[i], t_res[i])
            )

            # print('Position_Correlation_Score function without multi threading timing: {:.4f}'.format(t_worker_end-t_worker_start))

            t_new.start()
            t.append(t_new)

            # Measuring time
            t_worker_end = time.time()
            t_worker += t_worker_end - t_worker_start
            k += 1

        for n, thread in enumerate(t):
            thread.join()

        # This is another simple way of multi threading but takes more time than the harder approach
        # with concurrent.futures.ThreadPoolExecutor(max_workers=threads + 2) as executor:
        #     executor.map(worker, (test_scales_split[0], test_scales_split[1], test_scales_split[2]), # , test_scales_split[2]
        #                     (t_res[0], t_res[1], t_res[2])) # , t_res[2]

        for i in range(threads):
            for j in range(len(t_res[i])):
                scale_score.append(t_res[i][j][0])
                scale_track.append(t_res[i][j][1])
                pos_track.append(t_res[i][j][2])
    else:
        # Measure timing and total loops
        t_PCS = 0
        k = 0
        for i_scale in range(len(scale_sweep)):
            scale_ar_sweep = np.linspace(
                scale_sweep[i_scale] * (1 - scale_AR_sweep_range),
                scale_sweep[i_scale] * (1 + scale_AR_sweep_range),
                scale_AR_sweep_steps,
            )

            for i_ar in range(len(scale_ar_sweep)):
                img_b_scale = (
                    scale_sweep[i_scale],
                    scale_ar_sweep[i_ar],
                )  # candidate scale

                # generate scaled img_b
                img_b_scaled = cv2.resize(
                    img_b,
                    (
                        int(img_b.shape[1] * img_b_scale[1]),
                        int(img_b.shape[0] * img_b_scale[0]),
                    ),
                )
                if not SLO_frame:
                    img_b_scaled = (img_b_scaled > 0.01) * 1

                # the generated scaled image undergoes vein connection
                # connector = Vein_conncetor(img_b_scaled, l_max=8, l_min=7, l_step=2, theta_class=[1,2,3])
                # img_b_scaled = connector.run()

                # Sweep offset to find the correlation score and highest scoring offset
                t_PCS_start = time.time()
                # if k==8:
                #     print('afnsv')
                corr_score, best_offset = Position_Correlation_Score(
                    img_1=img_a,
                    img_2=img_b_scaled,
                    offset_sweep_range=offset_sweep_range,
                    offset_sweep_steps=offset_sweep_steps,
                    start_pos=start_pos,
                    SLO_frame=SLO_frame,
                )
                t_PCS_end = time.time()
                t_PCS += t_PCS_end - t_PCS_start
                k += 1
                # print('Position_Correlation_Score function without multi threading timing: {:.4f}'.format(t_PCS_end-t_PCS_start))
                # store the score value for this scaling factor weighted by the gradient magnitude
                # Larger gradient = faster dropoff of overlap when shifting img_b = more likely that this is a good vein match
                scale_score.append(corr_score)
                scale_track.append(img_b_scale)
                pos_track.append(best_offset)

                """
                img_b_framed, img_b_scaled = Reframe_Image(img=img_a, new_scale=img_b_scale, new_offset=best_offset,
                                                           frame_size=img_a.shape)
                plt.imshow(img_a, cmap='gray')
                plt.imshow(img_b_framed, cmap='hot', alpha=0.5)
                plt.title("score: {:.2f} (best={:.2f}), scale: {}".format(corr_score, np.array(scale_score).max(), img_b_scale))
                plt.show()
                """
        # print("[Initial_Align_Images] Scale {}: {},  Best Score: {}".format(i_scale, scale_sweep[i_scale], np.max(conv)))

    # Find best alignment scale and offset
    nmax = np.argmax(np.array(scale_score))
    best_scale = scale_track[nmax]
    best_offset = pos_track[nmax]
    print(
        "[Initial_Align_Images] Best alignment (score={}): scale={}, offset={}".format(
            np.max(np.array(scale_score)), best_scale, best_offset
        )
    )

    t_end = time.time()
    if threads > 0:
        print(
            "Whole Initial alignment with multi threading timing: {:.4f}".format(
                t_end - t_start
            )
        )
        # print('Average worker function with multi threading timing: {:.4f}'.format(t_worker/k))
    else:
        print(
            "Whole Initial alignment without multi thread timing: {:.4f}".format(
                t_end - t_start
            )
        )
        print(
            "Average Position_Correlation_Score function without multi threading timing: {:.4f}".format(
                t_PCS / k
            )
        )

    return (
        best_scale,
        np.array(best_offset) * subscaling_factor,
        np.max(np.array(scale_score)) * subscaling_factor,
    )


def Filter_Distant_Matches(
    matches, distance_threshold, kp1=None, kp2=None, pt1=None, pt2=None
):
    """
    Steps through matches and remove matches between landmarks which have coordinate distances greater than
    'ditance_threshold'
    :param kp1: image 1 landmarks
    :param kp2: image 2 landmarks
    :param matches: list of matches
    :param distance_threshold: threshold (in pixels) for the maximum coordinate difference
    :return: filtered list of matches
    """
    matches_dst_filter = []
    for m in matches:
        if pt1 is None:
            coord1 = kp1[m.queryIdx].pt
            coord2 = kp2[m.trainIdx].pt
        else:
            coord1 = pt1[m.queryIdx]
            coord2 = pt2[m.trainIdx]
        dst = np.sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2)
        if dst < distance_threshold:
            matches_dst_filter.append(m)
    return matches_dst_filter


def configure_HOG(
    winSize=(64, 64),
    blockSize=(16, 16),
    blockStride=(8, 8),
    cellSize=(8, 8),
    nbins=9,
    derivAperture=1,
    winSigma=4.0,
    histogramNormType=0,
    L2HysThreshold=2.0e-01,
    gammaCorrection=0,
    nlevels=64,
):
    """
    Initialize HOG description computer
    :param winSize: (see HOG documentation)
    :param blockSize: (see HOG documentation)
    :param blockStride: (see HOG documentation)
    :param cellSize: (see HOG documentation)
    :param nbins: (see HOG documentation)
    :param derivAperture: (see HOG documentation)
    :param winSigma: (see HOG documentation)
    :param histogramNormType: (see HOG documentation)
    :param L2HysThreshold: (see HOG documentation)
    :param gammaCorrection: (see HOG documentation)
    :param nlevels: (see HOG documentation)
    :return: hog object (cv2.HOGDescriptor)
    """
    hog = cv2.HOGDescriptor(
        winSize,
        blockSize,
        blockStride,
        cellSize,
        nbins,
        derivAperture,
        winSigma,
        histogramNormType,
        L2HysThreshold,
        gammaCorrection,
        nlevels,
    )
    return hog


def HOG_Discription(
    img,
    kp=None,
    pt=None,
    window_size=None,
    block_size=None,
    block_stride=None,
    cell_size=None,
):
    """
    Generate HOG (histogram) description of features
    :param img:
    :param pt:
    :param HOG_block_size:
    :param HOG_block_stride:
    :param HOG_cell_size:
    :return: list of feature matrix
    """
    if pt is None:
        pt = []
        for p in kp:
            pt.append((p.pt[0], p.pt[1]))

    hog = configure_HOG(
        winSize=window_size,
        blockSize=block_size,
        blockStride=block_stride,
        cellSize=cell_size,
        winSigma=2.0,
        nlevels=64,
    )

    hist_size = len(
        hog.compute(img, winStride, padding, (np.array(pt[0]).astype("int"),))
    )

    d_hog = hog.compute(
        img=img,
        winStride=winStride,
        padding=padding,
        locations=np.array(pt).astype("int"),
    ).reshape(len(pt), hist_size)
    return d_hog


def FAST_FeatureDetection(img):
    """
    Detect feature coordinates using FAST algorithm (good features to detect)
    :param img:
    :return: list of detected feature coordinates
    """
    fast = cv2.FastFeatureDetector_create()
    kp = fast.detect(img, None)
    return kp


def Location_Discriptor(img_ref, kp1, weight=1):
    d_loc = []
    for p in kp1:
        d_loc.append(np.array(p.pt)[np.newaxis])
    d_loc = np.concatenate(d_loc, 0) * weight
    return d_loc.astype("float32")


def AKAZE_HOG_Matching(
    img_ref, img_target, nOctaveLayers, match_distance_filter_factor
):
    AKAZE = cv2.AKAZE_create(nOctaveLayers=nOctaveLayers)
    KAZE = cv2.KAZE_create(extended=True, nOctaveLayers=7)
    kp1, d1 = AKAZE.detectAndCompute(img_ref, None)
    kp2, d2 = AKAZE.detectAndCompute(img_target, None)

    if len(kp1) < 3 or len(kp2) < 3:
        print(
            "Warning: (AKAZE_HOG_Matching) Fewer than 3 landmarks for AKAZE. Skipping."
        )
        return kp1, kp2, []

    d_loc1 = Location_Discriptor(img_ref, kp1, weight=0)
    d_loc2 = Location_Discriptor(img_target, kp2, weight=0)

    d1 = np.concatenate((d1, d_loc1), 1)
    d2 = np.concatenate((d2, d_loc2), 1)

    matcher = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
    matches = matcher.match(d1, d2)
    # print("# of total matches: {}".format(len(matches)))
    # matches.sort(key=lambda x: x.distance)
    matches = sorted(matches, key=lambda x: x.distance)
    matches = matches[:105]

    """    f, ax = plt.subplots(1, 2)
            ax[0].imshow(img_ref, cmap='gray')
            ax[1].imshow(img_target, cmap='gray')
            ax[0].scatter(kp1[matches[0].queryIdx].pt[0], kp1[matches[0].queryIdx].pt[1], c='r')
            ax[1].scatter(kp2[matches[0].trainIdx].pt[0], kp2[matches[0].trainIdx].pt[1], c='r')
            plt.show()
    """
    matches = Filter_Distant_Matches(
        kp1=kp1,
        kp2=kp2,
        matches=matches,
        distance_threshold=match_distance_filter_factor * img_ref.shape[0],
    )

    # Preview_Matches(img1=img_ref, img2=img_target, kp1=kp1, kp2=kp2, matches=matches,
    #                path=path + "matches_AKAZE.png")

    return kp1, kp2, matches


def FAST_HOG_Matching(img_ref, img_target, HOG_params, match_distance_filter_factor):
    # obtain landmark coordinates
    kp1 = FAST_FeatureDetection(img_ref)
    kp2 = FAST_FeatureDetection(img_target)

    if len(kp1) < 3 or len(kp2) < 3:
        print(
            "Warning: (FAST_HOG_Matching) Fewer than 3 landmarks for SIFT/HOG. Skipping."
        )
        return kp1, kp2, []

    # Compute HOG Descriptors
    d_hog1 = HOG_Discription(
        img_ref,
        kp1,
        window_size=HOG_params["HOG_window_size"],
        block_size=HOG_params["HOG_block_size"],
        block_stride=HOG_params["HOG_block_stride"],
        cell_size=HOG_params["HOG_cell_size"],
    )
    d_hog2 = HOG_Discription(
        img_target,
        kp2,
        window_size=HOG_params["HOG_window_size"],
        block_size=HOG_params["HOG_block_size"],
        block_stride=HOG_params["HOG_block_stride"],
        cell_size=HOG_params["HOG_cell_size"],
    )
    d_loc1 = Location_Discriptor(img_ref, kp1, weight=weight_HOG)
    d_loc2 = Location_Discriptor(img_target, kp2, weight=weight_HOG)

    d1 = np.concatenate((d_hog1, d_loc1), 1)
    d2 = np.concatenate((d_hog2, d_loc2), 1)

    # Perform Matching
    matcher = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
    matches = matcher.match(d1, d2)
    # print("# of total matches: {}".format(len(matches)))
    # matches.sort(key=lambda x: x.distance)
    matches = sorted(matches, key=lambda x: x.distance)
    matches = matches[:100]

    matches = Filter_Distant_Matches(
        kp1=kp1,
        kp2=kp2,
        matches=matches,
        distance_threshold=match_distance_filter_factor * img_ref.shape[0],
    )

    return kp1, kp2, matches


def SIFT_Matching(img_ref, img_target, SIFT_params, match_distance_filter_factor):
    # Get Sift Landmarks+Descriptors
    sift = cv2.SIFT_create(
        contrastThreshold=SIFT_params["SIFT_contrastThreshold"],
        sigma=SIFT_params["SIFT_sigma"],
        edgeThreshold=SIFT_params["SIFT_edgeThreshold"],
    )
    kp1, d1 = sift.detectAndCompute(img_ref, None)
    kp2, d2 = sift.detectAndCompute(img_target, None)
    if d1 is None or d2 is None:
        print(
            "Warning: (Registration_src) Fewer than 3 landmarks for SIFT/SIFT. Skipping."
        )
        return kp1, kp2, []
    else:
        d1 = (d1 * 255.0).astype("uint8")
        d2 = (d2 * 255.0).astype("uint8")
        d_loc1 = Location_Discriptor(img_ref, kp1, weight=weight_SIFT)  # 10
        d_loc2 = Location_Discriptor(img_target, kp2, weight=weight_SIFT)  # 10

        d1 = np.concatenate((d1, d_loc1), 1)
        d2 = np.concatenate((d2, d_loc2), 1)

        # Perform Matching
        matcher = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
        matches = matcher.match(d1, d2)
        # print("# of total matches: {}".format(len(matches)))
        # matches.sort(key=lambda x: x.distance)
        matches = sorted(matches, key=lambda x: x.distance)
        matches = matches[:100]

        matches = Filter_Distant_Matches(
            kp1=kp1,
            kp2=kp2,
            matches=matches,
            distance_threshold=match_distance_filter_factor * img_ref.shape[0],
        )
    return kp1, kp2, matches


def ORB_Matching(img_ref, img_target, ORB_params, match_distance_filter_factor):
    # Get ORB Landmarks+Descriptors
    orb_detector = cv2.ORB_create(
        ORB_params["ORB_count"],
        patchSize=ORB_params["ORB_patchSize"],
        nlevels=ORB_params["ORB_nlevels"],
        edgeThreshold=ORB_params["ORB_edgeThreshold"],
        scoreType=cv2.ORB_FAST_SCORE,
    )
    kp1, d1 = orb_detector.detectAndCompute(img_ref, None)
    kp2, d2 = orb_detector.detectAndCompute(img_target, None)
    # Perform Matching
    if d1 is not None and d2 is not None:
        d_loc1 = Location_Discriptor(img_ref, kp1, weight=weight_ORB)  # 2
        d_loc2 = Location_Discriptor(img_target, kp2, weight=weight_ORB)  # 2

        d1 = np.concatenate((d1, d_loc1), 1)
        d2 = np.concatenate((d2, d_loc2), 1)

        matcher = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
        matches = matcher.match(d1, d2)
        # matches.sort(key=lambda x: x.distance)
        matches = sorted(matches, key=lambda x: x.distance)
        matches = matches[:100]
    else:
        print(
            "Warning: ORB landmark search failed. ref image landmarks is None: {},"
            " target image landmarks is None: {}".format(d1 is None, d2 is None)
        )
        return kp1, kp2, []
    if len(matches) >= 10:
        matches = Filter_Distant_Matches(
            kp1=kp1,
            kp2=kp2,
            matches=matches,
            distance_threshold=match_distance_filter_factor * img_ref.shape[0],
        )
    return kp1, kp2, matches


def matchpoint_filter(
    img1, img2, points1, points2, kernel_size=11, angle_th=15, type="fa_slo"
):
    if len(img1.shape) == 3:
        img1 = img1[:, :, 0]
    if len(img2.shape) == 3:
        img2 = img2[:, :, 0]

    if type == "slo_slo":
        img1 = (skeletonize(img1 > 127) * 255).astype("uint8")
        img2 = (skeletonize(img2 > 127) * 255).astype("uint8")
        kernel_size = 21

    filtered_points1 = []
    filtered_points2 = []
    circles1 = np.zeros((img1.shape[0], img1.shape[1], 3))
    circles2 = np.zeros((img1.shape[0], img1.shape[1], 3))
    circles1[:, :, 0] = img1
    circles1[:, :, 1] = img1
    circles1[:, :, 2] = img1
    circles2[:, :, 0] = img2
    circles2[:, :, 1] = img2
    circles2[:, :, 2] = img2
    i = 1
    for p1, p2 in zip(points1, points2):
        cv2.circle(circles1, p1.astype("int"), 3, [255, 0, 0], 1)
        # cv2.putText(circles1, str(i), p1.astype('int'), cv2.FONT_HERSHEY_SIMPLEX, 1, [255,0,0])
        cv2.circle(circles2, p2.astype("int"), 3, [255, 0, 0], 1)
        # cv2.putText(circles2, str(i), p2.astype('int'), cv2.FONT_HERSHEY_SIMPLEX, 1, [255,0,0])
        i += 1

        kernel1 = img1[
            int(p1[1]) - kernel_size // 2 : int(p1[1]) + kernel_size // 2,
            int(p1[0]) - kernel_size // 2 : int(p1[0]) + kernel_size // 2,
        ]
        kernel2 = img2[
            int(p2[1]) - kernel_size // 2 : int(p2[1]) + kernel_size // 2,
            int(p2[0]) - kernel_size // 2 : int(p2[0]) + kernel_size // 2,
        ]

        line_points1 = np.argwhere(kernel1.transpose() == 255)
        line_points2 = np.argwhere(kernel2.transpose() == 255)

        if len(line_points1) < 2 and len(line_points2) < 2:
            filtered_points1.append(p1)
            filtered_points2.append(p2)
            continue
        elif (len(line_points1) >= 2 and len(line_points2) < 2) or (
            len(line_points1) < 2 and len(line_points2) >= 2
        ):
            continue

        vx1, vy1, x1, y1 = cv2.fitLine(line_points1, cv2.DIST_L2, 0, 0.01, 0.01)
        vx2, vy2, x2, y2 = cv2.fitLine(line_points2, cv2.DIST_L2, 0, 0.01, 0.01)

        angle1 = (180 / math.pi) * math.atan(vy1 / vx1)
        angle2 = (180 / math.pi) * math.atan(vy2 / vx2)

        cv2.putText(
            circles1,
            str(int(angle1)),
            p1.astype("int"),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            [255, 0, 0],
        )
        cv2.putText(
            circles2,
            str(int(angle2)),
            p2.astype("int"),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            [255, 0, 0],
        )

        if np.abs(angle1) - angle_th < np.abs(angle2) < np.abs(angle1) + angle_th:
            filtered_points1.append(p1)
            filtered_points2.append(p2)
        else:
            pass

    return np.array(filtered_points1), np.array(filtered_points2)


def Landmark_Description_Set(img_ref, img_target, path=None, type="fa_slo"):
    # ShiTomasi_HOG_Matching(img_ref, img_target)
    kp1_AKAZE, kp2_AKAZE, matches_AKAZE = AKAZE_HOG_Matching(
        img_ref, img_target, nOctaveLayers, match_distance_filter_factor
    )
    kp1_FAST, kp2_FAST, matches_FH = FAST_HOG_Matching(
        img_ref, img_target, HOG_params, match_distance_filter_factor
    )
    kp1_SIFT, kp2_SIFT, matches_SIFT = SIFT_Matching(
        img_ref, img_target, SIFT_params, match_distance_filter_factor
    )
    kp1_ORB, kp2_ORB, matches_ORB = ORB_Matching(
        img_ref, img_target, ORB_params, match_distance_filter_factor
    )

    p1 = []
    p2 = []
    if matches_FH is not None:
        for i in range(len(matches_FH)):
            p1.append(kp1_FAST[matches_FH[i].queryIdx].pt)
            p2.append(kp2_FAST[matches_FH[i].trainIdx].pt)

    if matches_SIFT is not None:
        for i in range(len(matches_SIFT)):
            p1.append(kp1_SIFT[matches_SIFT[i].queryIdx].pt)
            p2.append(kp2_SIFT[matches_SIFT[i].trainIdx].pt)
    if matches_ORB is not None:
        for i in range(len(matches_ORB)):
            p1.append(kp1_ORB[matches_ORB[i].queryIdx].pt)
            p2.append(kp2_ORB[matches_ORB[i].trainIdx].pt)
    if matches_AKAZE is not None:
        for i in range(len(matches_AKAZE)):
            p1.append(kp1_AKAZE[matches_AKAZE[i].queryIdx].pt)
            p2.append(kp2_AKAZE[matches_AKAZE[i].trainIdx].pt)

    p1 = np.array(p1)
    p2 = np.array(p2)

    if type == "fa_slo" or type == "slo_slo":
        p1, p2 = matchpoint_filter(img_ref, img_target, p1, p2, type=type)
    return p1, p2


def Apply_Mesh_Transform(img, src, dst, c_fill=0.0, subscaling_factor=1):
    """
    Apply the mesh morph transformation to image 'img'
    :param img:
    :type img:
    :param src:
    :type src:
    :param dst:
    :type dst:
    :return:
    :rtype:
    """

    shape_orig = img.shape
    if subscaling_factor > 1:
        img2 = cv2.dilate(np.copy(img), np.ones((2, 2), np.uint8), iterations=1)
        img2 = cv2.resize(
            img2,
            (
                int(img.shape[1] / subscaling_factor),
                int(img.shape[0] / subscaling_factor),
            ),
        )

    tform = PiecewiseAffineTransform()
    tform.estimate(src, dst)
    img_morphed = warp(img, tform, output_shape=img.shape, cval=c_fill)

    if subscaling_factor > 1:
        img_morphed = cv2.resize(img_morphed, shape_orig)

    return img_morphed


def apply_deformable(fixed, moving, outTx):
    fixed = sitk.GetImageFromArray(fixed)
    moving = sitk.GetImageFromArray(moving)

    fixed = sitk.Cast(fixed, sitk.sitkFloat32)
    moving = sitk.Cast(moving, sitk.sitkFloat32)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(100)
    resampler.SetTransform(outTx)

    out = resampler.Execute(moving)

    return sitk.GetArrayFromImage(out).astype("uint8")


def deformable1_stik(fixed, moving, n_iter=50, sigma=1.0, path=None):
    # Casting images to the sitk format
    fixed = sitk.GetImageFromArray(fixed)
    moving = sitk.GetImageFromArray(moving)

    fixed = sitk.Cast(fixed, sitk.sitkFloat32)
    moving = sitk.Cast(moving, sitk.sitkFloat32)

    # The basic Demons Registration Filter
    # Note there is a whole family of Demons Registration algorithms included in
    # SimpleITK
    demons = sitk.DemonsRegistrationFilter()
    demons.SetNumberOfIterations(n_iter)
    # Standard deviation for Gaussian smoothing of displacement field
    demons.SetStandardDeviations(sigma)

    # demons.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(demons))

    displacementField = demons.Execute(fixed, moving)

    print("-------")
    print(f"Number Of Iterations: {demons.GetElapsedIterations()}")
    print(f"RMS: {demons.GetRMSChange()}")

    outTx = sitk.DisplacementFieldTransform(displacementField)

    if path is not None:
        sitk.WriteTransform(outTx, path)

    if "SITK_NOSHOW" not in os.environ:
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(fixed)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(100)
        resampler.SetTransform(outTx)

        out = resampler.Execute(moving)

    return sitk.GetArrayFromImage(out).astype("uint8"), outTx


def Check_New_Registartion(img_ref, img_target1, img_target2, th=1.05):
    img_ref = img_ref / np.max(img_ref)
    img_target1 = img_target1 / np.max(img_target1)
    img_target2 = img_target2 / np.max(img_target2)
    if np.sum(img_ref * img_target2) > np.sum(img_ref * img_target1) * th:
        return 1
    else:
        return 0


def empty(img):
    return img


def Crop_Image_Window(img, window_size, window_center):
    """
    Crop an image to a specified window size and center
    :param img: image array
    :type img: numpy array [W,H]
    :param window_size: (W2,H2) window size in pixels
    :type window_size: numpy array [2]
    :param window_center: (X,Y) coordinate of window center
    :type window_center: numpy array [2]
    :return: cropped image
    :rtype: numpy array [W2,H2]
    """
    img_start_idx = [
        int(window_center[0] - window_size[1] / 2),
        int(window_center[1] - window_size[0] / 2),
    ]
    img_end_idx = [
        int(window_center[0] + window_size[1] / 2),
        int(window_center[1] + window_size[0] / 2),
    ]
    return img[img_start_idx[1] : img_end_idx[1], img_start_idx[0] : img_end_idx[0]]


def CompositeChannelImages(img_a, img_b, b_intensity=0.5):
    """
    Composite two images base on color channels. Images must be the same size.
    :param img_a: Background image (red channel)
    :type img_a: Numpy array [W,H,3]
    :param img_b: Foreground image (green/blue image)
    :type img_b: Numpy array [W,H,3]
    :param b_intensity: Color intensity of img b
    :type b_intensity: float 0->1
    :return: composite image
    :rtype:Numpy array [W,H,3]
    """
    img_composite = np.zeros((img_a.shape[0], img_a.shape[1], 3))
    if len(img_a.shape) == 3:
        img_a = np.mean(np.array(img_a), 2)
    if len(img_b.shape) == 3:
        img_b = np.mean(np.array(img_b), 2)
    img_composite[:, :, 0] = img_a / np.max(img_a)
    img_composite[:, :, 1] = b_intensity * (img_b / np.max(img_b))
    img_composite[:, :, 2] = b_intensity * (img_b / np.max(img_b))
    img_composite = (img_composite * 255.0).astype("uint8")

    return img_composite


def save_comparison_fig(ref_image, target_image, width, path):
    """
    Save side-by-side comparison image
    :param ref_image: left image
    :param target_image: right image
    :param width: image size (for text display)
    :param path: output path
    :return: (none)
    """
    fig, ax = plt.subplots(1, 2, figsize=(30, 15))
    ax[0].axis("off")
    ax[1].axis("off")
    ax[0].imshow(ref_image, cmap="gray")
    ax[1].imshow(target_image, cmap="gray")
    ax[0].text(
        10,
        width - 30,
        "FA",
        fontsize=15,
        color="white",
        bbox=dict(facecolor="Blue", alpha=0.5),
    )
    ax[1].text(
        width,
        width - 30,
        "SLO",
        fontsize=15,
        color="white",
        bbox=dict(facecolor="Blue", alpha=0.5),
    )
    plt.savefig(path)
    plt.close()


def Preview_Initial_Alignment(img_FA, img_SLO, align_scale, align_pos, save_path=None):
    """
    Show a channel composite preview of intial alignment images
    :param img_FA: FA image
    :type img_FA: numpy array [W1,H1]
    :param img_SLO: SLO image
    :type img_SLO: numpy array [W2,H2]
    :param align_scale: alignment scale (X,Y)
    :type align_scale: numpy array [2]
    :param align_pos: alignment offset (X,Y) in pixels
    :type align_pos: numpy array [2]
    :param save_path: save image path. If 'None', show pyplot image plot
    :type save_path: string
    :return: none
    :rtype:
    """
    img_b_framed, img_b_scaled = Reframe_Image(
        img=img_SLO,
        new_scale=align_scale,
        new_offset=align_pos,
        frame_size=img_FA.shape,
    )
    comp = CompositeChannelImages(img_FA, img_b_framed, b_intensity=1)
    if save_path is None:
        plt.figure(figsize=(10, 10))
        plt.imshow(comp)
        plt.show()
    else:
        comp = cv2.cvtColor(
            (comp / np.max(comp) * 255.0).astype("uint8"), cv2.COLOR_RGB2BGR
        )
        cv2.imwrite(save_path, comp)
    return comp


def preview_velocity_field(outTx, moving, path, n=8):
    x = np.arange(0, moving.shape[0], n)
    y = np.arange(0, moving.shape[1], n)

    Y, X = np.meshgrid(y, x, indexing="xy")

    field = sitk.GetArrayFromImage(outTx.GetDisplacementField())[::n, ::n, :]

    plt.figure()
    plt.quiver(Y, X, field[:, :, 1], field[:, :, 0], color="g")
    plt.title("Vector Field")
    # plt.show()
    plt.savefig(path)
    plt.close()
    return


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


def Reframe_Image(img, new_scale, new_offset, frame_size, img_scaled=None):
    """
    Reframe image by rescaling and translating across a larger canvas
    :param img: image to be reframes
    :type img: numpy array [W,H,3]
    :param new_scale: Fractional X, Y scale (0->1) to be applied to img
    :type new_scale: numpy array [2]
    :param new_offset: X, Y translation in pixels for positioning the img in the canvas
    :type new_offset: numpy array [2]
    :param frame_size: canvas dimensions (pixels) for reframining
    :type frame_size: numpy array [2]
    :return: img_framed: reframed image with scale and translation applied to img
        img_scaled: img after rescaling only
    :rtype:
    """

    if len(np.shape(img)) > 2:
        img = np.mean(np.copy(img), 2)

    if img_scaled is None:
        img_scaled = cv2.resize(
            img, (int(img.shape[1] * new_scale[1]), int(img.shape[0] * new_scale[0]))
        )
    img_framed = np.zeros(frame_size)
    img_start_idx = [
        int(new_offset[0] - img_scaled.shape[1] / 2),
        int(new_offset[1] - img_scaled.shape[0] / 2),
    ]
    img_end_idx = [
        int(new_offset[0] + img_scaled.shape[1] / 2),
        int(new_offset[1] + img_scaled.shape[0] / 2),
    ]
    if img_end_idx[1] >= frame_size[0]:
        img_end_idx[1] = frame_size[0] - 1
    if img_end_idx[0] >= frame_size[1]:
        img_end_idx[0] = frame_size[1]
    if (
        img_start_idx[1] < 0
        or img_start_idx[1] > frame_size[0]
        or img_start_idx[0] < 0
        or img_start_idx[0] > frame_size[1]
    ):
        print("Registration_src error: start/end indexes outside of image frame")
        print(
            "Start index: {}, end index: {}, frame size: {}".format(
                img_start_idx, img_end_idx, frame_size
            )
        )
        return img_framed
    else:
        img_crop = img_scaled.copy()
        display_crop_size = img_scaled.shape
        frame_crop_size = [
            (img_end_idx[1] - img_start_idx[1]),
            (img_end_idx[0] - img_start_idx[0]),
        ]
        if (img_end_idx[1] - img_start_idx[1]) < display_crop_size[0]:
            img_crop = img_crop[0 : (img_end_idx[1] - img_start_idx[1]), :]
        if (img_end_idx[0] - img_start_idx[0]) < display_crop_size[1]:
            img_crop = img_crop[:, 0 : (img_end_idx[0] - img_start_idx[0])]

        # compose rescaled image in frame:
        img_framed[
            img_start_idx[1] : img_end_idx[1], img_start_idx[0] : img_end_idx[0]
        ] = img_crop

    return img_framed, img_scaled


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(UNet, self).__init__()

        features = init_features
        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return torch.sigmoid(self.conv(dec1))

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )


def Get_unet_MRI(init_features=32, downscale_x4=False):
    """model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', in_channels=1, out_channels=1,
    init_features=init_features, pretrained=False)
    """

    model = UNet(in_channels=1, out_channels=1)

    # model.encoder1.enc1conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    # model.encoder2.enc2conv1 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

    if downscale_x4:
        model.pool1 = nn.MaxPool2d(
            kernel_size=4, stride=4, padding=0, dilation=1, ceil_mode=False
        )
        # model.pool2 = nn.MaxPool2d(kernel_size=4, stride=4, padding=0, dilation=1, ceil_mode=False)
        # model.pool3 = nn.MaxPool2d(kernel_size=4, stride=4, padding=0, dilation=1, ceil_mode=False)
        # model.pool4 = nn.MaxPool2d(kernel_size=3, stride=1, padding=0, dilation=1, ceil_mode=False)

        model.upconv1 = nn.ConvTranspose2d(
            init_features * 2,
            init_features,
            kernel_size=(4, 4),
            stride=(4, 4),
            padding=(0, 0),
        )
        # model.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(4, 4), padding=(0, 0))
        # model.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(4, 4), padding=(0, 0))
        # model.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0))

    return model
