import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.getcwd())
from scripts.SuperPoint.superpoint import (
    SuperPointFrontend,
    PointTracker,
)


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

        self.tracker = PointTracker(self.max_length, nn_thresh=self.fe.nn_thresh)
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

    def __call__(self, trg):
        # trg = cv2.resize(trg, (self.W, self.H), interpolation=self.interp)
        trg = trg.astype("float32") / 255.0

        trg_pts, trg_desc, trg_heatmap = self.fe.run(trg)

        trg_pts_useful = trg_pts[:2, :].transpose()
        trg_desc_useful = trg_desc[:, :].transpose()

        matches = self.matcher.match(self.ref_desc_useful, trg_desc_useful)
        matches = sorted(matches, key=lambda x: x.distance)
        matches = matches[:]
        self.p1 = np.array(
            [self.ref_pts_useful[matches[i].queryIdx] for i in range(len(matches))]
        )
        self.p2 = np.array(
            [trg_pts_useful[matches[i].trainIdx] for i in range(len(matches))]
        )

        self.success = True if self.p2.shape[0] >= 4 else False

        if not self.success:
            self.H_transform = None
            return (trg * 255).astype("uint8")

        self.H_transform, self.mask = cv2.findHomography(
            self.p2, self.p1, cv2.RANSAC, 15
        )

        if self.H_transform is None:
            self.success = False
            return (trg * 255).astype("uint8")

        img_target_transformed = cv2.warpPerspective(
            trg, self.H_transform, (trg.shape[1], trg.shape[0])
        )
        img_target_transformed = (img_target_transformed * 255).astype("uint8")

        return img_target_transformed
