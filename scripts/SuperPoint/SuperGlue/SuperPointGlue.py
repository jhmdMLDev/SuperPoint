import os
import sys
import torch

sys.path.insert(0, os.getcwd())
from scripts.SuperPoint.SuperGlue.SuperGlue import SuperGlue
from scripts.SuperPoint.SuperGlue.SuperPoint import SuperPoint
from scripts.SuperPoint.superpoint import SuperPointFrontend


class SuperPointGlue(torch.nn.Module):
    """Image Matching Frontend (SuperPoint + SuperGlue)"""

    def __init__(self, config):
        super().__init__()
        self.superpoint = SuperPointFrontend(
            weights_path=r"C:\Users\Javad\Desktop\Model_Check\Superpoint\epoch_22_model.pt",
            nms_dist=4,
            conf_thresh=0.015,
            nn_thresh=0.7,
            cuda=True,
        )
        self.superglue = SuperGlue(config["SuperGlue"]).cuda()
        self.data = {}
        self.has_reference = False
        self.placeholder_value = float("nan")

    def set_reference(self, ref):
        self.data["image0"] = torch.tensor(ref).cuda().unsqueeze(0).unsqueeze(0) / 255.0
        pts, desc, _ = self.superpoint.run(ref.astype("float32") / 255.0)
        self.data["keypoints0"] = (
            torch.tensor(pts[0:2, :]).transpose(1, 0).unsqueeze(0).cuda().float()
        )
        self.data["scores0"] = torch.tensor(pts[2, :]).unsqueeze(0).cuda().float()
        self.data["descriptors0"] = torch.tensor(desc).unsqueeze(0).cuda().float()
        self.has_reference = True
        for k in self.data:
            if isinstance(self.data[k], (list, tuple)):
                self.data[k] = torch.stack(self.data[k])

    def forward(self, target):
        if not self.has_reference:
            raise ValueError("No Reference Value")

        self.data["image1"] = (
            torch.tensor(target).cuda().unsqueeze(0).unsqueeze(0) / 255.0
        )

        pts, desc, _ = self.superpoint.run(target.astype("float32") / 255.0)
        self.data["keypoints1"] = (
            torch.tensor(pts[0:2, :]).transpose(1, 0).unsqueeze(0).cuda().float()
        )
        self.data["scores1"] = torch.tensor(pts[2, :]).unsqueeze(0).cuda().float()
        self.data["descriptors1"] = torch.tensor(desc).unsqueeze(0).cuda().float()

        # Perform the matching
        pred = {**self.data, **self.superglue(self.data)}

        pred["matchIdx"] = pred["matches0"][0]

        pred["p1"] = self.data["keypoints0"][0]
        pred["p2"] = self.data["keypoints1"][0][pred["matchIdx"]]

        matches = torch.where(pred["scores0"][0] > 0.015)[0]
        pred["p1"] = pred["p1"][matches]
        pred["p2"] = pred["p2"][matches]

        return pred


if __name__ == "__main__":
    import time
    import cv2
    import numpy as np

    cfg = {}
    cfg["Superpoint"] = {}
    cfg["Superpoint"][
        "weights"
    ] = r"C:\Users\Javad\Desktop\Model_Check\Superpoint\epoch_22_model.pt"

    cfg["SuperGlue"] = {}
    cfg["SuperGlue"]["weights"] = r"C:\Users\Javad\Downloads\superglue_indoor.pth"

    refPath = r"C:\Users\Javad\Downloads\19-07-2023-20230719T205729Z-001\processedData\SecondRun\raw_133813_1.png"
    trgPath = r"C:\Users\Javad\Downloads\19-07-2023-20230719T205729Z-001\processedData\SecondRun\raw_133813_270.png"

    ref = cv2.imread(refPath, 0)
    trg = cv2.imread(trgPath, 0)

    superpointglue = SuperPointGlue(cfg)
    superpointglue.set_reference(ref)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    t0 = time.time()
    pred = superpointglue(trg)
    p1 = np.array(pred["p1"].cpu())
    p2 = np.array(pred["p2"].cpu())
    H_transform, mask = cv2.findHomography(p2, p1, cv2.RANSAC, 15)

    img_target_transformed = cv2.warpPerspective(
        trg, H_transform, (trg.shape[1], trg.shape[0])
    )

    print("time: ", time.time() - t0)
    end_event.record()
    elapsed_time = start_event.elapsed_time(end_event)
    print(f"Elapsed time: {elapsed_time:.4f} ms")

    img_3d = np.dstack([ref, np.zeros_like(ref), img_target_transformed])

    cv2.imwrite(".sample/supergluereg.png", img_3d)
    cv2.imwrite(".sample/supergluetref.png", ref)
    cv2.imwrite(
        ".sample/supergluettransformed.png",
        img_target_transformed,
    )
