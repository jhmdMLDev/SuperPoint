from pathlib import Path
import torch
from torch import nn


def simple_nms(scores, nms_radius: int):
    """Fast Non-maximum suppression to remove nearby points"""
    assert nms_radius >= 0

    def max_pool(x):
        return torch.nn.functional.max_pool2d(
            x, kernel_size=nms_radius * 2 + 1, stride=1, padding=nms_radius
        )

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)
    for _ in range(2):
        supp_mask = max_pool(max_mask.float()) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == max_pool(supp_scores)
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    return torch.where(max_mask, scores, zeros)


def remove_borders(keypoints, scores, border: int, height: int, width: int):
    """Removes keypoints too close to the border"""
    mask_h = (keypoints[:, 0] >= border) & (keypoints[:, 0] < (height - border))
    mask_w = (keypoints[:, 1] >= border) & (keypoints[:, 1] < (width - border))
    mask = mask_h & mask_w
    return keypoints[mask], scores[mask]


def top_k_keypoints(keypoints, scores, k: int):
    if k >= len(keypoints):
        return keypoints, scores
    scores, indices = torch.topk(scores, k, dim=0)
    return keypoints[indices], scores


def sample_descriptors(keypoints, descriptors, s: int = 8):
    """Interpolate descriptors at keypoint locations"""
    b, c, h, w = descriptors.shape
    keypoints = keypoints - s / 2 + 0.5
    keypoints /= torch.tensor(
        [(w * s - s / 2 - 0.5), (h * s - s / 2 - 0.5)],
    ).to(
        keypoints
    )[None]
    keypoints = keypoints * 2 - 1  # normalize to (-1, 1)
    args = {"align_corners": True} if torch.__version__ >= "1.3" else {}
    descriptors = torch.nn.functional.grid_sample(
        descriptors, keypoints.view(b, 1, -1, 2), mode="bilinear", **args
    )
    descriptors = torch.nn.functional.normalize(
        descriptors.reshape(b, c, -1), p=2, dim=1
    )
    return descriptors


class SuperPoint(nn.Module):
    """SuperPoint Convolutional Detector and Descriptor

    SuperPoint: Self-Supervised Interest Point Detection and
    Description. Daniel DeTone, Tomasz Malisiewicz, and Andrew
    Rabinovich. In CVPRW, 2019. https://arxiv.org/abs/1712.07629

    """

    default_config = {
        "descriptor_dim": 256,
        "nms_radius": 4,
        "keypoint_threshold": 0.015,
        "max_keypoints": -1,
        "remove_borders": 4,
    }

    def __init__(self, config):
        super().__init__()
        self.config = {**self.default_config, **config}

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5 = 64, 64, 128, 128, 256

        self.conv1a = nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)

        self.convPa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convPb = nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)

        self.convDa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = nn.Conv2d(
            c5, self.config["descriptor_dim"], kernel_size=1, stride=1, padding=0
        )

        self.load_state_dict(torch.load(config["weights"]))

        mk = self.config["max_keypoints"]
        if mk == 0 or mk < -1:
            raise ValueError('"max_keypoints" must be positive or "-1"')

        print("Loaded SuperPoint model")

    def forward(self, data):
        """Compute keypoints, scores, descriptors for image"""
        # Shared Encoder
        x = self.relu(self.conv1a(data["image"]))
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

        # Compute the dense keypoint scores
        cPa = self.relu(self.convPa(x))
        scores = self.convPb(cPa)
        scores = torch.nn.functional.softmax(scores, 1)[:, :-1]
        b, _, h, w = scores.shape
        scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(b, h * 8, w * 8)
        scores = simple_nms(scores, self.config["nms_radius"])

        # Extract keypoints
        keypoints = [
            torch.nonzero(s > self.config["keypoint_threshold"]) for s in scores
        ]
        scores = [s[tuple(k.t())] for s, k in zip(scores, keypoints)]

        # Discard keypoints near the image borders
        keypoints, scores = list(
            zip(
                *[
                    remove_borders(k, s, self.config["remove_borders"], h * 8, w * 8)
                    for k, s in zip(keypoints, scores)
                ]
            )
        )

        # Keep the k keypoints with highest score
        if self.config["max_keypoints"] >= 0:
            keypoints, scores = list(
                zip(
                    *[
                        top_k_keypoints(k, s, self.config["max_keypoints"])
                        for k, s in zip(keypoints, scores)
                    ]
                )
            )

        # Convert (h, w) to (x, y)
        keypoints = [torch.flip(k, [1]).float() for k in keypoints]

        # Compute the dense descriptors
        cDa = self.relu(self.convDa(x))
        descriptors = self.convDb(cDa)
        descriptors = torch.nn.functional.normalize(descriptors, p=2, dim=1)

        # Extract descriptors
        descriptors = [
            sample_descriptors(k[None], d[None], 8)[0]
            for k, d in zip(keypoints, descriptors)
        ]

        return {
            "keypoints": keypoints,
            "scores": scores,
            "descriptors": descriptors,
        }


if __name__ == "__main__":
    import os
    import sys
    import cv2
    import time
    import ml_collections

    sys.path.insert(0, os.getcwd())
    from scripts.SuperPoint.SuperGlue.SuperGlue import SuperGlue

    cfgSuperGlue = ml_collections.ConfigDict()
    cfgSuperGlue.weights = r"C:\Users\Iman\Downloads\superglue_indoor.pth"

    superglue = SuperGlue(cfgSuperGlue).cuda()

    cfgSuperPoint = ml_collections.ConfigDict()
    cfgSuperPoint.weights = (
        r"C:\Users\Iman\Desktop\Model_Check\Superpoint\epoch_22_model.pt"
    )

    superpoint = SuperPoint(cfgSuperPoint).cuda()

    data = {}
    refPath = r"C:\Users\Iman\Downloads\19-07-2023-20230719T205729Z-001\processedData\SecondRun\raw_133813_1.png"
    ref = cv2.imread(refPath, 0)
    data["image"] = torch.tensor(ref).cuda().unsqueeze(0).unsqueeze(0) / 255.0
    data["image0"] = data["image"]

    superpointResultsRef = superpoint(data)

    trgPath = r"C:\Users\Iman\Downloads\19-07-2023-20230719T205729Z-001\processedData\SecondRun\raw_133813_270.png"
    trg = cv2.imread(trgPath, 0)
    data["image"] = torch.tensor(trg).cuda().unsqueeze(0).unsqueeze(0) / 255.0
    data["image1"] = data["image"]

    superpointResultsTrg = superpoint(data)

    data["descriptors0"] = superpointResultsRef["descriptors"][0].unsqueeze(0)
    data["descriptors1"] = superpointResultsTrg["descriptors"][0].unsqueeze(0)
    data["keypoints0"] = superpointResultsRef["keypoints"][0].unsqueeze(0)
    data["keypoints1"] = superpointResultsTrg["keypoints"][0].unsqueeze(0)
    data["scores0"] = superpointResultsRef["scores"][0].unsqueeze(0)
    data["scores1"] = superpointResultsTrg["scores"][0].unsqueeze(0)

    superglueRes = superglue(data)
