import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.geometry.transform import warp_perspective


class SuperPointLoss(nn.Module):
    def __init__(self, cfg):
        super(SuperPointLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.device = cfg.DEVICE
        self.margin = cfg.margin
        self.weight_keypoint = cfg.weight_keypoint
        self.weight_background = cfg.weight_background

    def keypoint_loss(self, ground_truth_mask, heatmap_mask):
        # Flatten the heatmaps
        pred_flat = heatmap_mask.flatten()
        true_flat = ground_truth_mask.flatten()
        # Compute the binary cross-entropy loss
        loss = nn.BCELoss(reduction="none")(pred_flat, true_flat)
        # Compute the weights for keypoints and background pixels
        weight = (
            true_flat * self.weight_keypoint + (1 - true_flat) * self.weight_background
        )
        # Apply the weights to the loss
        weighted_loss = loss * weight
        # Compute the average loss
        loss = torch.mean(weighted_loss)
        return loss

    def keypoint_warp_loss(self, heatmap_mask, warped_mask, homography):
        heatmap_mask_expanded = heatmap_mask.unsqueeze(0).unsqueeze(0)
        heatmap_transformed = (
            warp_perspective(
                heatmap_mask_expanded.float(),
                homography.to(self.device).float(),
                (512, 512),
                align_corners=False,
            )
            .squeeze(0)
            .squeeze(0)
        )
        return self.keypoint_loss(heatmap_transformed, warped_mask)

    def descriptor_loss(self, descriptors1, descriptors2):
        pairwise_distances = torch.cdist(descriptors1, descriptors2, p=2)
        rows, cols = pairwise_distances.size()
        mask = torch.eye(rows, cols, dtype=torch.bool).to(self.device)
        paired_loss = pairwise_distances[mask].pow(2).mean()
        unpaired_loss = pairwise_distances[~mask].pow(2).mean()
        loss = F.relu(self.margin - unpaired_loss) + paired_loss
        return loss
