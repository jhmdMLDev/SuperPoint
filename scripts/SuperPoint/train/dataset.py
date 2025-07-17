import os
import sys
import math
import time
from ast import literal_eval

import numpy as np
import pandas as pd
import cv2
import torch
from skimage import morphology


sys.path.insert(0, os.getcwd())
from scripts.SuperPoint.train.pseudo_ground_truth import create_homography_matrix
from scripts.SuperPoint.train.synthetic_floater_utils import cast_synthetic_floater


class SuperPointFineTuneDataset(torch.utils.data.Dataset):
    def __init__(self, cfg):
        """
        Initialize the dataset for fine-tuning SuperPoint model.

        Args:
            data_path (str): Path to the dataset.
            transform (callable, optional): Optional data transformations to apply.
        """
        self.data_path = cfg.DATA_DIRECTORY
        self.transform = cfg.transforms_image

        # Load the dataset
        self.load_dataset()

    def blur_box(self, image):
        # Copy the original image to avoid modifying it directly
        blurred_image = image.copy()
        x = np.random.randint(100, 400)
        y = np.random.randint(100, 400)
        width = np.random.randint(50, 100)
        height = np.random.randint(50, 100)
        blur_amount = int(2 * np.random.randint(5, 16) + 1)

        # Extract the region of interest (box) from the image
        box = blurred_image[y : y + height, x : x + width]

        # Apply Gaussian blur to the box region
        blurred_box = cv2.GaussianBlur(box, (blur_amount, blur_amount), 0)

        # Place the blurred box back into the original image
        blurred_image[y : y + height, x : x + width] = blurred_box

        return blurred_image

    def apply_random_blur(self, image):
        for _ in range(0, np.random.randint(0, 5)):
            image = self.blur_box(image)
        return image

    def load_dataset(self):
        # Implement your logic to load the dataset
        # and extract keypoints, descriptors, and corresponding images
        # from the data_path
        self.image_files = [
            ele
            for ele in os.listdir(self.data_path)
            if (ele.endswith("jpg")) or (ele.endswith("png"))
        ]
        csv_file = [ele for ele in os.listdir(self.data_path) if ele.endswith("csv")][0]
        self.image_file_list = []
        for image_name in self.image_files:
            image_path = os.path.join(self.data_path, image_name)
            self.image_file_list.append(image_path)

        self.df = pd.read_csv(os.path.join(self.data_path, csv_file))

    def __getitem__(self, index):
        """
        Get an item from the dataset.

        Args:
            index (int): Index of the item to retrieve.

        Returns:
            tuple: A tuple containing the image, keypoints, and descriptors at the given index.
        """
        image = cv2.imread(self.image_file_list[index], 0)
        values = self.df[self.df["Filename"] == self.image_files[index]].iloc[0, 1]

        homography_matrix = create_homography_matrix(
            theta=np.random.randint(-10, 10),
            tx=np.random.randint(-30, 30),
            ty=np.random.randint(-30, 30),
            sx=1 + (0.1 * np.random.random() - 0.05),
            sy=1 + (0.1 * np.random.random() - 0.05),
            shear=np.random.randint(-3, 3),
        )

        keypoints = np.array(literal_eval(values))

        points_homogeneous = np.column_stack((keypoints, np.ones(len(keypoints))))

        # Transform the points using the homography matrix
        transformed_points_homogeneous = np.dot(
            homography_matrix, points_homogeneous.T
        ).T

        transformed_points = (
            transformed_points_homogeneous[:, :2]
            / transformed_points_homogeneous[:, 2:]
        )
        transformed_points = transformed_points.astype("int64")

        warped_heatmap = np.zeros_like(image)
        out_of_field_points = []
        successful_points = []
        successful_pts = 0
        for i, point in enumerate(transformed_points):
            x, y = point
            if (
                max(x, y) > (image.shape[0] - 1)
                or min(x, y) < 0
                or (y, x) in successful_points
            ):
                out_of_field_points.append(i)
                continue
            warped_heatmap[y, x] = 255
            successful_points.append((y, x))

            successful_pts += 1

        heatmap = np.zeros_like(image)
        for i, point in enumerate(keypoints):
            x, y = point
            if i not in out_of_field_points:
                heatmap[y, x] = 255

        warped_image = cv2.warpPerspective(
            image, homography_matrix, (image.shape[0], image.shape[0])
        )

        if np.random.random() > 0.5:
            image = self.apply_random_blur(image)
            warped_image = self.apply_random_blur(warped_image)

        if np.random.random() > 0.5:
            image = cast_synthetic_floater(image)
            warped_image = cast_synthetic_floater(warped_image)

        data = {}
        data["image"] = self.transform(torch.tensor(image).unsqueeze(0) / 255.0)
        data["warped_image"] = self.transform(
            torch.tensor(warped_image).unsqueeze(0) / 255.0
        )
        data["heatmap"] = torch.tensor(heatmap) / 255.0
        data["warped_heatmap"] = torch.tensor(warped_heatmap) / 255.0
        data["affine_matrix"] = torch.tensor(homography_matrix)

        return data

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.image_file_list)


if __name__ == "__main__":
    import ml_collections
    from scripts.SuperPoint.train.config import (
        get_superpoint_train_cfg,
        get_superpoint_val_cfg,
    )

    cfg = get_superpoint_val_cfg("Test")

    dataset = SuperPointFineTuneDataset(cfg)
    data = dataset[200]
    image = (255 * np.array(data["image"].squeeze())).astype("uint8")
    print(image.shape)
    cv2.imwrite(cfg.CHECK_PATH + "/test_dataset.png", image)
