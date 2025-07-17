import os
import sys
import shutil
import random

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.getcwd())
from scripts.SuperPoint.main import ApplySuperPointReg

DEBUG = False
POINTS_PERCENTAGE = 75
seed_value = 42
# np.random.seed(seed_value)
# random.seed(seed_value)

#  screen -S Create_annotation -L -Logfile /home/ubuntu/Projects/Registration_Benchmarking/screenlogs/Create_annotation.log python scripts/SuperPoint/train/pseudo_ground_truth.py


def create_homography_matrix(theta, tx, ty, sx, sy, shear):
    # Convert rotation angle and shear from degrees to radians
    theta_rad = np.deg2rad(theta)
    shear_rad = np.deg2rad(shear)

    # Compute cosine and sine of the rotation angle and shear
    cos_theta = np.cos(theta_rad)
    sin_theta = np.sin(theta_rad)
    tan_shear = np.tan(shear_rad)

    # Create the homography matrix
    homography_matrix = np.array(
        [
            [sx * cos_theta, -sy * sin_theta + tan_shear * cos_theta, tx],
            [sx * sin_theta + tan_shear * sin_theta, sy * cos_theta, ty],
            [0, 0, 1],
        ]
    )

    return homography_matrix


def get_keypoints(image):
    apply_superpoint = ApplySuperPointReg(
        image, "/mnt-fs/efs-ml-model/SVO/pretrained/SuperPoint/epoch_22_model.pt"
    )

    keypoints = apply_superpoint.ref_pts_useful
    if keypoints is None:
        return None
    return apply_superpoint.ref_pts_useful.astype("int64")


def register_back_keypoints(keypoints, homography_matrix):
    inv_H = np.linalg.inv(homography_matrix)
    points_homogeneous = np.column_stack((keypoints, np.ones(len(keypoints))))

    # Transform the points using the homography matrix
    transformed_points_homogeneous = np.dot(inv_H, points_homogeneous.T).T

    # Convert the transformed points back to Cartesian coordinates
    transformed_points = (
        transformed_points_homogeneous[:, :2] / transformed_points_homogeneous[:, 2:]
    )

    return transformed_points.astype("int64")


def show_warped_keypoints(warp, keypoints):
    img_diplay = np.copy(warp)
    for point in keypoints:
        x, y = point
        cv2.circle(img_diplay, (x, y), radius=5, color=(0, 0, 255), thickness=-1)
    return img_diplay


def display_heatmap_on_img(keypoints_heatmap, image):
    heatmap = (
        255
        * (keypoints_heatmap - np.min(keypoints_heatmap))
        / (np.max(keypoints_heatmap) - np.min(keypoints_heatmap))
    )
    heatmap = cv2.dilate(heatmap, np.ones((5, 5)))
    image_3d = np.dstack([image, image, image])
    non_zero_arr = heatmap[heatmap != 0]
    image_3d[heatmap > np.percentile(non_zero_arr, POINTS_PERCENTAGE), :] = [0, 0, 255]
    return image_3d


def apply_keypoints_to_heatmap(keypoints_heatmap, transformed_keypoints):
    for point in transformed_keypoints:
        x, y = point
        if (max(x, y) < 512) and min(x, y) >= 0:
            keypoints_heatmap[y, x] += 1

    return keypoints_heatmap


def get_pseudo_ground_truth(image, iteration=100):
    keypoints_heatmap = np.zeros(image.shape)
    for _ in range(0, iteration):
        theta = np.random.randint(-10, 10)
        tx = np.random.randint(-30, 30)
        ty = np.random.randint(-30, 30)
        sx = 1 + (0.1 * np.random.random() - 0.05)
        sy = 1 + (0.1 * np.random.random() - 0.05)
        shear = np.random.randint(-3, 3)
        homography_matrix = create_homography_matrix(theta, tx, ty, sx, sy, shear)
        warp = cv2.warpPerspective(
            image, homography_matrix, (image.shape[0], image.shape[0])
        )
        keypoints = get_keypoints(warp)
        if len(keypoints) == 0:
            continue
        transformed_points = register_back_keypoints(keypoints, homography_matrix)
        keypoints_heatmap = apply_keypoints_to_heatmap(
            keypoints_heatmap, transformed_points
        )

        if DEBUG:
            img_diplay = show_warped_keypoints(warp, keypoints)
            cv2.imshow("Wapred", img_diplay)
            heatmap_img = display_heatmap_on_img(keypoints_heatmap, image)
            cv2.imshow("Heatmap", heatmap_img.astype("uint8"))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    if np.max(keypoints_heatmap) > np.min(keypoints_heatmap):
        keypoints_heatmap = (
            255
            * (keypoints_heatmap - np.min(keypoints_heatmap))
            / (np.max(keypoints_heatmap) - np.min(keypoints_heatmap))
        )
    else:
        return None
    non_zero_arr = keypoints_heatmap[keypoints_heatmap != 0]
    indices = np.where(
        keypoints_heatmap > np.percentile(non_zero_arr, POINTS_PERCENTAGE)
    )
    final_keypoints = np.column_stack((indices[1], indices[0]))
    return final_keypoints


def get_data_annotation(data_path):
    image_path_list = [
        ele
        for ele in os.listdir(data_path)
        if (ele.endswith("jpg")) or (ele.endswith("png"))
    ]

    filenames = []
    keypoint_arr = []
    loop = tqdm(image_path_list, leave=True)
    for image_path in loop:
        image = cv2.imread(os.path.join(data_path, image_path), 0)
        final_keypoints = get_pseudo_ground_truth(image)
        if final_keypoints is None:
            os.remove(os.path.join(data_path, image_path))
            continue
        filenames.append(image_path)
        keypoint_arr.append(final_keypoints)

    df = pd.DataFrame(columns=["Filename", "Points"])

    for filename, arr in zip(filenames, keypoint_arr):
        # Append a row to the dataframe
        df = df._append(
            {"Filename": filename, "Points": str(arr.tolist())}, ignore_index=True
        )

    # Save the dataframe as a CSV file
    df.to_csv(data_path + "/output.csv", index=False)


def copy_files(
    main_path=r"C:\Users\Javad\Desktop\Dataset\floater_dataset\dataset_train",
    dst_path=r"C:\Users\Javad\Desktop\Dataset\SuperpointDataset\train_val_mix",
):
    folders = os.listdir(main_path)
    for folder in folders:
        folder_path = os.path.join(main_path, folder)
        imagelist = [
            ele
            for ele in os.listdir(folder_path)
            if (ele.endswith("jpg")) or (ele.endswith("png"))
        ]
        selected_indices = random.sample(
            range(len(imagelist)), int(len(imagelist) * 0.2)
        )
        imagelist = [imagelist[i] for i in selected_indices]
        for image_path in imagelist:
            shutil.copy(os.path.join(folder_path, image_path), dst_path)


if __name__ == "__main__":
    data_path = r"/mnt-fs/efs-ml-data/SuperpointDataset/train"
    get_data_annotation(data_path)
