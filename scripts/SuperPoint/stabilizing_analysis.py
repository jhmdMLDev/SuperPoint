import os
import sys
import re
import math

import numpy as np
import matplotlib.pyplot as plt
import cv2


sys.path.insert(0, os.getcwd())
from scripts.SuperPoint.main import ApplySuperPointReg

weights_path = r"/efs/model_check2023/SUPERPOINT_TUNE/FloaterAugv4/epoch_22_model.pt"


def homography_check(matrix):
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


def depth_convert_percentile(img):
    img_values = np.sort(img.flatten())
    high_index = int(0.999 * len(img_values))
    img8U = cv2.convertScaleAbs(img, alpha=255.0 / img_values[high_index])
    return img8U


def sorted_alphanumeric(data):
    def convert(text):
        return int(text) if text.isdigit() else text.lower()

    def alphanum_key(key):
        return [convert(c) for c in re.split("([0-9]+)", key)]

    return sorted(data, key=alphanum_key)


def decompose_affine_homography(matrix):
    # Translation component
    tx = matrix[0, 2]
    ty = matrix[1, 2]

    # Extract rotation component
    theta = -math.atan2(matrix[0, 1], matrix[0, 0]) * 180 / math.pi

    # Extract scaling/shearing component
    sx = matrix[0, 0] / math.cos(theta * math.pi / 180)
    sy = matrix[1, 1] / math.cos(theta * math.pi / 180)
    s = (sx + sy) / 2

    t = abs(tx) + abs(ty)

    return t, theta


def get_homography_result(data_path, save_data):
    folder_names = [
        item
        for item in os.listdir(data_path)
        if os.path.isdir(os.path.join(data_path, item))
    ]
    result = {folder: {} for folder in folder_names}
    for folder in folder_names:
        image_names = sorted_alphanumeric(os.listdir(os.path.join(data_path, folder)))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_name = save_data + f"/{folder}.mp4"
        video_out = cv2.VideoWriter(video_name, fourcc, 8, (512, 512))
        for i, image_name in enumerate(image_names):
            ## debug
            if i % 30 != 0:
                continue
            image = cv2.imread(os.path.join(data_path, folder, image_name), -1)
            image = depth_convert_percentile(image)
            if i == 0:
                superpoint = ApplySuperPointReg(image, weights_path=weights_path)
                continue

            trg_img = superpoint(image)
            if superpoint.success:
                result[folder][str(i)] = superpoint.H_transform
                video_out.write(np.dstack([trg_img, trg_img, trg_img]))
        video_out.release()
    return result


def plot_subplots(affine_components, save_path):
    # Create a figure with four subplots
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    for i, (feature, component) in enumerate(affine_components.items()):
        for elements in component:
            avg_val = np.mean(elements["values"])
            label = elements["name"] + f" Averge {feature}: {round(avg_val, 2)}"
            axs[i].plot(elements["ids"], elements["values"], label=label)
            axs[i].set_ylabel(feature)
            axs[i].set_xlabel("Frame ID")
            axs[i].legend()
            feature_desc = (
                "Translation (pixel)" if feature == "T" else "Rotation (degree)"
            )
            axs[i].set_title(f"{feature_desc} values over all frames")
        axs[i].set_xticks(range(300)[::10])
    # Adjust the spacing between subplots
    plt.subplots_adjust(hspace=0.4)

    # Display the plot
    plt.savefig(save_path + "/results.png")


def analyze_movement_control(data_path, save_path):
    folder_names = [
        item
        for item in os.listdir(data_path)
        if os.path.isdir(os.path.join(data_path, item))
    ]
    homography_output = get_homography_result(data_path, save_path)
    affine_components = {"T": [], "Theta": []}
    features = list(affine_components.keys())
    for folder in folder_names:
        homography_dict = homography_output[folder]
        ids = list(homography_dict.keys())
        homogrophy = list(homography_dict.values())
        values = [decompose_affine_homography(h) for h in homogrophy]
        for i in range(0, len(features)):
            folder_dict = {}
            folder_dict["ids"] = ids
            folder_dict["name"] = folder
            folder_dict["values"] = [tpl[i] for tpl in values]
            affine_components[features[i]].append(folder_dict)

    plot_subplots(affine_components, save_path)


if __name__ == "__main__":
    data_path = "/efs/datasets/SuperpointDataset/stablizing_test"
    save_path = "/efs/model_inference/registration/stability_analysis"
    analyze_movement_control(data_path, save_path)
