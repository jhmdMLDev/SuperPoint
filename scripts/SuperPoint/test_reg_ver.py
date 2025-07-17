import os
import sys
import time
import re

import cv2
import numpy as np
import torch

sys.path.insert(0, os.getcwd())
from scripts.SuperPoint.main import ApplySuperPointReg
from scripts.robustness_eval.synthetic_blur import apply_blur
from scripts.SuperPoint.stabilizing_analysis import depth_convert_percentile
from scripts.registration_verification.phase_correlation import (
    apply_registration_verification,
    divide_image,
)


def sorted_alphanumeric(data):
    def convert(text):
        return int(text) if text.isdigit() else text.lower()

    def alphanum_key(key):
        return [convert(c) for c in re.split("([0-9]+)", key)]

    return sorted(data, key=alphanum_key)


def put_text_on_image(image, text):
    image3d = np.dstack([image, image, image])
    # Set the font properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    font_color = (0, 255, 0) if text == "True" else (0, 0, 255)
    line_thickness = 1

    # Define the position of the text (top left corner)
    position = (10, 30)

    # Put the text on the image
    cv2.putText(image3d, text, position, font, font_scale, font_color, line_thickness)

    return image3d


def display_table_with_image(image, data):
    # Create a blank canvas to overlay the table
    canvas = np.ones_like(image) * 255  # White canvas

    # Determine the table dimensions and cell size
    table_width = 200
    table_height = 150
    cell_width = table_width // 3
    cell_height = table_height // 9

    # Define the starting position of the table
    table_x = 10
    table_y = 10

    # Draw the table border
    cv2.rectangle(
        canvas,
        (table_x, table_y),
        (table_x + table_width, table_y + table_height),
        (0, 0, 0),
        2,
    )

    # Iterate over the data array and draw text for each element
    for i in range(9):
        for j in range(3):
            value = str(round(data[i, j], 1))
            cell_x = table_x + j * cell_width
            cell_y = table_y + (i + 1) * cell_height
            cv2.putText(
                canvas,
                value,
                (cell_x, cell_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
            )

    # Combine the image and the table
    result = np.hstack((image, canvas))
    return result


def register_folder(
    folder_path,
    save_path,
    save_folder_path=None,
    weights_path=r"/efs/model_check2023/SUPERPOINT_TUNE/FloaterAugv4/epoch_22_model.pt",
):
    imgs_path = sorted_alphanumeric(os.listdir(folder_path))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_name = save_path + f"/FifthRun.mp4"
    video_out = cv2.VideoWriter(video_name, fourcc, 8, (1024, 512))
    time_arr = []
    for i, img_path in enumerate(imgs_path):
        image = cv2.imread(os.path.join(folder_path, img_path), 0)
        image = depth_convert_percentile(image)
        if i == 0:
            apply_superpoint = ApplySuperPointReg(image, weights_path=weights_path)
            reference_patches = divide_image(image)
            image_disp = put_text_on_image(image, "Reference")
            video_out.write(image_disp)
            continue

        targ_transferred = apply_superpoint(image)
        t0 = time.time()
        success, results = apply_registration_verification(
            reference_patches, targ_transferred
        )
        time_arr.append(time.time() - t0)
        if save_folder_path is not None and success:
            image_name = img_path[:-5] + ".jpg"
            cv2.imwrite(os.path.join(save_folder_path, img_path), targ_transferred)
        image_disp = put_text_on_image(targ_transferred, str(success))
        image_disp_table = display_table_with_image(image_disp, results)
        video_out.write(image_disp_table)
    video_out.release()
    print(f"Measured Time is {np.mean(time_arr)}")


if __name__ == "__main__":
    # folder_path = r"/efs/datasets/EyeMovement/Ashley/NoTargetFocused"
    # save_path = r"/efs/model_inference/registration/regVer"
    folder_path = r"/efs/datasets/floater_dataset_edited/unregistered_new_data_2023/processedData/FifthRun"
    save_path = r"/efs/model_inference/registration/regVer"
    save_folder_path = r"/efs/datasets/floater_dataset_edited/dataset_train/99"
    register_folder(folder_path, save_path, save_folder_path)
