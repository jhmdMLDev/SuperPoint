import os
import sys
import argparse
from ast import literal_eval
import time

import ml_collections
import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
from scripts.TomokoTracker.main import TomokoTrakcer
from scripts.SuperPoint.main import ApplySuperPointReg
from registration_svo.tracking_eval import tracking_eval


def sorted_alphanumeric(data):
    def convert(text):
        return int(text) if text.isdigit() else text.lower()

    def alphanum_key(key):
        return [convert(c) for c in re.split("([0-9]+)", key)]

    return sorted(data, key=alphanum_key)


def read_annotation(filename, df):
    mask_target = np.zeros((512, 512), dtype="uint8")
    bbx = df[df.loc[:, "filename"] == filename]
    bbx = bbx.reset_index()
    for i in range(0, len(bbx)):
        box = literal_eval(bbx.loc[i, "region_shape_attributes"])
        if box != {}:
            x, y, w, h = (
                int(box["x"]),
                int(box["y"]),
                int(box["width"]),
                int(box["height"]),
            )
            mask_target = cv2.rectangle(mask_target, (x, y), (x + w, y + h), 255, -1)

    return mask_target


def annotation_output(
    annotation: pd.DataFrame, contours: list, image_name: str = "FA1.png"
) -> pd.DataFrame:
    """This function gets the annotation dataframe created by VIA and overwrites the image regions for the selected image name (in filename column).

    Args:
        annotation (pd.DataFrame): pandas dataframe created by VIA.
        contours (List): List of contours.
        image_name (str, optional): The image name existing in the dataframe. Defaults to "FA1.png".

    Returns:
        pd.DataFrame: The overwritten dataframe.
    """
    df = annotation.copy()
    output = pd.DataFrame(data=np.zeros((len(contours), 7)), columns=df.columns)
    sample = df[df.loc[:, "filename"] == image_name]
    sample = sample.iloc[0, :]
    sample.loc["region_count"] = int(len(contours))
    for i in range(0, len(contours)):
        contour = contours[i]
        w = np.max(contour, axis=0)[0, 0] - np.min(contour, axis=0)[0, 0]
        h = np.max(contour, axis=0)[0, 1] - np.min(contour, axis=0)[0, 1]
        x = np.mean(contour, axis=0)[0, 0] - w / 2
        y = np.mean(contour, axis=0)[0, 1] - h / 2
        row = sample.iloc[:]
        row.loc["region_id"] = int(i)

        shape = (
            '{"name":"rect","x":'
            + str(int(x))
            + ',"y":'
            + str(int(y))
            + ',"width":'
            + str(int(w))
            + ',"height":'
            + str(int(h))
            + "}"
        )
        row.loc["region_shape_attributes"] = shape
        output.iloc[i, :] = row

    if len(contours) == 0:
        row = sample.iloc[:]
        output.loc[0, :] = row

    output.iloc[:, 1] = output.iloc[:, 1].astype("int64")
    output.iloc[:, 3] = output.iloc[:, 1].astype("int64")
    output.iloc[:, 4] = output.iloc[:, 1].astype("int64")

    return output


def csv_builder(
    path: str, annotation: pd.DataFrame, contours_giantlist: list
) -> pd.DataFrame:
    """The function generates the dataframe with overwritten annotations.

    Args:
        path (str): path to image seq folder.
        annotation (pd.DataFrame): The csv file to overwrite.
        contours_giantlist (list): List of contours.

    Returns:
        pd.DataFrame: The overwritten dataframe.
    """
    list_of_images = [
        ele
        for ele in sorted_alphanumeric(os.listdir(path))
        if (ele.endswith("jpg") or ele.endswith("png"))
    ]
    for i, file in enumerate(list_of_images):
        contours = contours_giantlist[i]
        output = annotation_output(annotation, contours, image_name=file)
        if i == 0:
            df = output
        else:
            df = pd.concat([df, output])
    return df


def initialize_registration(ref_img, cfg):
    if cfg.module_name == "Tomoko":
        func = cfg.module(cfg.data_path)
        # func.set_temp_ref(ref_img)
        # func.run_powershell()
    else:
        func = cfg.module(ref_img)
    return func


def apply_mask_transform(mask, H):
    mask = cv2.warpPerspective(mask, H, (mask.shape[1], mask.shape[0]))
    mask = (mask * 255).astype("uint8")
    return mask


def apply_registraton_offline(cfg):
    image_list = sorted_alphanumeric(
        [
            ele
            for ele in os.listdir(cfg.data_path)
            if (ele.endswith("jpg") or ele.endswith("png"))
        ]
    )
    csv_file = [ele for ele in os.listdir(cfg.data_path) if ele.endswith("csv")][0]
    annotations = pd.read_csv(os.path.join(cfg.data_path, csv_file))
    ref_img = cv2.imread(os.path.join(cfg.data_path, image_list[0]), 0)

    registration_method = initialize_registration(ref_img, cfg)

    print(f"Reg method is {type(registration_method)}")
    cfg.time = []
    giant_contour_list = []
    for i in range(1, len(image_list)):
        image = cv2.imread(os.path.join(cfg.data_path, image_list[i]), 0)
        mask = read_annotation(image_list[i], annotations)
        if cfg.module_name == "Tomoko":
            trg = {}
            trg["filename"] = image_list[i]
            trg["image"] = image
        else:
            trg = image
        t0 = time.time()
        image_transformed = registration_method(trg)
        t1 = time.time() - t0
        cfg.time.append(t1)
        if not registration_method.success:
            print("Registration failed!")
            continue
        if not tracking_eval(ref_img, image_transformed):
            print("Registration failed to satisfy the evaluation")
            continue
        transformed_mask = apply_mask_transform(mask, registration_method.H_transform)
        mask_contours, _ = cv2.findContours(
            transformed_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        giant_contour_list.append(mask_contours)
        cv2.imwrite(os.path.join(cfg.save_path, image_list[i]), image_transformed)

        print(
            f"Image {image_list[i]} successfully transformed and saved to save folder"
        )

    print(f"Average registration time is: {np.mean(cfg.time)}")

    if os.path.exists(os.path.join(cfg.data_path, "/temp_ref.png")):
        os.remove(os.path.join(cfg.data_path, "/temp_ref.png"))

    df = csv_builder(cfg.save_path, annotations, giant_contour_list)
    cols = annotations.columns.tolist()
    df = df[cols]
    df.to_csv(cfg.save_path + "/annotaion.csv", index=False)


def get_args():
    cfg = ml_collections.ConfigDict()
    parser = argparse.ArgumentParser()
    parser.add_argument("--module_name", type=str)
    parser.add_argument("--data_path", type=str, default="1")
    parser.add_argument(
        "--save_path",
        type=str,
        default=r"C:\Users\Javad\Desktop\Dataset\SVOregcomparison",
    )

    args = parser.parse_args()
    cfg.module_name = args.module_name
    cfg.save_path = args.save_path
    cfg.data_path = os.path.join(
        r"C:\Users\Javad\Desktop\Dataset\Floater Data\Dataset_eval", f"{args.data_path}"
    )

    print(f"Data Folder: {cfg.data_path}")

    cfg.save_path = os.path.join(
        cfg.save_path, f"{cfg.module_name}_" + cfg.data_path[-1]
    )
    if not os.path.exists(cfg.save_path):
        os.mkdir(cfg.save_path)

    print(f"Save Folder: {cfg.save_path}")

    if not os.path.exists(cfg.data_path):
        raise ValueError("Could not find dataset path")

    if args.module_name == "Superpoint":
        cfg.module = ApplySuperPointReg
    elif args.module_name == "Tomoko":
        cfg.module = TomokoTrakcer
    else:
        raise NotImplementedError("Please set your module Superpoint or Tomoko")

    return cfg


def test_annotations(cfg):
    out = cv2.VideoWriter(
        r"C:\Users\Javad\Desktop\Model_inference/test_reg.mp4",
        cv2.VideoWriter_fourcc(*"DIVX"),
        8,
        (512, 512),
    )

    images = sorted_alphanumeric(
        [ele for ele in os.listdir(cfg.save_path) if ele.endswith("png")]
    )
    annotation = [ele for ele in os.listdir(cfg.save_path) if ele.endswith("csv")][0]

    df = pd.read_csv(os.path.join(cfg.save_path, annotation))

    for i in range(0, len(images)):
        image = cv2.imread(os.path.join(cfg.save_path, images[i]))
        mask = read_annotation(images[i], df)
        mask_contours, _ = cv2.findContours(
            mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        image = cv2.drawContours(image, mask_contours, -1, (255, 0, 0), 3)
        out.write(image)

    out.release()


if __name__ == "__main__":
    cfg = get_args()
    apply_registraton_offline(cfg)
    test_annotations(cfg)
