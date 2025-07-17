import os
import re
import json
from ast import literal_eval

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch


def sorted_alphanumeric(data):
    def convert(text):
        return int(text) if text.isdigit() else text.lower()

    def alphanum_key(key):
        return [convert(c) for c in re.split("([0-9]+)", key)]

    return sorted(data, key=alphanum_key)


class RegistrationDataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        super().__init__()
        self.data_path = data_path
        self.img_list = sorted_alphanumeric(
            [item for item in os.listdir(data_path) if item.endswith("jpg")]
        )

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_path, self.img_list[idx])
        img = cv2.imread(img_path, 0) / 255.0
        return torch.tensor(img).unsqueeze(0)

    def __len__(self):
        return len(self.img_list)


class RegistrationValidationDataset:
    def __init__(self, main_image_folder, main_annotation_folder):
        self.main_image_folder = main_image_folder
        self.main_annotation_folder = main_annotation_folder
        self.image_folders = sorted_alphanumeric(
            [ele for ele in os.listdir(main_image_folder)]
        )
        self.annotation_folders = sorted_alphanumeric(
            [ele for ele in os.listdir(main_annotation_folder)]
        )
        self.folder_pointer = 0

        self.num_img_folders = len(self.image_folders)

    @staticmethod
    def process_annotation(json_path):
        with open(json_path) as f:
            data = json.load(f)

        annotations = data["annotations"]

        points = np.zeros((8, 2))

        for ele in annotations:
            point_name = ele["name"]
            idx = literal_eval(point_name[-1]) - 1
            points[idx, 0] = ele["keypoint"]["x"]
            points[idx, 1] = ele["keypoint"]["y"]

        return points

    def set_reference_frame(self):
        img = cv2.imread(os.path.join(self.image_folder, self.images_ref_path[0]), 0)
        points = RegistrationValidationDataset.process_annotation(
            os.path.join(self.annotation_folder, self.annotation_ref_path[0])
        )
        return img, points

    def update(self):
        self.image_folder = os.path.join(
            self.main_image_folder, self.image_folders[self.folder_pointer]
        )
        self.annotation_folder = os.path.join(
            self.main_annotation_folder, self.annotation_folders[self.folder_pointer]
        )

        self.folder_pointer += 1

        if self.folder_pointer == self.num_img_folders:
            self.folder_pointer = 0

        # create image_list
        self.images_path_list = sorted_alphanumeric(
            [
                ele
                for ele in os.listdir(os.path.join(self.image_folder))
                if ele.endswith("_annotation.png")
            ]
        )

        self.annotation_path_list = sorted_alphanumeric(
            [
                ele
                for ele in os.listdir(self.annotation_folder)
                if ele.endswith("_annotation.json")
            ]
        )

        self.images_ref_path = sorted_alphanumeric(
            [
                ele
                for ele in os.listdir(os.path.join(self.image_folder))
                if ele.endswith("_reference.png")
            ]
        )

        self.annotation_ref_path = sorted_alphanumeric(
            [
                ele
                for ele in os.listdir(os.path.join(self.annotation_folder))
                if ele.endswith("_reference.json")
            ]
        )

        self.trackeranalysis = sorted_alphanumeric(
            [
                ele
                for ele in os.listdir(os.path.join(self.image_folder))
                if ele.endswith(".json")
            ]
        )[0]
        json_path = os.path.join(self.image_folder, self.trackeranalysis)
        with open(json_path) as f:
            self.analysis_json = json.load(f)

        self.ref_img, self.ref_points = self.set_reference_frame()

        self.img_count = len(self.images_path_list)

    def __getitem__(self, idx):
        img = cv2.imread(os.path.join(self.image_folder, self.images_path_list[idx]), 0)
        points = RegistrationValidationDataset.process_annotation(
            os.path.join(self.annotation_folder, self.annotation_path_list[idx])
        )
        return img, points


if __name__ == "__main__":
    ## model train
    # data_path = r"/efs/datasets/floater_dataset_edited/Tracker_sample/clean_non_reg"
    # registrationdataset = RegistrationDataset(data_path)
    # x = registrationdataset[0]
    # plt.imshow(x[0, 0], cmap="gray")
    # plt.savefig(r"/home/ubuntu/Projects/Floater_tracking/.samples/img32")
    # print(x.shape)

    ## reg val
    image_path = r"/efs/datasets/Registration_benchmark/images"
    annotation_path = r"/efs/datasets/Registration_benchmark/annotations"

    dataset = RegistrationValidationDataset(image_path, annotation_path)
    dataset.update()
    dataset.update()
    # print(dataset.ref_img.shape, dataset.ref_points.shape)
    img, points = dataset[0]

    arr = dataset.analysis_json["files"][3908]
    print(arr["path"])
    arr = np.array(arr["homography2Ref"]).reshape(3, 3)
    print(arr)
