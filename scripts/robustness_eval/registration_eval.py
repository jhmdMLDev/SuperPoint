import os
import sys
import argparse

import numpy as np
import cv2
import scipy
import ml_collections
import matplotlib.pyplot as plt
from prettytable import PrettyTable

sys.path.insert(0, os.getcwd())
from scripts.robustness_eval.synthetic_blur import apply_blur
from scripts.robustness_eval.dataset import RegistrationValidationDataset
from scripts.JavadTracker.main import JavadRegistration
from scripts.TomokoTracker.main import TomokoTrackerReader, TomokoTrakcer
from scripts.classical_landmark.main import (
    siftregistration,
    akazeregistration,
    orbregistration,
    surfregistration,
)
from scripts.SuperPoint.main import ApplySuperPointReg

# screen -S reg_performance python scripts/deeplearning_registration/registration_eval.py --feature_name Opacity_Size


def compute_auc(x, y):
    f = scipy.interpolate.interp1d(x, y)

    a = np.min(x)
    b = np.max(x)

    area, error = scipy.integrate.quad(f, a, b)

    return area


def plot_performance_per_feature(accuracy, features, feature_name, cfg):
    if feature_name == "Opacity_Size":
        xlabel = "Opacity Size Ratio"
        default = (
            f"with Opacity: {cfg.default_opacity}, Blur Kernel: {cfg.default_blur}"
        )
    elif feature_name == "Opacity_Level":
        xlabel = "Opacity Level"
        default = f"with Size: {cfg.default_size}, Blur Kernel: {cfg.default_blur}"
    elif feature_name == "Bluriness_Level":
        xlabel = "Blur Kernel Size"
        default = f"with Opacity: {cfg.default_opacity}, Size: {cfg.default_size}"
    else:
        raise ValueError("Not valid feature name")

    plt.figure(figsize=(12, 12))
    for name in cfg.method_names:
        x = features[name]
        area = 0
        if len(accuracy[name]) > 1:
            x_n = [ele - np.min(x) for ele in x]
            x_n = [ele / np.max(x_n) for ele in x_n]
            area = round(compute_auc(x_n, accuracy[name]), 2)
        label = name + " AUC: " + str(area)
        plt.plot(features[name], accuracy[name], label=label, marker="x", linestyle="-")

    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel("Registration Accuarcy " + default)
    plt.title("Registration Performance")
    plt.savefig(cfg.PLOT_PATH)


class RegistrationValidation:
    def __init__(self, cfg):
        self.feature_name = cfg.feature_name
        self.test_accuracy_only = cfg.test_accuracy_only
        self.methods = cfg.methods
        self.method_names = cfg.method_names
        self.method_weights = cfg.method_weights
        self.thresholds = cfg.threshold
        self.features = {ele: [] for ele in self.method_names}
        self.accuracy = {ele: [] for ele in self.method_names}
        self.h_error = {ele: [] for ele in self.method_names}
        self.dataset = RegistrationValidationDataset(
            cfg.image_path, cfg.annotation_path
        )

    def create_video(self, out, ref, targ_transferred):
        overlay = np.dstack(
            [255 - ref, np.zeros_like(targ_transferred), 255 - targ_transferred]
        )
        ref = np.dstack([ref, ref, ref])
        frame = np.concatenate([ref, overlay], axis=1)
        out.write(frame)

    def set_model(
        self,
        method_class,
        feature_size,
        opacity_level,
        bluriness_level,
        optic_disc_loc,
        method_weights,
        threshold,
    ):
        if (
            (not self.test_accuracy_only)
            and (bluriness_level > 1)
            and (feature_size != 0)
        ):
            self.ref, self.blurred_area = apply_blur(
                self.ref, feature_size, optic_disc_loc, opacity_level, bluriness_level
            )
        else:
            self.blurred_area = 0

        if self.method_name == "LandmarkDetectionVeinSEg":
            self.ref = cv2.rotate(self.ref, cv2.ROTATE_90_CLOCKWISE)

        elif self.method_name == "RetinaTracker":
            method_class = method_class(self.directory_folder)
            method_class.set_temp_ref(self.ref)
            method_class.run_powershell()

        elif "Superpoint" in self.method_name:
            method_class = method_class(self.ref, method_weights, threshold)

        else:
            method_class = method_class(self.ref)
        return method_class

    def rescale_points(self, pts, scale):
        dst = np.zeros(pts.shape)
        for i in range(0, pts.shape[0]):
            dst[i, :] = (scale * pts[i, :]).astype("int64")
        return dst

    def rotate_landmarks(self, pts):
        dst = np.zeros(pts.shape)
        for i in range(0, pts.shape[0]):
            dst[i, 0] = -pts[i, 1]
            dst[i, 1] = pts[i, 0]
        return dst.astype("int64")

    def registration_eval(self, H_register1, pts_trg, pts_ref, reg_done):
        if self.method_name == "LandmarkDetectionVeinSEg":
            pts_ref = self.rotate_landmarks(pts_ref)
            pts_trg = self.rotate_landmarks(pts_trg)

        if self.resize:
            pts_trg = self.rescale_points(pts_trg, scale=(768 / 512))
            pts_ref = self.rescale_points(pts_ref, scale=(768 / 512)) + 300

        if not reg_done:
            return 100

        p2_trg_transform = cv2.transform(pts_trg.reshape(1, -1, 2), H_register1)[
            :, :, 0:2
        ].squeeze(0)
        error = np.sqrt(np.sum((p2_trg_transform - pts_ref) ** 2)) / pts_ref.shape[0]
        error = min(error, 100)
        if self.resize:
            error = (512 / 768) * error
        return error

    def compare_homography(self, trg_name, trg, H_register1):
        retinatracker = TomokoTrackerReader(self.dataset.analysis_json)
        if not isinstance(trg, dict):
            trg = {"filename": trg_name, "image": trg}
        _ = retinatracker(trg)
        H_transform = retinatracker.H_transform
        error = np.sqrt(np.mean((H_transform - H_register1) ** 2))
        return error

    def apply_reg_parameter(
        self,
        cfg,
        feature_size,
        opacity_level,
        bluriness_level,
        iteration,
        method_class,
        method_name,
        method_weights,
        thresholds,
    ):
        self.method_name = method_name
        self.resize = True if method_name == "LandmarkDetectionVeinSEg" else False
        video_size = (1536, 768) if self.resize else (1024, 512)
        error_list = []
        h_error_list = []
        record_video = False
        if iteration % 5 == 0:
            record_video = True
            out = cv2.VideoWriter(
                os.path.join(
                    cfg.SAVE_PATH,
                    f"method{method_class.__name__}s{feature_size}_o{opacity_level}_b{bluriness_level}_reg.mp4",
                ),
                cv2.VideoWriter_fourcc(*"XVID"),
                5,
                video_size,
            )

        for fold_pointer in range(0, self.dataset.num_img_folders):
            self.dataset.update()
            self.directory_folder = os.path.join(
                self.dataset.main_image_folder, self.dataset.image_folders[fold_pointer]
            )
            self.ref = self.dataset.ref_img
            method_func = self.set_model(
                method_class,
                cfg.radius,
                cfg.opacity,
                cfg.bluriness_level,
                (255, 0),
                method_weights,
                thresholds,
            )
            self.ref = method_func.reference_image

            for idx in range(0, len(self.dataset.images_path_list)):
                trg, pts_trg = self.dataset[idx]
                trg_name = self.dataset.images_path_list[idx]
                if self.method_name == "RetinaTracker":
                    trg = {"filename": trg_name, "image": trg}

                if self.method_name == "LandmarkDetectionVeinSEg":
                    trg = cv2.rotate(trg, cv2.ROTATE_90_CLOCKWISE)

                targ_transferred = method_func(trg)
                error = self.registration_eval(
                    method_func.H_transform,
                    pts_trg,
                    self.dataset.ref_points,
                    method_func.success,
                )
                if cfg.compare_homography:
                    homography_error = self.compare_homography(
                        trg_name, trg, method_func.H_transform
                    )
                    h_error_list.append(homography_error)

                error_list.append(error)
                print(f"for image {trg_name} error is:", error)
                if record_video:
                    self.create_video(out, self.ref, targ_transferred)

        accuracy = max(0, (100 - np.mean(error_list))) / 100
        print("accuracy", accuracy)

        if cfg.compare_homography:
            self.h_error[method_name].append(np.mean(h_error_list))

        if self.feature_name == "Opacity_Size":
            feature = self.blurred_area
        elif self.feature_name == "Opacity_Level":
            feature = opacity_level
        elif self.feature_name == "Bluriness_Level":
            feature = bluriness_level
        else:
            raise ValueError("Not valid feature name")

        self.features[method_name].append(feature)
        self.accuracy[method_name].append(accuracy)
        print(
            f"method {method_name}: for the blurred area ratio of {self.features[method_name][-1]}, average accuracy is {self.accuracy[method_name][-1]}"
        )

    def output_accuracy_table(self):
        table = PrettyTable()
        table.field_names = ["Name", "Accuracy"]
        for name in self.method_names:
            table.add_row([name, "%" + str(round(100 * self.accuracy[name][-1], 4))])
        print(table)

    def output_h_error_table(self):
        table = PrettyTable()
        table.field_names = ["Name", "Homography Error to RT"]
        for name in self.method_names:
            table.add_row([name, str(round(self.h_error[name][-1], 4))])
        print(table)

    def __call__(self, cfg):
        for iteration in range(0, len(cfg.bluriness_level_list)):
            cfg.radius = (
                50 * iteration
                if (self.feature_name == "Opacity_Size")
                else cfg.default_size
            )
            cfg.opacity = (
                15 * iteration
                if (self.feature_name == "Opacity_Level")
                else cfg.default_opacity
            )
            cfg.bluriness_level = (
                cfg.bluriness_level_list[iteration]
                if (self.feature_name == "Bluriness_Level")
                else cfg.default_blur
            )
            print(
                f"Doing the calculation for radius {cfg.radius},  opacity {cfg.opacity} and blur kernel size {cfg.bluriness_level}"
            )

            for methods_idx in range(0, len(self.method_names)):
                self.apply_reg_parameter(
                    cfg,
                    cfg.radius,
                    cfg.opacity,
                    cfg.bluriness_level,
                    iteration,
                    self.methods[methods_idx],
                    self.method_names[methods_idx],
                    self.method_weights[methods_idx],
                    self.thresholds[methods_idx],
                )
            if cfg.compare_homography:
                self.output_h_error_table()

            if cfg.test_accuracy_only:
                self.output_accuracy_table()
                break

            plot_performance_per_feature(
                self.accuracy, self.features, self.feature_name, cfg
            )


def get_reg_config():
    config = ml_collections.ConfigDict()
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_name", type=str)
    args = parser.parse_args()
    config.feature_name = args.feature_name
    config.MAIN_PATH = r"C:\Users\Javad\Desktop\Model_inference\Registration"
    # config.MAIN_PATH = r"/efs/model_inference/registration/auc"
    config.SAVE_PATH = os.path.join(config.MAIN_PATH, config.feature_name)
    if not os.path.isdir(config.SAVE_PATH):
        os.mkdir(config.SAVE_PATH)
    config.PLOT_PATH = os.path.join(config.SAVE_PATH, "performance_plot.png")
    # config.image_path = r"/efs/datasets/Registration_benchmark/images"
    # config.annotation_path = r"/efs/datasets/Registration_benchmark/annotations"
    config.image_path = r"C:\Users\Javad\Desktop\Dataset\Registration_benchmark\images"
    config.annotation_path = (
        r"C:\Users\Javad\Desktop\Dataset\Registration_benchmark\annotations"
    )

    # config.methods = [JavadRegistration]
    # config.method_names = ["LandmarkDetectionVeinSEg"]

    config.methods = [TomokoTrakcer, ApplySuperPointReg]
    config.method_names = ["RetinaTracker", "Superpoint"]
    config.method_weights = [
        r"C:\Users\Javad\Desktop\Model_Check\Superpoint\superpoint_v1.pth",
        None,
    ]
    config.threshold = [
        0.015,
        0.2,
    ]

    # config.methods = [ApplySuperPointReg, TomokoTrakcer, siftregistration, akazeregistration, orbregistration, surfregistration]
    # config.method_names = ["Superpoint", "RetinaTrakcer", "SIFT", "AKAZE", "ORB", "SURF"]

    config.test_accuracy_only = True
    config.compare_homography = False

    config.bluriness_level_list = [
        1,
        3,
        5,
        7,
        11,
        13,
        15,
        17,
        19,
        21,
        23,
        25,
        27,
        29,
        31,
        33,
        35,
        37,
        39,
        41,
        43,
        45,
        47,
        49,
        51,
        53,
        55,
        57,
        59,
        61,
        63,
        65,
        67,
        69,
        71,
    ]
    config.default_opacity = 40
    config.default_size = 500
    config.default_blur = 15
    return config


if __name__ == "__main__":
    cfg = get_reg_config()
    RegistrationValidation(cfg)(cfg)
