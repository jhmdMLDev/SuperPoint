import argparse
import os
import sys

import cv2
import matplotlib.pyplot as plt
import ml_collections
import numpy as np
import scipy
import torch
import torch.nn.functional as F

sys.path.insert(0, os.getcwd())
from scripts.robustness_eval.dataset import RegistrationValidationDataset
from scripts.SuperPoint.main import ApplySuperPointReg
from scripts.SuperPoint.train.synthetic_floater_utils import cast_synthetic_floater

# screen -S reg_performance python scripts/deeplearning_registration/registration_eval.py --feature_name resolution

seed_value = 42
np.random.seed(seed_value)


def compute_auc(x, y):
    f = scipy.interpolate.interp1d(x, y)

    a = np.min(x)
    b = np.max(x)

    area, error = scipy.integrate.quad(f, a, b)

    return area


def plot_performance_per_feature(accuracy, features, param_name, cfg):
    if param_name == "Resolution":
        xlabel = "Average Kernel Size"
    elif param_name == "Floater":
        xlabel = "Floater Patch Size"
    else:
        raise ValueError("Not valid feature name")

    plt.figure(figsize=(12, 12))
    colors = ["b", "r"]
    markeredgecolors = ["r", "b"]
    markers = ["*", "o"]
    for i, name in enumerate(cfg.method_names):
        x = features[name]
        area = 0
        if len(accuracy[name]) > 1:
            x_n = [ele - np.min(x) for ele in x]
            x_n = [ele / np.max(x_n) for ele in x_n]
            area = round(compute_auc(x_n, accuracy[name]), 2)
        label = name + " AUC: " + str(area)
        plt.plot(
            features[name],
            accuracy[name],
            colors[i],
            label=label,
            marker=markers[i],
            markeredgecolor=markeredgecolors[i],
            linewidth=2,
            markersize=8,
        )

    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel("Registration Accuarcy")
    plt.title("Registration Performance")
    plt.savefig(cfg.PLOT_PATH)


class RegistrationParamSweep:
    def __init__(self, cfg):
        self.param_name = cfg.param_name
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

    def apply_param_sweep(self, image, param_val, param_name):
        if param_name == "Resolution":
            return self.change_resolution(image, param_val)
        elif param_name == "Floater":
            return self.apply_floater(image, param_val)
        else:
            raise NotImplemented("Not implemented yet!")

    def change_resolution(self, image, kernel_size):
        if kernel_size < 2:
            return image
        new_size = int(image.shape[0] / kernel_size)
        img_resized = cv2.resize(image, (new_size, new_size))
        result = cv2.resize(img_resized, (512, 512))
        return result

    def apply_floater(self, image, floater_size):
        if floater_size < 1:
            return image
        size = floater_size * 50
        result = cast_synthetic_floater(
            image, size=np.random.randint(size, size + 1, 2)
        )
        return result

    def create_video(self, out, ref, targ_transferred):
        track = np.dstack([targ_transferred, targ_transferred, targ_transferred])
        ref = np.dstack([ref, ref, ref])
        frame = np.concatenate([ref, track], axis=1)
        out.write(frame)

    def set_model(self, method_class, method_weights, threshold, param_val):
        self.ref = self.apply_param_sweep(self.ref, param_val, self.param_name)

        if "Superpoint" in self.method_name:
            method_class = method_class(self.ref, method_weights, threshold)

        else:
            method_class = method_class(self.ref)

        return method_class

    def registration_eval(self, H_register1, pts_trg, pts_ref, reg_done):
        if not reg_done:
            return 100

        p2_trg_transform = cv2.transform(pts_trg.reshape(1, -1, 2), H_register1)[
            :, :, 0:2
        ].squeeze(0)
        error = np.sqrt(np.sum((p2_trg_transform - pts_ref) ** 2)) / pts_ref.shape[0]
        error = min(error, 100)
        return error

    def apply_reg_parameter(
        self,
        cfg,
        param_val,
        iteration,
        method_class,
        method_name,
        method_weights,
        thresholds,
    ):
        self.method_name = method_name
        video_size = (1024, 512)
        error_list = []
        record_video = False
        if iteration % 5 == 0:
            record_video = True
            out = cv2.VideoWriter(
                os.path.join(
                    cfg.SAVE_PATH,
                    f"method{method_name}p{param_val}_reg.mp4",
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
                method_class, method_weights, thresholds, param_val
            )
            self.ref = method_func.reference_image

            for idx in range(0, len(self.dataset.images_path_list)):
                trg, pts_trg = self.dataset[idx]
                trg_name = self.dataset.images_path_list[idx]
                trg = self.apply_param_sweep(trg, param_val, self.param_name)
                targ_transferred = method_func(trg)
                error = self.registration_eval(
                    method_func.H_transform,
                    pts_trg,
                    self.dataset.ref_points,
                    method_func.success,
                )

                error_list.append(error)
                print(f"for image {trg_name} error is:", error)
                if record_video:
                    self.create_video(out, self.ref, targ_transferred)

        accuracy = max(0, (100 - np.mean(error_list))) / 100
        print("accuracy", accuracy)

        if self.param_name == "Resolution":
            feature = param_val

        elif self.param_name == "Floater":
            feature = param_val * 50
        else:
            raise ValueError("Not valid feature name")

        self.features[method_name].append(feature)
        self.accuracy[method_name].append(accuracy)
        print(
            f"method {method_name}: for the blurred area ratio of {self.features[method_name][-1]}, average accuracy is {self.accuracy[method_name][-1]}"
        )

    def __call__(self, cfg):
        for iteration in range(0, 11):
            # cfg.param = iteration if (self.param_name == "Resolution") else 0
            cfg.param = iteration if (self.param_name == "Floater") else 0

            for methods_idx in range(0, len(self.method_names)):
                self.apply_reg_parameter(
                    cfg,
                    iteration,
                    cfg.param,
                    self.methods[methods_idx],
                    self.method_names[methods_idx],
                    self.method_weights[methods_idx],
                    self.thresholds[methods_idx],
                )

            plot_performance_per_feature(
                self.accuracy, self.features, self.param_name, cfg
            )


def get_reg_config():
    config = ml_collections.ConfigDict()
    parser = argparse.ArgumentParser()
    parser.add_argument("--param_name", type=str)
    args = parser.parse_args()
    config.param_name = args.param_name
    # config.MAIN_PATH = r"C:\Users\Javad\Desktop\Model_inference\Registration"
    config.MAIN_PATH = (
        r"/efs/model_inference/registration/param_sweep/Experiment_Floaterv4_pretrained"
    )
    config.SAVE_PATH = os.path.join(config.MAIN_PATH, config.param_name)
    if not os.path.isdir(config.SAVE_PATH):
        os.mkdir(config.SAVE_PATH)
    config.PLOT_PATH = os.path.join(config.SAVE_PATH, "floater_pretrained_plot.png")
    config.image_path = r"/efs/datasets/Registration_benchmark/images"
    config.annotation_path = r"/efs/datasets/Registration_benchmark/annotations"
    # config.image_path = r"C:\Users\Javad\Desktop\Dataset\Registration_benchmark\images"
    # config.annotation_path = (
    #     r"C:\Users\Javad\Desktop\Dataset\Registration_benchmark\annotations"
    # )

    config.methods = [ApplySuperPointReg, ApplySuperPointReg]
    config.method_names = ["Superpoint_Pretrained", "Superpoint_Floaterv4"]
    config.method_weights = [
        r"/efs/model_check2023/Superpoint/superpoint_v1.pth",
        r"/efs/model_check2023/SUPERPOINT_TUNE/FloaterAugv4/epoch_22_model.pt",
    ]
    config.threshold = [
        0.015,
        0.015,
    ]
    return config


if __name__ == "__main__":
    cfg = get_reg_config()
    RegistrationParamSweep(cfg)(cfg)
