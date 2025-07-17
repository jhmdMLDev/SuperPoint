import os
import sys

import ml_collections

sys.path.insert(0, os.getcwd())
from scripts.SuperPoint.main import ApplySuperPointReg
from scripts.robustness_eval.registration_eval import RegistrationValidation
from scripts.SuperPoint.superpoint import SuperPointFrontend


def get_superpoint_eval_config(model_path, feature_name, epoch):
    config = ml_collections.ConfigDict()
    config.feature_name = feature_name
    config.MAIN_PATH = r"/efs/model_check2023/SUPERPOINT_TUNE/FloaterAugv4"
    # config.MAIN_PATH = r"C:\Users\Javad\Desktop\Model_inference\Registration"
    config.SAVE_PATH = os.path.join(
        config.MAIN_PATH, config.feature_name + "_" + str(epoch)
    )
    if not os.path.isdir(config.SAVE_PATH):
        os.mkdir(config.SAVE_PATH)
    config.PLOT_PATH = os.path.join(config.SAVE_PATH, "performance_plot.png")
    config.image_path = r"/efs/datasets/Registration_benchmark/images"
    config.annotation_path = r"/efs/datasets/Registration_benchmark/annotations"
    # config.image_path = r"C:\Users\Javad\Desktop\Dataset\Registration_benchmark\images"
    # config.annotation_path = (
    #     r"C:\Users\Javad\Desktop\Dataset\Registration_benchmark\annotations"
    # )
    config.CURRENT_MODEL_PATH = model_path

    superpoint_new = "Superpoint_Floaterv4"
    config.methods = [ApplySuperPointReg, ApplySuperPointReg]
    config.method_names = ["Superpoint_pretrained", superpoint_new]
    # config.method_weights = [
    #     r"/efs/model_check2023/Superpoint/superpoint_v1.pth",
    #     model_path,
    # ]

    config.method_weights = [
        r"/efs/model_check2023/Superpoint/superpoint_v1.pth",
        model_path,
    ]
    config.threshold = [
        0.015,
        0.015,
    ]

    config.test_accuracy_only = False
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


def superpoint_comparison(model_path, epoch):
    features = ["Bluriness_Level", "Opacity_Size", "Opacity_Level"]
    for feature in features:
        superpoint_eval_cfg = get_superpoint_eval_config(model_path, feature, epoch)
        superpoint_val = RegistrationValidation(superpoint_eval_cfg)
        superpoint_val(superpoint_eval_cfg)


if __name__ == "__main__":
    epoch = 20
    model_path = r"/efs/model_check2023/SUPERPOINT_TUNE/FloaterAugv4/epoch_22_model.pt"
    superpoint_comparison(model_path, epoch)
