import os

import ml_collections
import torch

from torchvision import transforms


def get_superpoint_train_cfg(folder_name, parent_folder_name="SUPERPOINT_TUNE"):
    config = ml_collections.ConfigDict()
    # USER PATH

    # config.BEST_MODEL_PATH = "/efs/model_check2023/Superpoint/superpoint_v1.pth"
    config.BEST_MODEL_PATH = (
        "/mnt-fs/efs-ml-model/SVO/pretrained/SuperPoint/epoch_22_model.pt"
    )

    config.CHECK_PATH = (
        f"/mnt-fs/efs-ml-model/Experiments/{parent_folder_name}/" + folder_name
    )
    if not os.path.isdir(config.CHECK_PATH):
        os.mkdir(config.CHECK_PATH)

    config.DATA_DIRECTORY = "/mnt-fs/efs-ml-data/SuperpointDataset/train"

    # PARAMETERS
    config.BATCH_SIZE = 1
    config.LEARNING_RATE = 0.00001
    config.WEIGHT_DECAY = 0.0001
    config.MOMENTUM = 0.9
    config.GAMMA = 0.8
    config.EPOCHS = 101
    config.NUM_WORKERS = 0
    config.PIN_MEMORY = True

    # loss
    config.margin = 2.0
    config.weight_keypoint = 100
    config.weight_background = 1

    # device
    config.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # transform
    config.transforms_image = transforms.Compose(
        [
            transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
            transforms.RandomAutocontrast(p=0.5),
            transforms.GaussianBlur(3, sigma=(0.1, 2.0)),
        ]
    )

    return config


def get_superpoint_val_cfg(folder_name, parent_folder_name="SUPERPOINT_TUNE"):
    config = get_superpoint_train_cfg(folder_name, parent_folder_name)
    config.DATA_DIRECTORY = "/mnt-fs/efs-ml-data/SuperpointDataset/validation"
    config.BATCH_SIZE = 1

    return config
