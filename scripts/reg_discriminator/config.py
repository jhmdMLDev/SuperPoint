import json
import os

import ml_collections
import torch
from torchvision import transforms


def get_train_config(folder_name, parent_folder_name="real_time"):
    config = ml_collections.ConfigDict()

    config.CHECK_PATH = f"/efs/model_check2023/{parent_folder_name}/" + folder_name
    if not os.path.isdir(config.CHECK_PATH):
        os.mkdir(config.CHECK_PATH)

    config.DATA_DIRECTORY = "/efs/datasets/floater_dataset_edited/dataset_train"

    # PARAMETERS
    config.LEARNING_RATE = 0.005
    config.BATCH_SIZE = 16
    config.WEIGHT_DECAY = 0.00001
    config.EPOCHS = 101
    config.NUM_WORKERS = 0
    config.PIN_MEMORY = True
    config.MOMENTUM = 0.9
    config.GAMMA = 0.8
    config.RELOAD = True
    config.BEST_MODEL_PATH = r"/efs/model_check2023/registration_disc/registration_disc3/seqlen15unetepoch_44_model.pt"

    # STATES
    config.ID = "seqlen15unet"

    # debug
    config.DEBUG_TRAIN = False

    # device
    config.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # transforms
    config.transforms_image = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
            transforms.RandomAutocontrast(p=0.5),
        ]
    )

    config.random_affine = transforms.Compose(
        [transforms.RandomAffine((-10, 10), translate=(0.05, 0.05), scale=(0.98, 1.02))]
    )

    save_hyperparams(config, "cfgtrain.json")

    return config


def get_val_config(folder_name, parent_folder_name="real_time"):
    config = get_train_config(folder_name, parent_folder_name)

    config.DATA_DIRECTORY = "/efs/datasets/floater_dataset_edited/dataset_val"

    config.NON_TRACKED_DATA = (
        r"/efs/datasets/floater_dataset_edited/Tracker_sample/clean_non_reg"
    )

    config.BATCH_SIZE = 16

    # transforms
    config.transforms_image = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    config.random_affine = transforms.Compose(
        [transforms.RandomAffine((-5, 5), translate=(0.05, 0.05), scale=(0.98, 1.02))]
    )

    save_hyperparams(config, "cfgval.json")

    return config


def save_hyperparams(config, file_name):
    cfg_json = config.to_json_best_effort()
    json_path = os.path.join(config.CHECK_PATH, file_name)
    with open(json_path, "w") as outfile:
        json.dump(cfg_json, outfile)
