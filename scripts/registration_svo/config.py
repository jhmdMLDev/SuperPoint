import os

import ml_collections
import torch
from torchvision import transforms


def get_regsvo_cfg(folder_name, parent_folder_name="SVOREG"):
    config = ml_collections.ConfigDict()
    # USER PATH

    config.BEST_MODEL_PATH = (
        "/efs/Model_checks/hyperparams/fullsize/Instance2/seqlen15unetepoch_8_model.pt"
    )

    config.CHECK_PATH = f"/efs/model_check2023/{parent_folder_name}/" + folder_name
    if not os.path.isdir(config.CHECK_PATH):
        os.mkdir(config.CHECK_PATH)

    config.DATA_DIRECTORY = "/efs/datasets/floater_dataset_edited/dataset_train"

    # PARAMETERS
    config.WINDOW_SIZE = 15
    config.SEQ_LEN = config.WINDOW_SIZE

    # device
    config.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # transforms
    config.transforms_image = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    config.transforms_mask = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    return config
