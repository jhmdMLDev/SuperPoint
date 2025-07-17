import os
import sys

import numpy as np
import cv2
import torch

sys.path.insert(0, os.getcwd())
from scripts.SuperPoint.train.model import SuperPointNet


def save_model_jit(
    path: str,
    model_path: str = r"C:\Users\Javad\Desktop\Model_Check\Superpoint\superpoint_v1.pth",
):
    """This function saves the model as jit file for cpp use.
    Args:
        path (str): save path.
        model_path (str, optional): Model path. Defaults to r"/efs/Model_checks/KD/KD_W15epoch_35_model.pt".
    """
    # model structure
    model = SuperPointNet()

    # load weights
    model.load_state_dict(
        torch.load(model_path, map_location=lambda storage, loc: storage)
    )

    # An example input you would normally provide to your model's forward() method.
    example = torch.rand(1, 1, 512, 512)

    # Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
    traced_script_module = torch.jit.trace(model, example)

    traced_script_module.save(path + "/superpoint.pt")


if __name__ == "__main__":
    path = r"C:\Users\Javad\Desktop\Model_Check\Model_jit_sp"
    model_path = r"C:\Users\Javad\Desktop\Model_Check\Superpoint\superpoint_v1.pth"
    save_model_jit(path, model_path)
