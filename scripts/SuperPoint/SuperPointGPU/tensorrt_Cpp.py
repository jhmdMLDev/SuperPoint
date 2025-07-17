import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torch_tensorrt

sys.path.insert(0, os.getcwd())
from scripts.SuperPoint.SuperPointGPU.model import benchmark


class SuperPointNet(torch.nn.Module):
    """Pytorch definition of SuperPoint Network."""

    def __init__(self):
        super(SuperPointNet, self).__init__()
        self.relu = torch.nn.ReLU(inplace=True)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        # Shared Encoder.
        self.conv1a = torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.conv1b = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv2a = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv2b = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv3a = torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3b = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv4a = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv4b = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        # Detector Head.
        self.convPa = torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.convPb = torch.nn.Conv2d(256, 65, kernel_size=1, stride=1, padding=0)
        # Descriptor Head.
        self.convDa = torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.convDb = torch.nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """Forward pass that jointly computes unprocessed point and descriptor
        tensors.
        Input
          x: Image pytorch tensor shaped N x 1 x H x W.
        Output
          semi: Output point pytorch tensor shaped N x 65 x H/8 x W/8.
          desc: Output descriptor pytorch tensor shaped N x 256 x H/8 x W/8.
        """
        # Shared Encoder.
        x = self.relu(self.conv1a(x))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        # Detector Head.
        cPa = self.relu(self.convPa(x))
        semi = self.convPb(cPa)
        # Descriptor Head.
        cDa = self.relu(self.convDa(x))
        desc = self.convDb(cDa)
        desc = F.normalize(desc, p=2.0, dim=1)

        return semi, desc


def save_model_jit(model_path, save_path):
    model = SuperPointNet()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    model.float()

    # An example input you would normally provide to your model's forward() method.
    example = torch.rand(1, 1, 512, 512)

    # Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
    traced_script_module = torch.jit.trace(model, example)

    traced_script_module.save(save_path + "/superpointRT_v2.pt")


def test_superpoint_tensorrt(model_path):
    model = SuperPointNet()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    model.cuda()

    trt_ts_module = torch_tensorrt.compile(
        model,
        inputs=[torch_tensorrt.Input((1, 1, 512, 512), dtype=torch.float32)],
        enabled_precisions=torch.float32,  # Run with FP32
        workspace_size=1 << 22,
        truncate_long_and_double=True,
    )

    benchmark(trt_ts_module, input_shape=(1, 1, 512, 512), nruns=100)


if __name__ == "__main__":
    model_path = r"/efs/model_check2023/SUPERPOINT_TUNE/FloaterAugv4/epoch_22_model.pt"
    save_path = r"/efs/model_check2023/SuperPointJit"
    save_model_jit(model_path, save_path)

    test_superpoint_tensorrt(model_path)
