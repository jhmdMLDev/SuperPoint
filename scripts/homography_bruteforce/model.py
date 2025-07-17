import time
import warnings

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="Default grid_sample and affine_grid behavior has changed to align_corners=False since 1.3.0. Please specify align_corners=True if the old behavior is desired.",
)


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class SpatialTransformerNetwork(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SpatialTransformerNetwork, self).__init__()
        self.out_channels = out_channels

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.conv4 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.bn4 = nn.BatchNorm2d(out_channels)

        self.conv5 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.bn5 = nn.BatchNorm2d(out_channels)

        self.fc1 = nn.Linear(out_channels * 512 * 512, 512)
        self.fc2 = nn.Linear(512, 6)

        self.tanh = nn.Tanh()

        self.theta_correct_mul = torch.tensor(
            [[[0.0, 0.0, 1], [0.0, 0.0, 1]]],
        ).cuda()

        self.theta_correct_add = torch.tensor(
            [[[0.0, -1.0, 0.0], [1.0, 0.0, 0.0]]],
        ).cuda()

        # test
        self._test_theta = (
            Variable(torch.randn(2, 3).type(torch.FloatTensor), requires_grad=True)
            .cuda()
            .unsqueeze(0)
        )

        self.no_translation_thetha = torch.tensor(
            [[[0.0, -1.0, 0.5], [1.0, 0.0, 0.5]]],
        ).cuda()

    def forward(self, x):
        # x1 = F.relu(self.bn1(self.conv1(x)))
        # x1 = F.relu(self.bn2(self.conv2(x1)))
        # x1 = F.relu(self.bn3(self.conv3(x1)))
        # x1 = F.relu(self.bn4(self.conv4(x1)))
        # x1 = F.relu(self.bn5(self.conv5(x1)))

        # x1 = x1.view(-1, self.out_channels * 512 * 512)

        # x1 = F.relu(self.fc1(x1))
        # x1 = self.tanh(self.fc2(x1))

        # theta = x1.view(-1, 2, 3)
        # theta = theta * self.theta_correct_mul + self.theta_correct_add
        # self._test_theta = self._test_theta*self.theta_correct_mul + self.theta_correct_add
        # theta = self._test_theta
        # theta = theta.view(-1, 2, 3)

        theta = self.no_translation_thetha.repeat(200, 1, 1)

        print("shape: ", theta.shape)
        # print(theta)
        t0 = time.time()
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid, align_corners=True)
        print("time: ", time.time() - t0)

        return x


if __name__ == "__main__":
    import cv2
    import numpy as np

    model = SpatialTransformerNetwork(in_channels=1, out_channels=1).cuda()
    # x = 1.0*(100*torch.rand((1, 1, 256, 256)).cuda() > 50)
    img = cv2.imread(r"/efs/datasets/floater_dataset_edited/dataset_val/89/322.jpg", 0)
    x = torch.tensor(img).unsqueeze(0).unsqueeze(0).cuda().repeat(200, 1, 1, 1)
    x = x / 255.0
    print(x.shape)
    y = model(x)
    y = model(x)
    y = model(x)
    y = model(x)
    y = model(x)
    print("max", torch.max(y))

    x = np.array(255 * x[0, 0].cpu()).astype("uint8")
    y = np.array(255 * y[0, 0].cpu()).astype("uint8")

    cv2.imwrite("/home/ubuntu/Projects/Floater_tracking/.samples/x.png", x)
    cv2.imwrite("/home/ubuntu/Projects/Floater_tracking/.samples/y.png", y)
