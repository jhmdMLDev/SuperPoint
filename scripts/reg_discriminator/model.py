import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 8, 8)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 128, 8, 8)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 4, 4)
        self.bn3 = nn.BatchNorm2d(256)
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        return self.flatten(x)


class RegistrationDiscriminator(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        self.encoder = Encoder(cfg)
        self.linear1 = nn.Linear(1024, 512)
        self.dropout1 = nn.Dropout(p=0.2)
        self.linear2 = nn.Linear(512, 64)
        self.dropout2 = nn.Dropout(p=0.2)
        self.linear3 = nn.Linear(64, 1)
        self.dropout3 = nn.Dropout(p=0.1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        x1 = self.encoder(x1)
        x2 = self.encoder(x2)
        x = torch.abs(x2 - x1)
        x = self.relu(self.linear1(self.dropout1(x)))
        x = self.relu(self.linear2(self.dropout2(x)))
        x = self.linear3(self.dropout3(x))
        x = self.sigmoid(x)
        return x


class RegistrationDiscriminatorInference(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        self.encoder = Encoder(cfg)
        self.linear1 = nn.Linear(1024, 512)
        self.dropout1 = nn.Dropout(p=0.2)
        self.linear2 = nn.Linear(512, 64)
        self.dropout2 = nn.Dropout(p=0.2)
        self.linear3 = nn.Linear(64, 1)
        self.dropout3 = nn.Dropout(p=0.1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def add_ref(self, x_ref):
        self.x_ref = self.encoder(x_ref)

    def forward(self, x):
        x = self.encoder(x)
        x = torch.abs(x - self.x_ref)
        x = self.relu(self.linear1(self.dropout1(x)))
        x = self.relu(self.linear2(self.dropout2(x)))
        x = self.linear3(self.dropout3(x))
        x = self.sigmoid(x)
        return x


if __name__ == "__main__":
    import time

    model = RegistrationDiscriminator().cuda()
    x1 = torch.rand((1000, 1, 512, 512)).cuda()
    x2 = torch.rand((1000, 1, 512, 512)).cuda()
    y = model(x1, x2)
    y = model(x1, x2)
    t0 = time.time()
    y = model(x1, x2)
    print("time: ", time.time() - t0)
    print("outshape:", y.shape)

    loss = torch.sum(y)
    loss.backward()
