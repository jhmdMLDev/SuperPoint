import torch


class SuperPointNet(torch.nn.Module):
    """Pytorch definition of SuperPoint Network."""

    def __init__(self):
        super(SuperPointNet, self).__init__()
        self.relu = torch.nn.ReLU(inplace=True)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256
        # Shared Encoder.
        self.conv1a = torch.nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = torch.nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = torch.nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = torch.nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = torch.nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = torch.nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = torch.nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = torch.nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)
        # Detector Head.
        self.convPa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convPb = torch.nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)
        # Descriptor Head.
        self.convDa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = torch.nn.Conv2d(c5, d1, kernel_size=1, stride=1, padding=0)

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
        dn = torch.norm(desc, p=2, dim=1)  # Compute the norm.
        desc = desc.div(torch.unsqueeze(dn, 1))  # Divide by norm to normalize.
        return semi, desc

    def get_heatmap(self, semi):
        H, W = 512, 512
        cell = 8
        dense = torch.exp(semi)  # Softmax.
        dense = dense / (torch.sum(dense, axis=0) + 0.00001)  # Should sum to 1.
        # Remove dustbin.
        nodust = dense[:-1, :, :]
        # Reshape to get full resolution heatmap.
        Hc = int(H / cell)
        Wc = int(W / cell)
        nodust_transposed = nodust.permute(1, 2, 0)
        heatmap = torch.reshape(nodust_transposed, [Hc, Wc, cell, cell])
        heatmap = heatmap.permute(0, 2, 1, 3)
        heatmap = torch.reshape(heatmap, [Hc * cell, Wc * cell])
        return heatmap

    def get_descriptors(self, coarse_desc, keypoint_heatmap):
        samp_pts = torch.nonzero(keypoint_heatmap > 0)[:, 1:].permute(1, 0)
        H, W = 512, 512
        D = coarse_desc.shape[1]
        samp_pts_mod = (samp_pts / 256.0) - 1.0
        samp_pts_mod = samp_pts_mod.transpose(0, 1).contiguous()
        samp_pts_mod = samp_pts_mod.view(1, 1, -1, 2).float().cuda()
        desc = torch.nn.functional.grid_sample(coarse_desc, samp_pts_mod)
        desc = desc.view(D, -1)
        desc_norm = desc.norm(dim=0, keepdim=True)
        desc_normalized = desc / desc_norm
        return desc_normalized


if __name__ == "__main__":
    import cv2

    image = cv2.imread(
        r"C:\Users\Javad\Desktop\Dataset\77\2022.01.27-15.40.15_0034.jpg", 0
    )
    x = torch.tensor(image).unsqueeze(0).unsqueeze(0) / 255.0
    model = SuperPointNet()
    semi, desc = model(x)
    print(semi.shape)
    print(desc.shape)
