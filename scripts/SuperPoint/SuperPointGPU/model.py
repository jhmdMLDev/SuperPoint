import time

import numpy as np
import torch
import torch.nn.functional as F


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
        self.threshold = 0.015

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

        return heatmap, desc


def postprocessing(semi):
    semi = torch.exp(semi.squeeze(0))
    semi /= torch.sum(semi, dim=0) + 0.0001
    semi = semi.narrow(0, 0, semi.size(0) - 1)
    heatmap = semi.permute(1, 2, 0)
    heatmap = heatmap.view(64, 64, 8, 8)
    heatmap = heatmap.permute(0, 2, 1, 3)
    heatmap = heatmap.reshape(64 * 8, 64 * 8)
    return heatmap


def simple_nms(scores, nms_radius: int):
    zeros = torch.zeros_like(scores)
    max_mask = scores == torch.nn.functional.max_pool2d(
        scores, kernel_size=nms_radius * 2 + 1, stride=1, padding=nms_radius
    )
    for _ in range(2):
        supp_mask = (
            torch.nn.functional.max_pool2d(
                max_mask.float(),
                kernel_size=nms_radius * 2 + 1,
                stride=1,
                padding=nms_radius,
            )
            > 0
        )
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == torch.nn.functional.max_pool2d(
            supp_scores, kernel_size=nms_radius * 2 + 1, stride=1, padding=nms_radius
        )
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    return torch.where(max_mask, scores, zeros)


def remove_borders(scores, border_remove: int):
    border_mask = torch.zeros(scores.shape, device=scores.device)
    border_mask[border_remove:-border_remove, border_remove:-border_remove] = 1
    scores = border_mask * scores
    return scores


def benchmark(
    model, input_shape=(1, 1, 512, 512), dtype="fp32", nwarmup=50, nruns=10000
):
    input_data = torch.randn(input_shape)
    input_data = input_data.to("cuda")
    if dtype == "fp16":
        input_data = input_data.half()

    print("Warm up ...")
    with torch.no_grad():
        for _ in range(nwarmup):
            features = model(input_data)
    torch.cuda.synchronize()
    print("Start timing ...")
    timings = []
    with torch.no_grad():
        for i in range(1, nruns + 1):
            start_time = time.time()
            features = model(input_data)
            torch.cuda.synchronize()
            end_time = time.time()
            timings.append(end_time - start_time)
            if i % 10 == 0:
                print(
                    "Iteration %d/%d, ave batch time %.2f ms"
                    % (i, nruns, np.mean(timings) * 1000)
                )

    print("Input shape:", input_data.size())
    print("Average batch time: %.2f ms" % (np.mean(timings) * 1000))


if __name__ == "__main__":
    import time
    import onnx
    import torch.backends.cudnn as cudnn

    cudnn.benchmark = True

    print("torch: ", torch.__version__)

    model = SuperPointNet().cuda()
    model.eval()

    import torch_tensorrt

    # Create an example input tensor with the correct shape and dtype
    trt_ts_module = torch_tensorrt.compile(
        model,
        inputs=[torch_tensorrt.Input((1, 1, 512, 512), dtype=torch.float32)],
        enabled_precisions=torch.float32,  # Run with FP32
        workspace_size=1 << 22,
        truncate_long_and_double=True,
    )

    benchmark(trt_ts_module, input_shape=(1, 1, 512, 512), nruns=100)

    benchmark(model, input_shape=(1, 1, 512, 512), nruns=100)

    model.half()
    trt_model_fp16 = torch_tensorrt.compile(
        model,
        inputs=[torch_tensorrt.Input((1, 1, 512, 512), dtype=torch.half)],
        enabled_precisions={torch.half},  # Run with FP16
        workspace_size=1 << 22,
    )

    benchmark(trt_model_fp16, input_shape=(1, 1, 512, 512), dtype="fp16", nruns=100)
