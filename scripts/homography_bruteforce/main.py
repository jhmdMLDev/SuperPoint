import os
import sys

import cv2
import numpy as np
import torch
import torch.optim as optim

sys.path.insert(0, os.getcwd())
from scripts.homography_bruteforce.dataset import RegistrationDataset
from scripts.homography_bruteforce.loss import NCCLoss
from scripts.homography_bruteforce.model import SpatialTransformerNetwork


def main(
    data_path=r"/efs/datasets/floater_dataset_edited/Tracker_sample/clean_non_reg",
    output_path=r"/efs/model_inference/floater/reg/sample.mp4",
    ep_len=1000,
):
    dataset = RegistrationDataset(data_path)
    model = SpatialTransformerNetwork(1, 1).cuda()
    # loss_fn = torch.nn.MSELoss()

    optimizer = optim.SGD(
        model.parameters(),
        lr=0.001,
        momentum=0.99,
        weight_decay=0.0005,
    )

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"DIVX"), 2, (512, 512))

    for idx in range(0, len(dataset)):
        if idx == 0:
            ref = dataset[idx].float().unsqueeze(0).cuda()
            continue

        img = dataset[idx]
        for it in range(0, ep_len):
            img_reg = model(img.float().unsqueeze(0).cuda())
            loss = NCCLoss(ref, img_reg.cuda())
            print("loss: ", loss)
            loss.backward(retain_graph=True)
            optimizer.step()

        registered_image = img_reg.cpu().squeeze(0).squeeze(0).detach().numpy()
        img_demo = 255 * registered_image
        img_demo3d = np.dstack([img_demo, img_demo, img_demo]).astype("uint8")
        print(f"image {idx} done!")
        out.write(img_demo3d)

    out.release()


if __name__ == "__main__":
    main()
