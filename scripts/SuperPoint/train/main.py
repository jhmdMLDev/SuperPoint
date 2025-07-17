import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import ml_collections
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from torchsummary import summary

sys.path.insert(0, os.getcwd())
from scripts.SuperPoint.train.config import (
    get_superpoint_train_cfg,
    get_superpoint_val_cfg,
)
from scripts.SuperPoint.train.model import SuperPointNet
from scripts.SuperPoint.train.loss import SuperPointLoss
from scripts.SuperPoint.train.dataset import SuperPointFineTuneDataset
from scripts.SuperPoint.train.train_utils import (
    train,
    model_checkpoint,
    loss_per_epoch,
    collect_samples,
    preview,
)

#  screen -S "FloaterAugv5" -L -Logfile ./screenlogs/FloaterAug.log python ./scripts/SuperPoint/train/main.py --save_folder_name "FloaterAugv5"


def main(save_folder_name):
    # cfg
    cfg_args = [True, save_folder_name]
    train_cfg = get_superpoint_train_cfg(folder_name=cfg_args[1])
    val_cfg = get_superpoint_val_cfg(folder_name=cfg_args[1])

    experiment_dictionary = ml_collections.ConfigDict()
    experiment_dictionary.params = {
        "save_folder_name": save_folder_name,
        "margin": train_cfg.margin,
        "keypoint_weight": train_cfg.weight_keypoint,
        "background_weight": train_cfg.weight_background,
    }
    experiment_dictionary.results = []

    # load the Unet model
    model = SuperPointNet()

    # # load weights
    model.load_state_dict(
        torch.load(train_cfg.BEST_MODEL_PATH, map_location=train_cfg.DEVICE)
    )

    # summary(model, (train_cfg.SEQ_LEN, 512, 512), device="cpu")
    model.to(train_cfg.DEVICE)

    # determine optimizer and loss function
    optimizer = optim.SGD(
        model.parameters(),
        lr=train_cfg.LEARNING_RATE,
        momentum=train_cfg.MOMENTUM,
        weight_decay=train_cfg.WEIGHT_DECAY,
    )

    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=train_cfg.GAMMA)
    loss_fn = SuperPointLoss(cfg=train_cfg)

    # Load dataset
    # dataset
    dataset_train = SuperPointFineTuneDataset(cfg=train_cfg)
    dataset_val = SuperPointFineTuneDataset(cfg=val_cfg)

    train_loader = DataLoader(
        dataset=dataset_train,
        batch_size=train_cfg.BATCH_SIZE,
        num_workers=train_cfg.NUM_WORKERS,
        shuffle=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        dataset=dataset_val,
        batch_size=val_cfg.BATCH_SIZE,
        num_workers=val_cfg.NUM_WORKERS,
        shuffle=True,
        drop_last=True,
    )

    print(
        f"The size of training dataset is {len(dataset_train)} and the size of val dataset is {len(dataset_val)}"
    )

    train_loss_list = []
    val_loss_list = []

    data_samples = collect_samples(dataset_val)

    experiment_dictionary_json = experiment_dictionary.to_json_best_effort()
    json_path = os.path.join(train_cfg.CHECK_PATH, "experiment_details.json")
    with open(json_path, "w") as outfile:
        json.dump(experiment_dictionary_json, outfile)

    for epoch in range(train_cfg.EPOCHS):
        if epoch % 2 == 0 and epoch != 0:
            model_checkpoint(model, epoch, train_cfg)
            loss_per_epoch(train_loss_list, val_loss_list, epoch, train_cfg)

        if epoch % 2 == 0:
            preview(data_samples, model, epoch, train_cfg)

        train_loss, val_loss = train(
            train_loader, val_loader, model, optimizer, loss_fn, epoch, train_cfg
        )

        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)


if __name__ == "__main__":
    # screen -S "FloaterAugv4" python scripts/SuperPoint/train/main.py --save_folder_name "FloaterAugv4"
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser()

    # Add arguments to the parser
    parser.add_argument("--save_folder_name", type=str)

    # Parse the command-line arguments
    args = parser.parse_args()

    main(save_folder_name=args.save_folder_name)
