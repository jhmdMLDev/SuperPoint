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
from reg_discriminator.config import (
    get_train_config,
    get_val_config,
)
from reg_discriminator.dataset import (
    RegistrationDiscDataset,
)
from reg_discriminator.model import (
    RegistrationDiscriminator,
)
from reg_discriminator.train import (
    collect_samples,
    model_save,
    model_visualization,
    train,
)


def loss_per_epoch(train_loss_list, val_loss_list, epoch, cfg):
    """This function plots loss per epoch for train and validation.

    Args:
        train_loss_list ([list]): list of train losses.
        val_loss_list ([type]):  list of val losses.
        epoch ([int]): epoch number for plot name
        check_path ([str]): the path to save the figure
        ID (str, optional): The unique id for the name of the figure. Defaults to "unet32_".
    """
    plt.figure(figsize=(4, 4))
    plt.plot(list(range(0, epoch)), train_loss_list, "b")
    plt.plot(list(range(0, epoch)), val_loss_list, "r")
    plt.legend(["train loss", "validation loss"])
    plt.title("loss per epoch")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.savefig(
        os.path.join(
            cfg.CHECK_PATH, cfg.ID + "epoch_" + str(epoch) + "_loss_per_epoch.png"
        )
    )


def main(
    save_folder_name,
    parent_folder,
):
    # cfg
    train_cfg = get_train_config(save_folder_name, parent_folder)
    val_cfg = get_val_config(save_folder_name, parent_folder)

    # load the Unet model
    model = RegistrationDiscriminator(train_cfg)
    model.to(train_cfg.DEVICE)

    # determine optimizer and loss function
    optimizer = optim.SGD(
        model.parameters(),
        lr=train_cfg.LEARNING_RATE,
        momentum=train_cfg.MOMENTUM,
        weight_decay=train_cfg.WEIGHT_DECAY,
    )

    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=train_cfg.GAMMA)
    loss_fn = nn.MSELoss()

    # Load dataset
    # dataset
    dataset_train = RegistrationDiscDataset(train_cfg)
    dataset_val = RegistrationDiscDataset(val_cfg)

    # select some sequences to preview
    list_of_refs, list_of_images, list_of_outputs = collect_samples(dataset_val)

    train_loader = DataLoader(
        dataset=dataset_train,
        batch_size=train_cfg.BATCH_SIZE,
        num_workers=train_cfg.NUM_WORKERS,
        shuffle=True,
        drop_last=True,
    )

    num_of_val_sample = len(dataset_val)
    print(
        f"The size of training dataset is {len(dataset_train)} and the size of val dataset is {len(dataset_val)}"
    )

    val_loader = DataLoader(
        dataset=dataset_val,
        batch_size=val_cfg.BATCH_SIZE,
        num_workers=val_cfg.NUM_WORKERS,
        shuffle=True,
        drop_last=True,
    )

    train_loss_list = []
    val_loss_list = []

    for epoch in range(train_cfg.EPOCHS):
        if epoch % 2 == 0:
            model_save(model, train_cfg, epoch)
            model_visualization(
                model,
                epoch,
                list_of_refs,
                list_of_images,
                list_of_outputs,
                train_cfg,
            )

        # plot
        if epoch % 2 == 0 and epoch != 0:
            loss_per_epoch(train_loss_list, val_loss_list, epoch, train_cfg)

        train_loss, val_loss = train(
            train_loader,
            val_loader,
            model,
            optimizer,
            loss_fn,
            epoch,
            scheduler,
            train_cfg,
        )

        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)


if __name__ == "__main__":
    # screen -S "model_reg" python ./scripts/deeplearning_registration/reg_discriminator/main.py --save_folder_name "registration_disc1" --parent_folder registration_disc
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser()

    # Add arguments to the parser
    parser.add_argument("--save_folder_name", type=str)
    parser.add_argument("--parent_folder", type=str)

    # Parse the command-line arguments
    args = parser.parse_args()

    main(
        save_folder_name=args.save_folder_name,
        parent_folder=args.parent_folder,
    )
