import os

import numpy as np
import cv2
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

torch.autograd.set_detect_anomaly(True)


def train(train_loader, val_loader, model, optimizer, loss_fn, epoch, cfg):
    loop = tqdm(train_loader, leave=True)
    train_loss = []
    print(f"epoch: {epoch+1} out of {cfg.EPOCHS}")
    mean_val_loss = "TBD"
    for batch_idx, data in enumerate(loop):
        optimizer.zero_grad()

        train_loss, loss = inference(model, data, train_loss, loss_fn, cfg)
        loss.backward()
        optimizer.step()

        if batch_idx == len(train_loader) - 1:
            val_loss = val(val_loader, model, cfg, loss_fn)
            mean_val_loss = np.mean(val_loss)

        mean_train_loss = np.mean(train_loss)
        update_loop_prefix(loop, mean_train_loss, mean_val_loss)

    return mean_train_loss, mean_val_loss


def val(val_loader, model, cfg, loss_fn):
    val_loss = []
    for batch_idx, data in enumerate(val_loader):
        val_loss, _ = inference(model, data, val_loss, loss_fn, cfg)
    return val_loss


def inference(model, data, loss_list, loss_fn, cfg):
    heatmap = {}
    descriptors = {}

    # get ground truth
    heatmap["ground_truth"], heatmap["ground_truth_warped"] = (
        data["heatmap"].to(cfg.DEVICE),
        data["warped_heatmap"].to(cfg.DEVICE),
    )
    # get the image scores and coarse desc
    pred_scores, pred_coarse_desc = model(data["image"].to(cfg.DEVICE))
    heatmap["predicted"] = model.get_heatmap(pred_scores.squeeze(0))
    descriptors["predicted"] = model.get_descriptors(
        pred_coarse_desc, heatmap["ground_truth"]
    )
    # get the warped image scores and coarse desc
    pred_w_scores, pred__coarse_w_desc = model(data["warped_image"].to(cfg.DEVICE))
    heatmap["predicted_warped"] = model.get_heatmap(pred_w_scores.squeeze(0))
    descriptors["predicted_warped"] = model.get_descriptors(
        pred__coarse_w_desc, heatmap["ground_truth_warped"]
    )

    loss_detector = loss_fn.keypoint_loss(heatmap["ground_truth"], heatmap["predicted"])
    loss_detector_warped = loss_fn.keypoint_loss(
        heatmap["ground_truth_warped"], heatmap["predicted_warped"]
    )
    loss_descriptors = loss_fn.descriptor_loss(
        descriptors["predicted"], descriptors["predicted_warped"]
    )
    loss_keypoint_warped = loss_fn.keypoint_warp_loss(
        heatmap["predicted"], heatmap["predicted_warped"], data["affine_matrix"]
    )

    loss = (
        0.3 * loss_detector
        + 0.3 * loss_detector_warped
        + 0.3 * loss_descriptors
        + 0.1 * loss_keypoint_warped
    )

    loss_list.append(loss.item())

    return loss_list, loss


def update_loop_prefix(loop, mean_train_loss, mean_val_loss):
    loop.set_postfix(train_loss=mean_train_loss, val_loss=mean_val_loss)


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
    plt.savefig(os.path.join(cfg.CHECK_PATH, "_loss_per_epoch.png"))


def model_checkpoint(model, epoch, cfg):
    # save the model
    check_point_name = "epoch_" + str(epoch) + "_model.pt"
    save_path = os.path.join(cfg.CHECK_PATH, check_point_name)
    cfg.LAST_MODEL_PATH = save_path
    output_save = open(save_path, mode="wb")
    torch.save(model.state_dict(), output_save)


def overlay_kp(image, heatmap, gt_heatmap):
    preview_img = (
        (255 * image).squeeze(0).squeeze(0).cpu().detach().numpy().astype("uint8")
    )
    preview_img_3d = np.dstack([preview_img, preview_img, preview_img])
    heatmap_numpy = heatmap.cpu().detach().numpy()
    coordinates = np.where(heatmap_numpy > 0.015)

    gt_numpy = gt_heatmap.cpu().detach().numpy()
    coordinates_gt = np.where(gt_numpy > 0.015)

    for y, x in zip(coordinates_gt[0], coordinates_gt[1]):
        preview_img_3d = cv2.circle(
            preview_img_3d, (x, y), radius=5, color=(0, 255, 0), thickness=2
        )

    for y, x in zip(coordinates[0], coordinates[1]):
        preview_img_3d = cv2.putText(
            preview_img_3d,
            "x",
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.25,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

    return preview_img_3d


def preview(data_samples, model, epoch, cfg):
    for i, data in enumerate(data_samples):
        heatmap = {}
        # get ground truth
        heatmap["ground_truth"], heatmap["ground_truth_warped"] = (
            data["heatmap"].to(cfg.DEVICE),
            data["warped_heatmap"].to(cfg.DEVICE),
        )
        # get the image scores and coarse desc
        pred_scores, pred_coarse_desc = model(data["image"].to(cfg.DEVICE).unsqueeze(0))
        heatmap["predicted"] = model.get_heatmap(pred_scores.squeeze(0))

        # get the warped image scores and coarse desc
        pred_w_scores, pred__coarse_w_desc = model(
            data["warped_image"].to(cfg.DEVICE).unsqueeze(0)
        )
        heatmap["predicted_warped"] = model.get_heatmap(pred_w_scores.squeeze(0))

        preview_img = overlay_kp(
            data["image"], heatmap["predicted"], heatmap["ground_truth"]
        )
        preview_img_warped = overlay_kp(
            data["warped_image"],
            heatmap["predicted_warped"],
            heatmap["ground_truth_warped"],
        )

        preview_final = np.concatenate([preview_img, preview_img_warped], axis=1)

        filename = "/sample" + str(i) + "_epoch_" + str(epoch) + ".jpg"

        cv2.imwrite(cfg.CHECK_PATH + filename, preview_final)


def collect_samples(dataset_val):
    indices = [100, 500, 800]
    data_samples = [dataset_val[i] for i in indices]
    return data_samples
