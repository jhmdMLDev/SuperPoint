import os
import sys

import cv2
import numpy as np
import torch
from tqdm import tqdm


def update_loop_prefix(loop, mean_train_loss, mean_val_loss):
    loop.set_postfix(train_loss=mean_train_loss, val_loss=mean_val_loss)


def train(train_loader, val_loader, model, optimizer, loss_fn, epoch, scheduler, cfg):
    loop = tqdm(train_loader, leave=True)
    train_loss = []
    print(f"epoch: {epoch+1} out of {cfg.EPOCHS}")
    mean_val_loss = "TBD"
    for batch_idx, (x1_batch, x2_batch, y_batch) in enumerate(loop):
        model.train()
        optimizer.zero_grad()

        train_loss, loss = inference(
            x1_batch, x2_batch, y_batch, model, loss_fn, cfg, train_loss
        )

        loss.backward()
        optimizer.step()

        if batch_idx == len(train_loader) - 1:
            val_loss = val(val_loader, model, cfg, loss_fn)
            mean_val_loss = np.mean(val_loss)

        mean_train_loss = np.mean(train_loss)
        update_loop_prefix(loop, mean_train_loss, mean_val_loss)

    scheduler.step()
    return mean_train_loss, mean_val_loss


def val(val_loader, model, cfg, loss_fn):
    val_loss = []
    model.eval()
    for x1_batch, x2_batch, y_batch in val_loader:
        val_loss, _ = inference(
            x1_batch, x2_batch, y_batch, model, loss_fn, cfg, val_loss
        )
    return val_loss


def inference(x1_batch, x2_batch, y_batch, model, loss_fn, cfg, loss_list):
    y_pred = model(x1_batch.to(cfg.DEVICE), x2_batch.to(cfg.DEVICE))
    loss = loss_fn(y_pred, y_batch.to(cfg.DEVICE).float())
    loss_list.append(loss.item())
    return loss_list, loss


def collect_samples(dataset_val):
    idx_list = [10, 200, 300, 800, 1200, 1400, 1500, 1800]
    list_of_refs = []
    list_of_images = []
    list_of_outputs = []

    for ele in idx_list:
        x1, x2, y = dataset_val[ele]
        list_of_refs.append(x1)
        list_of_images.append(x2)
        list_of_outputs.append(y)

    return list_of_refs, list_of_images, list_of_outputs


def model_visualization(
    model, epoch, list_of_refs, list_of_images, list_of_outputs, cfg
):
    """This function checks the model and saves the model weights and shows some image inferences.

    Args:
        model ([torch model]): The torch deep learning model
        epoch ([int]): The current epoch
        list_of_images ([list]): List of images for inference and preview
        list_of_ground_truth ([List]): List of labels for inference and preview
    """
    # show annotated image
    for i in range(0, len(list_of_refs)):
        ref = list_of_refs[i]
        img = list_of_images[i]
        out = list_of_outputs[i].item()
        reference_img = np.array(255 * ref[0], dtype="uint8")
        target_img = np.array(255 * img[0], dtype="uint8")

        # inference
        x1 = ref.unsqueeze(0)
        x2 = img.unsqueeze(0)
        model.to(cfg.DEVICE)
        model.eval()
        with torch.no_grad():
            y_pred = model(x1.to(cfg.DEVICE), x2.to(cfg.DEVICE))

        y_pred = y_pred.squeeze(0).cpu().item()

        registration_success = 1 - y_pred

        # visualize
        preview_3d = np.dstack(
            [
                reference_img,
                np.zeros(reference_img.shape),
                (0.5 * target_img).astype("uint8"),
            ]
        )
        ground_truth = "registered pair" if out == 0 else "failed pair"
        text = f"Registrtion success prediction {round(registration_success, 2)} for {ground_truth}"
        preview_padded = 255 * np.ones(
            tuple([sum(x) for x in zip((100, 0, 0), preview_3d.shape)])
        )
        preview_padded[100:, :] = preview_3d
        preview_padded = cv2.putText(
            preview_padded,
            text,
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            1,
            cv2.LINE_AA,
        )

        filename = "/" + cfg.ID + "image" + str(i) + "_epoch_" + str(epoch) + ".jpg"

        cv2.imwrite(cfg.CHECK_PATH + filename, preview_padded)


def model_save(model, cfg, epoch):
    # save the model
    check_point_name = cfg.ID + "epoch_" + str(epoch) + "_model.pt"
    save_path = os.path.join(cfg.CHECK_PATH, check_point_name)
    if epoch != 0:
        cfg.LAST_MODEL_PATH = save_path
    output_save = open(save_path, mode="wb")
    torch.save(model.state_dict(), output_save)
