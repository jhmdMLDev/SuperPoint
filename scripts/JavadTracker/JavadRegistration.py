import os
import sys
import time

import cv2
import numpy as np
import SimpleITK as sitk
import torch
from PIL import Image
from skimage.morphology import skeletonize, thin
from torch import nn
from torchvision import transforms

sys.path.insert(0, os.getcwd())
from scripts.JavadTracker.Javadutils import *


def DL_SLO_Segment_Veins(
    slo,
    model_chkpt_path="/efs/model_check2023/VeinSeg/SLO_VeinSeg.model",
    gpu_id=0,
    threshold=120,
    small_obj_filter=300,
    USE_THRESH=True,
    FILTER_SMALL_HOLES=True,
    SHRINK_VEINS=True,
):
    """
    Use ML model to generate vein segmentation mask for given SLO image
    :param slo: SLO image (cv2 image numpy array of size [W,H]. If >1 channel is given, it will be converted to gray)
    :param model_chkpt_path: (string) path to ML model state dictionary
    :param gpu_id: gpu index to use (if there are multiple devices) If there is not GPU, this parameter will be ignored
        and the CPU will be used
    :param small_obj_filter: When removing pixel "islands" (disconnected regions remaining after thresholding), this number
        specifies the minimum allowed pixel "island" area/size which is not removed
    :param threshold: (int 0->255) the lower this value, the lower the applied threshold value, revealing more of the
        vein structure but potentially more artifacts/noise
    :return: vein segmentation mask (numpy array [W,H])
    """
    from skimage import morphology

    if len(slo.shape) == 3:
        slo = cv2.cvtColor(np.copy(slo), cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.6, tileGridSize=(12, 12))
    slo = clahe.apply(slo)

    device = torch.device(
        "cuda:{}".format(gpu_id) if torch.cuda.is_available() else "cpu"
    )

    input_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize([0], [0.225])]
    )

    model = Get_unet_MRI(init_features=32, downscale_x4=True)
    model = model.to(device)

    # load model checkpoint:
    model.load_state_dict(torch.load(model_chkpt_path, map_location=device))
    model.eval()

    img = Image.fromarray(slo)
    batch_input = input_transforms(img).to(device).unsqueeze(0)
    pred = model(batch_input)

    vein_mask = pred[0][0].detach().cpu().numpy()
    vein_mask = (vein_mask / np.max(vein_mask) * 255.0).astype("uint8")

    if not USE_THRESH:
        return vein_mask

    if threshold == 120:
        _, BW = cv2.threshold(vein_mask, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, BW = cv2.threshold(vein_mask, threshold, 255, cv2.THRESH_BINARY)
    BW = BW > 0.5
    BW2 = morphology.remove_small_objects(BW, min_size=small_obj_filter, connectivity=1)

    if FILTER_SMALL_HOLES:
        BW2 = ~morphology.remove_small_objects(~BW2, min_size=2000, connectivity=1)

    vein_mask = np.array(BW2 * 255.0).astype("uint8")

    # Shrink SLO veins to match smaller FA vein thickness
    if SHRINK_VEINS:
        vein_mask = cv2.morphologyEx(
            vein_mask, cv2.MORPH_ERODE, np.ones((4, 4), np.uint8)
        )

    return vein_mask


def SLO_SLO_registration_transform(
    img_SLO1_color,
    img_SLO2_color,
    subscale=4,
    original=False,
    skel=False,
    THIN=True,
    small_obj_filter=800,
    th=50,
    pad=600,
    iter_stik=400,
    LNR_REGISTRATION=True,
    SLO_frame=False,
    LNR_grid_size=7,
    LNR_iterations=2,
    SLO_vein_seg_model_path=r"/efs/model_check2023/VeinSeg/SLO_VeinSeg.model",
    path=None,
    logger=None,
    path_preview_save=None,
):
    t_rigid1 = time.time()

    Skel_func = empty
    if THIN:
        Skel_func = thin
    elif skel:
        Skel_func = skeletonize

    if path is not None:
        cv2.imwrite(path + "\SLO1.png", img_SLO1_color)
        cv2.imwrite(path + "\SLO2.png", img_SLO2_color)

    img_SLO2 = np.array(img_SLO2_color)
    img_SLO2 = (img_SLO2 / np.max(img_SLO2) * 255.0).astype("uint8")
    img_SLO1 = np.array(img_SLO1_color)
    img_SLO1 = (img_SLO1 / np.max(img_SLO1) * 255.0).astype("uint8")

    if not SLO_frame:
        t1 = time.time()
        img_SLO2_veins_org = DL_SLO_Segment_Veins(
            slo=img_SLO2,
            gpu_id=0,
            threshold=th,
            small_obj_filter=small_obj_filter,
            model_chkpt_path=SLO_vein_seg_model_path,
            USE_THRESH=True,
            SHRINK_VEINS=False,
        )
        t2 = time.time()
        # print("Vein Segmentation: {:.2f}".format(t2 - t1))
        img_SLO2_veins = (Skel_func(img_SLO2_veins_org > 127) * 255).astype(
            "uint8"
        )  # if (skel or THIN) else img_SLO2_veins_org        # img_SLO2_veins_org    # cv2.normalize(skeletonize((img_SLO2_veins_org>127)*1)*1, None, 255,0, cv2.NORM_MINMAX, cv2.CV_8UC1)
        if path is not None:
            cv2.imwrite(path + "/SLO2_Veins.png", img_SLO2_veins_org)
            # cv2.imwrite(path + '\__SLO2_Veins_ed.png', img_SLO2_veins_org)
        if logger is not None:
            logger.debug("First image vein mask generated")
    else:
        img_SLO2_veins_org = img_SLO2
        img_SLO2_veins = img_SLO2

    if not SLO_frame:
        # FA vein segmentation of whole image (macro) before cropping. Used for initial alignment
        img_SLO1_veins_org = DL_SLO_Segment_Veins(
            slo=img_SLO1,
            gpu_id=0,
            threshold=th,
            small_obj_filter=small_obj_filter,
            model_chkpt_path=SLO_vein_seg_model_path,
            USE_THRESH=True,
            SHRINK_VEINS=False,
        )
        img_SLO1_veins = (Skel_func(img_SLO1_veins_org > 127) * 255).astype(
            "uint8"
        )  # if (skel or THIN) else img_SLO1_veins_org         # img_SLO2_veins_org    # cv2.normalize(skeletonize((img_SLO2_veins_org>127)*1)*1, None, 255,0, cv2.NORM_MINMAX, cv2.CV_8UC1)
        if path is not None:
            cv2.imwrite(path + "/SLO1_Veins.png", img_SLO1_veins_org)
        if logger is not None:
            logger.debug("Second image vein mask generated")
    else:
        img_SLO1_veins_org = img_SLO1
        img_SLO1_veins = img_SLO1

    img_SLO1_veins = cv2.copyMakeBorder(
        img_SLO1_veins, pad, pad, pad, pad, cv2.BORDER_CONSTANT, 0
    )
    img_SLO1_veins_org_padded = cv2.copyMakeBorder(
        img_SLO1_veins_org, pad, pad, pad, pad, cv2.BORDER_CONSTANT, 0
    )
    img_SLO1_color_padded = cv2.copyMakeBorder(
        img_SLO1_color, pad, pad, pad, pad, cv2.BORDER_CONSTANT, 0
    )
    if path is not None:
        cv2.imwrite(path + "/SLO1_Veins_padded.png", img_SLO1_veins)

    t1 = time.time()
    align_scale, align_pos, _ = Initial_Align_Images(
        img_FA_vein=img_SLO1_veins,
        img_SLO_vein=img_SLO2_veins,
        subscaling_factor=2,
        scale_sweep_range=0.04,
        scale_sweep_steps=9,
        scale_AR_sweep_range=0.10,
        scale_AR_sweep_steps=10,
        offset_sweep_range=100,
        offset_sweep_steps=21,
        start_scale=1,
        start_pos=None,
        threads=3,
    )

    # align_scale = (1, 1)  #### Change
    # align_pos = np.array(img_SLO1_veins_org_padded.shape)//2

    t2 = time.time()
    # print("Initial Alignment: {:.2f}".format(t2 - t1))

    if path is not None:
        prv_initial_alignment = Preview_Initial_Alignment(
            img_FA=img_SLO1_veins_org_padded,
            img_SLO=img_SLO2_veins_org,
            align_scale=align_scale,
            align_pos=align_pos,
            save_path=path + "/initial_alignment.png",
        )
    if logger is not None:
        logger.debug("Initial allignment is done")

    _, img_SLO2_veins_scaled = Reframe_Image(
        img=img_SLO2_veins,
        new_scale=align_scale,
        new_offset=align_pos,
        frame_size=img_SLO1_veins.shape,
    )
    if path is not None:
        cv2.imwrite(path + "/cropped_SLO.png", img_SLO2_veins_scaled)

    # Obtain cropped FA image with a slightly increased (10%) border size
    SLO_window_size = np.multiply(img_SLO2_veins.shape, align_scale)
    register_ref_window_size = np.multiply(SLO_window_size, (1, 1))  #### Change to 1.1
    register_ref_window_size = (
        int(register_ref_window_size[0]),
        int(register_ref_window_size[1]),
    )  #### Change, substitute [1] and [0]

    if original:
        img_SLO2_veins = img_SLO2
        img_SLO1_veins = img_SLO1

    img_SLO1_veins_cropped = Crop_Image_Window(
        img=img_SLO1_veins,
        window_size=register_ref_window_size,
        window_center=align_pos,
    )
    if path is not None:
        cv2.imwrite(path + "/cropped_SLO1.png", img_SLO1_veins_cropped)

    img_target = img_SLO2_veins_scaled  # img_SLO2_veins_scaled    #cv2.normalize((img_SLO2_veins_scaled>1)*1, None, 255,0, cv2.NORM_MINMAX, cv2.CV_8UC1)
    img_ref = img_SLO1_veins_cropped

    # Display normalized images
    if path is not None:
        save_comparison_fig(
            img_ref, img_target, img_ref.shape[0], path + "/normalized.png"
        )
        cv2.imwrite(path + "\img_target_land1.png", img_target)
        cv2.imwrite(path + "\img_ref_land1.png", img_ref)

    # Search for landmarks and match based on several descriptor algorithms
    img_SLO1_veins_org_land = Crop_Image_Window(
        img=img_SLO1_veins_org_padded,
        window_size=register_ref_window_size,
        window_center=align_pos,
    )
    img_ref_land = (Skel_func(img_SLO1_veins_org_land > 127) * 255).astype("uint8")

    _, img_SLO2_veins_org_land = Reframe_Image(
        img=img_SLO2_veins_org,
        new_scale=align_scale,
        new_offset=align_pos,
        frame_size=img_SLO1_veins.shape,
    )
    img_target_land = (Skel_func(img_SLO2_veins_org_land > 127) * 255).astype("uint8")

    # if (skel or THIN) and not original and subscale>1: # and 0:
    #     connector = Vein_conncetor(img_target_land, l_max=25//(subscale-1), l_step=2, l_min=1, theta_class=[1,2,3]) # , l_step=2
    #     img_target_land = connector.run()
    #     connector = Vein_conncetor(img_ref_land, l_max=25//(subscale-1), l_step=2, l_min=1, theta_class=[1,2,3]) # , l_step=2
    #     img_ref_land = connector.run()

    if logger is not None:
        logger.debug("Rigid registration is started")
    t1 = time.time()
    p1, p2 = Landmark_Description_Set(
        img_ref=img_ref_land, img_target=img_target_land, path=path, type="slo_slo"
    )
    t2 = time.time()
    # print("Landmark Detection and Matching: {:.2f}".format(t2 - t1))
    if logger is not None:
        logger.debug("Landmarks are found")

    # Calculate registration homography matrix
    # Note: This homography matrix assumes input image dimensions of (img_SLO2.shape*align_scale)

    if len(p2) < 4 or len(p1) < 4:
        print(
            "Error: (Registration_src) fewer than 4 matches found with reference"
            " ({}) or target ({})".format(len(p2), len(p1))
        )
        if path is not None:
            f, ax = plt.subplots(1, 2, figsize=(15, 15))
            ax[0].imshow(img_SLO1, cmap="gray")
            ax[1].imshow(img_SLO2, cmap="gray")
            ax[0].axis("off")
            ax[1].axis("off")
            plt.savefig(path + "/Registration_Summary.png")
            plt.close()
        if logger is not None:
            logger.debug("Not enough landmarks were found, quiting the registration")
        return img_SLO2_color[:, :, 0], None

    H_register1, mask = cv2.findHomography(p2, p1, cv2.RANSAC, 15)
    t_rigid2 = time.time()
    # print("Rigid Registration time: {:.2f}".format(t_rigid2 - t_rigid1))
    # H_register1 = np.eye(3) #####Change it

    # detect if the register registration is invalid. If so, return the initial alignment homography matrix
    # angle between image plane and transformation plane normals
    theta = -np.arctan2(H_register1[0, 1], H_register1[0, 0]) * 180 / 3.14159
    rigid_transform_valid = True
    if abs(theta) > 20:
        rigid_transform_valid = False
    # rigid_transform_valid = True
    if not rigid_transform_valid:
        print("Invalid rigid transform. Returning initial alignment transform instead.")
        H_register1 = np.eye(3)
        if logger is not None:
            logger.debug(
                "Invalid rigid transform. Returning initial alignment transform instead."
            )

    img_target_transformed = cv2.warpPerspective(
        img_target_land, H_register1, (img_ref_land.shape[1], img_ref_land.shape[0])
    )
    if (
        0
        and not original
        and not Check_New_Registartion(
            img_ref_land, img_target_land, img_target_transformed, th=1.05
        )
    ):
        H_register1 = np.eye(3)
        print("Not improved enough. Returning initial alignment transform instead.")
        if logger is not None:
            logger.debug(
                "Not improved enough. Returning initial alignment transform instead."
            )

    # Construct homography for scaling the target image to the cropped size before applying registration
    H_preregister_scale = np.eye(3)
    H_preregister_scale[0, 0] *= align_scale[1]
    H_preregister_scale[1, 1] *= align_scale[0]

    # Obtain homography matrix for cropping FA to SLO alignment window size
    H_crop_FA = Crop_Image_Homography(
        window_size=register_ref_window_size, window_center=align_pos
    )

    # Combine preregistration and registration
    H_register2 = np.matmul(H_register1, H_preregister_scale)

    # Combine registration homography with the inverse of FA cropping homography to transform the registered image
    # into non-cropped FA space
    H_register_full = np.matmul(np.linalg.inv(H_crop_FA.copy()), H_register2.copy())

    if path is not None:
        img_SLO2_veins_transformed = cv2.warpPerspective(
            img_SLO2_veins_org,
            H_register_full,
            (img_SLO1_veins.shape[1], img_SLO1_veins.shape[0]),
        )

        img_SLO2_color_transformed = cv2.warpPerspective(
            img_SLO2_color,
            H_register_full,
            (img_SLO1_veins.shape[1], img_SLO1_veins.shape[0]),
        )

        prv_rigid = Preview_Registration_Overlay(
            img_base=img_SLO1_veins_org,
            img_transformed=img_SLO2_veins_transformed[pad:-pad, pad:-pad],
            intensity=1,
            path=path + "/rigid_registration_veins_preview.png",
        )
        prv_rigid_color = Preview_Registration_Overlay(
            img_base=img_SLO1_color,
            img_transformed=img_SLO2_color_transformed[pad:-pad, pad:-pad],
            intensity=1,
            path=path + "/rigid_registration_preview.png",
        )

    output = img_SLO2_color_transformed[pad:-pad, pad:-pad]
    return output, H_register_full


if __name__ == "__main__":
    ref = cv2.imread(
        "/efs/datasets/floater_dataset_edited/Tracker_sample/clean_non_reg/322.jpg"
    )
    trg = cv2.imread(
        "/efs/datasets/floater_dataset_edited/Tracker_sample/clean_non_reg/323.jpg"
    )

    slo_size = (768, 768)
    ref = cv2.resize(ref, slo_size)
    trg = cv2.resize(trg, slo_size)

    SLO_reg, _ = SLO_SLO_registration_transform(
        ref,
        trg,
        pad=300,
        iter_stik=500,
        LNR_REGISTRATION=False,
        path="/home/ubuntu/Projects/Floater_tracking/.samples/Javad/test",
    )
    print(SLO_reg.shape)

    img_3d = np.dstack([ref[:, :, 0], np.zeros_like(ref[:, :, 0]), SLO_reg])
    print("img_3d size: ", img_3d.shape)

    cv2.imwrite(
        "/home/ubuntu/Projects/Floater_tracking/.samples/Javad/test/reg.png", img_3d
    )
