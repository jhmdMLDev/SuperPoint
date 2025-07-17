import os
import sys
import re
from ast import literal_eval

import numpy as np
import pandas as pd
import cv2
import torch


def sorted_alphanumeric(data):
    def convert(text):
        return int(text) if text.isdigit() else text.lower()

    def alphanum_key(key):
        return [convert(c) for c in re.split("([0-9]+)", key)]

    return sorted(data, key=alphanum_key)


def create_dataset_folder(data_dir, data_folders, window_size, use_fixed_annotation):
    # appending the image directories in the list
    image_sequences = []
    bboxes = []
    len_folders = []
    for i in range(len(data_folders)):
        files = sorted_alphanumeric(os.listdir(data_dir + "/" + data_folders[i]))
        if len(files) <= window_size:
            continue
        fixed_annot_idx = None

        for j in range(len(files)):
            if files[j].endswith("fixed_annotations.csv"):
                fixed_annot_idx = j
                break

        if fixed_annot_idx is not None:
            files.pop(fixed_annot_idx)
        for j in range(len(files)):
            if files[j].endswith(".csv"):
                df = pd.read_csv(data_dir + "/" + data_folders[i] + "/" + files[j])
                df = df.loc[:, ["filename", "region_shape_attributes"]]

        image_sequence = []
        bbox_sequence = []
        for j in range(len(files)):
            if files[j].endswith(".jpg"):
                image_path = data_dir + "/" + data_folders[i] + "/" + files[j]
                image_sequence.append(image_path)
                bbx = df[df.loc[:, "filename"] == files[j]]
                bbox_sequence.append(bbx)

        image_sequences.append(image_sequence)
        bboxes.append(bbox_sequence)
        len_folders.append(len(image_sequence) - window_size + 1)
    return image_sequences, bboxes, len_folders


class RegistrationSVOdataset(torch.utils.data.Dataset):
    def __init__(self, cfg):
        self.data_folders = sorted_alphanumeric(os.listdir(cfg.DATA_DIRECTORY))
        self.data_folders = [
            ele for ele in self.data_folders if ele not in cfg.EXCLUSION_LIST
        ]
        self.transforms_image = cfg.transforms_image
        self.transforms_mask = cfg.transforms_mask

        self.image_sequences, self.bboxes, self.len_folders = create_dataset_folder(
            cfg.DATA_DIRECTORY,
            self.data_folders,
            self.window_size,
            self.have_contour_masks,
            self.use_fixed_annotation,
        )

    def obtain_rec_mask(self, folder_idx, frame_num):
        mask_target = np.zeros((512, 512), dtype="uint8")
        df = self.bboxes[folder_idx][frame_num]
        df = df.reset_index()
        for i in range(0, len(df)):
            box = literal_eval(df.loc[i, "region_shape_attributes"])
            if box != {}:
                x, y, w, h = (
                    int(box["x"]),
                    int(box["y"]),
                    int(box["width"]),
                    int(box["height"]),
                )
                mask_target = cv2.rectangle(
                    mask_target, (x, y), (x + w, y + h), 255, -1
                )

        return mask_target

    def apply_post_processing(self, image, transform, seq):
        image = np.array(image).astype("uint8")
        torch.set_rng_state(self.rng_state)  # load saved RNG state
        image_data = transform(image)
        seq.append(np.array(image_data))

    def __getitem__(self, idx):
        # real data
        self.image_seq = []
        self.mask_seq = []

        # get the folder index and start frame
        folder_idx, start_frame = self.find_index(idx)

        # change manual seed each time
        rng_state_val = torch.random.manual_seed(torch.random.seed())
        # change rng state
        # save RNG state to ensure consistent augmentation with image and mask
        self.rng_state = torch.get_rng_state()
        self.folder_idx = folder_idx
        for frame_num in range(start_frame, start_frame + self.window_size):
            # obtain image
            image = cv2.imread(self.image_sequences[folder_idx][frame_num], 0)
            mask_target = self.obtain_rec_mask(folder_idx, frame_num)

            # Final step (transform and append to sequence)
            self.apply_post_processing(image, self.transforms_image, self.image_seq)
            self.apply_post_processing(mask_target, self.transforms_mask, self.mask_seq)

        # output the sequence in torch tensor
        batch_input = torch.tensor(np.array(self.image_seq)).squeeze(1)
        batch_output = torch.tensor(np.array(self.mask_seq)).squeeze(1)

        return batch_input, batch_output

    def __len__(self):
        """This function gives the len of the dataset.

        Returns:
            [int]: len of dataset object
        """

        return sum(self.len_folders)
