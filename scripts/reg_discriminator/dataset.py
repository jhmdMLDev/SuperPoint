import os
import re

import cv2
import numpy as np
import torch


def sorted_alphanumeric(data):
    def convert(text):
        return int(text) if text.isdigit() else text.lower()

    def alphanum_key(key):
        return [convert(c) for c in re.split("([0-9]+)", key)]

    return sorted(data, key=alphanum_key)


def create_dataset_folder(data_dir, data_folders):
    # appending the image directories in the list
    image_sequences = []
    len_folders = []
    for i in range(len(data_folders)):
        files = sorted_alphanumeric(os.listdir(data_dir + "/" + data_folders[i]))
        image_sequence = []
        for j in range(len(files)):
            if files[j].endswith(".jpg"):
                image_path = data_dir + "/" + data_folders[i] + "/" + files[j]
                image_sequence.append(image_path)

        image_sequences.append(image_sequence)
        len_folders.append(len(image_sequence) - 1)
    return image_sequences, len_folders


def randomcrop(img):
    if np.random.random() < 0.5:
        return img

    bg = np.zeros(img.shape).astype("uint8")
    padx = np.random.randint(15, 30)
    pady = np.random.randint(15, 30)

    bg[padx:-padx, pady:-pady] = img[padx:-padx, pady:-pady]
    return bg


class RegistrationDiscDataset(torch.utils.data.Dataset):
    def __init__(self, cfg):
        self.data_folders = sorted_alphanumeric(os.listdir(cfg.DATA_DIRECTORY))
        self.image_sequences = []
        self.len_folders = []
        self.transforms_image = cfg.transforms_image
        self.random_affine = cfg.random_affine
        self.demo = False
        self.folder_ = []
        self.debug_train = cfg.DEBUG_TRAIN

        # appending the image directories in the list
        self.image_sequences, self.len_folders = create_dataset_folder(
            cfg.DATA_DIRECTORY,
            self.data_folders,
        )

    def find_index(self, idx):
        """This function gives the correct index to get the right image and sequence

        Args:
            idx ([int]): index of the sequence

        Returns:
            [int]: folder index and starting frame for sequence
        """
        index_range = list(np.cumsum(self.len_folders))
        index_range.insert(0, 0)
        for i in range(0, len(index_range) - 1):
            if index_range[i] <= idx < index_range[i + 1]:
                return i, idx - index_range[i]
            if idx >= index_range[-1]:
                return i, idx - index_range[-1]

    def apply_post_processing(self, image, transform, seq):
        image = np.array(image).astype("uint8")
        torch.set_rng_state(self.rng_state)  # load saved RNG state
        image_data = transform(image)
        seq.append(np.array(image_data))

    def __getitem__(self, idx):
        if self.debug_train:
            idx = np.random.randint(0, sum(self.len_folders))

        # get the folder index and start frame
        folder_idx, start_frame = self.find_index(idx)

        # label
        label = 1 if np.random.random() > 0.5 else 0

        frames = np.random.choice(
            len(self.image_sequences[folder_idx]), size=(2,), replace=False
        )

        ref = cv2.imread(self.image_sequences[folder_idx][frames[0]], 0)
        image = cv2.imread(self.image_sequences[folder_idx][frames[1]], 0)

        # do some random crop
        image = randomcrop(image) if label == 0 else image

        # output the sequence in torch tensor
        self.rng_state = torch.get_rng_state()
        batch_ref = self.transforms_image(ref)
        torch.set_rng_state(self.rng_state)
        batch_img = self.transforms_image(image)

        # output
        batch_img = self.random_affine(batch_img) if label == 1 else batch_img
        batch_out = torch.tensor([label])

        return batch_ref, batch_img, batch_out

    def __len__(self):
        """This function gives the len of the dataset.

        Returns:
            [int]: len of dataset object
        """
        if self.debug_train:
            return 10

        return sum(self.len_folders)


if __name__ == "__main__":
    import os
    import sys

    import matplotlib.pyplot as plt

    sys.path.insert(0, os.getcwd())
    from scripts.deeplearning_registration.reg_discriminator.config import (
        get_train_config,
    )

    cfg = get_train_config(folder_name="Test")
    dataset = RegistrationDiscDataset(cfg)

    print("len dataset is: ", len(dataset))

    # dataset.demo = True
    x1, x2, y = dataset[20000]

    print("x1 shape: ", x1.shape)
    print("x2 shape: ", x2.shape)
    print("y shape: ", y.shape)

    print("label: ", y)

    plt.imshow(x1[0], cmap="gray")
    plt.savefig("/home/ubuntu/Projects/Floater_tracking/.samples/x1.png")
    plt.imshow(x2[0], cmap="gray")
    plt.savefig("/home/ubuntu/Projects/Floater_tracking/.samples/x2.png")
