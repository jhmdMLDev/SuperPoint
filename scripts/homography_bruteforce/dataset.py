import os

import cv2
import torch


def sorted_alphanumeric(data):
    def convert(text):
        return int(text) if text.isdigit() else text.lower()

    def alphanum_key(key):
        return [convert(c) for c in re.split("([0-9]+)", key)]

    return sorted(data, key=alphanum_key)


class RegistrationDataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        super().__init__()
        self.data_path = data_path
        self.img_list = sorted_alphanumeric(
            [item for item in os.listdir(data_path) if item.endswith("jpg")]
        )

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_path, self.img_list[idx])
        img = cv2.imread(img_path, 0) / 255.0
        return torch.tensor(img).unsqueeze(0)

    def __len__(self):
        return len(self.img_list)
