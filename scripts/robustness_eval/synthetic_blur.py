import os
import sys

import cv2
import numpy as np


def create_blur(radius, center, opacity=40, bluriness=5):
    np.random.seed(0)
    img = np.zeros((512, 512), np.uint8)
    img = cv2.circle(img, center, radius, 255, -1)
    blurred_area = np.mean(img / 255.0)
    noise_mask = np.random.randint(opacity, opacity + 10, img.shape) / 255.0
    img = noise_mask * img
    img = cv2.GaussianBlur(img.astype("uint8"), (bluriness, bluriness), 0)
    img = 255 - img
    return img / 255.0, blurred_area


def apply_blur(img, radius, center, opacity=40, bluriness=5):
    if radius == 0:
        img.astype("uint8"), 0
    blur, blurred_area = create_blur(radius, center, opacity, bluriness)
    image_blur = cv2.GaussianBlur(img.astype("uint8"), (bluriness, bluriness), 0)
    image_blur = 1.0 * (blur < 1) * image_blur
    img[np.nonzero(image_blur)] = image_blur[np.nonzero(image_blur)]
    img = blur * img
    return img.astype("uint8"), blurred_area
