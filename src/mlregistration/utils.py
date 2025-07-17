import cv2
import numpy as np


def segment_vein(image):
    if image is None:
        raise ValueError("Image not found or unable to read the image.")

    # Apply a Gaussian blur to the image
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # Use adaptive thresholding to highlight the veins
    veins = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    # Apply morphological operations to clean up the segmentation
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(veins, cv2.MORPH_CLOSE, kernel, iterations=2)

    return cleaned
