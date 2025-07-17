import time
from functools import wraps


import numpy as np
import cv2
from skimage import morphology
from scipy.ndimage import convolve

seed_value = 42
np.random.seed(seed_value)


def timer(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()
        # print(f"func:{f.__name__} took: {te-ts} sec")
        return result

    return wrap


@timer
def create_floater_mask(patch):
    mask = np.zeros((512, 512))
    edge = min(512 - patch.shape[0], 512 - patch.shape[1])
    mask[edge : edge + patch.shape[0], edge : edge + patch.shape[1]] = patch
    mask = cv2.GaussianBlur(mask, (7, 7), 3, 3)
    mask = 255 * (mask > 80)
    return mask


@timer
def create_floater_patch(size=(40, 40)):
    random_thresh = 0.65
    floater_patch = 255 * (np.random.random(size) > random_thresh)
    return floater_patch


@timer
def apply_dilate_erode(floater_patch):
    kernel_size = tuple(np.random.randint(5, 10, 2))
    kernel = np.ones(kernel_size, np.uint8)
    floater_patch = cv2.erode(floater_patch.astype("uint8"), (2, 2), iterations=1)
    floater_patch = cv2.dilate(floater_patch.astype("uint8"), kernel, iterations=3)
    return floater_patch


@timer
def fill_floater_patch(floater_patch, iterations):
    for _ in range(iterations):
        floater_patch = apply_dilate_erode(floater_patch)
    return floater_patch


@timer
def process_floater_patch(size_of_patch=(60, 60)):
    patch = create_floater_patch(size_of_patch)
    patch = collapse_waveform(patch)
    patch = cv2.medianBlur(patch, 3)
    patch = patch > 0
    patch = morphology.remove_small_objects(patch, min_size=50)
    patch = 255 * patch
    patch = fill_floater_patch(patch, iterations=1)
    patch = cv2.GaussianBlur(patch, (7, 7), 3, 3)
    patch = 255 * (patch > 80)
    return patch


# @timer
# def collapse_waveform(floater_patch):
#     floater_patch_updated = floater_patch.copy()
#     for i in range(1, floater_patch.shape[0] - 1):
#         for j in range(1, floater_patch.shape[1] - 1):
#             if np.random.random() > (
#                 1 - np.sum(floater_patch[i - 1 : i + 1, j - 1 : j + 1] / 255) / 9
#             ):
#                 floater_patch_updated[i, j] = 255

#     floater_patch_updated = floater_patch_updated.astype("uint8")
#     return floater_patch_updated


@timer
def collapse_waveform(floater_patch):
    random_values = np.random.random(floater_patch.shape)
    kernel = np.ones((3, 3)) / 9
    blurred_patch = convolve(floater_patch / 255.0, kernel, mode="constant")

    mask = 255 * (random_values > (1 - blurred_patch)).astype("uint8")

    return mask


@timer
def get_wf_collapse_obj(size=np.random.randint(100, 160, 2)):
    patch = process_floater_patch(size_of_patch=tuple(size))
    mask = create_floater_mask(patch).astype("uint8")
    matrix = cv2.getRotationMatrix2D(
        (mask.shape[0] / 2, mask.shape[1] / 2), np.random.randint(-45, 45), 1
    )
    mask = cv2.warpAffine(mask, matrix, (mask.shape[1], mask.shape[0]))
    return mask


@timer
def init_floater(floater_patch, T):
    floater_patch = np.roll(floater_patch, T[0], axis=0)
    floater_patch = np.roll(floater_patch, T[1], axis=1)
    return floater_patch


@timer
def create_floater_patch_list(number_of_floaters, size):
    floater_patch_list = []
    translate_region = np.arange(0, 400) - 200
    T = np.random.choice(translate_region, (number_of_floaters, 2), False)
    for i in range(number_of_floaters):
        floater_patch = get_wf_collapse_obj(size)
        floater_patch = init_floater(floater_patch, T[i])
        floater_patch_list.append(floater_patch)
    return floater_patch_list


@timer
def preprocess_mask(mask):
    # Taking a matrix of size 5 as the kernel
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    # mask = cv2.erode(mask, kernel, iterations=1)
    return mask


@timer
def mask_blank_out(mask, image):
    image_foreground = 1.0 * (image > 1)
    return image_foreground * mask


@timer
def fill_mask(bg, image, floater_mask_list):
    for mask in floater_mask_list:
        mask = preprocess_mask(mask)
        mask = mask_blank_out(mask, image)
        contours, _ = cv2.findContours(
            mask.astype("uint8"), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        bg = cv2.drawContours(bg, contours, -1, 255, -1)
    return bg


@timer
def add_noise(floater_patch, floater_mask, iterations):
    for _ in range(iterations):
        floater_patch = floater_patch + floater_mask * np.random.randint(
            -20, 20, floater_mask.shape
        )
        floater_patch = cv2.GaussianBlur(floater_patch.astype("uint8"), (31, 31), 3)
    return floater_patch


@timer
def apply_floater(floater_patch, image, offset=0):
    if np.max(floater_patch.astype("float32")) == 0:
        return image
    floater_mask = floater_patch > 0
    floater_patch = (
        0.2
        * (floater_patch.astype("float32"))
        / (np.max(floater_patch.astype("float32")))
    )
    floater_patch = (floater_patch + offset) / (offset + 1)
    floater_patch[floater_mask == 0] = 1
    floater_patch = 255 * floater_patch
    floater_patch = add_noise(floater_patch, floater_mask, 5)
    floater_patch = floater_patch / 255
    # floater_patch[floater_mask==0] = 1
    image_with_floater = floater_patch * (image.astype("float32"))
    image_with_floater = image_with_floater.astype("uint8")
    return image_with_floater


@timer
def find_mask(floater_patch_list):
    floater_mask_list = []
    for floater_patch in floater_patch_list:
        mask = (255 * (floater_patch > 0)).astype("uint8")
        floater_mask_list.append(mask)
    return floater_mask_list


@timer
def floater_frame_apply(image, floater_patch_list):
    floater_mask_list = find_mask(floater_patch_list)
    for j, floater_patch in enumerate(floater_patch_list):
        image = apply_floater(floater_patch, image)
    mask = fill_mask(np.zeros(image.shape, dtype="uint8"), image, floater_mask_list)
    mask = 1.0 * (mask > 0)
    return image, mask


@timer
def cast_synthetic_floater(image, size=np.random.randint(210, 250, 2)):
    floater_patch_list = create_floater_patch_list(3, size)
    image_floater, _ = floater_frame_apply(image, floater_patch_list)
    return image_floater
