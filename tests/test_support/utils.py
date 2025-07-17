import numpy as np
from scipy.ndimage import gaussian_filter
import cv2


def entropy(image, bins=256):
    """
    Compute the entropy of an image.

    Parameters:
    - image: Input image.
    - bins: Number of bins for the histogram.

    Returns:
    - entropy_value: Entropy of the image.
    """
    # Compute histogram
    hist, _ = np.histogram(image.ravel(), bins=bins, range=(0, bins))

    # Normalize histogram
    hist = hist / hist.sum()

    # Compute entropy
    entropy_value = -np.sum(hist * np.log(hist + 1e-10))

    return entropy_value


def mutual_information(image1, image2, bins=256):
    """
    Compute Mutual Information (MI) between two images.

    Parameters:
    - image1: First image.
    - image2: Second image.
    - bins: Number of bins for the histogram.

    Returns:
    - MI: Mutual Information value.
    """
    # Compute joint histogram
    joint_hist = np.histogram2d(image1.ravel(), image2.ravel(), bins=bins)[0]

    # Normalize histograms
    joint_hist /= joint_hist.sum()
    p1 = joint_hist.sum(axis=1)
    p2 = joint_hist.sum(axis=0)

    # Compute MI
    MI = 0
    for i in range(bins):
        for j in range(bins):
            if joint_hist[i, j] > 0:
                MI += joint_hist[i, j] * np.log(
                    joint_hist[i, j] / (p1[i] * p2[j] + 1e-10)
                )

    return MI


def assert_homography_close(
    estimated_homography, true_homography, translation_tol=3.0, other_tol=0.1
):
    """
    Assert that the estimated homography is close to the true homography matrix within specified tolerances.

    Parameters:
    - estimated_homography: The estimated homography matrix.
    - true_homography: The true homography matrix.
    - translation_tol: Tolerance for translation values.
    - other_tol: Tolerance for other values.
    """

    estimated_homography = np.array(estimated_homography)
    true_homography = np.array(true_homography)

    # Tolerance for translation values (last column)
    translation_diff = np.abs(estimated_homography[:, 2] - true_homography[:, 2])
    np.testing.assert_allclose(
        translation_diff,
        np.zeros_like(translation_diff),
        atol=translation_tol,
        err_msg="Translation values are not close.",
    )

    # Tolerance for other values (excluding last column)
    non_translation_diff = np.abs(
        estimated_homography[:, :-1] - true_homography[:, :-1]
    )

    np.testing.assert_allclose(
        non_translation_diff,
        np.zeros_like(non_translation_diff),
        atol=other_tol,
        err_msg="Non-translation values are not close.",
    )


def elastic_transform(image, alpha, sigma, random_state=None):
    if random_state is None:
        random_state = np.random.RandomState(42)  # Set a seed for reproducibility

    shape = image.shape
    dx = (
        gaussian_filter(
            (random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0
        )
        * alpha
    )
    dy = (
        gaussian_filter(
            (random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0
        )
        * alpha
    )

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    map_x = np.float32(x + dx)
    map_y = np.float32(y + dy)

    distorted_image = cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )

    return distorted_image


def save_images_side_by_side(image1, image2, save_path):
    """
    Save two images side by side.

    Parameters:
    - image1 (numpy.ndarray): The first image.
    - image2 (numpy.ndarray): The second image.
    - save_path (str): The path to save the combined image.
    """
    save_path = save_path[:-4] + "_comparison.png"
    # Ensure both images have the same height
    if image1.shape[0] != image2.shape[0]:
        height = min(image1.shape[0], image2.shape[0])
        image1 = image1[:height, :]
        image2 = image2[:height, :]

    # Stack images side by side
    combined_image = np.hstack((image1, image2))

    # Save the combined image
    cv2.imwrite(save_path, combined_image)
