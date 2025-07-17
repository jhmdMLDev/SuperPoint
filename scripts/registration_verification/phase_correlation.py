import cv2
import numpy as np
import threading
import time


def measure_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time of {func.__name__}: {execution_time} seconds")
        return result

    return wrapper


# Divide the image into 9 patches
# @measure_time
def divide_image(image):
    height, width = image.shape[:2]
    patch_height = height // 3
    patch_width = width // 3
    patches = []
    for i in range(3):
        for j in range(3):
            patch = image[
                i * patch_height : (i + 1) * patch_height,
                j * patch_width : (j + 1) * patch_width,
            ]
            patches.append(patch)
    return patches


# @measure_time
def flatten_tuple(data):
    if isinstance(data, tuple):
        return [item for sublist in data for item in flatten_tuple(sublist)]
    else:
        return [data]


# Thread worker function for phase correlation
# @measure_time
def phase_correlation_worker(reference, target, results, idx):
    translation = cv2.phaseCorrelate(
        reference.astype("float32"), target.astype("float32")
    )
    results[idx] = flatten_tuple(translation)


def verify_result(results_arr):
    patch_success = []
    for i in range(results_arr.shape[0]):
        if results_arr[i, 2] < 0.5 or np.max(results_arr[i, 0:2]) > 10:
            patch_success.append(False)
        else:
            patch_success.append(True)
    if (np.sum(np.array(patch_success)) / results_arr.shape[0]) > 0.3:
        return True
    else:
        return False


# @measure_time
def apply_registration_verification(reference_patches, target_image):
    # Divide the image into patches
    target_patches = divide_image(target_image)

    # Initialize results array
    results = [None] * len(reference_patches)

    # Create threads for each patch
    threads = []
    for i, (ref_patch, target_patch) in enumerate(
        zip(reference_patches, target_patches)
    ):
        thread = threading.Thread(
            target=phase_correlation_worker, args=(ref_patch, target_patch, results, i)
        )
        thread.start()
        threads.append(thread)

    # Wait for all threads to finish
    for thread in threads:
        thread.join()

    results_arr = np.abs(np.array(results))
    success = verify_result(results_arr)
    return success, results_arr


if __name__ == "__main__":
    reference_image = cv2.imread(
        r"C:\Users\Javad\Desktop\Dataset\77\2022.01.27-15.40.14_0904.jpg", 0
    )
    target_image = cv2.imread(
        r"C:\Users\Javad\Desktop\Dataset\77\2022.01.27-15.40.14_0969.jpg", 0
    )
    reference_patches = divide_image(reference_image)
    Success = apply_registration_verification(reference_patches, target_image)
    print("Success: ", Success)
