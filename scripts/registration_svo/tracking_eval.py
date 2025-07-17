import cv2


def partition_img(img, n=3):
    if n == 0:
        raise ValueError("Division by Zero!")

    h, w = img.shape
    patch_list = []
    for i in range(0, n):
        for j in range(0, n):
            img_patch = img[
                int(i * h / n) : int((i + 1) * h / n),
                int(j * w / n) : int((j + 1) * w / n),
            ]
            patch_list.append(img_patch)
    return patch_list


def get_phase_correlation_translation(ref, trg):
    # Apply phase correlation
    ref, trg = ref.astype("float32"), trg.astype("float32")
    translation, response = cv2.phaseCorrelate(ref, trg)
    tx = -translation[0]
    ty = -translation[1]
    return tx, ty, response


def tracking_eval(reference_img, transformed_img):
    ref_img_patches = partition_img(reference_img)
    transformed_img_patches = partition_img(transformed_img)

    for i in range(0, len(ref_img_patches)):
        tx, ty, response = get_phase_correlation_translation(
            ref_img_patches[i], transformed_img_patches[i]
        )
        if abs(tx) > 5 or abs(ty) > 5 or response < 0.3:
            return False
    return True


if __name__ == "__main__":
    import numpy as np

    img = (255 * np.random.random((512, 512))).astype("uint8")
    img_2 = np.roll(img, (3, 3))
    is_registered = tracking_eval(img, img_2)
    print(is_registered)
