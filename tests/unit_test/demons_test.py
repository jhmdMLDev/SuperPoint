import os
import inspect

import unittest
import pytest
import numpy as np
import cv2

from mlregistration.superpoint import Preview_Registration_Overlay
from mlregistration.demons import demons_registration
from tests.test_support.utils import (
    entropy,
    mutual_information,
    assert_homography_close,
    elastic_transform,
    save_images_side_by_side,
)


class DemonsUnitTest(unittest.TestCase):
    def setUp(self):
        # Create a sample image
        image_path = "./tests/test_data/superpoint_unittest"
        self.vis_path = "./tests/.vis/demons_unit"
        if not os.path.exists(self.vis_path):
            os.mkdir(self.vis_path)
        self.images_paths = [
            os.path.join(image_path, item)
            for item in os.listdir(image_path)
            if item.endswith("png") or item.endswith("jpg")
        ]

    def test_identity_transformation(self):
        for image_path in self.images_paths:
            # get the image
            image = cv2.imread(image_path, 0)
            # define homography
            M = np.float32([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            # get transformed image
            tgt_image = cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]))
            # get the transformation
            estimated_transformed_target = demons_registration(
                fixed_image_np=image,
                moving_image_np=tgt_image,
                iterations=2000,
                sigma=1,
            )
            # test image
            save_path = (
                self.vis_path
                + f"/{inspect.currentframe().f_code.co_name}_{os.path.basename(image_path)}.png"
            )
            Preview_Registration_Overlay(
                estimated_transformed_target, image, 0.5, save_path
            )
            save_images_side_by_side(estimated_transformed_target, tgt_image, save_path)

    def test_translation_transformation(self):
        for image_path in self.images_paths:
            # Read the image
            image = cv2.imread(image_path, 0)
            # Define the translation homography matrix
            tx, ty = 10, 20
            M = np.float32([[1, 0, tx], [0, 1, ty], [0, 0, 1]])
            M_inv = np.linalg.inv(M)
            # get transformed image
            tgt_image = cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]))
            # get the transformation
            estimated_transformed_target = demons_registration(
                fixed_image_np=image,
                moving_image_np=tgt_image,
                iterations=2000,
                sigma=1,
            )
            # test image
            save_path = (
                self.vis_path
                + f"/{inspect.currentframe().f_code.co_name}_{os.path.basename(image_path)}.png"
            )
            Preview_Registration_Overlay(
                estimated_transformed_target, image, 0.5, save_path
            )
            save_images_side_by_side(estimated_transformed_target, tgt_image, save_path)

    def test_rotation_transformation(self):
        for image_path in self.images_paths:
            # Read the image
            image = cv2.imread(image_path, 0)
            # Define the rotation homography matrix
            angle = 5
            center = (image.shape[1] // 2, image.shape[0] // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            M = np.vstack([M, [0, 0, 1]])  # Convert to 3x3 matrix
            M_inv = np.linalg.inv(M)
            # get transformed image
            tgt_image = cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]))
            # get the transformation
            estimated_transformed_target = demons_registration(
                fixed_image_np=image,
                moving_image_np=tgt_image,
                iterations=2000,
                sigma=1,
            )
            # test image
            save_path = (
                self.vis_path
                + f"/{inspect.currentframe().f_code.co_name}_{os.path.basename(image_path)}.png"
            )
            Preview_Registration_Overlay(
                estimated_transformed_target, image, 0.5, save_path
            )
            save_images_side_by_side(estimated_transformed_target, tgt_image, save_path)

    def test_scaling_transformation(self):
        for image_path in self.images_paths:
            # Read the image
            image = cv2.imread(image_path, 0)
            # Define the scaling homography matrix
            sx, sy = 1.2, 1.2
            M = np.float32([[sx, 0, 0], [0, sy, 0], [0, 0, 1]])
            M_inv = np.linalg.inv(M)
            # get transformed image
            tgt_image = cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]))
            # get the transformation
            estimated_transformed_target = demons_registration(
                fixed_image_np=image,
                moving_image_np=tgt_image,
                iterations=2000,
                sigma=1,
            )
            # test image
            save_path = (
                self.vis_path
                + f"/{inspect.currentframe().f_code.co_name}_{os.path.basename(image_path)}.png"
            )
            Preview_Registration_Overlay(
                estimated_transformed_target, image, 0.5, save_path
            )
            save_images_side_by_side(estimated_transformed_target, tgt_image, save_path)

    def test_mixed_transformation(self):
        for image_path in self.images_paths:
            # Read the image
            image = cv2.imread(image_path, 0)
            # Define the mixed homography matrix (rotation + scale + translation)
            angle = 15
            scale = 1.2
            tx, ty = 15, -10
            center = (image.shape[1] // 2, image.shape[0] // 2)
            M = cv2.getRotationMatrix2D(center, angle, scale)
            M[0, 2] += tx
            M[1, 2] += ty
            M = np.vstack([M, [0, 0, 1]])  # Convert to 3x3 matrix
            M_inv = np.linalg.inv(M)
            # get transformed image
            tgt_image = cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]))
            # get the transformation
            estimated_transformed_target = demons_registration(
                fixed_image_np=image,
                moving_image_np=tgt_image,
                iterations=2000,
                sigma=1,
            )
            # test image
            save_path = (
                self.vis_path
                + f"/{inspect.currentframe().f_code.co_name}_{os.path.basename(image_path)}.png"
            )
            Preview_Registration_Overlay(
                estimated_transformed_target, image, 0.5, save_path
            )
            save_images_side_by_side(estimated_transformed_target, tgt_image, save_path)

    def test_elastic_transformation(self):
        for image_path in self.images_paths:
            # Read the image
            image = cv2.imread(image_path, 0)
            # Define elastic transform
            tgt_image = elastic_transform(
                image, 20, 2.0, random_state=np.random.RandomState(42)
            )
            # get the transformation
            estimated_transformed_target = demons_registration(
                fixed_image_np=image,
                moving_image_np=tgt_image,
                iterations=2000,
                sigma=1,
            )
            # test image
            save_path = (
                self.vis_path
                + f"/{inspect.currentframe().f_code.co_name}_{os.path.basename(image_path)}.png"
            )
            Preview_Registration_Overlay(
                estimated_transformed_target, image, 0.5, save_path
            )
            save_images_side_by_side(estimated_transformed_target, tgt_image, save_path)


if __name__ == "__main__":
    unittest.main()
