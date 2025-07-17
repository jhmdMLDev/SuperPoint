import unittest
import pytest

import cv2
import numpy as np
import os

from mlregistration.utils import segment_vein


class VeinSeg(unittest.TestCase):
    def setUp(self):
        # Create a sample image
        self.image_path = "./tests/test_data/superpoint_unittest/image1.png"
        self.vis_path = "./tests/.vis/vein_seg_unit"
        if not os.path.exists(self.vis_path):
            os.makedirs(self.vis_path)

    def test_vein_segmentation(self):
        image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        segmented_vein = segment_vein(image)

        # Save the result for visual inspection
        cv2.imwrite(os.path.join(self.vis_path, "segmented_vein.png"), segmented_vein)

        # Check if the segmented image is not empty
        self.assertFalse(np.all(segmented_vein == 0), "The segmented image is empty.")

        # Check if the output has the same shape as the input
        original_image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        self.assertEqual(
            segmented_vein.shape, original_image.shape, "The shapes do not match."
        )
