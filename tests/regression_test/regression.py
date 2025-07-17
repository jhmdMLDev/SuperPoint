import os
import sys
import argparse
import h5py as h5

import numpy as np
import cv2
from slopreprocessing import preprocessing

from tests.test_support.slo_extract import extract_SLO
from tests.test_support.logger import setup_logger
from tests.test_support.utils import assert_homography_close
from mlregistration.superpoint import ApplySuperPointReg, Preview_Registration_Overlay


class APIRegressionTest:
    def __init__(self, slo_file_path, tracking_file_path, logger, vis_path):
        # load slo data
        self.slo_data = extract_SLO(slo_file_path)
        self.slo_frames = self.slo_data["frames"]  # list of SLO frames (numpy arrays)
        self.slo_frame_ids = self.slo_data["frame ID"]  # frame ID for each frame

        logger.info(
            f"self.slo_frames.shape: {len(self.slo_frames)}, self.slo_frame_ids.shape: {len(self.slo_frame_ids)}"
        )

        # tracking file loading
        self.tracker_data = h5.File(tracking_file_path, "r")
        self.affine_mats = np.array(
            self.tracker_data["TRACKING"]["affineHomography"]
        )  # [2,3,# of SLO frames]
        self.affine_mats_ids = [
            row[0] for row in self.tracker_data["TRACKING"]["tracking"]
        ]  # Frame ID for each process SLO frame
        self.tracker_error_code = [
            value[12] for value in self.tracker_data["TRACKING"]["tracking"]
        ]

        logger.info(
            f" self.affine_mats.shape: {self.affine_mats.shape},  self.affine_mats_ids: {len(self.affine_mats_ids)}, self.tracker_error_code.shape: {len(self.tracker_error_code)}"
        )

        # loading superpoint model
        model_path = "./tests/test_data/model/epoch_22_model.pt"
        self.vis_path = vis_path
        if not os.path.exists(self.vis_path):
            os.mkdir(self.vis_path)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for MP4
        self.video_writer_offline = cv2.VideoWriter(
            self.vis_path + "/output_offline.mp4", fourcc, 20.0, (512, 512)
        )
        self.video_writer_api = cv2.VideoWriter(
            self.vis_path + "/output_api.mp4", fourcc, 20.0, (512, 512)
        )

        self.reference = preprocessing(self.slo_data["reference"])
        assert self.reference.shape == (
            512,
            512,
        ), "Error: The shape of 'self.reference' is not (512, 512)."
        assert (
            self.reference.dtype == np.uint8
        ), "Error: The shape of 'self.reference' is not unit8."

        self.reference = cv2.rotate(self.reference, cv2.ROTATE_90_COUNTERCLOCKWISE)
        self.superpoint = ApplySuperPointReg(self.reference, model_path)

    def test_homography_match(self):
        for i in range(len(self.slo_frames)):
            slo_frame_id = self.slo_frame_ids[i]
            if slo_frame_id not in self.affine_mats_ids:
                continue

            tracking_idx = self.affine_mats_ids.index(slo_frame_id)

            if self.tracker_error_code[tracking_idx] != 0:
                blank_image = np.zeros((512, 512, 3), dtype="uint8")
                self.video_writer_offline.write(blank_image)
                self.video_writer_api.write(blank_image)
                continue

            api_homography = self.affine_mats[:, :, tracking_idx]

            frame = preprocessing(self.slo_frames[i])
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            offline_transformed_target = self.superpoint(frame)
            offline_transformed_target_rotate = cv2.rotate(
                offline_transformed_target, cv2.ROTATE_90_COUNTERCLOCKWISE
            )
            self.video_writer_offline.write(
                cv2.cvtColor(offline_transformed_target_rotate, cv2.COLOR_GRAY2RGB)
            )

            if not self.superpoint.success:
                logger.error(f"Offline superpoint failed at frame id {slo_frame_id}!")
                continue

            offline_homography = self.superpoint.H_transform
            first_two_columns_close = np.allclose(
                api_homography[:, :2], offline_homography[:2, :2], atol=1e-1
            )
            translation_difference = np.abs(
                api_homography[:, 2] - offline_homography[:2, 2]
            )
            translation_close = np.all(translation_difference <= 3)

            if first_two_columns_close and translation_close:
                logger.info(
                    f"The homography matrices are close based on the specified tolerances in frame {i}."
                )
            else:
                logger.error("The homography matrices are not close.")
                if not first_two_columns_close:
                    logger.error(
                        f"Difference in the first two columns in frame {str(slo_frame_id)} by {(api_homography[:, :2] - offline_homography[:2, :2]).tolist()}",
                    )
                if not translation_close:
                    logger.error(
                        f"Difference in the translation part in frame {str(slo_frame_id)} by {translation_difference}",
                    )

            reference_rotate = cv2.rotate(
                self.reference, cv2.ROTATE_90_COUNTERCLOCKWISE
            )
            save_path = self.vis_path + f"/offline_frame_{slo_frame_id}.png"
            Preview_Registration_Overlay(
                offline_transformed_target_rotate, reference_rotate, 0.5, save_path
            )
            save_path = self.vis_path + f"/api_frame_{slo_frame_id}.png"
            api_transformed_target = cv2.warpAffine(frame, api_homography, frame.shape)
            api_transformed_target = cv2.rotate(
                api_transformed_target, cv2.ROTATE_90_COUNTERCLOCKWISE
            )

            api_transformed_target_rgb = cv2.cvtColor(
                api_transformed_target, cv2.COLOR_GRAY2RGB
            )
            self.video_writer_api.write(api_transformed_target_rgb)

            Preview_Registration_Overlay(
                api_transformed_target, reference_rotate, 0.5, save_path
            )

        self.video_writer_offline.release()
        self.video_writer_api.release()

    def __call__(self):
        self.test_homography_match()


if __name__ == "__main__":
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="Regression Test on Panama Data")

    # Add arguments
    parser.add_argument("--slo_hdf5_path", type=str, help="Path to the slo stream")
    parser.add_argument(
        "--tracking_hdf5_path", type=str, help="Path to the tracking file"
    )
    parser.add_argument(
        "--logger_output",
        type=str,
        help="Path to the logged output",
        default="./tests/.vis/api_regression",
    )

    # Parse the arguments
    args = parser.parse_args()

    if not os.path.exists(args.logger_output):
        os.mkdir(args.logger_output)

    logger = setup_logger(
        name="regression_log",
        filename=args.logger_output + "/regression_log.json",
        model="Superpoint",
    )

    # Use the arguments
    print(f"SLO file: {args.slo_hdf5_path}")
    print(f"TRACK file: {args.tracking_hdf5_path}")
    print(f"Logging level: {args.logger_output}")

    APIRegressionTest(
        args.slo_hdf5_path, args.tracking_hdf5_path, logger, args.logger_output
    )()
