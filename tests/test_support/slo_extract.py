import numpy as np
import h5py as h5
import argparse
import struct
import cv2

"""
Functions for extracting SLO frames from .h5 (hdf5) files.
Use function "extract_SLO(filename)" to extract from a .h5 file to a list of numpy arrays
"""


class SLO:
    dim = []
    data = []

    def setData(self, slogroup):
        if "FrameDim" in slogroup.attrs.keys():
            self.dim = slogroup.attrs["FrameDim"]
        else:
            self.dim = slogroup.attrs["FRAMEDIM"]
        if "sloStream" in slogroup.keys():
            self.data = slogroup["sloStream"]
        else:
            self.data = slogroup["slostream"]


def process_slo_frame(frame_raw, slo16bit=False):
    frame = np.rot90(frame_raw, 3)
    low = 0.0  # np.percentile(frame, 40)
    high = np.percentile(frame, 99.9)
    if slo16bit:
        return frame

    # Convert to 8bit
    # Contrast enhancement
    # 40 percentile --> 0 , 99.7 percentile --> 255
    if high <= 0.0:
        frame8U = np.zeros(frame.shape).astype("uint8")
    else:
        frame8U = cv2.convertScaleAbs(
            frame, alpha=(255.0 / (high - low)), beta=(-low * (255.0 / (high - low)))
        )
    return frame8U


def extract_SLO(filename, slo16bit=True, reference_only=False):
    """
    Loads and extracts an SLO h5 file.
    output:
    :param filename: path to .h5 file
    :param slo16bit: if 'True', skip the 16->8 bit conversion and return the raw frame data
    :param reference_only: if 'True', only extract the reference frame
    :return: {"frames": [list of 2D images], "metadata", "frame ts", "frame ID", "reference": 2D reference image}
    """

    sloraw = h5.File(filename, "r")

    slostream = SLO()
    slostream.setData(sloraw["SLOSTREAM"])

    metadata = dict()
    metadata["laserPower"] = (
        sloraw["SLOSTREAM"]["laserPower"]
        if "laserPower" in sloraw["SLOSTREAM"].keys()
        else None
    )
    metadata["lensDiopter"] = (
        sloraw["SLOSTREAM"]["lensDiopter"]
        if "lensDiopter" in sloraw["SLOSTREAM"].keys()
        else None
    )
    metadata["sharpness"] = (
        sloraw["SLOSTREAM"]["sharpness"]
        if "sharpness" in sloraw["SLOSTREAM"].keys()
        else None
    )

    frames = []
    for i in range(slostream.dim[0]):
        frames.append(process_slo_frame(slostream.data[:, :, i], slo16bit=slo16bit))
        if reference_only:
            break
        """
        frame = np.rot90(frame, 3)
        # Contrast enhancement
        # 40 percentile --> 0 , 99.7 percentile --> 255
        if slo16bit:
            frames.append(frame)
        else:
            low = 0.0 #np.percentile(frame, 40)
            high = np.percentile(frame, 99.9)

            # Convert to 8bit
            if high <= 0.:
                frame8U = np.zeros(frame.shape).astype("uint8")
            else:
                frame8U = cv2.convertScaleAbs(frame, alpha=(255.0 / (high - low)), beta=(-low * (255.0 / (high - low))))
            frames.append(frame8U)"""

    # Get bbox coordinates/size
    if "contour" in sloraw["SLOSTREAM"].keys():
        bbox_data = np.array(sloraw["SLOSTREAM"]["contour"])
        if len(bbox_data) == 4:
            bbox = {
                "x": int(np.min((bbox_data[0], bbox_data[2]))),
                "y": int(np.min((bbox_data[1], bbox_data[3]))),
                "w": int(np.abs(bbox_data[0] - bbox_data[2])),
                "h": int(np.abs(bbox_data[1] - bbox_data[3])),
            }
        elif len(bbox_data) == 8:
            bbox = {
                "x": int(
                    np.min((bbox_data[0], bbox_data[2], bbox_data[4], bbox_data[6]))
                ),
                "y": int(
                    np.min((bbox_data[1], bbox_data[3], bbox_data[5], bbox_data[7]))
                ),
                "w": int(np.abs(bbox_data[0] - bbox_data[2])),
                "h": int(np.abs(bbox_data[1] - bbox_data[7])),
                "p1": (bbox_data[0], bbox_data[1]),
                "p2": (bbox_data[2], bbox_data[3]),
                "p3": (bbox_data[4], bbox_data[5]),
                "p4": (bbox_data[6], bbox_data[7]),
            }
        elif len(bbox_data) > 0:
            bbox = {"contour": bbox_data}

        else:
            bbox = None
    else:
        bbox = None

    if "referenceFrame" in sloraw["SLOSTREAM"].keys():
        ref_frame = np.array(sloraw["SLOSTREAM"]["referenceFrame"]).reshape(
            frames[0].shape
        )
        ref_frame = process_slo_frame(ref_frame, slo16bit)
    else:
        ref_frame = None

    frameIDs = None
    if "frameIDs" in sloraw["SLOSTREAM"].keys():
        frameIDs = list(np.array(sloraw["SLOSTREAM"]["frameIDs"]))

    result = {
        "frames": frames,
        "bbox": bbox,
        "metadata": metadata,
        "frame ts": list(np.array(sloraw["SLOSTREAM"]["sloTimestamps"])),
        "frame ID": frameIDs,
        "reference": ref_frame,
    }

    return result


def flipThroughSLO(filename):
    frames = extract_SLO(filename)["frames"]
    for i in range(len(frames)):
        cv2.imshow("test", frames[i])
        cv2.waitKey(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flip through SLO")
    parser.add_argument("--slo_file", help="SLO file to flip through")

    args = parser.parse_args()

    # flipThroughSLO(args.slo_file)
    flipThroughSLO(
        r"C:\OneDrive\Documents\Chris\Machine Learning\Python\ML_OCT_Volume_Preview\data\patient_Areeba\2024.01.03\SLOStream_13_2024.01.03-14.26.16_0190.h5"
    )
