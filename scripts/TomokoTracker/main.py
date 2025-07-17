import os
import json
import subprocess
from ast import literal_eval

import numpy as np
import cv2


class TomokoTrackerReader:
    def __init__(self, analysis_file):
        self.analysis_file = analysis_file

    def annotation_error_check(self, file_num):
        filenum_annot = self.analysis_file["files"][file_num - 1]["path"].split(".")[0]
        filenum_annot = literal_eval(filenum_annot)
        return filenum_annot == file_num

    def read_homography(self, trg_filename):
        experiment = [
            ele for ele in self.analysis_file["files"] if ele["path"] == trg_filename
        ][0]
        if "homography2Ref" in experiment:
            homography = experiment["homography2Ref"]
            homography = np.array(homography).reshape(3, 3)
        else:
            homography = None
        return homography

    def __call__(self, trg):
        self.H_transform = self.read_homography(trg["filename"])

        if self.H_transform is None:
            self.success = False
            return np.zeros_like(trg["image"])

        self.success = True
        img_target_transformed = cv2.warpPerspective(
            trg["image"],
            self.H_transform,
            (trg["image"].shape[1], trg["image"].shape[0]),
        )

        return img_target_transformed


class TomokoTrakcer:
    def __init__(self, data_path) -> None:
        self.data_path = data_path
        self.json_path = os.path.join(data_path, "TrackingAnalysisV2.json")

    def set_temp_ref(self, ref):
        self.reference_image = ref
        cv2.imwrite(self.data_path + "./temp_ref.png", ref)

    def run_powershell(self, reference_image="temp_ref.png"):
        command1 = "cd " + self.data_path
        command2 = (
            "C:/Users/Javad/Downloads/ProcessTracker-20230324T194538Z-001/ProcessTracker/bin2/trackerEvaluation.exe 7 "
            + self.data_path
            + " "
            + reference_image
        )
        full_command = f'powershell.exe -Command "{command1}; {command2};"'
        completed_process = subprocess.run(
            full_command, shell=True, capture_output=True
        )
        print(completed_process.stdout.decode())

    def __call__(self, trg):
        with open(self.json_path) as f:
            analysis_file = json.load(f)

        tomokotrackerreader = TomokoTrackerReader(analysis_file)
        img_target_transformed = tomokotrackerreader(trg)
        self.success = tomokotrackerreader.success
        self.H_transform = tomokotrackerreader.H_transform
        return img_target_transformed
