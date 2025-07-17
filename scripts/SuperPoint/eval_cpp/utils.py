import json
from ast import literal_eval

import numpy as np
import cv2


def read_results(text_file_path):
    filename_matrix_dict = {}

    with open(text_file_path, "r") as f:
        matrix_str = ""
        for line in f:
            line = line.strip()  # remove trailing whitespaces and newline character
            if len(line) == 0:
                continue
            if line[0] == "C":
                filename = line[:-1]
            else:
                matrix_str += line
            if line[-1] == "]":
                matrix_str = "[" + matrix_str + "]"
                matrix_str = matrix_str.replace(";", "], [")
                matrix = np.array(literal_eval(matrix_str))
                matrix_str = ""
                filename_matrix_dict[filename] = matrix

    return filename_matrix_dict


def get_registration_acc(H_register1, pts_trg, pts_ref):
    p2_trg_transform = cv2.transform(pts_trg.reshape(1, -1, 2), H_register1)[
        :, :, 0:2
    ].squeeze(0)
    error = np.sqrt(np.sum((p2_trg_transform - pts_ref) ** 2)) / pts_ref.shape[0]
    error = min(error, 100)
    accuracy = max(0, (100 - error)) / 100
    return accuracy


def process_annotation(json_path):
    with open(json_path) as f:
        data = json.load(f)

    annotations = data["annotations"]

    points = np.zeros((8, 2))

    for ele in annotations:
        point_name = ele["name"]
        idx = literal_eval(point_name[-1]) - 1
        points[idx, 0] = ele["keypoint"]["x"]
        points[idx, 1] = ele["keypoint"]["y"]

    return points


def evaluate(text_file_path, annotation_paths, reference_json_path):
    filename_matrix_dict = read_results(text_file_path)
    accuracy_list = []
    ref_pts = process_annotation(reference_json_path)
    count = 0
    for image_path in filename_matrix_dict.keys():
        json_path = annotation_paths + "/" + image_path.split("\\")[-1][:-3] + "json"
        target_pts = process_annotation(json_path)
        homography = filename_matrix_dict[image_path]
        accuracy = get_registration_acc(homography, target_pts, ref_pts)
        accuracy_list.append(accuracy)
        count += 1

    print(f"The accuracy on the folder with {count} images is {np.mean(accuracy_list)}")

    return np.mean(accuracy_list)


if __name__ == "__main__":
    text_file_path = (
        r"C:\Users\Javad\Desktop\Dataset\cpp_test_sp\homography_matrices.txt"
    )
    annotation_paths = (
        r"C:\Users\Javad\Desktop\Dataset\Registration_benchmark\annotations\PMP_01_001"
    )
    reference_json_path = r"C:\Users\Javad\Desktop\Dataset\Registration_benchmark\annotations\PMP_01_001\2202_reference.json"

    # text_file_path = r"C:\Users\Javad\Desktop\Dataset\cpp_test_sp\homography_matrices2.txt"
    # annotation_paths = r"C:\Users\Javad\Desktop\Dataset\Registration_benchmark\annotations\PMP_01_005_02"
    # reference_json_path = r"C:\Users\Javad\Desktop\Dataset\Registration_benchmark\annotations\PMP_01_005_02\3407_reference.json"
    accuracy = evaluate(text_file_path, annotation_paths, reference_json_path)
