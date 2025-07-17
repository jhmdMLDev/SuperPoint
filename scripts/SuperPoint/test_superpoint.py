import os
import sys
import time

import cv2
import numpy as np
import torch

sys.path.insert(0, os.getcwd())
from scripts.SuperPoint.main import ApplySuperPointReg
from scripts.robustness_eval.synthetic_blur import apply_blur

if __name__ == "__main__":
    ref = cv2.imread(
        r"C:\Users\Javad\Desktop\Dataset\77\2022.01.27-15.40.14_0904.jpg", 0
    )
    trg = cv2.imread(
        r"C:\Users\Javad\Desktop\Dataset\77\2022.01.27-15.40.18_0658.jpg", 0
    )

    apply_superpoint = ApplySuperPointReg(ref)

    targ_transferred = apply_superpoint(trg)

    targ_transferred = apply_superpoint(trg)

    t0 = time.time()
    targ_transferred = apply_superpoint(trg)
    print(time.time() - t0)

    # print(apply_superpoint.H_transform)
    # print(apply_superpoint.p1.shape)
    # # targ_transferred = cv2.resize(targ_transferred, (512, 512))

    # img_3d = np.dstack([ref, np.zeros_like(ref), targ_transferred])

    # cv2.imwrite(
    #     "C:\Projects\Registration_Benchmarking\.sample/reg.png", img_3d
    # )
    # cv2.imwrite(
    #     "C:\Projects\Registration_Benchmarking\.sample/ref.png", ref
    # )
    # cv2.imwrite(
    #     "C:\Projects\Registration_Benchmarking\.sample/targ_transferred.png",
    #     targ_transferred,
    # )
