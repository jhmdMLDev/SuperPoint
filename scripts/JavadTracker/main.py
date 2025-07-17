import cv2
from scripts.JavadTracker.JavadRegistration import (
    SLO_SLO_registration_transform,
)


class JavadRegistration:
    def __init__(self, reference_image, method="sift"):
        self.reference_image = cv2.resize(reference_image, (768, 768))
        self.debug_path = r"/home/ubuntu/Projects/Floater_tracking/.samples/Javad/debug"

    def __call__(self, trg):
        trg = cv2.resize(trg, (768, 768))
        warped_image, self.H_transform = SLO_SLO_registration_transform(
            self.reference_image,
            trg,
            pad=300,
            iter_stik=500,
            LNR_REGISTRATION=False,
            path=self.debug_path,
        )

        if self.H_transform is None:
            self.H_transform = np.eye(3)
            self.success = False
            return trg
        else:
            self.success = True

        return warped_image
