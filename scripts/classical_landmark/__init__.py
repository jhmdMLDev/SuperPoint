class LandMarkRegistration:
    def __init__(self, reference_image, method="sift"):
        if method == "sift":
            self.method = cv2.ORB_create()
        elif method == "akaze":
            self.method = cv2.AKAZE_create()
        elif method == "orb":
            self.method = cv2.ORB_create()
        elif method == "surf":
            self.method = cv2.SIFT_create()
        else:
            raise ValueError("Method not supported")

        self.reference_image = reference_image

        # detect keypoints and compute descriptors for the reference image
        self.keypoints_ref, self.descriptors_ref = self.method.detectAndCompute(
            reference_image, None
        )

        # create a Brute-Force Matcher object
        self.matcher = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

    def __call__(self, target_image):
        # detect keypoints and compute descriptors for the target image
        keypoints_target, descriptors_target = self.method.detectAndCompute(
            target_image, None
        )

        # match the descriptors using KNN (k=2)
        matches = self.matcher.match(self.descriptors_ref, descriptors_target)

        # apply ratio test to keep only the good matches
        matches = sorted(matches, key=lambda x: x.distance)

        good_matches = []
        for i, m in enumerate(matches):
            if 0.75 * m.distance < matches[0].distance:
                good_matches.append(m)

        # get the keypoints for the good matches
        self.p1 = np.float32(
            [self.keypoints_ref[m.queryIdx].pt for m in good_matches]
        ).reshape(-1, 1, 2)
        self.p2 = np.float32(
            [keypoints_target[m.trainIdx].pt for m in good_matches]
        ).reshape(-1, 1, 2)

        self.success = True if self.p2.shape[0] >= 4 else False

        if not self.success:
            self.H_transform = np.eye(3)
            return target_image.astype("uint8")

        # estimate the transformation between the reference and target images
        self.H_transform, self.mask = cv2.findHomography(
            self.p1, self.p2, cv2.RANSAC, 15
        )

        if self.H_transform is None:
            self.success = False
            self.H_transform = np.eye(3)
            return target_image.astype("uint8")

        cv2.imwrite(
            "/home/ubuntu/Projects/Floater_tracking/.samples/landmark_reg/main_target.png",
            target_image,
        )
        warped_image = cv2.warpPerspective(
            target_image, self.H_transform, target_image.shape[1::-1]
        )

        # return the warped image and the transformation matrix
        return warped_image


def siftregistration(reference_image):
    return LandMarkRegistration(reference_image, method="sift")


def akazeregistration(reference_image):
    return LandMarkRegistration(reference_image, method="akaze")


def orbregistration(reference_image):
    return LandMarkRegistration(reference_image, method="orb")


def surfregistration(reference_image):
    return LandMarkRegistration(reference_image, method="surf")
