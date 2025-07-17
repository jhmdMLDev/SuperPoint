if __name__ == "__main__":
    import cv2
    import numpy as np
    import h5py as h5
    from tests.test_support.slo_extract import extract_SLO
    from tqdm import tqdm

    slo_h5 = r"\\192.168.1.100\Retina Images ML\PanamaStudyData\AWS\29\1714434464679\SLOStream_1_2024.04.29-22.01.31.6831138_0000.h5"
    tracker_h5 = r"\\192.168.1.100\Retina Images ML\PanamaStudyData\AWS\29\1714434464679\Tracking_2024.04.29-22.01.41.3606406_0000..h5"

    # Load the affine matrices (a set of 2x3 matrices, one for each SLO frame)
    tracker_data = h5.File(tracker_h5, "r")
    affine_mats = np.array(
        tracker_data["TRACKING"]["affineHomography"]
    )  # [2,3,# of SLO frames]
    affine_mats_ts = tracker_data["TRACKING"][
        "trackingimestamps"
    ]  # timestamp for each processed SLO frame
    affine_mats_frameID = [
        row[0] for row in tracker_data["TRACKING"]["tracking"]
    ]  # Frame ID for each process SLO frame
    tracker_error_code = [value[12] for value in tracker_data["TRACKING"]["tracking"]]

    # Load the SLO frames
    SLO_data = extract_SLO(slo_h5)
    print("SLO_data: ", SLO_data.keys())
    SLO_frames = SLO_data["frames"]  # list of SLO frames (numpy arrays)
    SLO_frame_ts = SLO_data["frame ts"]  # timestamp for each frame
    SLO_frameIDs = SLO_data["frame ID"]
    print("SLO_frameIDs: ", SLO_data["frame ID"])

    # Not every frame will have an affine matrix (eg. if patient blinks, it will get skipped by ML and no matrix will be
    # saved) so we need to check that the frame timestamp and the tracking/affine timestamp match

    # Also, there will be more affine_mats in each tracking file than there are SLO frames in a single SLO .h5
    # this is because SLO frames in each .h5 file contain the frames from a 40-second period when the user clicks 'capture'
    # so the same patient's eye may be split into multiple SLO frames, and the SLO frames in between aren't saved
    # The ML tracker/registration is always running from the moment they start imaging the patient, so will constantly be
    # adding data to the "tracking" .h5 file both within the SLO capture window and outside, all to the same .h5
    # there will usually be 2 tracking .h5 files, since it restarts when they switch to the other eye.

    # Use Frame ID to match tracking/affine results with the SLO frame

    # For each tracker result, we need to know which SLO frameID is associated with it.
    # Every tracker result should have an SLO frame, but they might not all be in the same SLO file.

    """    slo_frame_idx = []
        # Populate list C with the indices of B that match with A
        for trackID_value in affine_mats_frameID:
            if trackID_value in SLO_frameIDs:
                slo_frame_idx.append(SLO_frameIDs.index(trackID_value))
            else:
                # If the value is not in B, append None or handle accordingly
                slo_frame_idx.append(None)
    # "slo_frame_idx" represents the index of the SLO frame associated with the tracker data
    """
    tracks_matched = 0
    slo_frame_idx = []
    for frameID_value in tqdm(SLO_frameIDs):
        if frameID_value in affine_mats_frameID:
            slo_frame_idx.append(affine_mats_frameID.index(frameID_value))
            tracks_matched += 1
        else:
            # If the value is not in B, append None or handle accordingly
            slo_frame_idx.append(None)
    print("{} of {} tracks matched".format(tracks_matched, len(SLO_frames)))
    # "slo_frame_idx" represents the index of the tracker data associated with each SLO frame
    # "None" indicates no tracking data for that frame

    """# Crop the index array to the first 'None' value
    # This is in case SLO data is split into multiple studies
    try:
        slo_frame_idx = slo_frame_idx[:slo_frame_idx.index(None)]
    except Exception as e:
        # No 'None' values in list
        pass"""

    SLO_registered = []
    for i in tqdm(range(len(slo_frame_idx))):
        if slo_frame_idx[i] is None or not tracker_error_code[i] == 0:
            frame = np.zeros(SLO_frames[0].shape).astype("uint8")
        else:
            M = affine_mats[:, :, slo_frame_idx[i]]
            frame = cv2.warpAffine(SLO_frames[i], M, SLO_frames[i].shape)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        cv2.putText(frame, "frame " + str(i), (20, 20), 1, 1, (0, 255, 0))
        SLO_registered.append(frame)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Change the codec as needed
    out = cv2.VideoWriter(
        "registration.mp4",
        fourcc,
        30.0,
        (SLO_frames[0].shape[0], SLO_frames[0].shape[1]),
    )

    for frame in SLO_registered:
        # cv2.imshow("", frame)
        # cv2.waitKey()
        out.write(frame)
    out.release()

    print("Done.")
