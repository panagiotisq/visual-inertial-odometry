import cv2
import numpy as np
import os
import re
from tqdm import tqdm

BASE_INPUT_DIR_RECTIFIED = "rectified/"
BASE_OUTPUT_DIR_FEATURES = "features/"
CALIBRATION_DATA_DIR="calibration_data_rectified/"
SIFT_NFEATURES = 500
SIFT_NOCTAVELAYERS = 3
SIFT_CONTRAST_THRESHOLD = 0.02
SIFT_EDGE_THRESHOLD = 15
SIFT_SIGMA = 1.6
LOWE_RATIO = 0.75
STEREO_VERTICAL_DISPARITY_THRESHOLD = 2.0
RANSAC_REPROJ_THRESHOLD_TEMPORAL = 1.0
RANSAC_CONFIDENCE_TEMPORAL = 0.999

k0_file_path = os.path.join(CALIBRATION_DATA_DIR, "K0_rectified.npy")
K0 = np.load(k0_file_path)
FOCAL_LENGTH_CAM0 = K0[0,0]
PRINCIPAL_POINT_CAM0 = (K0[0,2], K0[1,2])

output_dir_kp = os.path.join(BASE_OUTPUT_DIR_FEATURES, "keypoints")
output_dir_desc = os.path.join(BASE_OUTPUT_DIR_FEATURES, "descriptors")
output_dir_stereo_matches = os.path.join(BASE_OUTPUT_DIR_FEATURES, "stereo_matches_filtered")
output_dir_temporal_matches = os.path.join(BASE_OUTPUT_DIR_FEATURES, "temporal_matches_filtered")

os.makedirs(output_dir_kp, exist_ok=True)
os.makedirs(output_dir_desc, exist_ok=True)
os.makedirs(output_dir_stereo_matches, exist_ok=True)
os.makedirs(output_dir_temporal_matches, exist_ok=True)

class FeatureMatcher:
    def natural_sort_key(self, s):
        """
        Περιγραφή: Ταξινομεί αλφαριθμητικά που περιέχουν αριθμούς με φυσικό τρόπο.
        Είσοδος: s (str) - το αλφαριθμητικό προς ταξινόμηση.
        Έξοδος: list - μια λίστα από ακέραιους ή αλφαριθμητικά για χρήση στην ταξινόμηση.
        """
        return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

    def detect_and_save_sift_features(self, image_type_prefix="rect_left_"):
        """
        Περιγραφή: Εντοπίζει χαρακτηριστικά SIFT και αποθηκεύει τα keypoints και τους descriptors.
        Είσοδος: image_type_prefix (str) - το πρόθεμα των ονομάτων αρχείων εικόνας ('rect_left_' ή 'rect_right_').
        Έξοδος: None - αποθηκεύει αρχεία .npy με keypoints και descriptors.
        """
        sift = cv2.SIFT_create(
            nfeatures=SIFT_NFEATURES,
            nOctaveLayers=SIFT_NOCTAVELAYERS,
            contrastThreshold=SIFT_CONTRAST_THRESHOLD,
            edgeThreshold=SIFT_EDGE_THRESHOLD,
            sigma=SIFT_SIGMA
        )
        image_files = sorted([f for f in os.listdir(BASE_INPUT_DIR_RECTIFIED) if f.startswith(image_type_prefix)], key=self.natural_sort_key)
        print(f"Processing {len(image_files)} images for prefix '{image_type_prefix}'...")

        for img_name in tqdm(image_files, desc=f"SIFT for {image_type_prefix}"):
            img_path = os.path.join(BASE_INPUT_DIR_RECTIFIED, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                print(f"Warning: Could not read image {img_path}")
                continue

            keypoints, descriptors = sift.detectAndCompute(img, None)
            frame_id = img_name.split("_")[-1].split(".")[0]

            output_prefix = "left" if "left" in image_type_prefix else "right"

            if keypoints:
                kp_array = np.array([[
                    kp.pt[0], kp.pt[1], kp.size, kp.angle, kp.response, kp.octave, kp.class_id
                ] for kp in keypoints])
                np.save(os.path.join(output_dir_kp, f"kp_{output_prefix}_{frame_id}.npy"), kp_array)
                np.save(os.path.join(output_dir_desc, f"desc_{output_prefix}_{frame_id}.npy"), descriptors)
            else:
                np.save(os.path.join(output_dir_kp, f"kp_{output_prefix}_{frame_id}.npy"), np.array([]).reshape(0,7))
                np.save(os.path.join(output_dir_desc, f"desc_{output_prefix}_{frame_id}.npy"), np.array([]).reshape(0,128))

    def stereo_matching(self):
        """
        Περιγραφή: Πραγματοποιεί στερεοσκοπική αντιστοίχιση μεταξύ αριστερών και δεξιών εικόνων, εφαρμόζοντας φίλτρα Lowe's Ratio και κάθετης διαφοράς.
        Είσοδος: None - χρησιμοποιεί αρχεία keypoints και descriptors που έχουν ήδη αποθηκευτεί.
        Έξοδος: None - αποθηκεύει τα φιλτραρισμένα στερεοσκοπικά ζεύγη σε αρχεία .npy.
        """
        print("--- Part 2: Stereo Matching (Lowe's Ratio + Vertical Disparity Check) ---")
        bf_stereo = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        frame_ids_stereo = sorted(list(set([f.split("_")[-1].split(".")[0] for f in os.listdir(output_dir_desc) if "_left_" in f])), key=self.natural_sort_key)

        for frame_id in tqdm(frame_ids_stereo, desc="Stereo Matching Frames"):
            good_stereo_matches_final = []
            try:
                kp_left_array = np.load(os.path.join(output_dir_kp, f"kp_left_{frame_id}.npy"))
                desc_left = np.load(os.path.join(output_dir_desc, f"desc_left_{frame_id}.npy"))
                kp_right_array = np.load(os.path.join(output_dir_kp, f"kp_right_{frame_id}.npy"))
                desc_right = np.load(os.path.join(output_dir_desc, f"desc_right_{frame_id}.npy"))

                if not (kp_left_array.ndim == 2 and kp_left_array.shape[0] > 0 and desc_left.ndim == 2 and desc_left.shape[0] > 0 and kp_left_array.shape[0] == desc_left.shape[0] and \
                        kp_right_array.ndim == 2 and kp_right_array.shape[0] > 0 and desc_right.ndim == 2 and desc_right.shape[0] > 0 and kp_right_array.shape[0] == desc_right.shape[0]):
                    np.save(os.path.join(output_dir_stereo_matches, f"stereo_matches_filtered_{frame_id}.npy"), np.array(good_stereo_matches_final, dtype=int).reshape(0,2))
                    continue
            except FileNotFoundError:
                np.save(os.path.join(output_dir_stereo_matches, f"stereo_matches_filtered_{frame_id}.npy"), np.array(good_stereo_matches_final, dtype=int).reshape(0,2))
                continue

            matches_stereo_knn = bf_stereo.knnMatch(desc_left, desc_right, k=2)

            for m_item in matches_stereo_knn:
                if len(m_item) < 2: continue
                m, n = m_item
                if m.distance < LOWE_RATIO * n.distance:
                    idx_left = m.queryIdx
                    idx_right = m.trainIdx
                    y_left = kp_left_array[idx_left][1]
                    y_right = kp_right_array[idx_right][1]
                    if abs(y_left - y_right) < STEREO_VERTICAL_DISPARITY_THRESHOLD:
                        good_stereo_matches_final.append([idx_left, idx_right])

            np.save(os.path.join(output_dir_stereo_matches, f"stereo_matches_filtered_{frame_id}.npy"), np.array(good_stereo_matches_final, dtype=int))
        print("Stereo matching complete.\n")

    def temporal_matching(self):
        """
        Περιγραφή: Πραγματοποιεί χρονική αντιστοίχιση μεταξύ διαδοχικών καρέ της αριστερής κάμερας, εφαρμόζοντας φίλτρα Lowe's Ratio και RANSAC με Essential Matrix.
        Είσοδος: None - χρησιμοποιεί αρχεία keypoints και descriptors που έχουν ήδη αποθηκευτεί.
        Έξοδος: None - αποθηκεύει τα φιλτραρισμένα χρονικά ζεύγη σε αρχεία .npy.
        """
        print("--- Part 3: Temporal Matching (Left Cam, Lowe's Ratio + RANSAC E-Matrix) ---")
        bf_temporal = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        frame_ids_temporal = sorted(list(set([f.split("_")[-1].split(".")[0] for f in os.listdir(output_dir_desc) if "_left_" in f])), key=self.natural_sort_key)

        for i in tqdm(range(len(frame_ids_temporal) - 1), desc="Temporal Matching Frames (Left Cam)"):
            frame_id_t = frame_ids_temporal[i]
            frame_id_t_plus_1 = frame_ids_temporal[i+1]

            good_temporal_matches_ransac_filtered = []

            try:
                kp_left_t_array = np.load(os.path.join(output_dir_kp, f"kp_left_{frame_id_t}.npy"))
                desc_left_t = np.load(os.path.join(output_dir_desc, f"desc_left_{frame_id_t}.npy"))
                kp_left_t_plus_1_array = np.load(os.path.join(output_dir_kp, f"kp_left_{frame_id_t_plus_1}.npy"))
                desc_left_t_plus_1 = np.load(os.path.join(output_dir_desc, f"desc_left_{frame_id_t_plus_1}.npy"))

                if not (kp_left_t_array.ndim == 2 and kp_left_t_array.shape[0] > 0 and desc_left_t.ndim == 2 and desc_left_t.shape[0] > 0 and kp_left_t_array.shape[0] == desc_left_t.shape[0] and \
                        kp_left_t_plus_1_array.ndim == 2 and kp_left_t_plus_1_array.shape[0] > 0 and desc_left_t_plus_1.ndim == 2 and desc_left_t_plus_1.shape[0] > 0 and kp_left_t_plus_1_array.shape[0] == desc_left_t_plus_1.shape[0]):
                    np.save(os.path.join(output_dir_temporal_matches, f"temporal_matches_left_filtered_{frame_id_t}_to_{frame_id_t_plus_1}.npy"), np.array(good_temporal_matches_ransac_filtered, dtype=int).reshape(0,2))
                    continue
            except FileNotFoundError:
                np.save(os.path.join(output_dir_temporal_matches, f"temporal_matches_left_filtered_{frame_id_t}_to_{frame_id_t_plus_1}.npy"), np.array(good_temporal_matches_ransac_filtered, dtype=int).reshape(0,2))
                continue

            matches_temporal_knn = bf_temporal.knnMatch(desc_left_t, desc_left_t_plus_1, k=2)

            initial_good_temporal_matches_indices = []
            pts_t_for_ransac = []
            pts_t_plus_1_for_ransac = []

            for m_item in matches_temporal_knn:
                if len(m_item) < 2: continue
                m, n = m_item
                if m.distance < LOWE_RATIO * n.distance:
                    idx_t = m.queryIdx
                    idx_t_plus_1 = m.trainIdx

                    initial_good_temporal_matches_indices.append([idx_t, idx_t_plus_1])
                    pts_t_for_ransac.append(kp_left_t_array[idx_t][:2])
                    pts_t_plus_1_for_ransac.append(kp_left_t_plus_1_array[idx_t_plus_1][:2])

            if len(initial_good_temporal_matches_indices) >= 5 :
                pts_t_np = np.array(pts_t_for_ransac, dtype=np.float32)
                pts_t_plus_1_np = np.array(pts_t_plus_1_for_ransac, dtype=np.float32)

                E, mask_e = cv2.findEssentialMat(pts_t_np, pts_t_plus_1_np,
                                                 cameraMatrix=K0,
                                                 method=cv2.RANSAC,
                                                 prob=RANSAC_CONFIDENCE_TEMPORAL,
                                                 threshold=RANSAC_REPROJ_THRESHOLD_TEMPORAL)

                if mask_e is not None:
                    for i in range(len(mask_e)):
                        if mask_e[i, 0]:
                            good_temporal_matches_ransac_filtered.append(initial_good_temporal_matches_indices[i])

            np.save(os.path.join(output_dir_temporal_matches, f"temporal_matches_left_filtered_{frame_id_t}_to_{frame_id_t_plus_1}.npy"), np.array(good_temporal_matches_ransac_filtered, dtype=int))

        print("Temporal matching (left camera) complete.\n")


if __name__ == "__main__":
    matcher = FeatureMatcher()

    print("--- Part 1: SIFT Feature Detection and Description ---")
    matcher.detect_and_save_sift_features(image_type_prefix="rect_left_")
    matcher.detect_and_save_sift_features(image_type_prefix="rect_right_")
    print("SIFT feature detection and description complete.\n")

    matcher.stereo_matching()
    matcher.temporal_matching()

    print("All processing finished. Filtered matches saved.")