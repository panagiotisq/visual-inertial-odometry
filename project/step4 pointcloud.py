import cv2
import numpy as np
import os
import re
from tqdm import tqdm

CALIBRATION_DATA_DIR = "calibration_data_rectified/"
BASE_INPUT_DIR_KP = "features/keypoints/"
BASE_INPUT_DIR_STEREO_MATCHES = "features/stereo_matches_filtered/"
BASE_OUTPUT_DIR_POINTCLOUD = "features/pointclouds/"
os.makedirs(BASE_OUTPUT_DIR_POINTCLOUD, exist_ok=True)

try:
    P_cam0_rect = np.load(os.path.join(CALIBRATION_DATA_DIR, "P1.npy"))
    P_cam1_rect = np.load(os.path.join(CALIBRATION_DATA_DIR, "P2.npy"))
    print("Projection matrices P1 (P_cam0_rect) and P2 (P_cam1_rect) loaded successfully.")
except FileNotFoundError:
    print(f"ERROR: Projection matrices P1.npy and/or P2.npy not found in {CALIBRATION_DATA_DIR}")
    print("Please ensure you have run undistortion.py and it saved these files (P1.npy, P2.npy).")
    exit()

try:
    K0_rectified = np.load(os.path.join(CALIBRATION_DATA_DIR, "K0_rectified.npy"))
    print("K0_rectified loaded (may be used for PnP later).")
except FileNotFoundError:
    print(f"Warning: K0_rectified.npy not found in {CALIBRATION_DATA_DIR}. This will be needed for PnP.")
    K0_rectified = None

class PointCloudGenerator:
    def natural_sort_key(self, s):
        """
        Περιγραφή: Ταξινομεί αλφαριθμητικά που περιέχουν αριθμούς με φυσικό τρόπο.
        Είσοδος: s (str) - το αλφαριθμητικό προς ταξινόμηση.
        Έξοδος: list - μια λίστα από ακέραιους ή αλφαριθμητικά για χρήση στην ταξινόμηση.
        """
        return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

    def generate_point_clouds(self):
        """
        Περιγραφή: Δημιουργεί σύννεφα σημείων 3D από τα αντιστοιχισμένα 2D keypoints των στερεοσκοπικών εικόνων.
        Είσοδος: None - φορτώνει keypoints και αντιστοιχίες από προκαθορισμένους φακέλους.
        Έξοδος: None - αποθηκεύει τα 3D σημεία και τους δείκτες προέλευσής τους σε αρχεία .npy.
        """
        try:
            kp_left_files = [f for f in os.listdir(BASE_INPUT_DIR_KP) if f.startswith("kp_left_") and f.endswith(".npy")]
            if not kp_left_files:
                print(f"No keypoint files found in {BASE_INPUT_DIR_KP} starting with 'kp_left_'")
                exit()
            frame_ids_stereo = sorted(list(set([re.search(r'kp_left_(\d+)\.npy', f).group(1) for f in kp_left_files if re.search(r'kp_left_(\d+)\.npy', f)])), key=self.natural_sort_key)
            print(f"Found {len(frame_ids_stereo)} frame IDs to process for point cloud generation.")
        except FileNotFoundError:
            print(f"Error: Keypoint directory {BASE_INPUT_DIR_KP} not found.")
            exit()
        except Exception as e:
            print(f"Error listing frame IDs: {e}")
            exit()

        for frame_id in tqdm(frame_ids_stereo, desc="Generating Point Clouds"):
            kp_left_path = os.path.join(BASE_INPUT_DIR_KP, f"kp_left_{frame_id}.npy")
            kp_right_path = os.path.join(BASE_INPUT_DIR_KP, f"kp_right_{frame_id}.npy")
            stereo_matches_path = os.path.join(BASE_INPUT_DIR_STEREO_MATCHES, f"stereo_matches_filtered_{frame_id}.npy")

            try:
                kp_left_array = np.load(kp_left_path)
                kp_right_array = np.load(kp_right_path)
                stereo_match_indices = np.load(stereo_matches_path)
            except FileNotFoundError:
                print(f"Warning: Skipping frame {frame_id}. Missing keypoints or matches file.")
                np.save(os.path.join(BASE_OUTPUT_DIR_POINTCLOUD, f"points_3d_{frame_id}.npy"), np.array([]).reshape(0,3))
                np.save(os.path.join(BASE_OUTPUT_DIR_POINTCLOUD, f"source_indices_left_{frame_id}.npy"), np.array([], dtype=int))
                continue

            if not (kp_left_array.ndim == 2 and kp_left_array.shape[0] > 0 and \
                    kp_right_array.ndim == 2 and kp_right_array.shape[0] > 0 and \
                    stereo_match_indices.ndim == 2 and stereo_match_indices.shape[0] > 0 and stereo_match_indices.shape[1] == 2) :
                np.save(os.path.join(BASE_OUTPUT_DIR_POINTCLOUD, f"points_3d_{frame_id}.npy"), np.array([]).reshape(0,3))
                np.save(os.path.join(BASE_OUTPUT_DIR_POINTCLOUD, f"source_indices_left_{frame_id}.npy"), np.array([], dtype=int))
                continue

            points1_for_triangulation_list = []
            points2_for_triangulation_list = []
            source_2d_indices_left = []

            for i in range(stereo_match_indices.shape[0]):
                idx_l = stereo_match_indices[i, 0]
                idx_r = stereo_match_indices[i, 1]

                if idx_l < kp_left_array.shape[0] and idx_r < kp_right_array.shape[0]:
                    points1_for_triangulation_list.append(kp_left_array[idx_l, :2])
                    points2_for_triangulation_list.append(kp_right_array[idx_r, :2])
                    source_2d_indices_left.append(idx_l)
                else:
                    pass

            if not points1_for_triangulation_list:
                np.save(os.path.join(BASE_OUTPUT_DIR_POINTCLOUD, f"points_3d_{frame_id}.npy"), np.array([]).reshape(0,3))
                np.save(os.path.join(BASE_OUTPUT_DIR_POINTCLOUD, f"source_indices_left_{frame_id}.npy"), np.array([], dtype=int))
                continue

            points1_np = np.array(points1_for_triangulation_list, dtype=np.float32).T
            points2_np = np.array(points2_for_triangulation_list, dtype=np.float32).T

            points_4d_hom = cv2.triangulatePoints(P_cam0_rect, P_cam1_rect, points1_np, points2_np)

            w = points_4d_hom[3, :]
            epsilon = 1e-8
            valid_w_mask = np.abs(w) > epsilon

            points_3d_cartesian_all = np.zeros((3, points_4d_hom.shape[1]), dtype=np.float32)
            if np.any(valid_w_mask):
                points_3d_cartesian_all[:, valid_w_mask] = points_4d_hom[:3, valid_w_mask] / w[valid_w_mask]

            points_3d_cartesian_all = points_3d_cartesian_all.T

            source_2d_indices_left_np = np.array(source_2d_indices_left, dtype=int)
            source_2d_indices_left_valid_w = source_2d_indices_left_np[valid_w_mask]
            points_3d_cartesian_valid_w = points_3d_cartesian_all[valid_w_mask, :]

            FINAL_FILTER_MIN_DEPTH = 0.1
            FINAL_FILTER_MAX_DEPTH = 50.0

            valid_depth_mask = (points_3d_cartesian_valid_w[:, 2] > FINAL_FILTER_MIN_DEPTH) & \
                               (points_3d_cartesian_valid_w[:, 2] < FINAL_FILTER_MAX_DEPTH)

            final_3d_points = points_3d_cartesian_valid_w[valid_depth_mask]
            final_source_2d_indices_left = source_2d_indices_left_valid_w[valid_depth_mask]

            np.save(os.path.join(BASE_OUTPUT_DIR_POINTCLOUD, f"points_3d_{frame_id}.npy"), final_3d_points)
            np.save(os.path.join(BASE_OUTPUT_DIR_POINTCLOUD, f"source_indices_left_{frame_id}.npy"), final_source_2d_indices_left)

        print(f"Point cloud generation complete. Results saved in {BASE_OUTPUT_DIR_POINTCLOUD}")

if __name__ == "__main__":
    generator = PointCloudGenerator()
    generator.generate_point_clouds()