import numpy as np
import os
import re
from tqdm import tqdm

class VisualOdometry:
    def natural_sort_key(self, s):
        """
        Περιγραφή: Ταξινομεί αλφαριθμητικά που περιέχουν αριθμούς με φυσικό τρόπο.
        Είσοδος: s (str) - Το αλφαριθμητικό προς ταξινόμηση.
        Έξοδος: list - Μια λίστα από ακέραιους ή αλφαριθμητικά για χρήση στην ταξινόμηση.
        """
        return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

    def skew_symmetric(self, v):
        """
        Περιγραφή: Δημιουργεί έναν αντισύμμετρο (skew-symmetric) πίνακα από ένα 3-διάστατο διάνυσμα.
        Είσοδος: v (numpy.array) - Ένα 3-διάστατο διάνυσμα.
        Έξοδος: numpy.array - Ο 3x3 αντισύμμετρος πίνακας.
        """
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])

    def estimate_rigid_transform_3d_kabsch(self, points_source, points_target):
        """
        Περιγραφή: Υπολογίζει τον άκαμπτο μετασχηματισμό (περιστροφή R και μετάφραση t) μεταξύ δύο συνόλων 3D σημείων χρησιμοποιώντας τον αλγόριθμο Kabsch.
        Είσοδος:
            points_source (numpy.array) - Nx3 πίνακας 3D σημείων πηγής.
            points_target (numpy.array) - Nx3 πίνακας 3D σημείων στόχου.
        Έξοδος:
            R (numpy.array) - Ο 3x3 πίνακας περιστροφής.
            t (numpy.array) - Ο 3x1 διάνυσμα μετάφρασης.
        """
        if points_source.shape[0] != points_target.shape[0]:
            print("Warning: Mismatch in number of points for Kabsch. Returning identity.")
            return np.eye(3), np.zeros((3, 1))
        if points_source.shape[0] < 3:
            return np.eye(3), np.zeros((3, 1))

        centroid_source = np.mean(points_source, axis=0)
        centroid_target = np.mean(points_target, axis=0)
        centered_source = points_source - centroid_source
        centered_target = points_target - centroid_target
        H = centered_source.T @ centered_target
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        if np.linalg.det(R) < 0:
            Vt_corrected = Vt.copy()
            Vt_corrected[-1, :] *= -1
            R = Vt_corrected.T @ U.T
        t = centroid_target - R @ centroid_source
        return R, t.reshape(3, 1)

    def estimate_covariance(self, points_source, points_target, R, t_col):
        """
        Περιγραφή: Υπολογίζει τον πίνακα συνδιακύμανσης 6x6 για τον μετασχηματισμό (R, t).
        Είσοδος:
            points_source (numpy.array) - Nx3 πίνακας 3D σημείων πηγής.
            points_target (numpy.array) - Nx3 πίνακας 3D σημείων στόχου.
            R (numpy.array) - Ο 3x3 πίνακας περιστροφής.
            t_col (numpy.array) - Ο 3x1 διάνυσμα μετάφρασης.
        Έξοδος:
            cov (numpy.array) - Ο 6x6 πίνακας συνδιακύμανσης.
        """
        if points_source.shape[0] < 6:
            return np.diag([1e3] * 6)

        t_flat = t_col.flatten()
        transformed_points = (R @ points_source.T).T + t_flat
        residuals = points_target - transformed_points

        rmse = np.sqrt(np.mean(np.sum(residuals**2, axis=1)))
        sigma2 = rmse**2
        if sigma2 < 1e-12: sigma2 = 1e-12

        H_info = np.zeros((6, 6))
        for i in range(points_source.shape[0]):
            p_i = points_source[i, :]
            J_i = np.hstack((-np.eye(3), self.skew_symmetric(R @ p_i)))
            H_info += J_i.T @ J_i

        try:
            cov = sigma2 * np.linalg.pinv(H_info)
        except np.linalg.LinAlgError:
            cov = np.diag([1e3] * 6)
        return cov

    def calculate_relative_motion_and_cov_for_pair(self, frame_id_prev, frame_id_curr,
                                               pointcloud_dir="features/pointclouds/",
                                               temporal_match_dir="features/temporal_matches_filtered/"):
        """
        Περιγραφή: Υπολογίζει τον σχετικό μετασχηματισμό (T_vo) και τον πίνακα συνδιακύμανσής του (Cov_vo) μεταξύ δύο διαδοχικών καρέ.
        Είσοδος:
            frame_id_prev (str) - Το ID του προηγούμενου καρέ.
            frame_id_curr (str) - Το ID του τρέχοντος καρέ.
            pointcloud_dir (str) - Ο φάκελος που περιέχει τα 3D σύννεφα σημείων.
            temporal_match_dir (str) - Ο φάκελος που περιέχει τις χρονικές αντιστοιχίες.
        Έξοδος:
            tuple (numpy.array, numpy.array) - Το T_vo (4x4 πίνακας) και ο Cov_vo (6x6 πίνακας) ή (None, None) αν αποτύχει ο υπολογισμός.
        """
        try:
            points_3d_prev = np.load(os.path.join(pointcloud_dir, f"points_3d_{frame_id_prev}.npy"))
            source_indices_prev = np.load(os.path.join(pointcloud_dir, f"source_indices_left_{frame_id_prev}.npy"))
            points_3d_curr = np.load(os.path.join(pointcloud_dir, f"points_3d_{frame_id_curr}.npy"))
            source_indices_curr = np.load(os.path.join(pointcloud_dir, f"source_indices_left_{frame_id_curr}.npy"))
            temporal_matches_path = os.path.join(temporal_match_dir, f"temporal_matches_left_filtered_{frame_id_prev}_to_{frame_id_curr}.npy")
            temporal_match_indices_2d = np.load(temporal_matches_path)
        except FileNotFoundError:
            return None, None

        if points_3d_prev.ndim !=2 or points_3d_prev.shape[0] == 0 or points_3d_prev.shape[1] != 3 or \
           points_3d_curr.ndim !=2 or points_3d_curr.shape[0] == 0 or points_3d_curr.shape[1] != 3 or \
           temporal_match_indices_2d.ndim != 2 or temporal_match_indices_2d.shape[0] == 0 or temporal_match_indices_2d.shape[1] !=2:
            return None, None

        map_prev_kp_idx_to_3d_idx = {kp_idx_2d: i for i, kp_idx_2d in enumerate(source_indices_prev)}
        map_curr_kp_idx_to_3d_idx = {kp_idx_2d: i for i, kp_idx_2d in enumerate(source_indices_curr)}

        matched_points_prev_list, matched_points_curr_list = [], []
        for i in range(temporal_match_indices_2d.shape[0]):
            idx_2d_prev = temporal_match_indices_2d[i, 0]
            idx_2d_curr = temporal_match_indices_2d[i, 1]
            if idx_2d_prev in map_prev_kp_idx_to_3d_idx and idx_2d_curr in map_curr_kp_idx_to_3d_idx:
                actual_3d_idx_prev = map_prev_kp_idx_to_3d_idx[idx_2d_prev]
                actual_3d_idx_curr = map_curr_kp_idx_to_3d_idx[idx_2d_curr]
                if actual_3d_idx_prev < points_3d_prev.shape[0] and actual_3d_idx_curr < points_3d_curr.shape[0]:
                    matched_points_prev_list.append(points_3d_prev[actual_3d_idx_prev])
                    matched_points_curr_list.append(points_3d_curr[actual_3d_idx_curr])

        if len(matched_points_prev_list) < 10:
            return None, None

        points_target_np = np.array(matched_points_prev_list)
        points_source_np = np.array(matched_points_curr_list)

        R_prev_from_curr, t_prev_from_curr = self.estimate_rigid_transform_3d_kabsch(points_source_np, points_target_np)

        if np.array_equal(R_prev_from_curr, np.eye(3)) and np.array_equal(t_prev_from_curr, np.zeros((3,1))):
            return None, None

        R_curr_from_prev = R_prev_from_curr.T
        t_curr_from_prev = -R_prev_from_curr.T @ t_prev_from_curr

        T_vo = np.eye(4)
        T_vo[:3, :3] = R_curr_from_prev
        T_vo[:3, 3] = t_curr_from_prev.flatten()

        Cov_vo = self.estimate_covariance(points_source_np, points_target_np, R_curr_from_prev, t_curr_from_prev)

        return T_vo, Cov_vo

if __name__ == '__main__':
    vo_processor = VisualOdometry()

    POINTCLOUD_INPUT_DIR = "features/pointclouds/"
    TEMPORAL_MATCH_INPUT_DIR = "features/temporal_matches_filtered/"
    VO_OUTPUT_DIR = "features/relative_motions_vo/"
    VO_COV_OUTPUT_DIR = "features/vo_covariance/"

    os.makedirs(VO_OUTPUT_DIR, exist_ok=True)
    os.makedirs(VO_COV_OUTPUT_DIR, exist_ok=True)

    print(f"Starting batch processing...")

    try:
        pc_files = [f for f in os.listdir(POINTCLOUD_INPUT_DIR) if f.startswith("points_3d_") and f.endswith(".npy")]
        if not pc_files:
            print(f"No point cloud files found in {POINTCLOUD_INPUT_DIR}. Exiting.")
            exit()

        frame_ids_available = sorted(
            list(set([re.search(r'points_3d_(\d+)\.npy', f).group(1) for f in pc_files if re.search(r'points_3d_(\d+)\.npy', f)])),
            key=vo_processor.natural_sort_key
        )

        if len(frame_ids_available) < 2:
            print("Need at least two frames to calculate relative motion.")
            exit()

        print(f"Found {len(frame_ids_available)} frames. Processing {len(frame_ids_available) - 1} pairs.")

        successful_transforms = 0
        for i in tqdm(range(len(frame_ids_available) - 1), desc="Processing frame pairs"):
            prev_id = frame_ids_available[i]
            curr_id = frame_ids_available[i+1]

            T_vo_step, Cov_vo_step = vo_processor.calculate_relative_motion_and_cov_for_pair(prev_id, curr_id,
                                                                          pointcloud_dir=POINTCLOUD_INPUT_DIR,
                                                                          temporal_match_dir=TEMPORAL_MATCH_INPUT_DIR)

            if T_vo_step is not None:
                output_filename_T = f"T_vo_{curr_id}_from_{prev_id}.npy"
                np.save(os.path.join(VO_OUTPUT_DIR, output_filename_T), T_vo_step)

                output_filename_Cov = f"Cov_vo_{curr_id}_from_{prev_id}.npy"
                np.save(os.path.join(VO_COV_OUTPUT_DIR, output_filename_Cov), Cov_vo_step)

                successful_transforms += 1

        print(f"\nBatch processing complete. Successfully computed and saved {successful_transforms} transforms and covariances.")

    except Exception as e:
        print(f"An unexpected error occurred during batch processing: {e}")
        import traceback
        traceback.print_exc()