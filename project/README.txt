
Αυτό το project υλοποιεί ένα Visual-Inertial Odometry pipeline χρησιμοποιώντας εικόνες από stereo κάμερα και δεδομένα IMU. Παρακάτω βρίσκονται οδηγίες για το πώς να εκτελέσετε τον κώδικα.


## Απαιτήσεις
- Απαιτούμενες βιβλιοθήκες:
  - numpy
  - pandas
  - opencv-python (cv2)
  - scipy
  - matplotlib
  - tqdm
- Dataset: https://vision.in.tum.de/tumvi/exported/euroc/512_16/dataset-corridor4_512_16.tar
Τοποθετήστε τον φάκελο του dataset `dataset-corridor4_512_16` στον ίδιο φάκελο με τα scripts

## Pipeline Overview
1. Image undistortion and rectification
2. Feature detection and description
3. Feature matching (stereo and temporal)
4. 3D point cloud generation
5. Visual Odometry (VO) estimation
6. Extended Kalman Filter (EKF) for sensor fusion

## Execution Order
Run the scripts in the following order:

1. `step1 undistortion.py`
   - Undistorts and rectifies stereo images
   - Outputs:
     - Undistorted images in `cam0_images_undistorted/` and `cam1_images_undistorted/`
     - Rectified images in `rectified/`
     - Calibration data in `calibration_data_rectified/`

2. `step2 feature detection and description.py`
   - Detects features in rectified images
   - Outputs:
     - Keypoints in `features/keypoints/`
     - Descriptors in `features/descriptors/`

3. `step3 feature matching between cams and frames.py`
   - Performs stereo and temporal feature matching
   - Outputs:
     - Stereo matches in `features/stereo_matches_filtered/`
     - Temporal matches in `features/temporal_matches_filtered/`

4. `step4 pointcloud.py`
   - Generates 3D point clouds from stereo matches
   - Outputs:
     - Point clouds in `features/pointclouds/`

5. `step5 TVO.py`
   - Computes relative motions between frames (Visual Odometry)
   - Outputs:
     - Transformations in `features/relative_motions_vo/`
     - Covariance matrices in `features/vo_covariance/`

6. `downsample_inertial.py`
   - Downsamples IMU data to match camera frame rate
   - Outputs:
     - Downsampled inertial vectors in `features/processed/inertial_vectors.npy`

7. `ekf.py`
   - Fuses visual and inertial data using Extended Kalman Filter
   - Outputs:
     - Estimated trajectory in `features/processed/ekf_trajectory.npy`
     - Displays 3D trajectory plot


## Expected Runtime
The complete pipeline may take several minutes to run, with most time spent on image processing and feature matching steps.