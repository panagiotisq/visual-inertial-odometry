import cv2
import numpy as np
import os
import re
from tqdm import tqdm

input_dir = "rectified/"
output_dir_kp = "features/keypoints/"
output_dir_desc = "features/descriptors/"
os.makedirs(output_dir_kp, exist_ok=True)
os.makedirs(output_dir_desc, exist_ok=True)

sift = cv2.SIFT_create(
    nfeatures=500,
    nOctaveLayers=3,
    contrastThreshold=0.02,
    edgeThreshold=15,
    sigma=1.6
)

class FeatureDetectionAndDescription:

    def natural_sort_key(self, s):
        return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

    def process_images(self, image_files, image_type):
        print(f"\nProcessing {len(image_files)} {image_type} images...")
        for img_name in tqdm(image_files, desc=f"Processing {image_type} images"):
            img_path = os.path.join(input_dir, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                print(f"Warning: Could not read image {img_path}")
                continue

            self.detect_and_compute_features(img, img_name, image_type)

    def detect_and_compute_features(self, img, img_name, image_type):
        keypoints, descriptors = sift.detectAndCompute(img, None)

        frame_id = img_name.split("_")[-1].split(".")[0]
        kp_output_path = os.path.join(output_dir_kp, f"kp_{image_type}_{frame_id}.npy")
        desc_output_path = os.path.join(output_dir_desc, f"desc_{image_type}_{frame_id}.npy")

        if keypoints:
            kp_array = np.array([[
                kp.pt[0], kp.pt[1], kp.size, kp.angle, kp.response, kp.octave, kp.class_id
            ] for kp in keypoints])
            np.save(kp_output_path, kp_array)
            np.save(desc_output_path, descriptors)
        else:
            np.save(kp_output_path, np.array([]))
            np.save(desc_output_path, np.array([]))

if __name__ == "__main__":
    feature_detector = FeatureDetectionAndDescription()

    left_image_files = sorted([f for f in os.listdir(input_dir) if f.startswith("rect_left_")], key=feature_detector.natural_sort_key)
    feature_detector.process_images(left_image_files, "left")

    right_image_files = sorted([f for f in os.listdir(input_dir) if f.startswith("rect_right_")], key=feature_detector.natural_sort_key)
    feature_detector.process_images(right_image_files, "right")

    print("\nFeature detection and description complete for all images.")