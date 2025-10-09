import numpy as np
import pandas as pd
import os
# Load the IMU CSV
imu_df = pd.read_csv("dataset-corridor4_512_16\mav0\imu0\data.csv", comment='#')

# Extract the 6D vector: [wx, wy, wz, ax, ay, az]
imu_data = imu_df.iloc[:, 1:7].values

# Downsample every 10 steps by taking the mean
N = 10
num_blocks = imu_data.shape[0] // N
downsampled = np.array([
    np.mean(imu_data[i * N:(i + 1) * N], axis=0)
    for i in range(num_blocks)
])

# Save the result
output_dir = "features/processed"
os.makedirs(output_dir, exist_ok=True)
np.save("features/processed/inertial_vectors.npy", downsampled)
print(f"Saved {downsampled.shape[0]} downsampled vectors to features/processed/inertial_vectors.npy")
