import numpy as np
import os
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# === Load Data ===

# Load downsampled inertial vectors (N x 6): [wx, wy, wz, ax, ay, az]
imu_vectors = np.load("features/processed/inertial_vectors.npy")

# Load relative VO (Visual Odometry) transformations and compute cumulative pose
vo_dir = "features/relative_motions_vo/"
vo_files = sorted(os.listdir(vo_dir))
pose_list = [np.eye(4)]  # Initial pose is identity matrix

for f in vo_files:
    T = np.load(os.path.join(vo_dir, f))  # 4x4 transformation matrix
    pose_list.append(pose_list[-1] @ T)   # Update pose by chaining transformations

# Load VO covariance matrices (6x6 each)
qcov_dir = "features/vo_covariance/"
qcov_files = sorted(os.listdir(qcov_dir))
qcov_matrices = [
    np.diag(np.diag(np.load(os.path.join(qcov_dir, f))))  # Only use diagonal
    for f in qcov_files
]

print("found", len(qcov_matrices), "qcov_matrices")

dt = 0.05  # Time step (s) — corresponds to 20 Hz frequency

# === Utility Functions ===

def matrix_to_pose6d(T):
    """
    Converts a 4x4 transformation matrix into a 6D pose vector [x, y, z, roll, pitch, yaw].
    Args:
        T (np.ndarray): 4x4 transformation matrix.
    Returns:
        np.ndarray: 6-element pose vector.
    """
    pos = T[:3, 3]
    rot = R.from_matrix(T[:3, :3]).as_euler('xyz')
    return np.concatenate([pos, rot])

vo_poses = np.array([matrix_to_pose6d(p) for p in pose_list])  # List of VO poses

# === EKF Functions ===

def euler_to_rot(euler):
    """
    Converts Euler angles to a rotation matrix.
    Args:
        euler (np.ndarray): 3-element array of Euler angles [roll, pitch, yaw].
    Returns:
        np.ndarray: 3x3 rotation matrix.
    """
    return R.from_euler('xyz', euler).as_matrix()

def imu_dynamics(mu, u, dt=0.05):
    """
    Predicts the next state using IMU measurements.
    Args:
        mu (np.ndarray): Current state [x, y, z, roll, pitch, yaw, vx, vy, vz].
        u (np.ndarray): IMU input [wx, wy, wz, ax, ay, az].
        dt (float): Time step.
    Returns:
        np.ndarray: Predicted next state.
    """
    pos = mu[0:3]
    ori = mu[3:6]
    vel = mu[6:9]

    omega = u[0:3]
    acc_body = u[3:6]

    # Convert angular velocity to world frame and integrate
    R_body = euler_to_rot(ori)
    omega_world = R_body @ omega
    new_ori = ori + omega_world * dt  # Approximate integration

    # Transform acceleration and subtract gravity
    R_world = euler_to_rot(new_ori)
    acc_world = R_world @ acc_body - np.array([0, 0, 9.81])

    new_vel = vel + acc_world * dt
    new_pos = pos + new_vel * dt + 0.5 * acc_world * dt**2

    return np.concatenate([new_pos, new_ori, new_vel])

def imu_jacobian(mu, u, dt=0.05, eps=1e-5):
    """
    Computes Jacobian of the IMU dynamics using finite differences.
    Args:
        mu (np.ndarray): Current state.
        u (np.ndarray): IMU input.
        dt (float): Time step.
        eps (float): Small perturbation for numerical derivative.
    Returns:
        np.ndarray: Jacobian matrix (9x9).
    """
    n = len(mu)
    J = np.zeros((n, n))
    f_mu = imu_dynamics(mu, u, dt)

    for i in range(n):
        mu_perturbed = mu.copy()
        mu_perturbed[i] += eps
        f_perturbed = imu_dynamics(mu_perturbed, u, dt)
        J[:, i] = (f_perturbed - f_mu) / eps

    return J

def vo_observation(mu):
    """
    Observation function: maps full state to observed VO pose.
    Args:
        mu (np.ndarray): Current full state.
    Returns:
        np.ndarray: VO observation [x, y, z, roll, pitch, yaw].
    """
    return mu[0:6]

def vo_observation_jacobian(mu):
    """
    Computes Jacobian of the VO observation model.

    Args:
        mu (np.ndarray): Current state.

    Returns:
        np.ndarray: 6x9 Jacobian matrix.
    """
    H = np.zeros((6, 9))
    H[0:6, 0:6] = np.eye(6)
    return H

def kalman_step(mu, sig, u, z, f, df, h, dh, R, Q):
    """
    Executes a single step of the Extended Kalman Filter.
    Args:
        mu (np.ndarray): Current state estimate.
        sig (np.ndarray): Current covariance estimate.
        u (np.ndarray): IMU input.
        z (np.ndarray): Measurement (VO pose).
        f (function): Process model.
        df (function): Jacobian of process model.
        h (function): Measurement model.
        dh (function): Jacobian of measurement model.
        R (np.ndarray): Process noise covariance.
        Q (np.ndarray): Measurement noise covariance.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Updated state and covariance.
    """
    # Predict step
    mu_bar = f(mu, u)
    F = df(mu, u)
    sig_bar = F @ sig @ F.T + R

    # Update step
    H = dh(mu_bar)
    K = sig_bar @ H.T @ np.linalg.inv(H @ sig_bar @ H.T + Q)

    mu_new = mu_bar + K @ (z - h(mu_bar))
    sig_new = (np.eye(len(mu)) - K @ H) @ sig_bar

    return mu_new, sig_new

# === EKF Initialization ===

N = min(len(imu_vectors), len(vo_poses), len(qcov_matrices))
mu = np.zeros(9)              # Initial state
sig = np.eye(9) * 0.001       # Initial uncertainty

# IMU noise parameters
accel_noise_density = 0.0028  # m/s^1.5
gyro_noise_density = 0.00016  # rad/s^0.5

# Discrete-time variances
accel_var = (accel_noise_density ** 2) / dt
gyro_var = (gyro_noise_density ** 2) / dt


# R_cov: process noise covariance matrix (9x9) based on IMU noise — e.g., errors in acceleration and angular velocity integration.
R_cov = np.diag([
    0.25 * dt**4 * accel_var,  # position x
    0.25 * dt**4 * accel_var,  # position y
    0.25 * dt**4 * accel_var,  # position z
    dt**2 * gyro_var,          # orientation x
    dt**2 * gyro_var,          # orientation y
    dt**2 * gyro_var,          # orientation z
    dt**2 * accel_var,         # velocity x
    dt**2 * accel_var,         # velocity y
    dt**2 * accel_var          # velocity z
])
R_cov = np.eye(9) * 1e-4

# Measurement noise covariance (based on VO)
Q_cov = np.eye(6) * 1e-8

print(R_cov)

states = [mu.copy()]

# === EKF Loop ===
for k in range(N):
    u = imu_vectors[k]
    z = vo_poses[k]
    # Q_cov_k = qcov_matrices[k]  # Use corresponding VO covariance matrix

    mu, sig = kalman_step(
        mu, sig, u, z,
        imu_dynamics, imu_jacobian,
        vo_observation, vo_observation_jacobian,
        R_cov, Q_cov  #  Q_cov_k
    )
    states.append(mu.copy())

# Save final EKF result
ekf_output = np.array(states)
np.save("features/processed/ekf_trajectory.npy", ekf_output)
ekf_output.shape

# === Visualization ===
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot estimated EKF trajectory
ax.plot(ekf_output[:, 0], ekf_output[:, 1], ekf_output[:, 2], label='EKF Trajectory', color='b')

ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('3D Estimated Trajectory')
ax.legend()
ax.grid(True)

plt.show()
