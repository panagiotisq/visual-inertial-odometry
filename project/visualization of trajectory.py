import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # Για 3D plot
import os
import re
from tqdm import tqdm # Προαιρετικό, για progress bar

# --- Βοηθητική Συνάρτηση Ταξινόμησης ---
def natural_sort_key(s):
    # Βοηθά στην ταξινόμηση ονομάτων αρχείων που περιέχουν αριθμούς
    # π.χ. "file1.npy", "file2.npy", "file10.npy"
    # διασφαλίζοντας ότι το "2" έρχεται πριν το "10".
    # Αυτή η συγκεκριμένη ψάχνει για το {curr_id} στο όνομα αρχείου.
    match = re.search(r'T_vo_(\d+)_from_(\d+)\.npy', s)
    if match:
        return [int(match.group(1))] # Ταξινόμηση με βάση το curr_id
    return [text.lower() if not text.isdigit() else int(text) for text in re.split(r'(\d+)', s)]


# --- Ρυθμίσεις ---
VO_TRANSFORMS_DIR = "features/relative_motions_vo/" # Ο φάκελος όπου αποθηκεύτηκαν οι T_vo
VISUALIZATION_OUTPUT_DIR = "visualizations/"
os.makedirs(VISUALIZATION_OUTPUT_DIR, exist_ok=True)

# --- Κύρια Λογική ---
def plot_trajectory():
    print(f"Loading VO transformations from: {VO_TRANSFORMS_DIR}")
    
    try:
        transform_files = [f for f in os.listdir(VO_TRANSFORMS_DIR) if f.startswith("T_vo_") and f.endswith(".npy")]
        if not transform_files:
            print(f"No VO transformation files found in {VO_TRANSFORMS_DIR}. Cannot plot trajectory.")
            return
            
        # Ταξινόμηση των αρχείων με βάση το 'curr_id' για να εξασφαλιστεί η σωστή σειρά
        transform_files.sort(key=natural_sort_key)
        print(f"Found {len(transform_files)} transformation files: {transform_files[:5]}...") # Εκτύπωση των πρώτων 5 για έλεγχο

    except FileNotFoundError:
        print(f"Error: Directory not found: {VO_TRANSFORMS_DIR}")
        return
    except Exception as e:
        print(f"An error occurred while listing files: {e}")
        return

    global_poses = []
    current_global_pose = np.eye(4) # Η κάμερα ξεκινά από την αρχή (World = C0)
    global_poses.append(current_global_pose.copy()) # Αποθήκευση της αρχικής πόζας

    print("\nCalculating global poses by accumulating relative transformations...")
    for i, filename in enumerate(tqdm(transform_files, desc="Accumulating Poses")):
        try:
            T_vo_step = np.load(os.path.join(VO_TRANSFORMS_DIR, filename)) # Αυτός είναι ο T_{C_k <- C_{k-1}}
            
            # Έλεγχος αν ο πίνακας είναι 4x4
            if T_vo_step.shape != (4, 4):
                print(f"Warning: File {filename} does not contain a valid 4x4 matrix. Skipping.")
                continue

            # Η νέα ολική πόζα είναι η προηγούμενη ολική πόζα επί τον σχετικό μετασχηματισμό
            # T_W_Ck = T_W_Ck-1 @ T_Ck-1_Ck  (αν T_vo_step είναι T_Ck_from_Ck-1)
            # Στον κώδικα calculate_relative_motion_vo.py, το T_vo ήταν T_curr_from_prev, άρα είναι σωστό.
            current_global_pose = current_global_pose @ T_vo_step
            global_poses.append(current_global_pose.copy())

        except Exception as e:
            print(f"Error loading or processing file {filename}: {e}")
            # Αν ένα αρχείο είναι κατεστραμμένο, μπορείς να επιλέξεις να σταματήσεις ή να συνεχίσεις
            # continue 

    if len(global_poses) < 2:
        print("Not enough poses to plot a trajectory.")
        return

    # Εξαγωγή των συντεταγμένων θέσης (x, y, z) από κάθε ολική πόζα
    positions = np.array([pose[:3, 3] for pose in global_poses]) # Κάθε γραμμή είναι [x, y, z]
    x_coords = positions[:, 0]
    y_coords = positions[:, 1]
    z_coords = positions[:, 2]

    print(f"\nPlotting trajectory with {len(positions)} points.")

    # --- 2D Plot (Top-Down X-Y) ---
    plt.figure(figsize=(10, 8))
    plt.plot(x_coords, y_coords, marker='o', linestyle='-', markersize=3, label="VO Trajectory")
    plt.scatter(x_coords[0], y_coords[0], color='green', s=100, label='Start', zorder=5) # Μεγαλύτερη κουκκίδα για την αρχή
    plt.scatter(x_coords[-1], y_coords[-1], color='red', s=100, label='End', zorder=5)   # Μεγαλύτερη κουκκίδα για το τέλος
    plt.xlabel("X (world frame)")
    plt.ylabel("Y (world frame)")
    plt.title("Visual Odometry Trajectory (Top-Down X-Y View)")
    plt.grid(True)
    plt.axis('equal') # Σημαντικό για να μην παραμορφώνεται η τροχιά
    plt.legend()
    plt.savefig(os.path.join(VISUALIZATION_OUTPUT_DIR, "vo_trajectory_2D_XY.png"))
    print(f"2D X-Y trajectory plot saved to: {os.path.join(VISUALIZATION_OUTPUT_DIR, 'vo_trajectory_2D_XY.png')}")
    # plt.show() # Μπορείς να το ενεργοποιήσεις αν θέλεις να εμφανιστεί άμεσα

    # --- 2D Plot (Side X-Z) ---
    plt.figure(figsize=(10, 8))
    plt.plot(x_coords, z_coords, marker='o', linestyle='-', markersize=3, label="VO Trajectory")
    plt.scatter(x_coords[0], z_coords[0], color='green', s=100, label='Start', zorder=5)
    plt.scatter(x_coords[-1], z_coords[-1], color='red', s=100, label='End', zorder=5)
    plt.xlabel("X (world frame)")
    plt.ylabel("Z (world frame - Depth)")
    plt.title("Visual Odometry Trajectory (Side X-Z View)")
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    plt.savefig(os.path.join(VISUALIZATION_OUTPUT_DIR, "vo_trajectory_2D_XZ.png"))
    print(f"2D X-Z trajectory plot saved to: {os.path.join(VISUALIZATION_OUTPUT_DIR, 'vo_trajectory_2D_XZ.png')}")
    # plt.show()

    # --- 3D Plot ---
    fig3d = plt.figure(figsize=(12, 10))
    ax3d = fig3d.add_subplot(111, projection='3d')
    ax3d.plot(x_coords, y_coords, z_coords, marker='o', linestyle='-', markersize=2, label="VO Trajectory")
    ax3d.scatter(x_coords[0], y_coords[0], z_coords[0], color='green', s=100, label='Start', depthshade=True)
    ax3d.scatter(x_coords[-1], y_coords[-1], z_coords[-1], color='red', s=100, label='End', depthshade=True)
    ax3d.set_xlabel("X (world frame)")
    ax3d.set_ylabel("Y (world frame)")
    ax3d.set_zlabel("Z (world frame)")
    ax3d.set_title("Visual Odometry Trajectory (3D View)")
    ax3d.legend()
    # Προσπάθεια για ισομετρική προβολή στο 3D, αν και η matplotlib έχει περιορισμούς
    # For a somewhat 'equal' aspect ratio in 3D:
    max_range = np.array([x_coords.max()-x_coords.min(), y_coords.max()-y_coords.min(), z_coords.max()-z_coords.min()]).max() / 2.0
    mid_x = (x_coords.max()+x_coords.min()) * 0.5
    mid_y = (y_coords.max()+y_coords.min()) * 0.5
    mid_z = (z_coords.max()+z_coords.min()) * 0.5
    ax3d.set_xlim(mid_x - max_range, mid_x + max_range)
    ax3d.set_ylim(mid_y - max_range, mid_y + max_range)
    ax3d.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.savefig(os.path.join(VISUALIZATION_OUTPUT_DIR, "vo_trajectory_3D.png"))
    print(f"3D trajectory plot saved to: {os.path.join(VISUALIZATION_OUTPUT_DIR, 'vo_trajectory_3D.png')}")
    
    print("\nShowing all plots...")
    plt.show() # Εμφάνιση όλων των plots στο τέλος

if __name__ == '__main__':
    plot_trajectory()