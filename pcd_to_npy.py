import open3d as o3d
import numpy as np
import os

pcd_dir = "/home/daniel/hm_ws/class/pcd"
pcd_files = sorted([f for f in os.listdir(pcd_dir) if f[-4:]==".pcd" and f[0]!="f"])
save_dir = "/home/daniel/hm_ws/class/lidar_npy"
os.makedirs(save_dir, exist_ok=True)

all_clouds = []

for file in pcd_files:
    file_path = os.path.join(pcd_dir, file)
    pcd = o3d.io.read_point_cloud(file_path)
    points = np.asarray(pcd.points)   # Nx3 array
    all_clouds.append(points)
    
    # Use same filename (without .pcd) for .npy
    filename_npy = os.path.join(save_dir, file.replace(".pcd", ".npy"))
    np.save(filename_npy, points)
    
    print(f"Saved {filename_npy} with shape {points.shape}")

# Example: access first frame
first_frame = all_clouds[0]
print(first_frame[:5])  # first 5 points
