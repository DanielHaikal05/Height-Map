import open3d as o3d
import numpy as np
import cv2
import os

# Update this to your CEAR folder
DEPTH_PATH = "/home/daniel/Downloads/between_buildings_day_trot/depth/1702746358277337_depth_rgb.png"

# Read the 16-bit depth image (millimeters)
depth_raw = cv2.imread(DEPTH_PATH, cv2.IMREAD_UNCHANGED).astype(np.float32)

print("Depth image shape:", depth_raw.shape)
print("Min depth:", np.nanmin(depth_raw), "Max depth:", np.nanmax(depth_raw))

# Convert to meters if needed
depth_m = depth_raw / 1000.0  # many RealSense sensors store mm

# Define intrinsics (example: RealSense D435)
FX = FY = 615.0  # approximate focal lengths
CX = depth_raw.shape[1] / 2.0
CY = depth_raw.shape[0] / 2.0

intrinsics = o3d.camera.PinholeCameraIntrinsic(
    width=depth_raw.shape[1],
    height=depth_raw.shape[0],
    fx=FX,
    fy=FY,
    cx=CX,
    cy=CY
)

# Create a point cloud from the depth map
depth_o3d = o3d.geometry.Image(depth_m)
pcd = o3d.geometry.PointCloud.create_from_depth_image(
    depth_o3d,
    intrinsics,
    depth_scale=1.0,  # because we already converted to meters
    stride=1
)

# Optional: remove NaNs and far points
pcd = pcd.voxel_down_sample(voxel_size=0.02)
pcd.remove_non_finite_points()

# Visualize
o3d.visualization.draw_geometries([pcd])
