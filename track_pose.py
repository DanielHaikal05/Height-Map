import glob
import numpy as np
import open3d as o3d
import cv2

FX, FY, cx, cy = 382.7142944335938, 382.7142944335938, 316.7781372070312, 241.8565368652344 
transforms={
    "depth_to_rgb":
   [[0.999986115848177, 0.00201034610955854, 0.00487099778244396, -0.059056371],
    [-0.00202387960198555, 0.999994101493743, 0.00277504758125118, 0.00020030836],
    [-0.00486539024472426, -0.00278486736512245, 0.9999842861223, 0.00059907947],
    [0.0, 0.0, 0.0, 1.0]],
    "rgb_to_robot": np.linalg.inv(
   [[0.0, -1.0, 0.0, -0.012],
    [0.0, 0.0, -1.0, 0.132],
    [1.0, 0.0, 0.0, -0.1],
    [0.0, 0.0, 0.0, 1.0]]),
}

T_depth_to_robot=transforms["rgb_to_robot"] @ transforms["depth_to_rgb"]

def depth_to_points(depth, max_depth=2000, fx=FX, fy=FY, cx=cx, cy=cy):
    h, w = depth.shape
    us, vs = np.meshgrid(np.arange(w), np.arange(h))
    valid=(depth>0) & (depth<max_depth) & (np.isfinite(depth))
    z = depth[valid]
    x = (us[valid] - cx) * z / fx
    y = (vs[valid] - cy) * z / fy
    return np.stack([x, y, z], axis=-1).reshape(-1, 3)

def visualize_depth(path,index,step):
    depth_files=sorted(glob.glob(f"{path}/depth_2/*.png"))
    pose_files=sorted(glob.glob(f"{path}/poses/*.npy"))

    assert index+step<len(depth_files), "Index and/or Step too large"

    frame1, frame2 = depth_to_points(cv2.imread(depth_files[index],cv2.IMREAD_UNCHANGED))/1000, depth_to_points(cv2.imread(depth_files[index+step],cv2.IMREAD_UNCHANGED))/1000
    frame1,frame2 = np.hstack([frame1,np.ones((len(frame1),1))]),np.hstack([frame2,np.ones((len(frame2),1))])
    frame1,frame2 = (T_depth_to_robot@frame1.T).T, (T_depth_to_robot@frame2.T).T
    pose1, pose2 = np.load(pose_files[index],allow_pickle=True), np.load(pose_files[index+step],allow_pickle=True)
    #rel_pose=np.linalg.inv(pose2) @ pose1
    rel_pose=np.linalg.inv(pose2) @ pose1
    frame1=(rel_pose@frame1.T).T
    frame1,frame2= frame1[:,:3], frame2[:,:3]

    pcd1,pcd2 = o3d.geometry.PointCloud(),o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(frame1)
    pcd2.points = o3d.utility.Vector3dVector(frame2)
    pcd1.paint_uniform_color([1,0,0])
    pcd2.paint_uniform_color([0,1,0])
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([pcd1,pcd2,axes])

    print(pose1)
    print(pose2)
    print(rel_pose)

visualize_depth("class",1000,300)