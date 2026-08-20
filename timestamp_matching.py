import numpy as np
import glob
import os
import cv2

path="class"
lidar_files=sorted(glob.glob(f"{path}/lidar_npy/*.npy"))
depth_files=sorted(glob.glob(f"{path}/raw_depth/*_depth_depth.png"))
lidar_times=[int(os.path.basename(l)[:-4]) for l in lidar_files]
depth=[int(os.path.basename(d)[:-16]) for d in depth_files]
depth_times=[d for d in depth if d>=(lidar_times[0]/1000.0) and d<=(lidar_times[-1]/1000.0)]
ld=iter(lidar_times)
dp=iter(depth_times)

def closest_matching(path):
    os.makedirs(f"{path}/depth_2",exist_ok=True)
    os.makedirs(f"{path}/lidar_npy2",exist_ok=True)
    j1,j2=next(ld),next(ld)
    for d_t in depth_times:
        while(abs(d_t-j2/1000.0)<abs(d_t-j1/1000.0)):
            try:
                j1,j2=j2,next(ld)
            except StopIteration:
                return
        np.save(f"{path}/lidar_npy2/{d_t}.npy",np.load(f"{path}/lidar_npy/{j1}.npy",allow_pickle=True),allow_pickle=True)
        cv2.imwrite(f"{path}/depth_2/{d_t}.png",cv2.imread(f"{path}/raw_depth/{d_t}_depth_depth.png",cv2.IMREAD_UNCHANGED))


def check_identical(path):
    l=sorted(int(os.path.basename(ld)[:-4]) for ld in glob.glob(f"{path}/lidar_npy2/*.npy"))
    d=sorted(int(os.path.basename(dp)[:-4]) for dp in glob.glob(f"{path}/depth_2/*.png"))
    print(f"Same length? {len(l)==len(d)}")     
    print(f"Identical? {l==d}")

closest_matching(path)
check_identical(path)





















def lerp_frames(td,tl1,tl2,l1,l2):
    a=(td-tl1)/(tl2-tl1)
    return (1-a)*l1+a*l2

def interpolate_lidars():
    i=next(dp)
    j1,j2=next(ld),next(ld)
    l1,l2=lidar[j1],lidar[j2]
    while True:
        while(i<=j2 and i>j1):
            f2=lerp_frames(i,j1,j2,l1,l2)
            print(f"Saving interpolated LiDAR: lidar_npy/{i}.npy")
            np.save(f"lidar_npy/{i}.npy",f2.astype(np.float32),allow_pickle=True)
            try:
                i=next(dp)
            except StopIteration:
                return
        try:
            j1,j2=j2,next(ld)
            l1,l2=l2,lidar[j2]
        except StopIteration:
            return