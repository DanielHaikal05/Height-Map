import cv2
import numpy as np
import glob
import os

i=0
for path in sorted(glob.glob("class/raw_depth/*_depth_depth.png")):
    ts=os.path.basename(path).split('_')[0]
    depth_m=cv2.imread(path,cv2.IMREAD_ANYDEPTH)/1000.0
    depth_m[depth_m==0]=np.nan
    i+=1
    if(i%100==0):
        print(f"Image {i} done")
    np.save(f"class/depth_npy/{ts}.npy",depth_m,allow_pickle=True)