import os
import glob
import numpy as np
import pandas as pd

def normalize_quat_xyzw(q):
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    return q / (n + 1e-12)

def slerp_xyzw(q0, q1, t):
    # q = [qx,qy,qz,qw]
    q0 = normalize_quat_xyzw(q0)
    q1 = normalize_quat_xyzw(q1)

    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    # Nearly identical -> lerp
    if dot > 0.9995:
        return normalize_quat_xyzw(q0 + t * (q1 - q0))

    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta = np.sin(theta)
    w0 = np.sin((1.0 - t) * theta) / sin_theta
    w1 = np.sin(t * theta) / sin_theta
    return w0 * q0 + w1 * q1

def quat_xyzw_to_R(qx, qy, qz, qw):
    # normalize
    n = np.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
    qw, qx, qy, qz = qw/(n+1e-12), qx/(n+1e-12), qy/(n+1e-12), qz/(n+1e-12)
    return np.array([
        [1 - 2*(qy*qy + qz*qz),     2*(qx*qy - qw*qz),     2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz),         1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy),         2*(qy*qz + qw*qx),     1 - 2*(qx*qx + qy*qy)]
    ], dtype=np.float64)

def build_T_world_from_body(x, y, z, qx, qy, qz, qw):
    R = quat_xyzw_to_R(qx, qy, qz, qw)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3,  3] = [x, y, z]
    return T

def build_poses(path,
                source_file="FasterLIO.txt",
                target_folder="depth_2",
                out_folder="poses"):
    # --- load FasterLIO ---
    columns = ['timestamp_s', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']
    df = pd.read_csv(
        os.path.join(path, source_file),
        sep=r"\s+",
        header=None,
        comment="#",          # ignores the "#timestamp ..." line
        names=columns
    )

    ts_s = df['timestamp_s'].to_numpy(dtype=np.float64)
    ts_us = np.round(ts_s * 1e6).astype(np.int64)  # KEEP DECIMALS

    pos = df[['x','y','z']].to_numpy(dtype=np.float64)
    quat = df[['qx','qy','qz','qw']].to_numpy(dtype=np.float64)  # xyzw

    # sort by time (important for searchsorted)
    order = np.argsort(ts_us)
    ts_us = ts_us[order]
    pos   = pos[order]
    quat  = quat[order]

    # --- target timestamps from filenames ---
    target_files = sorted(glob.glob(os.path.join(path, target_folder, "*.png")))
    target_ts_us = np.array([int(os.path.splitext(os.path.basename(f))[0]) for f in target_files], dtype=np.int64)

    # --- output folder ---
    out_dir = os.path.join(path, out_folder)
    os.makedirs(out_dir, exist_ok=True)

    # --- interpolate at each target timestamp ---
    for t in target_ts_us:
        # clamp outside range
        if t <= ts_us[0]:
            p = pos[0]
            q = quat[0]
        elif t >= ts_us[-1]:
            p = pos[-1]
            q = quat[-1]
        else:
            j = int(np.searchsorted(ts_us, t))
            i = j - 1
            t0, t1 = ts_us[i], ts_us[j]
            a = float((t - t0) / (t1 - t0 + 1e-12))

            p = (1.0 - a) * pos[i] + a * pos[j]
            q = slerp_xyzw(quat[i], quat[j], a)

        T = build_T_world_from_body(p[0], p[1], p[2], q[0], q[1], q[2], q[3])
        np.save(os.path.join(out_dir, f"{t}.npy"), T)

    print(f"Saved {len(target_ts_us)} interpolated poses to {out_dir}")

build_poses("class")
        
