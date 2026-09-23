#!/usr/bin/env python3
"""N2: realistic metric anchors from the datasets' own sensors.

TUM RGB-D : registered depth in the init window -> mean depth (metres).
EuRoC    : stereo cam0/cam1 SGBM disparity (rectified) -> mean depth via
           z = fx * baseline / disparity.

The anchor is a single scalar mean depth used by `run_wired.py --anchor-mean`
(gauge mean set from the sensor, not the FM). Emits outputs/anchors.json.

Runs in the rs3d venv (cv2/numpy/yaml).
"""
import os, json, argparse
import numpy as np
import cv2
import yaml

import caliprior_paths as _CP  # path resolution (env-overridable)
from caliprior_common import TUM5 as TUM, EUROC5 as EUROC
P = _CP.PROJECT_ROOT
EU = _CP.EUROC_ROOT
TU = _CP.TUM_ROOT


def tum_anchor(seq, W=15):
    base = f"{TU}/rgbd_dataset_{seq}"
    rgb = [l.split() for l in open(f"{base}/rgb.txt") if not l.startswith("#")]
    dep = [l.split() for l in open(f"{base}/depth.txt") if not l.startswith("#")]
    dep_ts = np.array([float(r[0]) for r in dep])
    vals = []
    for ts, fn in rgb[:W]:
        k = int(np.searchsorted(dep_ts, float(ts)).clip(0, len(dep_ts) - 1))
        d = cv2.imread(f"{base}/{dep[k][1]}", cv2.IMREAD_UNCHANGED)
        if d is None:
            continue
        z = d.astype(np.float32) / 5000.0
        h, w = z.shape
        c = z[h // 4:3 * h // 4, w // 4:3 * w // 4]
        v = c[(c > 0.3) & (c < 8.0)]
        if v.size > 100:
            vals.append(float(v.mean()))
    return (float(np.mean(vals)), len(vals)) if vals else (None, 0)


def euroc_anchor(seq, W=12):
    base = f"{EU}/{seq}/mav0"
    with open(f"{base}/cam0/sensor.yaml") as f:
        s0 = yaml.safe_load(f)
    with open(f"{base}/cam1/sensor.yaml") as f:
        s1 = yaml.safe_load(f)
    def K_of(s):
        fx, fy, cx, cy = s["intrinsics"]
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    K0, D0 = K_of(s0), np.array(s0["distortion_coefficients"])
    K1, D1 = K_of(s1), np.array(s1["distortion_coefficients"])
    T0 = np.array(s0["T_BS"]["data"]).reshape(4, 4)
    T1 = np.array(s1["T_BS"]["data"]).reshape(4, 4)
    T01 = np.linalg.inv(T1) @ T0            # cam0 -> cam1
    R, t = T01[:3, :3], T01[:3, 3]
    baseline = float(np.linalg.norm(t))
    f0 = sorted(os.listdir(f"{base}/cam0/data"))
    f1 = sorted(os.listdir(f"{base}/cam1/data"))
    ts1 = {n.split(".")[0]: n for n in f1}
    sample = cv2.imread(f"{base}/cam0/data/{f0[0]}", cv2.IMREAD_GRAYSCALE)
    h, w = sample.shape[:2]
    R0, R1, P0, P1, _, _, _ = cv2.stereoRectify(
        K0, D0, K1, D1, (w, h), R, t, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0.0)
    m0 = cv2.initUndistortRectifyMap(K0, D0, R0, P0, (w, h), cv2.CV_32FC1)
    m1 = cv2.initUndistortRectifyMap(K1, D1, R1, P1, (w, h), cv2.CV_32FC1)
    sgbm = cv2.StereoSGBM_create(minDisparity=0, numDisparities=128,
                                 blockSize=7, P1=8 * 7 * 7, P2=32 * 7 * 7,
                                 uniquenessRatio=10, speckleWindowSize=80,
                                 speckleRange=2, disp12MaxDiff=1)
    fx, vals = P0[0, 0], []
    for n0 in f0[:W]:
        key = n0.split(".")[0]
        n1 = ts1.get(key)
        if n1 is None:
            continue
        L = cv2.remap(cv2.imread(f"{base}/cam0/data/{n0}", cv2.IMREAD_GRAYSCALE),
                      m0[0], m0[1], cv2.INTER_LINEAR)
        Rt = cv2.remap(cv2.imread(f"{base}/cam1/data/{n1}", cv2.IMREAD_GRAYSCALE),
                       m1[0], m1[1], cv2.INTER_LINEAR)
        disp = sgbm.compute(L, Rt).astype(np.float32) / 16.0
        v = disp[disp > 1.0]
        if v.size > 500:
            z = fx * baseline / v
            z = z[(z > 0.3) & (z < 12.0)]
            if z.size > 500:
                vals.append(float(np.median(z)))
    return (float(np.mean(vals)), len(vals)) if vals else (None, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{P}/outputs/anchors.json")
    ap.add_argument("--seqs", default=None, help="comma list to restrict")
    a = ap.parse_args()
    out = {}
    if os.path.exists(a.out):
        out = json.load(open(a.out))
    seqs = (a.seqs.split(",") if a.seqs else TUM + EUROC)
    for seq in seqs:
        if seq in TUM:
            m, n = tum_anchor(seq)
            mod = "rgbd"
        else:
            m, n = euroc_anchor(seq)
            mod = "stereo"
        out[seq] = {"anchor_mean": m, "n_frames": n, "modality": mod}
        print(f"{seq:32s} {mod:6s} anchor_mean={m} (n={n})")
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
