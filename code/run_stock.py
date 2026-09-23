#!/usr/bin/env python3
"""Phase 0 stock baseline: run unmodified HI-SLAM2 on an EuRoC sequence.

EuRoC-specific handling demo.py lacks: grayscale PNGs (-> RGB stack),
radial-tangential undistort, real-ns timestamps, GT ATE + scale metrics,
and keyframe-state dumps for the Phase-2 adapter. Tracking itself is
stock Hi2 (Omnidata prior + JDSA); no prior injection here.

Usage (inside hislam2 env, see env_setup.sh):
  python run_stock.py --seq MH_01_easy --out outputs/stock_MH_01
"""
import os, sys, argparse, json
import numpy as np
import cv2
import torch
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (100000, rlimit[1]))

import caliprior_paths as _CP  # path resolution (env-overridable)
sys.path.insert(0, _CP.HISLAM2_ROOT)
sys.path.insert(0, _CP.HISLAM2_ROOT + "/hislam2")
from hi2 import Hi2
import lietorch
from caliprior_common import umeyama_ate

MIRROR = _CP.EUROC_ROOT
MIRROR_KITTI = _CP.KITTI_ROOT
MIRROR_KITTI_G = _CP.KITTI_GREY_ROOT
RES = 341 * 640  # same working resolution as demo.py


def load_cam_yaml(seq):
    import yaml
    with open(f"{MIRROR}/{seq}/mav0/cam0/sensor.yaml") as f:
        y = yaml.safe_load(f)
    fx, fy, cx, cy = y["intrinsics"]
    dist = np.array(y["distortion_coefficients"], dtype=np.float64)
    return np.array([fx, fy, cx, cy]), dist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="MH_01_easy")
    ap.add_argument("--dataset", default="euroc", choices=["euroc", "kitti"])
    ap.add_argument("--n", type=int, default=800, help="kitti frames")
    ap.add_argument("--workwh", default=None, help="force working W H (kitti res-match)")
    ap.add_argument("--config", default=_CP.HISLAM2_ROOT + "/config/euroc_config.yaml")
    ap.add_argument("--weights", default=_CP.HISLAM2_ROOT + "/pretrained_models/droid.pth")
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--length", type=int, default=100000)
    ap.add_argument("--buffer", type=int, default=-1)
    a = ap.parse_args()
    a.out = os.path.abspath(a.out)
    a.config = os.path.abspath(a.config)
    a.weights = os.path.abspath(a.weights)
    os.makedirs(a.out, exist_ok=True)
    # HI-SLAM2 loads its Omnidata checkpoints via paths relative to its root.
    os.chdir(_CP.HISLAM2_ROOT)

    dist_on, is_rgb = True, False
    K, imgdir, files, cam_ns, gt_ns, gt_p = None, None, None, None, None, None
    if a.dataset == "euroc":
        (fx, fy, cx, cy), dist = load_cam_yaml(a.seq)
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        imgdir = f"{MIRROR}/{a.seq}/mav0/cam0/data"
        files = sorted(os.listdir(imgdir))[a.start:a.start + a.length]
        cam_ns = np.array([int(f.split(".")[0]) for f in files])
        gt = np.loadtxt(f"{MIRROR}/{a.seq}/mav0/state_groundtruth_estimate0/data.csv",
                        delimiter=",", skiprows=1)
        gt_ns, gt_p = gt[:, 0].astype(np.int64), gt[:, 1:4]
    elif a.dataset == "kitti":
        with open(f"{MIRROR_KITTI}/sequences/00/calib.txt") as f:
            for line in f:
                if line.startswith("P2:"):
                    P2 = np.array(line.split()[1:], float).reshape(3, 4)
        fx, fy, cx, cy = P2[0, 0], P2[1, 1], P2[0, 2], P2[1, 2]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        dist = np.zeros(4)
        imgdir = f"{MIRROR_KITTI}/sequences/00/image_2"
        files = [f"{i:06d}.png" for i in range(a.n)]
        cam_ns = np.arange(a.n)
        T = []
        with open(f"{MIRROR_KITTI_G}/00/poses.txt") as f:
            for i, line in enumerate(f):
                if i >= a.n:
                    break
                T4 = np.eye(4)
                T4[:3] = np.array(line.split(), float).reshape(3, 4)
                T.append(T4[:3, 3])
        gt_ns, gt_p, dist_on, is_rgb = cam_ns.copy(), np.array(T), False, True

    torch.multiprocessing.set_start_method("spawn")
    args = type("A", (), {})()
    args.config, args.weights, args.buffer = a.config, a.weights, a.buffer
    args.output = a.out
    args.gsvis, args.droidvis, args.gtdepthdir = False, False, None

    hi2, images0 = None, None
    N = len(files)
    args.buffer = min(1000, N // 10 + 150) if args.buffer < 0 else args.buffer
    for t, fn in enumerate(files):
        if is_rgb:
            img = cv2.imread(os.path.join(imgdir, fn))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            g = cv2.imread(os.path.join(imgdir, fn), cv2.IMREAD_UNCHANGED)
            if dist_on:
                g = cv2.undistort(g, K, dist)
            img = np.stack([g, g, g], -1)
        h0, w0 = img.shape[:2]
        if a.workwh:
            w1, h1 = (int(x) for x in a.workwh.split())
            assert w1 % 8 == 0 and h1 % 8 == 0, "HI-SLAM2 needs %8 working dims"
        else:
            h1 = int(h0 * np.sqrt(RES / (h0 * w0))); h1 -= h1 % 8
            w1 = int(w0 * np.sqrt(RES / (h0 * w0))); w1 -= w1 % 8
        img = cv2.resize(img, (w1, h1))
        intr = torch.tensor([fx * w1 / w0, fy * h1 / h0,
                             cx * w1 / w0, cy * h1 / h0])
        image = torch.as_tensor(img).permute(2, 0, 1)
        if hi2 is None:
            args.image_size = [image.shape[1], image.shape[2]]
            hi2 = Hi2(args)
        hi2.track(t, image[None], intrinsics=intr[None], is_last=(t == N - 1))
        if t % 500 == 0:
            print(f"[{t}/{N}] kf={hi2.video.counter.value}", flush=True)

    hi2.terminate()
    n = hi2.video.counter.value
    kf_idx = hi2.video.tstamp[:n].cpu().numpy().astype(int)
    poses_wc = lietorch.SE3(hi2.video.poses[:n]).inv().matrix().data.cpu().numpy()  # T_wc (n,3,4)
    kf_ns = cam_ns[kf_idx]
    np.savetxt(f"{a.out}/traj_kf.txt",
               np.concatenate([kf_ns[:, None] / 1e9, poses_wc.reshape(n, -1)], axis=1))
    np.savez(f"{a.out}/kf_state.npz", kf_idx=kf_idx, kf_ns=kf_ns,
             poses_wc=poses_wc,
             dscales=hi2.video.dscales[:n].cpu().numpy())

    # ATE: nearest-GT association + Umeyama (reports scale factor s: est->GT)
    gi = np.searchsorted(gt_ns, kf_ns).clip(0, len(gt_ns) - 1)
    rmse, s = umeyama_ate(poses_wc[:, :3, 3], gt_p[gi])
    metrics = {"seq": a.seq, "n_frames": N, "n_kf": int(n),
               "ate_rmse_m": float(rmse), "umeyama_scale_est2gt": s,
               "scale_err_pct": float(abs(s - 1.0) * 100.0)}
    json.dump(metrics, open(f"{a.out}/metrics.json", "w"), indent=2)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
