#!/usr/bin/env python3
"""Phase 1: MapAnything prior caches for the 5-seq MVP subset.

Per sequence: sliding-chunk inference (150 views/chunk, memory-efficient),
stores per-frame metric depth_z + confidence + metric_scaling_factor at a
fixed 560-longest resolution whose dims are multiples of 56 (14 for ViT
patches AND 8 for HI-SLAM2's exact [3::8] prior downsampling). Preprocessing
(undistort + resize + center-crop) is done HERE with exact K bookkeeping, so
the wired-in runner consumes pixels and K verbatim -- no loader-side geometry.

K-crop lesson from ma_sanity: never let load_images change dims under a K.

Layout: outputs/ma_cache/<SEQ>.npz with
  depth (N,H,W) float16  metric Z-depth in camera frame
  conf  (N,H,W) float16  per-pixel confidence
  msf   (N,)   float32   metric scaling factor of the view's chunk
  chunk (N,)   int16     chunk id (msf is per-chunk global)
  K     (3,3)  float64   intrinsics of the cached pixels
  frame_id (N,) int64    dataset frame index (for GT association)
plus manifest.json (orig size, crop offsets, dataset, chunk size).

Runs in the rs3d venv: ${VENV_ROOT}/bin/python
"""
import os, sys, json, argparse
import numpy as np
import cv2
import torch

import caliprior_paths as _CP  # path resolution (env-overridable)
MIRROR_EUROCs = _CP.EUROC_ROOT
MIRROR_TUM = _CP.TUM_ROOT
MIRROR_KITTI = _CP.KITTI_ROOT
MIRROR_KITTI_G = _CP.KITTI_GREY_ROOT
CHUNK = 150
LONG = 560

def fit_14(h, w):
    # multiples of lcm(14, 8) = 56: ViT patches AND HI-SLAM2's ht//8 buffers
    # with its [3::8] prior downsampling (exact match iff dims % 8 == 0)
    return (h // 56) * 56, (w // 56) * 56


def resize_crop(gray_or_rgb, K, is_rgb=False):
    """Undistort must happen BEFORE (caller). Resize longest->518, then
    center-crop to multiples of 14. Returns rgb, K_new, (top, left)."""
    h0, w0 = gray_or_rgb.shape[:2]
    s = LONG / max(h0, w0)
    h1, w1 = round(h0 * s), round(w0 * s)
    img = cv2.resize(gray_or_rgb, (w1, h1))
    h2, w2 = fit_14(h1, w1)
    top, left = (h1 - h2) // 2, (w1 - w2) // 2
    img = img[top:top + h2, left:left + w2]
    if not is_rgb:
        img = np.stack([img, img, img], -1)
    K1 = K.copy() * s
    K1[2, 2] = 1.0  # scaling a pinhole matrix must keep K33 = 1
    K1[0, 2] -= left
    K1[1, 2] -= top
    return img, K1


def euroc_frames(seq, cam="cam0"):
    import yaml
    base = f"{MIRROR_EUROCs}/{seq}/mav0/{cam}"
    with open(f"{base}/sensor.yaml") as f:
        y = yaml.safe_load(f)
    fx, fy, cx, cy = y["intrinsics"]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    dist = np.array(y["distortion_coefficients"], dtype=np.float64)
    files = sorted(os.listdir(f"{base}/data"))

    def gen():
        for i, fn in enumerate(files):
            g = cv2.imread(f"{base}/data/{fn}", cv2.IMREAD_UNCHANGED)
            g = cv2.undistort(g, K, dist)
            rgb, K1 = resize_crop(g, K)
            yield i, rgb, K1
    return files, K, gen


TUM_CAL = {
    # freiburg1/2: radial-tangential distortion (the CamIntrinsics_fr*.txt files
    # carry K but zero distortion); freiburg3 is already undistorted.
    "freiburg1": ([[517.3, 0, 318.6], [0, 516.5, 255.3], [0, 0, 1]],
                  [0.2624, -0.9531, -0.0054, 0.0026, 1.1633]),
    "freiburg2": ([[520.9, 0, 325.1], [0, 521.0, 249.7], [0, 0, 1]],
                  [0.2312, -0.7849, -0.0033, -0.0001, 0.9172]),
    "freiburg3": ([[535.4, 0, 320.1], [0, 539.2, 247.6], [0, 0, 1]],
                  [0.0, 0.0, 0.0, 0.0, 0.0]),
}


def tum_frames(seq):
    K = np.array(TUM_CAL[seq.split("_")[0]][0])
    dist = np.array(TUM_CAL[seq.split("_")[0]][1])
    base = f"{MIRROR_TUM}/rgbd_dataset_{seq}"
    rows = [l.split() for l in open(f"{base}/rgb.txt") if not l.startswith("#")]

    def gen():
        for i, (ts, fn) in enumerate(rows):
            img = cv2.imread(f"{base}/{fn}")
            img = cv2.undistort(img, K, dist)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            rgb, K1 = resize_crop(rgb, K, is_rgb=True)
            yield i, rgb, K1
    return rows, K, gen


def kitti_frames(n=800):
    with open(f"{MIRROR_KITTI}/sequences/00/calib.txt") as f:
        for line in f:
            if line.startswith("P2:"):
                P2 = np.array(line.split()[1:], float).reshape(3, 4)
    K = P2[:, :3].copy()
    base = f"{MIRROR_KITTI}/sequences/00/image_2"

    def gen():
        for i in range(n):
            img = cv2.imread(f"{base}/{i:06d}.png")
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            rgb, K1 = resize_crop(rgb, K, is_rgb=True)
            yield i, rgb, K1
    return n, K, gen


def infer_chunk(model, items, H, W):
    import tempfile
    from mapanything.utils.image import load_images
    rgbs = [it[1] for it in items]
    with tempfile.TemporaryDirectory() as d:
        paths = []
        for k, rgb in enumerate(rgbs):
            p = f"{d}/{k:04d}.png"
            cv2.imwrite(p, rgb[:, :, ::-1])
            paths.append(p)
        views = load_images(paths, resize_mode="fixed_size", size=(W, H))
    for v, it in zip(views, items):
        _, _, h, w = v["img"].shape
        assert (h, w) == (H, W), f"loader changed dims {(h,w)} != {(H,W)}"
        v["intrinsics"] = torch.from_numpy(it[2]).float()[None].to("cuda")
    with torch.no_grad():
        preds = model.infer(views, memory_efficient_inference=True,
                            minibatch_size=1, use_amp=True, amp_dtype="bf16")
    out = []
    for p in preds:
        out.append((p["depth_z"][0, :, :, 0].cpu().numpy(),
                    p["conf"][0].cpu().numpy(),
                    float(p["metric_scaling_factor"].reshape(-1)[0])))
    return out


def build_seq(model, name, gen, n, outdir, k_scale=1.0):
    import glob
    depth_l, conf_l, msf_l, chk_l, fid_l = [], [], [], [], []
    items, cid, H, W, Kref = [], 0, None, None, None
    for i, rgb, K in gen():
        if k_scale != 1.0:
            K = K.copy()
            K[0, 0] *= k_scale
            K[1, 1] *= k_scale
        if H is None:
            H, W = rgb.shape[:2]
            Kref = K
        items.append((i, rgb, K))
        if len(items) == CHUNK:
            for (ii, _, _), (dep, cnf, m) in zip(items, infer_chunk(model, items, H, W)):
                depth_l.append(dep.astype(np.float16))
                conf_l.append(cnf.astype(np.float16))
                msf_l.append(m)
                chk_l.append(cid)
                fid_l.append(ii)
            print(f"{name} chunk {cid}: {len(depth_l)}/{n} frames", flush=True)
            items, cid = [], cid + 1
    if items:
        for (ii, _, _), (dep, cnf, m) in zip(items, infer_chunk(model, items, H, W)):
            depth_l.append(dep.astype(np.float16))
            conf_l.append(cnf.astype(np.float16))
            msf_l.append(m)
            chk_l.append(cid)
            fid_l.append(ii)
        print(f"{name} chunk {cid}: {len(depth_l)}/{n} frames", flush=True)
    np.savez_compressed(
        f"{outdir}/{name}.npz",
        depth=np.stack(depth_l), conf=np.stack(conf_l),
        msf=np.array(msf_l, np.float32), chunk=np.array(chk_l, np.int16),
        frame_id=np.array(fid_l, np.int64), K=Kref)
    return {"name": name, "n": len(depth_l), "H": H, "W": W,
            "K": Kref.tolist(), "chunks": cid + 1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default=None, help="comma list of seq keys")
    ap.add_argument("--longest", type=int, default=560)
    ap.add_argument("--chunk", type=int, default=150)
    ap.add_argument("--k-scale", type=float, default=1.0,
                    help="E3 blind-spot: scale fx,fy by this before inference to "
                         "induce a constant teacher scale bias")
    ap.add_argument("--suffix", default="",
                    help="E3 blind-spot: output name suffix (e.g. _k120)")
    a = ap.parse_args()
    global LONG, CHUNK
    LONG, CHUNK = a.longest, a.chunk
    os.makedirs(a.out, exist_ok=True)

    from mapanything.models import MapAnything
    model = MapAnything.from_pretrained("facebook/map-anything-apache").to("cuda")
    model.eval()

    seqs = {}
    for seq in ["MH_01_easy", "V1_02_medium", "MH_02_easy",
                "V1_01_easy", "V2_01_easy",
                "MH_03_medium", "MH_04_difficult", "MH_05_difficult",
                "V1_03_difficult", "V2_02_medium", "V2_03_difficult"]:
        files, K, gen = euroc_frames(seq)
        seqs[seq] = (gen, len(files))
    for seq in ["freiburg1_xyz", "freiburg1_desk", "freiburg2_xyz",
                "freiburg2_desk", "freiburg3_long_office_household"]:
        rows, K, gen = tum_frames(seq)
        seqs[seq] = (gen, len(rows))
    n_k, K, gen = kitti_frames(800)
    seqs["kitti00_800"] = (gen, n_k)

    if a.only:
        seqs = {k: v for k, v in seqs.items() if k in a.only.split(",")}
    manifest = {}
    for name, (gen, n) in seqs.items():
        oid = f"{name}{a.suffix}"
        manifest[oid] = build_seq(model, oid, gen, n, a.out,
                                  k_scale=a.k_scale)
    mpath = f"{a.out}/manifest.json"
    if os.path.exists(mpath):
        old = json.load(open(mpath))
        old.update(manifest)
        manifest = old
    json.dump(manifest, open(mpath, "w"), indent=2)
    print(json.dumps({k: v["n"] for k, v in manifest.items()}, indent=2))


if __name__ == "__main__":
    main()