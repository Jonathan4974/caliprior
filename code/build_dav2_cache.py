#!/usr/bin/env python3
"""N3: Depth-Anything-V2 Metric-Indoor caches in the HI-SLAM2 prior format.

Second teacher for the generality test. DA-V2 is single-image (no multi-view
chunking); we run it per frame on the same working-res RGB as `build_ma_cache`
(undistort + resize + 56-crop) and resize the metric depth back to the working
grid, so it is drop-in for `run_wired.py`.

Cache layout mirrors build_ma_cache: depth (N,H,W) float16 metric metres,
conf=1, msf=1, chunk=0, frame_id, K. `--only` selects sequences.

Runs in the rs3d venv.
"""
import os, argparse, json
import numpy as np
import torch

import build_ma_cache as bmc
from caliprior_common import EUROC5, TUM5

MID = "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"


def build(model, proc, name, gen, outdir, H=None):
    depth_l, conf_l, msf_l, chk_l, fid_l = [], [], [], [], []
    Kref = None
    for i, rgb, K in gen():
        if Kref is None:
            Kref = K
        im = proc(images=rgb, return_tensors="pt").to("cuda")
        with torch.no_grad():
            d = model(**im).predicted_depth[0]
        d = torch.nn.functional.interpolate(
            d[None, None].float(), size=rgb.shape[:2],
            mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
        d = np.where(np.isfinite(d) & (d > 0.05), d, 0.0).astype(np.float16)
        depth_l.append(d)
        conf_l.append(np.ones_like(d))
        msf_l.append(1.0)
        chk_l.append(0)
        fid_l.append(i)
        if len(depth_l) % 500 == 0:
            print(f"{name}: {len(depth_l)} frames", flush=True)
    np.savez_compressed(
        f"{outdir}/{name}.npz",
        depth=np.stack(depth_l), conf=np.stack(conf_l).astype(np.float16),
        msf=np.array(msf_l, np.float32), chunk=np.array(chk_l, np.int16),
        frame_id=np.array(fid_l, np.int64), K=Kref)
    return {"name": name, "n": len(depth_l), "H": depth_l[0].shape[0],
            "W": depth_l[0].shape[1], "K": Kref.tolist(), "chunks": 1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default=None)
    ap.add_argument("--model", default=MID,
                    help="HF model id (default: DA-V2 Metric-Indoor-Large)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    proc = AutoImageProcessor.from_pretrained(a.model)
    model = AutoModelForDepthEstimation.from_pretrained(a.model).to("cuda").eval()
    print(f"teacher: {a.model}", flush=True)

    seqs = {}
    for seq in EUROC5:
        files, K, gen = bmc.euroc_frames(seq)
        seqs[seq] = (gen, len(files))
    for seq in TUM5:
        rows, K, gen = bmc.tum_frames(seq)
        seqs[seq] = (gen, len(rows))
    if a.only:
        seqs = {k: v for k, v in seqs.items() if k in a.only.split(",")}

    manifest = {}
    for name, (gen, n) in seqs.items():
        manifest[name] = build(model, proc, name, gen, a.out)
    mpath = f"{a.out}/manifest.json"
    if os.path.exists(mpath):
        old = json.load(open(mpath))
        old.update(manifest)
        manifest = old
    json.dump(manifest, open(mpath, "w"), indent=2)
    print(json.dumps({k: v["n"] for k, v in manifest.items()}, indent=2))


if __name__ == "__main__":
    main()
