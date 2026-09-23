#!/usr/bin/env python3
"""T2.5: DA3METRIC-LARGE caches in the HI-SLAM2 prior format.

Third metric teacher (Depth Anything 3 metric head, Apache-2.0). Single-image;
we run it per frame on the same working-res RGB as `build_ma_cache`
(undistort + resize + 56-crop) and resize the metric depth back to the working
grid, so it is drop-in for `run_wired.py`.

`pred.depth` is already in metres for DA3METRIC-LARGE (the API applies the
focal * net / 300 conversion; verified: mean 1.05 m on f1_xyz). Cache layout
mirrors build_dav2_cache: depth (N,H,W) float16 metres, conf, msf, chunk,
frame_id, K. `--only` selects sequences. Runs in the rs3d venv.
"""
import sys, os, types, argparse, json
sys.modules.setdefault("gsplat", types.ModuleType("gsplat"))
import numpy as np
import torch

import build_ma_cache as bmc
from caliprior_common import EUROC5, TUM5

MID = "depth-anything/DA3METRIC-LARGE"


def build(model, name, gen, outdir):
    depth_l, conf_l, msf_l, chk_l, fid_l = [], [], [], [], []
    Kref = None
    for i, rgb, K in gen():
        if Kref is None:
            Kref = K
        p = model.inference([rgb], intrinsics=K[None])
        d = np.asarray(p.depth)[0]
        d = torch.nn.functional.interpolate(
            torch.from_numpy(d)[None, None].float(), size=rgb.shape[:2],
            mode="bilinear", align_corners=False)[0, 0].numpy()
        conf = np.asarray(p.conf)[0] if getattr(p, "conf", None) is not None else None
        if conf is not None and conf.shape != rgb.shape[:2]:
            conf = torch.nn.functional.interpolate(
                torch.from_numpy(conf)[None, None].float(), size=rgb.shape[:2],
                mode="bilinear", align_corners=False)[0, 0].numpy()
        else:
            conf = np.ones(rgb.shape[:2], np.float32)
        d = np.where(np.isfinite(d) & (d > 0.05), d, 0.0).astype(np.float16)
        depth_l.append(d)
        conf_l.append(conf.astype(np.float16))
        msf_l.append(1.0)
        chk_l.append(0)
        fid_l.append(i)
        if len(depth_l) % 200 == 0:
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
    ap.add_argument("--model", default=MID)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    from depth_anything_3.api import DepthAnything3
    model = DepthAnything3.from_pretrained(a.model).to("cuda").eval()
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
        manifest[name] = build(model, name, gen, a.out)
    mpath = f"{a.out}/manifest.json"
    if os.path.exists(mpath):
        old = json.load(open(mpath)); old.update(manifest); manifest = old
    json.dump(manifest, open(mpath, "w"), indent=2)
    print(json.dumps({k: v["n"] for k, v in manifest.items()}, indent=2))


if __name__ == "__main__":
    main()
