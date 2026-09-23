#!/usr/bin/env python3
"""P3.4: local (intra-sequence) drift — does the FM's per-keyframe scale drift
reach the reconstruction's local metric geometry, after a global anchor?

The gauge-once result killed *global* online states. The only place a
temporal state could still have a lever is local geometry: if the FM depth's
local scale drifts within a sequence, the dense depth factor (alpha=0.01) might
bend the reconstruction's local depth. If the reconstruction's local scale
tracks the FM's local scale, an online state is useful; if it is absorbed (as
the global case was), the open question is closed negative.

On TUM (registered GT depth) we compute, per keyframe:
  ma_ratio_k  = median( MA_depth_k  / GT_depth_k )   # FM's local metric bias
  rec_ratio_k = median( rec_depth_k / GT_depth_k )   # vehicle's local metric bias
with rec_depth = 1/disps from kf_state (anchored metric gauge). Then report
intra-sequence drift of each and their correlation.

GT is loaded on demand at the ~100 keyframes only (not all frames). Emits
outputs/local_drift.json.
"""
import os, json
import numpy as np
import cv2

import caliprior_paths as _CP  # path resolution (env-overridable)
P = _CP.PROJECT_ROOT
T = _CP.TUM_ROOT
K0 = np.array([[517.3, 0, 318.6], [0, 516.5, 255.3], [0, 0, 1]])
DIST = np.array([0.2624, -0.9531, -0.0054, 0.0026, 1.1633])
LONG = 560
SEQS = ["freiburg1_xyz", "freiburg1_desk"]
# runs that saved `disps` (post-dump change); the anchor is a global scalar and
# does not affect the intra-sequence drift/correlation measured here.
RUNS = {"freiburg1_xyz": "anchorA_freiburg1_xyz",
        "freiburg1_desk": "anchorA_freiburg1_desk"}


def fit_56(h, w):
    return (h // 56) * 56, (w // 56) * 56


def gt_loader(seq, cache_shape):
    base = f"{T}/rgbd_dataset_{seq}"
    rgb_rows = [l.split() for l in open(f"{base}/rgb.txt") if not l.startswith("#")]
    dep_rows = [l.split() for l in open(f"{base}/depth.txt") if not l.startswith("#")]
    dep_ts = np.array([float(r[0]) for r in dep_rows])
    sample = cv2.imread(f"{base}/{dep_rows[0][1]}", cv2.IMREAD_UNCHANGED)
    h0, w0 = sample.shape[:2]
    m1, m2 = cv2.initUndistortRectifyMap(K0, DIST, None, K0, (w0, h0), cv2.CV_32FC1)
    s = LONG / max(h0, w0)
    h1, w1 = round(h0 * s), round(w0 * s)
    h2, w2 = fit_56(h1, w1)
    top, left = (h1 - h2) // 2, (w1 - w2) // 2

    def get(i):
        ts = float(rgb_rows[i][0])
        k = int(np.searchsorted(dep_ts, ts).clip(0, len(dep_ts) - 1))
        gt = cv2.imread(f"{base}/{dep_rows[k][1]}", cv2.IMREAD_UNCHANGED)
        if gt is None:
            return None
        gtu = cv2.remap(gt, m1, m2, cv2.INTER_NEAREST)
        gtu = cv2.resize(gtu, (w1, h1), interpolation=cv2.INTER_NEAREST)
        gtu = gtu[top:top + h2, left:left + w2]
        if gtu.shape != cache_shape:
            return None
        return gtu.astype(np.float32) / 5000.0
    return get


def half_slope(x):
    x = np.asarray(x, float)
    n = len(x)
    if n < 4:
        return float("nan"), float("nan"), float("nan")
    return (float(np.median(x[:n // 2])), float(np.median(x[n // 2:])),
            float(np.polyfit(np.arange(n), x, 1)[0]))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default=None, help="analyze a single sequence")
    ap.add_argument("--run", default=None, help="run dir name under outputs/")
    a = ap.parse_args()
    seqs = [a.seq] if a.seq else SEQS
    out = {}
    for seq in seqs:
        rundir = a.run if a.run else RUNS[seq]
        run = f"{P}/outputs/{rundir}/kf_state.npz"
        if not os.path.exists(run):
            print(f"skip {seq}: no run")
            continue
        cache = np.load(f"{P}/outputs/ma_cache/{seq}.npz")
        ma = cache["depth"].astype(np.float32)
        getgt = gt_loader(seq, tuple(ma.shape[1:]))
        s = np.load(run)
        disps, kf = s["disps"], s["kf_idx"]
        ma_r, rec_r = [], []
        for i, k in enumerate(kf):
            G0 = getgt(int(k))
            if G0 is None:
                continue
            REC = 1.0 / np.clip(disps[i].astype(np.float64), 1e-6, None)
            h8, w8 = REC.shape
            G = cv2.resize(G0, (w8, h8), interpolation=cv2.INTER_NEAREST)
            MA = cv2.resize(ma[k], (w8, h8), interpolation=cv2.INTER_AREA)
            m = (G > 0.3) & (G < 6.0) & (REC > 0.05) & (MA > 0.1)
            if m.sum() < 100:
                continue
            ma_r.append(float(np.median(MA[m] / G[m])))
            rec_r.append(float(np.median(REC[m] / G[m])))
        ma_r, rec_r = np.array(ma_r), np.array(rec_r)
        if len(rec_r) < 4:
            print(f"skip {seq}: too few valid kfs")
            continue
        c = float(np.corrcoef(ma_r, rec_r)[0, 1]) if len(rec_r) > 2 else float("nan")
        mh = half_slope(ma_r)
        rh = half_slope(rec_r)
        key = seq if a.run is None else f"{seq}__{rundir}"
        out[key] = {
            "run": rundir,
            "n_kf": len(rec_r),
            "ma_mean": float(np.mean(ma_r)), "rec_mean": float(np.mean(rec_r)),
            "ma_half1": mh[0], "ma_half2": mh[1], "ma_slope": mh[2],
            "rec_half1": rh[0], "rec_half2": rh[1], "rec_slope": rh[2],
            "rec_std": float(np.std(rec_r)), "ma_std": float(np.std(ma_r)),
            "corr_ma_rec": c,
            "pass_through_slope": float(np.polyfit(ma_r, rec_r, 1)[0]),
        }
        d = out[key]
        print(f"{key}: n={d['n_kf']} ma {d['ma_half1']:.3f}->{d['ma_half2']:.3f} "
              f"(slope {d['ma_slope']:+.2e}, std {d['ma_std']:.3f}) | rec "
              f"{d['rec_half1']:.3f}->{d['rec_half2']:.3f} (slope "
              f"{d['rec_slope']:+.2e}, std {d['rec_std']:.3f}) | corr={c:+.2f} "
              f"pass={d['pass_through_slope']:+.2f}")
    op = f"{P}/outputs/local_drift_{a.run}.json" if a.run else f"{P}/outputs/local_drift.json"
    json.dump(out, open(op, "w"), indent=2)
    print(f"wrote {op}")


if __name__ == "__main__":
    main()
