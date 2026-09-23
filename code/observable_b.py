#!/usr/bin/env python3
"""Candidate-B observable analysis (CPU, offline).

Observable per keyframe: r_k = median( D_MA / D_backend ), where the backend
ran PRIOR-FREE at the fixed gauge (mean depth 2.0), so D_backend is an
FM-independent, cross-domain-stable reference. r_k therefore tracks the FM's
absolute metric scale (up to one global constant shared across sequences).

Tests: level/stability per domain, half1 vs half2 drift, and correlation with
the measured FM bias (TUM ~0.90) and MA init-frame mean depth.
"""
import os, json, argparse
import numpy as np
import cv2

import caliprior_paths as _CP  # path resolution (env-overridable)
P = _CP.PROJECT_ROOT
SEQS = ["MH_01_easy", "V1_02_medium", "freiburg1_xyz", "freiburg1_desk"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{P}/outputs/observable_b.json")
    ap.add_argument("--window", type=int, default=20,
                    help="init-window size (kf) for r_init; the gauge is set "
                         "at warmup, so the observable that predicts it must be "
                         "an early median, not the whole-sequence median")
    a = ap.parse_args()
    out = {}
    for seq in SEQS:
        if not os.path.exists(f"{P}/outputs/noprior_{seq}/kf_state.npz"):
            print(f"skip {seq}: no prior-free run", flush=True)
            continue
        z = np.load(f"{P}/outputs/ma_cache/{seq}.npz")
        depth = z["depth"].astype(np.float32)
        s = np.load(f"{P}/outputs/noprior_{seq}/kf_state.npz")
        disps = s["disps"]
        kf = s["kf_idx"]
        ratios, ratios_log = [], []
        for i, k in enumerate(kf):
            D_B = 1.0 / np.clip(disps[i].astype(np.float64), 1e-6, None)
            h8, w8 = D_B.shape
            D_MA = cv2.resize(depth[k], (w8, h8), interpolation=cv2.INTER_AREA)
            m = (D_MA > 0.1) & (D_B > 0.1) & (D_MA < 80)
            if m.sum() < 50:
                continue
            r = D_MA[m] / D_B[m]
            ratios.append(float(np.median(r)))
            ratios_log.append(float(np.median(np.log(r))))
        ratios = np.array(ratios)
        rl = np.array(ratios_log)
        n = len(ratios)
        w = min(a.window, n)
        out[seq] = {
            "n": n,
            "median": float(np.median(ratios)),
            "r_init": float(np.median(ratios[:w])),
            "r_init_n": int(w),
            "r_init_log": float(np.median(rl[:w])),
            "std": float(np.std(ratios)),
            "p10": float(np.percentile(ratios, 10)),
            "p90": float(np.percentile(ratios, 90)),
            "half1": float(np.median(ratios[:n // 2])),
            "half2": float(np.median(ratios[n // 2:])),
            "log_std": float(np.std(rl)),
            "trend_log": float(np.polyfit(np.arange(n), rl, 1)[0]) if n > 3 else 0.0,
        }
        print(seq, json.dumps(out[seq]))
    json.dump(out, open(a.out, "w"), indent=2)
    print("\n== candidate-B observable: D_MA / D_backend (prior-free) ==")
    for seq, d in out.items():
        print(f"{seq:16s} med={d['median']:6.3f} r_init={d['r_init']:6.3f} "
              f"std={d['std']:5.3f} h1={d['half1']:6.3f} h2={d['half2']:6.3f} "
              f"logstd={d['log_std']:.3f}")


if __name__ == "__main__":
    main()
