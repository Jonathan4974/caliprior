#!/usr/bin/env python3
"""P3.2 / N1.3: OOD trust gate from the observable's DISPERSION, N sequences.

The mean of r_init = D_MA/D_backend does not order the required metric
correction across domains. Its *dispersion* over the init window does
separate domains: when the FM's metric scale is trustworthy, r is tight; when
the FM is OOD, r scatters.

Two independent OOD labels (no single circular label):
  A) metric truth: |scale_ma - 1| > tau_a, where scale_ma is the gauge-anchored
     trajectory scale vs GT trajectory (from outputs/gaugeMA_<seq>/metrics.json).
  B) GT-free FM signal: MapAnything confidence statistic from the cache.

Reports AUROC of the dispersion gate against each label, plus the raw table.
Emits outputs/gate_ood.json (+ png).
"""
import os, json, argparse
import numpy as np
import cv2

import caliprior_paths as _CP  # path resolution (env-overridable)
from caliprior_common import auroc
P = _CP.PROJECT_ROOT
ALL = ["freiburg1_xyz", "freiburg1_desk", "freiburg2_xyz", "freiburg2_desk",
       "freiburg3_long_office_household",
       "MH_01_easy", "V1_02_medium", "MH_02_easy", "V1_01_easy", "V2_01_easy",
       # E3(i) gate expansion: six remaining EuRoC sequences (mostly negatives)
       "MH_03_medium", "MH_04_difficult", "MH_05_difficult",
       "V1_03_difficult", "V2_02_medium", "V2_03_difficult"]
TAU_A = 0.30  # |scale-1| above this => FM metric scale is wrong


def r_series(seq, window, cache_dir, prefix):
    kf = f"{P}/outputs/{prefix}noprior_{seq}/kf_state.npz"
    cache = f"{cache_dir}/{seq}.npz"
    if not (os.path.exists(kf) and os.path.exists(cache)):
        return None
    z = np.load(cache)
    depth = z["depth"].astype(np.float32)
    s = np.load(kf)
    disps, idx = s["disps"], s["kf_idx"]
    ratios = []
    for i, k in enumerate(idx):
        D_B = 1.0 / np.clip(disps[i].astype(np.float64), 1e-6, None)
        h8, w8 = D_B.shape
        D_MA = cv2.resize(depth[k], (w8, h8), interpolation=cv2.INTER_AREA)
        m = (D_MA > 0.1) & (D_B > 0.1) & (D_MA < 80)
        if m.sum() < 50:
            continue
        ratios.append(float(np.median(D_MA[m] / D_B[m])))
    if len(ratios) < 5:
        return None
    return np.array(ratios), window


def conf_stat(seq, cache_dir):
    cache = f"{cache_dir}/{seq}.npz"
    if not os.path.exists(cache):
        return None
    c = np.load(cache)["conf"].astype(np.float32)
    return {"conf_mean": float(c.mean()), "conf_med": float(np.median(c)),
            "conf_p10": float(np.percentile(c, 10))}


def scale_ma(seq, prefix):
    f = f"{P}/outputs/{prefix}gaugeMA_{seq}/metrics.json"
    if not os.path.exists(f):
        return None
    return float(json.load(open(f))["umeyama_scale_est2gt"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--cache-dir", default=f"{P}/outputs/ma_cache")
    ap.add_argument("--prefix", default="", help="run dir prefix, e.g. dav2_")
    ap.add_argument("--out", default=f"{P}/outputs/gate_ood.json")
    a = ap.parse_args()
    rows = {}
    for seq in ALL:
        rs = r_series(seq, a.window, a.cache_dir, a.prefix)
        if rs is None:
            print(f"skip {seq}: incomplete")
            continue
        r, w = rs
        w = min(w, len(r))
        sm = scale_ma(seq, a.prefix)
        cs = conf_stat(seq, a.cache_dir)
        rows[seq] = {"n": len(r),
                     "r_mean": float(r[:w].mean()),
                     "log_std": float(np.std(np.log(r[:w]))),
                     "std": float(r[:w].std()),
                     "scale_ma": sm,
                     "label_a": (None if sm is None else int(abs(sm - 1.0) > TAU_A)),
                     "conf_med": (None if cs is None else cs["conf_med"])}
        rr = rows[seq]
        print(f"{seq:32s} r_mean={rr['r_mean']:6.3f} log_std={rr['log_std']:6.4f} "
              f"scale_ma={rr['scale_ma']} label_a={rr['label_a']} "
              f"conf_med={rr['conf_med']}")

    keys = list(rows)
    disp = np.array([rows[k]["log_std"] for k in keys])
    # label A
    idxA = [i for i, k in enumerate(keys) if rows[k]["label_a"] is not None]
    auA = float("nan")
    if idxA:
        auA = auroc(disp[idxA], np.array([rows[keys[i]]["label_a"] for i in idxA]))
    # label B: median split of FM confidence (low conf = untrusted)
    confs = np.array([rows[k]["conf_med"] for k in keys if rows[k]["conf_med"] is not None])
    auB = float("nan")
    thrB = float(np.median(confs)) if len(confs) else float("nan")
    idxB = [i for i, k in enumerate(keys) if rows[k]["conf_med"] is not None]
    if idxB:
        labB = np.array([int(rows[keys[i]]["conf_med"] < thrB) for i in idxB])
        # gate flags OOD when dispersion HIGH; low conf = OOD. Align signs by
        # negating dispersion so "higher score = more OOD" is consistent.
        auB = auroc(np.array([-disp[i] for i in idxB]), labB)
    # threshold sweep on label A
    best = (float("nan"), float("nan"))
    if idxA:
        la = np.array([rows[keys[i]]["label_a"] for i in idxA])
        ths = np.linspace(disp[idxA].min(), disp[idxA].max(), 200)
        best = max(((float(((disp[idxA] > t).astype(int) == la).mean()), float(t))
                    for t in ths), key=lambda x: x[0])
    out = {"window": a.window, "tau_a": TAU_A, "per_seq": rows,
           "auroc_disp_vs_labelA_metric": auA,
           "auroc_disp_vs_labelB_confidence": auB,
           "labelB_conf_threshold": thrB,
           "best_thr_labelA": best[1], "best_acc_labelA": best[0]}
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"\nAUROC(dispersion vs metric-truth label A) = {auA:.3f}")
    print(f"AUROC(-dispersion vs FM-confidence label B) = {auB:.3f} "
          f"(median-split conf thr={thrB:.3f})")
    print(f"best thr(A)={best[1]:.4f} acc={best[0]:.2f} -> {a.out}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4.6, 3.2))
        for k in keys:
            d = rows[k]
            c = ("tab:red" if d.get("label_a") == 1 else
                 "tab:blue" if d.get("label_a") == 0 else "tab:gray")
            ax.scatter(d["r_mean"], d["log_std"], c=c)
            ax.annotate(k, (d["r_mean"], d["log_std"]), fontsize=6,
                        xytext=(3, 3), textcoords="offset points")
        if not np.isnan(best[1]):
            ax.axhline(best[1], ls="--", c="gray")
        ax.set_xlabel("mean r_init"); ax.set_ylabel("log-std r_init (gate score)")
        ax.set_title("FM trust gate (red = metric-truth OOD)")
        fig.tight_layout()
        png = (f"{P}/outputs/gate_ood_{a.prefix.rstrip('_')}.png" if a.prefix
               else f"{P}/outputs/gate_ood.png")
        fig.savefig(png, dpi=150)
        print(f"wrote {png}")
    except Exception as e:
        print(f"(no png: {e})")


if __name__ == "__main__":
    main()
