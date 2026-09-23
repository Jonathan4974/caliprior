#!/usr/bin/env python3
"""E3(iii)/(iv): positive-class sensitivity and a WHY-MA diagnostic.

(iii) The trust gate's positive class is "teacher metric scale is wrong",
      |scale-1| > tau. Here we sweep tau for all three teachers and report the
      dispersion AUROC (and the positive count) so the paper's tau=0.30 is not
      the whole story.

(iv)  Hypothesis for why MapAnything's dispersion carries signal while
      DA-V2/DA3METRIC's does not: a multi-view teacher is temporally
      consistent, so the dispersion of r_init = D_teacher/D_backend reflects
      the DOMAIN, whereas a single-image teacher injects per-frame scale noise
      that swamps the domain signal. GT-free proxy: the teacher's own
      frame-to-frame median-depth stability over the init window. We report it
      next to the gate AUROC; this is a diagnostic, not a proof.
"""
import json
import os
import numpy as np

import caliprior_paths as _CP  # path resolution (env-overridable)
from caliprior_common import auroc
OUT = f"{_CP.PROJECT_ROOT}/outputs"
TEACHERS = [
    ("magic",  "gate_ood.json",       "ma_cache",   ""),
    ("dav2",   "gate_ood_dav2.json",  "dav2_cache", "dav2_"),
    ("da3m",   "gate_ood_da3m.json",  "da3m_cache", "da3m_"),
]


def tau_sweep(path):
    d = json.load(open(path))["per_seq"]
    rows = [(k, v["log_std"], v["scale_ma"]) for k, v in d.items()
            if v.get("scale_ma") is not None]
    taus = [0.1, 0.2, 0.3, 0.4, 0.5]
    out = []
    for t in taus:
        lab = [int(abs(s - 1.0) > t) for _, _, s in rows]
        au = auroc([x[1] for x in rows], lab)
        out.append((t, sum(lab), au))
    return len(rows), out


def teacher_self_stability(seq, cache_dir):
    """GT-free: std of per-frame log(median teacher depth / window median)."""
    f = os.path.join(OUT, cache_dir, f"{seq}.npz")
    if not os.path.exists(f):
        return None
    depth = np.load(f)["depth"].astype(np.float32)
    n = min(20, len(depth))
    meds = []
    for k in range(n):
        d = depth[k]
        v = d[(d > 0.1) & (d < 80)]
        if v.size > 50:
            meds.append(np.median(v))
    if len(meds) < 5:
        return None
    meds = np.array(meds)
    return float(np.std(np.log(meds / np.median(meds))))


def main():
    print("== E3(iii): tau sensitivity (positive class |scale-1| > tau) ==")
    for name, gatef, _, _ in TEACHERS:
        g = os.path.join(OUT, gatef)
        if not os.path.exists(g):
            print(f"{name}: missing {gatef}")
            continue
        n, out = tau_sweep(g)
        cells = "  ".join(f"tau={t:.1f}: n={c},AUROC={a:.3f}" for t, c, a in out)
        print(f"{name:6s} (N={n})  {cells}")

    print("\n== E3(iv): teacher temporal self-stability vs gate AUROC ==")
    small = ["freiburg1_xyz", "freiburg1_desk"]
    for name, gatef, cache_dir, _ in TEACHERS:
        g = os.path.join(OUT, gatef)
        if not os.path.exists(g):
            continue
        au = json.load(open(g)).get("auroc_disp_vs_labelA_metric", float("nan"))
        st = [teacher_self_stability(s, cache_dir) for s in small]
        st = [x for x in st if x is not None]
        if not st:
            continue
        print(f"{name:6s} gate AUROC={au:.3f}  teacher init-window "
              f"log-med-depth std over {small}: "
              + ", ".join(f"{s}={x:.4f}" for s, x in zip(small, st)))


if __name__ == "__main__":
    main()
