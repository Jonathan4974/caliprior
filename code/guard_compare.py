#!/usr/bin/env python3
"""K-sensitivity: unguarded vs stability-guarded per-regime sigma.

Unguarded: outputs/e2reg_regimeK{2,3,4}_<seq>   (c_lo=0.05)
Guarded:   outputs/e2regG_regimeK{2,3,4}_<seq>  (c_lo=0.20, 5<s time cap)

Reports median held-out NIS (target 1), divergence fraction (solves with
NIS>10), and the final c_k, so we can see whether the guard restores
convergence and what it costs in calibration.
"""
import json, os, argparse
import numpy as np

import caliprior_paths as _CP  # path resolution (env-overridable)
OUT = f"{_CP.PROJECT_ROOT}/outputs"
SEQS = ["MH_01_easy", "V1_02_medium"]
KS = [2, 3, 4]


def load(prefix, k, seq):
    f = os.path.join(OUT, f"{prefix}regimeK{k}_{seq}", "metrics.json")
    return json.load(open(f)) if os.path.exists(f) else None


def stats(m):
    if m is None:
        return None
    tr = m.get("nis_trace") or []
    fa = m.get("freeze_at", 0) or 0
    nt = m.get("n_track_solves", 0) or 0
    v = np.array([x[1] for x in tr if fa > 0 and x[0] > fa
                  and (nt == 0 or x[0] <= nt)])
    if v.size == 0:
        return dict(med=None, div=None, n=0, c=m.get("c_regime_final"))
    return dict(med=float(np.median(v)), div=float(np.mean(v > 10.0)),
                n=int(v.size), c=m.get("c_regime_final"))


def main():
    ap = argparse.ArgumentParser()
    a = ap.parse_args()
    print(f"{'seq':14s} {'K':>2s} | {'unguarded med':>13s} {'div':>6s} "
          f"{'c_k':>22s} | {'guarded med':>11s} {'div':>6s} {'c_k':>22s}")
    for seq in SEQS:
        for k in KS:
            u = stats(load("e2reg_", k, seq))
            g = stats(load("e2regG_", k, seq))
            def cell(s):
                if s is None:
                    return f"{'DNC':>13s} {'-':>6s} {'-':>22s}"
                med = "none" if s["med"] is None else f"{s['med']:.3f}"
                div = "-" if s["div"] is None else f"{s['div']:.2f}"
                c = "-" if s["c"] is None else str([round(x, 2) for x in s["c"]])
                return f"{med:>13s} {div:>6s} {c:>22s}"
            print(f"{seq:14s} {k:2d} | {cell(u)} | {cell(g)}")
    print("\nDNC = did not converge / no metrics. Unguarded K2/K4 hung; the guard "
          "bounds c_k and each run is time-capped.")


if __name__ == "__main__":
    main()
