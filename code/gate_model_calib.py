#!/usr/bin/env python3
"""Per-model gate calibration (C2 generality).

The gate is `dispersion(r_init) > thr`, with the positive class defined by the
teacher's scale error |scale_ma - 1| > tau. The paper says "model-specific, must
be validated per model". Here we ask the sharper question: for each teacher, is
the decision rule a *stable per-model constant*, or is the dispersion ranking
itself weak (so no calibration rescues it)?

Uses existing outputs only: gate_ood{,_dav2,_da3m}.json, common 10-sequence set.
Reports per model: AUROC at the shared tau=0.30; best AUROC over a tau grid;
argmax tau*; bootstrap 95% CI of the best AUROC; and a leave-one-sequence-out
estimate of tau* (fit on 9, report the held-out fold's tau*).
"""
import json, os, argparse
import numpy as np

import caliprior_paths as _CP  # path resolution (env-overridable)
from caliprior_common import auroc
OUT = f"{_CP.PROJECT_ROOT}/outputs"
MODELS = [("MapAnything", "gate_ood.json"),
          ("DA-V2", "gate_ood_dav2.json"),
          ("DA3METRIC", "gate_ood_da3m.json")]
SHARED_TAU = 0.30
TAUS = np.round(np.arange(0.05, 0.71, 0.025), 4)


def load(name):
    d = json.load(open(os.path.join(OUT, name)))["per_seq"]
    rows = [(k, v["log_std"], v["scale_ma"]) for k, v in d.items()
            if v.get("scale_ma") is not None]
    return rows  # (seq, dispersion, scale)


def auroc_at(rows, tau):
    lab = [int(abs(s - 1.0) > tau) for _, _, s in rows]
    return auroc([r[1] for r in rows], lab), sum(lab), len(rows)


def boot_ci(rows, tau, n=4000, seed=0):
    lab = np.array([int(abs(s - 1.0) > tau) for _, _, s in rows])
    disp = np.array([r[1] for r in rows])
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(rows), len(rows))
        if 0 < lab[i].sum() < len(lab):
            vals.append(auroc(disp[i], lab[i]))
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def loo_tau(rows):
    best = []
    for i in range(len(rows)):
        tr = rows[:i] + rows[i + 1:]
        aus = [(auroc_at(tr, t)[0], t) for t in TAUS]
        aus = [(a, t) for a, t in aus if a == a]
        if not aus:
            continue
        best.append(max(aus)[1])
    return np.array(best)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(OUT, "gate_model_calib.json"))
    a = ap.parse_args()
    res = {}
    print(f"{'model':12s} {'N':>3s} {'AUROC@0.30':>11s} {'bestAUROC':>10s} "
          f"{'tau*':>6s} {'[95% CI]':>16s} {'LOSO tau*':>16s}")
    for name, fn in MODELS:
        p = os.path.join(OUT, fn)
        if not os.path.exists(p):
            print(f"{name:12s} missing {fn}")
            continue
        rows = load(fn)
        au_shared, npos_s, n = auroc_at(rows, SHARED_TAU)
        sweep = [(auroc_at(rows, t)[0], t) for t in TAUS]
        sweep = [(au, t) for au, t in sweep if au == au]
        au_best, tau_best = max(sweep)
        lo, hi = boot_ci(rows, tau_best)
        lt = loo_tau(rows)
        res[name] = dict(n=n, npos_at_shared=npos_s,
                         auroc_shared_tau=au_shared, auroc_best=au_best,
                         tau_best=tau_best, ci95=[lo, hi],
                         loso_tau_mean=float(lt.mean()),
                         loso_tau_std=float(lt.std()),
                         loso_tau_min=float(lt.min()),
                         loso_tau_max=float(lt.max()))
        print(f"{name:12s} {n:3d} {au_shared:11.3f} {au_best:10.3f} "
              f"{tau_best:6.3f} [{lo:.3f},{hi:.3f}] "
              f"{lt.mean():.3f}+-{lt.std():.3f} [{lt.min():.2f},{lt.max():.2f}]")
    json.dump(res, open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")
    print("reading: a stable per-model constant shows up as a small LOSO tau* "
          "std; a weak ranking shows up as low AUROC at every tau.")


if __name__ == "__main__":
    main()
