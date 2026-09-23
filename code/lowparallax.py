#!/usr/bin/env python3
"""T2.6: low-parallax probe for the trajectory anchor.

The trajectory anchor needs a metric *translation*. As the baseline shrinks the
PnP translation is poorly conditioned, so the scale estimate should degrade.
This is the observable face of the classical result that a depth-scale state is
exactly unobservable at zero translation (gauge_null.py).

For each sequence: take many pairs across the whole trajectory, split them by
backend baseline into low/mid/high terciles, and report the anchor scale and its
bootstrap CI per tercile, plus the per-pair correlation between r and baseline.
Emits outputs/lowparallax.json (+ png). CPU only (rs3d venv).
"""
import os, json
import numpy as np
import build_traj_anchor as B

import caliprior_paths as _CP  # path resolution (env-overridable)
from caliprior_common import TUM5 as TUM, EUROC5 as EUROC
P = _CP.PROJECT_ROOT


def corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main():
    out = {}
    for seq in TUM + EUROC:
        run = f"{P}/outputs/gaugeMA_{seq}/kf_state.npz"
        mf = f"{P}/outputs/gaugeMA_{seq}/metrics.json"
        if not (os.path.exists(run) and os.path.exists(mf)):
            continue
        s1 = float(json.load(open(mf))["umeyama_scale_est2gt"])
        n_kf = int(np.load(run)["kf_idx"].shape[0])
        try:
            rows = B.pairs_for(seq, run, B.auto_pairs(n_kf))
        except Exception as e:
            print(f"skip {seq}: {type(e).__name__}: {e}")
            continue
        if len(rows) < 9:
            print(f"skip {seq}: only {len(rows)} pairs")
            continue
        r = np.array([x[4] for x in rows], float)
        te = np.array([x[3] for x in rows], float)
        q1, q2 = np.quantile(te, [1 / 3, 2 / 3])
        bins = {"low": te <= q1, "mid": (te > q1) & (te <= q2), "high": te > q2}
        rec = {"s1": s1, "n_pairs": len(rows), "corr_r_baseline": corr(r, te),
               "baseline_q": [float(q1), float(q2)], "terciles": {}}
        for name, m in bins.items():
            if m.sum() < 3:
                continue
            rr = r[m]
            med = float(np.median(rr))
            lo, hi, _ = B.bootstrap_ci(rr, s1)
            rec["terciles"][name] = {
                "n": int(m.sum()), "baseline_mean": float(te[m].mean()),
                "r_median": med, "r_rel_scatter": float(np.std(rr) / med),
                "scale": s1 / med, "scale_ci95": [lo, hi],
                "scale_ci_halfwidth": 0.5 * (hi - lo)}
        out[seq] = rec
        t = rec["terciles"]
        print(f"{seq:32s} corr(r,base)={rec['corr_r_baseline']:+.2f}  " +
              "  ".join(f"{k}: s={t[k]['scale']:.3f}±{t[k]['scale_ci_halfwidth']:.3f}"
                        f"(n={t[k]['n']})" if k in t else f"{k}:-"
                        for k in ("low", "mid", "high")))

    # aggregate low vs high CI width over sequences that have both
    lows, highs = [], []
    for rec in out.values():
        t = rec["terciles"]
        if "low" in t:
            lows.append(t["low"]["scale_ci_halfwidth"])
        if "high" in t:
            highs.append(t["high"]["scale_ci_halfwidth"])
    agg = {"mean_ci_low": float(np.mean(lows)) if lows else None,
           "mean_ci_high": float(np.mean(highs)) if highs else None,
           "n_seq": len(out)}
    json.dump({"per_seq": out, "aggregate": agg},
              open(f"{P}/outputs/lowparallax.json", "w"), indent=2)
    print(f"\naggregate CI half-width: low-baseline {agg['mean_ci_low']:.3f} "
          f"vs high-baseline {agg['mean_ci_high']:.3f} (n={agg['n_seq']})")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
        for seq, rec in out.items():
            ax[0].scatter(np.array([rec["terciles"][k]["baseline_mean"]
                                    for k in rec["terciles"]]),
                          np.array([rec["terciles"][k]["scale_ci_halfwidth"]
                                    for k in rec["terciles"]]),
                          label=seq[:10])
        ax[0].set_xlabel("mean backend baseline (a.u.)")
        ax[0].set_ylabel("scale CI half-width")
        ax[0].set_title("(a) scale uncertainty vs parallax")
        ax[0].legend(fontsize=6)
        names = ["low", "mid", "high"]
        ax[1].bar(np.arange(3) - 0.2, [np.mean([rec["terciles"][k]["scale"]
                 for rec in out.values() if k in rec["terciles"]]) for k in names],
                  0.4, label="scale")
        ax[1].axhline(1.0, ls=":", c="gray")
        ax[1].set_xticks(np.arange(3)); ax[1].set_xticklabels(names)
        ax[1].set_title("(b) mean anchor scale by baseline tercile")
        fig.tight_layout()
        fig.savefig(f"{P}/outputs/lowparallax.png", dpi=150)
        print(f"wrote {P}/outputs/lowparallax.png")
    except Exception as e:
        print(f"(no png: {e})")
    print(f"wrote {P}/outputs/lowparallax.json")


if __name__ == "__main__":
    main()
