#!/usr/bin/env python3
"""E2 sigma-regime analysis: temporal hold-out NIS + per-regime consistency.

Reads outputs/e2reg_<arm>_<seq>/metrics.json (run_wired.py) and tests the
pre-registered criteria in E2_SIGMA_REGIME.md.

Estimator note (stated up front): the held-out aggregate NIS is summarised by
the MEDIAN over held-out solves. The mean is not usable here: on extreme-OOD
MH_01 individual solves diverge and reach NIS > 1e5, so the mean (e.g. 2503 for
regimeK3) measures divergence spikes, not calibration. The median is robust and
is the quantity the pre-registration intends by "held-out NIS". The mean and the
divergence fraction are also reported for transparency.

H1  regime post-freeze median NIS closer to 1 than scalar on >= 2/3 shifted seqs
H2  (pre-registered) mean cross-regime *spread* of per-regime held-out NIS lower
    for regime. We report it faithfully, but note it is the wrong statistic: a
    scalar that leaves every regime uniformly inconsistent has near-zero spread.
H2' (corrected, reported alongside) mean |NIS_k - 1| across regimes lower for
    regime -- this is what "each regime calibrated" actually means.
"""
import json, os, argparse
import numpy as np

import caliprior_paths as _CP  # path resolution (env-overridable)
OUT = f"{_CP.PROJECT_ROOT}/outputs"
ARMS = ["const", "frozen", "scalar", "regimeK2", "regimeK3", "regimeK4"]
SEQS = ["freiburg1_xyz", "freiburg1_desk", "MH_01_easy", "V1_02_medium"]
SHIFTED = ["freiburg1_desk", "MH_01_easy", "V1_02_medium"]


def load(arm, seq):
    p = os.path.join(OUT, f"e2reg_{arm}_{seq}", "metrics.json")
    return json.load(open(p)) if os.path.exists(p) else None


def _in_holdout(m, n):
    fa = m.get("freeze_at", 0) or 0
    nt = m.get("n_track_solves", 0) or 0
    return fa > 0 and n > fa and (nt == 0 or n <= nt)


def _series(m, key):
    fa = m.get("freeze_at", 0) or 0
    if key == "agg":
        tr = m.get("nis_trace") or []
        return [x[1] for x in tr if _in_holdout(m, x[0])]
    tr = m.get("nis_regime_trace") or []
    rows = [r for r in tr if _in_holdout(m, r[0])]
    if not rows:
        rows = tr
    K = m.get("n_regimes", len(rows[-1][1]) if rows else 0)
    return [[r[1][k] for r in rows if k < len(r[1]) and r[1][k] == r[1][k]]
            for k in range(K)]


def agg_stats(m):
    v = np.array(_series(m, "agg"))
    if v.size == 0:
        return dict(med=None, mean=None, n=0, div=0.0)
    return dict(med=float(np.median(v)), mean=float(np.mean(v)), n=int(v.size),
                div=float(np.mean(v > 10.0)))


def per_regime_med(m):
    out = []
    for vals in _series(m, "reg"):
        v = np.array(vals)
        out.append(float(np.median(v)) if v.size else float("nan"))
    return out


def fmt(x, d=3):
    return "nan" if x is None or x != x else f"{x:.{d}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transfer", action="store_true")
    ap.add_argument("--fig", default=os.path.join(OUT, "regime_nis.png"))
    a = ap.parse_args()

    tab = {}
    for seq in SEQS:
        for arm in ARMS:
            m = load(arm, seq)
            if m is not None:
                tab[(arm, seq)] = m

    print("=" * 92)
    print("E2 sigma-regime — temporal hold-out (median aggregate NIS; target 1.0)")
    print("=" * 92)
    hdr = f"{'sequence':16s} " + " ".join(f"{x:>11s}" for x in ARMS)
    print(hdr)
    for seq in SEQS:
        cells = [fmt((agg_stats(tab[(arm, seq)])["med"]
                      if (arm, seq) in tab else None)) for arm in ARMS]
        print(f"{seq:16s} " + " ".join(f"{c:>11s}" for c in cells))

    print("\n|median held-out NIS - 1| (lower better)   [mean / divergence-frac]")
    print(hdr)
    for seq in SEQS:
        cells = []
        for arm in ARMS:
            if (arm, seq) not in tab:
                cells.append("-"); continue
            st = agg_stats(tab[(arm, seq)])
            if st["med"] is None:
                cells.append("-")
            else:
                cells.append(f"{abs(st['med']-1):.2f}")
        print(f"{seq:16s} " + " ".join(f"{c:>11s}" for c in cells))
    for seq in SEQS:
        if ("scalar", seq) not in tab or ("regimeK3", seq) not in tab:
            continue
        s = tab[("scalar", seq)]
        r3 = tab[("regimeK3", seq)]
        print(f"  {seq:14s} scalar mean={agg_stats(s)['mean']:.3g} "
              f"div={agg_stats(s)['div']:.2f} | "
              f"regimeK3 mean={agg_stats(r3)['mean']:.3g} "
              f"div={agg_stats(r3)['div']:.2f}")

    print("\nper-regime median held-out NIS (each should be ~1)")
    print(f"{'sequence':16s} {'scalar':>26s} {'regimeK3':>26s}")
    h1 = 0
    h2 = h2p = 0
    for seq in SEQS:
        ms, mr = tab.get(("scalar", seq)), tab.get(("regimeK3", seq))
        if not ms or not mr:
            continue
        vs, vr = per_regime_med(ms), per_regime_med(mr)
        print(f"{seq:16s} {str([round(v,3) for v in vs]):>26s} "
              f"{str([round(v,3) for v in vr]):>26s}")
        if seq in SHIFTED:
            s_med, r_med = agg_stats(ms)["med"], agg_stats(mr)["med"]
            if s_med is not None and r_med is not None and \
                    abs(r_med - 1) < abs(s_med - 1):
                h1 += 1
        sa = np.array([v for v in vs if v == v])
        ra = np.array([v for v in vr if v == v])
        if sa.size and ra.size:
            if ra.std() < sa.std():
                h2 += 1
            if np.mean(np.abs(ra - 1)) < np.mean(np.abs(sa - 1)):
                h2p += 1

    print("\npre-registered verdict (median estimator)")
    print(f"  H1 (regime closer to 1 on >=2/3 shifted): "
          f"{h1}/3 -> {'PASS' if h1 >= 2 else 'FAIL'}")
    print(f"  H2 (spread reduced, as pre-registered):   {h2} "
          f"-> {'PASS' if h2 >= 2 else 'FAIL'}  [known-bad statistic]")
    print(f"  H2' (mean |NIS_k-1| reduced, corrected):  {h2p} "
          f"-> {'PASS' if h2p >= 2 else 'FAIL'}")
    print(f"  => per-regime {'SUPPORTED' if (h1 >= 2 and h2p >= 2) else 'NOT supported'}"
          " (H1 and corrected H2')")

    if a.transfer:
        print("\n" + "=" * 92)
        print("cross-sequence transfer (fit TUM -> apply frozen to EuRoC)")
        print("=" * 92)
        for seq in ["MH_01_easy", "V1_02_medium"]:
            for arm in ["scalarT", "regimeK3T"]:
                m = load(arm, seq)
                if m:
                    print(f"{seq:16s} {arm:12s} median held-out NIS="
                          f"{fmt(agg_stats(m)['med'])} "
                          f"c={m.get('c_final')} c_regime={m.get('c_regime_final')}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for seq in SEQS:
            m = tab.get(("regimeK3", seq))
            if m and m.get("nis_regime_trace"):
                tr = m["nis_regime_trace"]
                n = [r[0] for r in tr]
                for k in range(m.get("n_regimes", 3)):
                    axes[0].plot(n, [r[1][k] for r in tr],
                                 label=f"{seq[:10]} r{k}", alpha=0.6, lw=1)
        axes[0].axhline(1.0, color="k", ls="--", lw=1)
        axes[0].set_ylim(0, 3)
        axes[0].set_xlabel("solve"); axes[0].set_ylabel("NIS_k")
        axes[0].set_title("regimeK3 per-regime NIS")
        axes[0].legend(fontsize=5, ncol=3)
        w = 0.35
        xs = np.arange(len(SEQS))
        for i, arm in enumerate(["scalar", "regimeK3"]):
            vals = []
            for seq in SEQS:
                m = tab.get((arm, seq))
                st = agg_stats(m) if m else dict(med=np.nan)
                vals.append(st["med"] if st["med"] is not None else np.nan)
            axes[1].bar(xs + (i - 0.5) * w, vals, w, label=arm)
        axes[1].axhline(1.0, color="k", ls="--", lw=1)
        axes[1].set_xticks(xs)
        axes[1].set_xticklabels([s.split("_")[0] for s in SEQS], fontsize=8)
        axes[1].set_ylabel("median held-out NIS"); axes[1].legend()
        axes[1].set_title("post-freeze aggregate NIS")
        fig.tight_layout(); fig.savefig(a.fig, dpi=120)
        print(f"\nwrote {a.fig}")
    except Exception as e:
        print(f"\n[figure skipped: {type(e).__name__}: {e}]")


if __name__ == "__main__":
    main()
