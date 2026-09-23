#!/usr/bin/env python3
"""T1.3: robustness of the dispersion trust gate.

Three attacks a reviewer will make on AUROC 0.92 with n=10:
  (a) small-sample noise  -> sequence-level bootstrap CI + leave-one-out range;
  (b) window choice       -> AUROC vs the init-window length;
  (c) length confound     -> the bootstrap frame count n varies 11..106 across
      sequences, and dispersion could be tracking n rather than shift. Control
      by using a FIXED frame budget for every sequence and by reporting
      Spearman(log_std, n).

Emits outputs/gate_robust.json (+ png). CPU only.
"""
import json, argparse
import numpy as np
import gate_ood as G

import caliprior_paths as _CP  # path resolution (env-overridable)
P = _CP.PROJECT_ROOT


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


def collect(window, cache_dir, prefix, seqs):
    rows = {}
    for seq in seqs:
        rs = G.r_series(seq, window, cache_dir, prefix)
        if rs is None:
            continue
        r, w = rs
        w = min(w, len(r))
        rows[seq] = {"n": len(r), "log_std": float(np.std(np.log(r[:w]))),
                     "label_a": int(abs(G.scale_ma(seq, prefix) - 1.0) > G.TAU_A)}
    return rows


def auroc_from(rows, keys=None):
    keys = keys if keys is not None else list(rows)
    s = np.array([rows[k]["log_std"] for k in keys])
    l = np.array([rows[k]["label_a"] for k in keys])
    if l.sum() == 0 or (1 - l).sum() == 0:
        return float("nan")
    return G.auroc(s, l)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=f"{P}/outputs/ma_cache")
    ap.add_argument("--prefix", default="")
    ap.add_argument("--seqs", default=None,
                    help="comma list; default = gate_ood.ALL (16). Use the "
                         "original ten for the 10-seq artifact.")
    ap.add_argument("--out", default=f"{P}/outputs/gate_robust.json")
    a = ap.parse_args()

    seqs = a.seqs.split(",") if a.seqs else list(G.ALL)
    base = collect(20, a.cache_dir, a.prefix, seqs)
    keys = list(base)
    au = auroc_from(base)

    # (a) bootstrap over sequences + leave-one-out
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(5000):
        idx = rng.integers(0, len(keys), len(keys))
        kk = [keys[i] for i in idx]
        v = auroc_from(base, kk)
        if not np.isnan(v):
            boots.append(v)
    boots = np.array(boots)
    loo = [auroc_from(base, [k for k in keys if k != kx]) for kx in keys]

    # (b) window sweep
    windows = [5, 10, 15, 20, 30, 40, 60, 100]
    sweep = {}
    for w in windows:
        rows = collect(w, a.cache_dir, a.prefix, seqs)
        sweep[str(w)] = {"auroc": auroc_from(rows),
                         "mean_logstd": float(np.mean([r["log_std"]
                                                       for r in rows.values()]))}

    # (c) fixed frame budget (min n across sequences) + length correlation
    nmin = min(r["n"] for r in base.values())
    fixed = {}
    for seq in keys:
        rs = G.r_series(seq, 10 ** 9, a.cache_dir, a.prefix)
        r = rs[0][:nmin]
        fixed[seq] = {"log_std": float(np.std(np.log(r))),
                      "label_a": base[seq]["label_a"]}
    au_fixed = auroc_from(fixed)
    rho = spearman([base[k]["log_std"] for k in keys],
                   [base[k]["n"] for k in keys])

    out = {"n_seq": len(keys), "auroc_window20": au,
           "auroc_boot_mean": float(boots.mean()),
           "auroc_boot_ci95": [float(np.percentile(boots, 2.5)),
                               float(np.percentile(boots, 97.5))],
           "auroc_loo_min": float(np.nanmin(loo)),
           "auroc_loo_max": float(np.nanmax(loo)),
           "loo": {k: v for k, v in zip(keys, loo)},
           "window_sweep": sweep,
           "fixed_budget_n": int(nmin), "auroc_fixed_budget": au_fixed,
           "spearman_logstd_vs_n": rho,
           "per_seq": base}
    json.dump(out, open(a.out, "w"), indent=2)

    print(f"sequences: {len(keys)}")
    print(f"AUROC (window 20)      = {au:.3f}")
    print(f"bootstrap 95% CI       = [{out['auroc_boot_ci95'][0]:.3f}, "
          f"{out['auroc_boot_ci95'][1]:.3f}]  (mean {boots.mean():.3f})")
    print(f"leave-one-out AUROC    = [{out['auroc_loo_min']:.3f}, "
          f"{out['auroc_loo_max']:.3f}]")
    print(f"fixed-budget (n={nmin})  = {au_fixed:.3f}")
    print(f"Spearman(log_std, n)   = {rho:.3f}")
    print("window sweep:")
    for w, d in sweep.items():
        print(f"  w={w:>4s}  AUROC={d['auroc']:.3f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
        ws = [int(w) for w in sweep]
        ax[0].plot(ws, [sweep[str(w)]["auroc"] for w in ws], "o-")
        ax[0].axhline(0.5, ls=":", c="gray")
        ax[0].set_xscale("log"); ax[0].set_ylim(0.4, 1.02)
        ax[0].set_xlabel("init window (frames)"); ax[0].set_ylabel("AUROC")
        ax[0].set_title("(b) window sensitivity")
        ax[1].hist(boots, bins=30, color="tab:blue", alpha=0.7)
        ax[1].axvline(au, c="k", lw=2)
        ax[1].axvline(out["auroc_boot_ci95"][0], c="k", ls="--")
        ax[1].axvline(out["auroc_boot_ci95"][1], c="k", ls="--")
        ax[1].set_xlabel("bootstrap AUROC"); ax[1].set_title("(a) bootstrap CI")
        fig.tight_layout()
        png = f"{P}/outputs/gate_robust{png_suffix(a.prefix)}.png"
        fig.savefig(png, dpi=150)
        print(f"wrote {png}")
    except Exception as e:
        print(f"(no png: {e})")
    print(f"wrote {a.out}")


def png_suffix(prefix):
    return f"_{prefix.rstrip('_')}" if prefix else ""


if __name__ == "__main__":
    main()
