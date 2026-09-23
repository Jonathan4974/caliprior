#!/usr/bin/env python3
"""Tier-1/2 consolidated figure: anchor coverage, gate robustness, protocol ATE,
low-parallax certificate. Reads the recorded JSON outputs. CPU only."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import caliprior_paths as _CP  # path resolution (env-overridable)
from caliprior_common import jload as jl
P = _CP.PROJECT_ROOT
FIG = f"{P}/outputs/figs"
os.makedirs(FIG, exist_ok=True)


def main():
    TA, PA, GR, GP = (jl(f"{P}/outputs/{f}") for f in
                      ["traj_anchor.json", "protocol_ate_16.json",
                       "gate_robust_16.json", "gauge_parallax.json"])
    fig, ax = plt.subplots(2, 2, figsize=(11.5, 7.6))

    # A. per-seq FM vs trajectory-anchor scale error (all 16; failures hatched)
    seqs = list(TA)
    x = np.arange(len(seqs)); w = 0.4
    ax[0, 0].bar(x - w / 2, [TA[s]["fm_only_err"] for s in seqs], w, label="FM-only")
    bars = ax[0, 0].bar(x + w / 2, [TA[s]["scale_err"] for s in seqs], w,
                        label="traj anchor")
    ok = [bool(TA[s].get("success", True)) for s in seqs]
    for b, o in zip(bars, ok):
        if not o:
            b.set_hatch("///"); b.set_edgecolor("crimson")
    ax[0, 0].set_xticks(x); ax[0, 0].set_xticklabels([s[:9] for s in seqs],
                                                     fontsize=5, rotation=40)
    ax[0, 0].set_ylabel("scale error"); ax[0, 0].legend(fontsize=7)
    ax[0, 0].set_title(f"A. trajectory anchor ({sum(ok)}/{len(seqs)})")

    # B. protocol ATE
    rows = PA.get("per_seq", {})
    seqs2 = list(rows)
    x = np.arange(len(seqs2))
    ax[0, 1].bar(x - w / 2, [rows[s]["ate_fm"] for s in seqs2], w, label="FM-only")
    ax[0, 1].bar(x + w / 2, [rows[s]["ate_protocol"] for s in seqs2], w, label="gate+anchor")
    ax[0, 1].set_xticks(x); ax[0, 1].set_xticklabels([s[:9] for s in seqs2],
                                                     fontsize=5, rotation=40)
    ax[0, 1].set_ylabel("rigid ATE (m)"); ax[0, 1].legend(fontsize=7)
    ax[0, 1].set_title(f"B. trust-gated protocol ATE ({len(seqs2)} seq)")

    # C. gate robustness: window sweep + bootstrap CI
    ws = GR.get("window_sweep", {})
    if ws:
        kk = sorted(ws, key=int)
        ax[1, 0].plot([int(k) for k in kk], [ws[k]["auroc"] for k in kk], "o-")
        ax[1, 0].axhline(0.5, ls=":", c="gray")
        ax[1, 0].set_xscale("log"); ax[1, 0].set_ylim(0.4, 1.02)
        ci = GR.get("auroc_boot_ci95", [np.nan, np.nan])
        ax[1, 0].set_title(f"C. gate: AUROC {GR.get('auroc_window20', float('nan')):.2f} "
                           f"(boot CI [{ci[0]:.2f},{ci[1]:.2f}])")
        ax[1, 0].set_xlabel("init window (frames)"); ax[1, 0].set_ylabel("AUROC")

    # D. low-parallax certificate
    rws = GP.get("rows", [])
    if rws:
        b = [r["baseline"] for r in rws]; d = [r["depth_only_delta_px"] for r in rws]
        ax[1, 1].plot(b, d, "o-")
        ax[1, 1].set_xscale("symlog", linthresh=1e-4)
        ax[1, 1].set_xlabel("translation |t|"); ax[1, 1].set_ylabel("10% depth-scale delta (px)")
        ax[1, 1].set_title("D. depth-scale observability vs parallax")

    fig.tight_layout()
    out = f"{FIG}/caliprior_tier12.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.02)
    from PIL import Image; Image.open(out).convert("RGB").save(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
