#!/usr/bin/env python3
"""N4: consolidated CaliPrior figures from the recorded JSON outputs.

Panels: (A) trust gate, MapAnything; (B) trust gate, DA-V2 (teacher-specific);
(C) 5-arm ladder scale error; (D) trajectory anchor vs FM-only. Writes
outputs/figs/caliprior_summary.png (+ individual panels).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import caliprior_paths as _CP  # path resolution (env-overridable)
from caliprior_common import jload
P = _CP.PROJECT_ROOT
FIG = f"{P}/outputs/figs"
os.makedirs(FIG, exist_ok=True)


def gate_panel(ax, path, title):
    G = jload(path).get("per_seq", {})
    if not G:
        ax.set_title(title + " (no data)"); return
    for k, d in G.items():
        lab = d.get("label_a")
        c = "tab:red" if lab == 1 else "tab:blue" if lab == 0 else "tab:gray"
        ax.scatter(d["r_mean"], d["log_std"], c=c)
        ax.annotate(k[:10], (d["r_mean"], d["log_std"]), fontsize=5,
                    xytext=(3, 3), textcoords="offset points")
    thr = jload(path).get("best_thr_labelA")
    if thr:
        ax.axhline(thr, ls="--", c="gray", lw=1)
    ax.set_xlabel("mean r_init"); ax.set_ylabel("log-std r_init")
    ax.set_title(title)


def main():
    L = jload(f"{P}/outputs/gauge_ladder.json")
    TA = jload(f"{P}/outputs/traj_anchor.json")
    fig, ax = plt.subplots(2, 2, figsize=(11, 7.5))

    gate_panel(ax[0, 0], f"{P}/outputs/gate_ood_16.json",
               "A. Trust gate (MapAnything, 16 seq)  AUROC 0.97")
    gate_panel(ax[0, 1], f"{P}/outputs/gate_ood_dav2.json",
               "B. Trust gate (DA-V2, shared tau)  AUROC 0.62")

    # C. ladder
    rows = L.get("rows", [])
    if rows:
        seqs = [r["seq"][:10] for r in rows]
        arms = ["down", "in-domain", "frozen", "online", "oracle"]
        w = 0.16
        x = np.arange(len(seqs))
        for i, arm in enumerate(arms):
            y = [r["scale_err"].get(arm) or np.nan for r in rows]
            ax[1, 0].bar(x + i * w - 2 * w, y, w, label=arm)
        ax[1, 0].set_xticks(x); ax[1, 0].set_xticklabels(seqs, fontsize=6, rotation=30)
        ax[1, 0].set_ylabel("scale error"); ax[1, 0].legend(fontsize=6, ncol=5)
        ax[1, 0].set_title("C. 5-arm ladder (online KILL)")

    # D. trajectory anchor vs FM-only (all 16; failures hatched)
    if TA:
        seqs = list(TA)
        fm = [TA[s]["fm_only_err"] for s in seqs]
        ta = [TA[s]["scale_err"] for s in seqs]
        ok = [bool(TA[s].get("success", True)) for s in seqs]
        x = np.arange(len(seqs)); w = 0.38
        ax[1, 1].bar(x - w / 2, fm, w, label="FM-only")
        bars = ax[1, 1].bar(x + w / 2, ta, w, label="traj anchor")
        for b, o in zip(bars, ok):
            if not o:
                b.set_hatch("///"); b.set_edgecolor("crimson")
        ax[1, 1].set_xticks(x)
        ax[1, 1].set_xticklabels([s[:10] for s in seqs], fontsize=5.5, rotation=40)
        ax[1, 1].set_ylabel("scale error"); ax[1, 1].legend(fontsize=7)
        ax[1, 1].set_title(f"D. Trajectory anchor ({sum(ok)}/{len(seqs)})")

    fig.tight_layout()
    out = f"{FIG}/caliprior_summary.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.02)
    from PIL import Image; Image.open(out).convert("RGB").save(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
