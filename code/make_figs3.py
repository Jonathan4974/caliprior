#!/usr/bin/env python3
"""Mechanism figure: the four results that were table/text-only.

Panels: (A) the gate is model-specific (per-model calibration);
(B) the gate is blind to a constant teacher bias; (C) scalar vs per-regime
sigma; (D) raising the prior weight never reaches the anchor. Reads the
recorded JSON/text outputs. CPU only.
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import caliprior_paths as _CP  # path resolution (env-overridable)
from caliprior_common import jload as jl
P = _CP.PROJECT_ROOT
FIG = f"{P}/outputs/figs"
os.makedirs(FIG, exist_ok=True)


def scale(seq, tag):
    f = f"{P}/outputs/{tag}{seq}/metrics.json"
    if not os.path.exists(f):
        return None
    return float(json.load(open(f))["umeyama_scale_est2gt"])


def panel_a(ax):
    C = jl(f"{P}/outputs/gate_model_calib.json")
    models = ["MapAnything", "DA-V2", "DA3METRIC"]
    if not C:
        ax.set_title("A. gate is model-specific (no data)"); return
    x = np.arange(len(models)); w = 0.38
    shared = [C[m]["auroc_shared_tau"] for m in models]
    best = [C[m]["auroc_best"] for m in models]
    ax.bar(x - w / 2, shared, w, label="shared threshold")
    ax.bar(x + w / 2, best, w, label="per-model $\\tau^*$")
    for i, m in enumerate(models):
        ax.text(x[i] + w / 2, best[i] + 0.02, f"$\\tau^*$={C[m]['tau_best']:.2f}\n"
                f"$\\pm${C[m]['loso_tau_std']:.2f}", ha="center", fontsize=6)
    ax.axhline(0.5, ls=":", c="gray")
    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=8)
    ax.set_ylim(0.4, 1.12); ax.set_ylabel("gate AUROC")
    ax.legend(fontsize=7, loc="lower right")
    ax.set_title("A. the gate is model-specific")


def panel_b(ax):
    G = jl(f"{P}/outputs/gate_ood.json").get("per_seq", {})
    K = jl(f"{P}/outputs/gate_ood_k120.json").get("per_seq", {})
    seqs = [s for s in ["freiburg1_xyz", "freiburg1_desk", "MH_01_easy"] if s in K]
    if not seqs:
        ax.set_title("B. gate blind to constant bias (no data)"); return
    x = np.arange(len(seqs)); w = 0.38
    un = [G[s]["log_std"] for s in seqs]
    pe = [K[s]["log_std"] for s in seqs]
    ax.bar(x - w / 2, un, w, label="unperturbed teacher")
    ax.bar(x + w / 2, pe, w, label="intrinsics $\\times$1.20 ($\\sim$5% bias)")
    ax.axhline(0.0625, ls="--", c="crimson", lw=1)
    ax.text(len(seqs) - 0.5, 0.075, "gate threshold", color="crimson",
            fontsize=6, ha="right")
    for i, s in enumerate(seqs):
        u, p = scale(s, "gaugeMA_"), scale(s, "e3_k120_gaugeMA_")
        if u and p:
            ax.text(x[i], max(un[i], pe[i]) + 0.008, f"{u:.3f}$\\to${p:.3f}",
                    ha="center", fontsize=6)
    ax.set_ylim(0, max(un + pe) * 1.32)
    ax.set_xticks(x); ax.set_xticklabels([s[:11] for s in seqs], fontsize=7)
    ax.set_ylabel("gate log-std $r_{init}$")
    ax.legend(fontsize=7)
    ax.set_title("B. gate is blind to a constant teacher bias\n(labels: adopted scale)")


def parse_regime_nis(path):
    rows = {}
    for line in open(path):
        if line.lstrip().startswith("|median"):   # second table starts here
            break
        t = line.split()
        if len(t) == 7 and t[0] in ("freiburg1_xyz", "freiburg1_desk",
                                    "MH_01_easy", "V1_02_medium"):
            rows[t[0]] = {k: (np.nan if v == "nan" else float(v))
                          for k, v in zip(
                              ["const", "frozen", "scalar", "K2", "K3", "K4"],
                              t[1:7])}
    return rows


def panel_c(ax):
    R = parse_regime_nis(f"{P}/outputs/regime_nis.txt")
    if not R:
        ax.set_title("C. scalar vs per-regime $\\sigma$ (no data)"); return
    seqs = list(R)
    arms = ["frozen", "scalar", "K3"]
    lab = ["conf, no adapt", "scalar", "per-regime K3"]
    w = 0.26
    x = np.arange(len(seqs))
    for i, (a, l) in enumerate(zip(arms, lab)):
        ax.bar(x + i * w - w, [R[s][a] for s in seqs], w, label=l)
    ax.axhline(1.0, ls="--", c="k", lw=1)
    ax.text(len(seqs) - 0.55, 1.06, "target NIS = 1", fontsize=6)
    top = max(R[s][a] for s in seqs for a in ("scalar", "K3", "K2")
              if not np.isnan(R[s][a]))
    for i, s in enumerate(seqs):
        if np.isnan(R[s]["K2"]):
            ax.text(x[i], top + 0.08, "K2/K4\nDNC", ha="center", fontsize=5.5,
                    color="crimson")
    ax.set_ylim(0, top + 0.28)
    ax.set_xticks(x); ax.set_xticklabels([s[:11] for s in seqs], fontsize=7)
    ax.set_ylabel("median held-out NIS")
    ax.legend(fontsize=7)
    ax.set_title("C. scalar calibrates; per-regime is fragile")


def panel_d(ax):
    seqs = ["freiburg1_xyz", "freiburg1_desk", "V1_02_medium"]
    alphas = [0.01, 1.0, 10.0]
    tags = {0.01: "_a0.01_", 1.0: "_", 10.0: "_a10.0_"}
    means = {}
    for arm in ("const", "conf"):
        ys = []
        for a in alphas:
            errs = []
            for s in seqs:
                v = scale(s, f"e1_metric{arm}{tags[a]}")
                if v is not None:
                    errs.append(abs(v - 1.0))
            ys.append(np.mean(errs) if errs else np.nan)
        means[arm] = ys
    anch = np.mean([abs(jl(f"{P}/outputs/traj_anchor.json")[s]["scale_traj_anchor"] - 1)
                    for s in seqs])
    ax.plot(alphas, means["const"], "o-", label="metric, scalar $\\alpha$")
    ax.plot(alphas, means["conf"], "s-", label="metric, confidence-weighted")
    ax.axhline(anch, ls="--", c="green", lw=1.2)
    ax.text(10.0, anch + 0.02, "trajectory anchor", color="green",
            fontsize=6.5, ha="right")
    ax.set_xscale("log"); ax.set_xticks(alphas)
    ax.set_xticklabels(["0.01", "1", "10"])
    ax.set_xlabel("prior weight $\\alpha$"); ax.set_ylabel("mean |scale$-$1|")
    ax.legend(fontsize=7)
    ax.set_title("D. raising the prior weight never reaches the anchor")


def panel_e(ax):
    A = jl(f"{P}/outputs/anchor_compare.json")
    if not A:
        ax.set_title("E. depth-mean vs trajectory anchor (no data)"); return
    arms = ["fm_err", "depth_anchor_err", "traj_anchor_err"]
    lab = ["teacher\n(FM-only)", "depth-mean\nanchor", "trajectory\nanchor"]
    col = ["tab:blue", "tab:orange", "tab:green"]
    vals = [np.mean([A[s][a] for s in A]) for a in arms]
    ax.bar(np.arange(3), vals, 0.6, color=col)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.012, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(np.arange(3)); ax.set_xticklabels(lab, fontsize=7)
    ax.set_ylabel("mean |scale$-$1|"); ax.set_ylim(0, max(vals) * 1.25)
    ax.set_title("E. the right anchor is trajectory scale, not depth mean")


def main():
    fig = plt.figure(figsize=(15.5, 7.6))
    a = fig.add_subplot(2, 3, 1)
    b = fig.add_subplot(2, 3, 2)
    c = fig.add_subplot(2, 3, 3)
    d = fig.add_subplot(2, 3, 4)
    e = fig.add_subplot(2, 3, (5, 6))
    panel_a(a)
    panel_b(b)
    panel_c(c)
    panel_d(d)
    panel_e(e)
    fig.tight_layout()
    out = f"{FIG}/caliprior_mechanisms.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.02)
    from PIL import Image; Image.open(out).convert("RGB").save(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
