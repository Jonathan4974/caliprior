"""Shared constants and helpers for the CaliPrior scripts.

Sequence groups and the few small utilities (JSON load, AUROC, Umeyama ATE,
ground-truth loaders) that were duplicated across the analysis and run scripts.
"""
import json
import os

import numpy as np

import caliprior_paths as _CP

# --- sequence groups -------------------------------------------------------
TUM5 = ["freiburg1_xyz", "freiburg1_desk", "freiburg2_xyz", "freiburg2_desk",
        "freiburg3_long_office_household"]
EUROC5 = ["MH_01_easy", "MH_02_easy", "V1_01_easy", "V1_02_medium", "V2_01_easy"]
EUROC6_EXTRA = ["MH_03_medium", "MH_04_difficult", "MH_05_difficult",
                "V1_03_difficult", "V2_02_medium", "V2_03_difficult"]
EUROC11 = EUROC5 + EUROC6_EXTRA
ALL10 = TUM5 + EUROC5
ALL16 = TUM5 + EUROC11
CORE4 = ["freiburg1_xyz", "freiburg1_desk", "MH_01_easy", "V1_02_medium"]


# --- small helpers ---------------------------------------------------------
def jload(path, default=None):
    """Load JSON, returning `default` if the file does not exist."""
    if not os.path.exists(path):
        return default
    with open(path) as f:
        return json.load(f)


def auroc(scores, labels):
    """AUROC of `scores` against binary `labels` (ties count as 0.5)."""
    scores, labels = np.asarray(scores, float), np.asarray(labels, int)
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float((pos[:, None] > neg[None, :]).mean()
                 + 0.5 * (pos[:, None] == neg[None, :]).mean())


def umeyama_ate(p_est, p_gt):
    """Similarity-aligned ATE (m) and the estimated->GT scale factor."""
    mu_e, mu_g = p_est.mean(0), p_gt.mean(0)
    E, G = p_est - mu_e, p_gt - mu_g
    U, _, Vt = np.linalg.svd(E.T @ G)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    s = (G * (E @ R.T)).sum() / (E ** 2).sum()
    p_al = s * (p_est @ R.T) + (mu_g - s * (R @ mu_e))
    return float(np.sqrt(((p_al - p_gt) ** 2).sum(1).mean())), float(s)


# --- ground truth ----------------------------------------------------------
def load_euroc_gt(seq):
    gt = np.loadtxt(f"{_CP.EUROC_ROOT}/{seq}/mav0/state_groundtruth_estimate0/"
                    "data.csv", delimiter=",", skiprows=1)
    return gt[:, 0].astype(np.int64), gt[:, 1:4]


def load_tum_gt(seq):
    base = f"{_CP.TUM_ROOT}/rgbd_dataset_{seq}"
    rgb = [l.split() for l in open(f"{base}/rgb.txt") if not l.startswith("#")]
    g = [l.split() for l in open(f"{base}/groundtruth.txt")
         if not l.startswith("#")]
    gts = np.array([float(r[0]) for r in g])
    gtp = np.array([[float(x) for x in r[1:4]] for r in g])
    ts = np.array([float(r[0]) for r in rgb])
    gi = np.searchsorted(gts, ts).clip(0, len(gts) - 1)
    return ts, gtp[gi]


def load_kitti_gt(n=800):
    T = [np.eye(4) for _ in range(n)]
    with open(f"{_CP.KITTI_GREY_ROOT}/00/poses.txt") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            T[i][:3] = np.array(line.split(), float).reshape(3, 4)
    return np.array([t[:3, 3] for t in T])
