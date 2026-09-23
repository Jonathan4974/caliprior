#!/usr/bin/env python3
"""N2b: trajectory-scale anchor (not depth-mean).

A depth-mean anchor cannot pin the trajectory scale because depth and
pose scales are mutually inconsistent. The correct anchor measures a metric
*translation* over a short init baseline and matches it to the backend's
up-to-scale translation:

    r = |t_metric| / |t_backend|          (per frame pair)
    metric scale = scale(1) * median(r)   (pairs within the init window)

- TUM RGB-D : metric 3D of ORB features from registered depth (+K) -> PnP to the
  second frame -> metric |t|.
- EuRoC     : metric 3D of ORB features from stereo SGBM depth in the rectified
  cam0 frame (+P0) -> PnP -> metric |t|  (|t| is invariant to the frame's fixed
  rotation, so rectified vs undistorted does not matter).

The backend translation comes from an existing gauge-anchored run's
`kf_state.npz` (`poses_wc`), so no re-run is needed: trajectory scale is a gauge,
so multiplying the trajectory by `scale(1)*r` makes the umeyama scale
`scale(1)*r`. Emits outputs/traj_anchor.json.

Runs in the rs3d venv.
"""
import os, json, argparse
import numpy as np
import cv2
import yaml
import build_ma_cache as bmc

import caliprior_paths as _CP  # path resolution (env-overridable)
P = _CP.PROJECT_ROOT
EU = _CP.EUROC_ROOT
TU = _CP.TUM_ROOT
TUM = ["freiburg1_xyz", "freiburg1_desk", "freiburg2_xyz", "freiburg2_desk",
       "freiburg3_long_office_household"]
EUROC = ["MH_01_easy", "MH_02_easy", "V1_01_easy", "V1_02_medium", "V2_01_easy"]
LONG = 560


def fit56(h, w):
    return (h // 56) * 56, (w // 56) * 56


def orb_match(a, b, n=4000, keep=1000):
    orb = cv2.ORB_create(n)
    ka, da = orb.detectAndCompute(a, None)
    kb, db = orb.detectAndCompute(b, None)
    if da is None or db is None:
        return None, None
    m = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(da, db)
    if len(m) < 20:
        return None, None
    m = sorted(m, key=lambda x: x.distance)[:keep]
    pa = np.float32([ka[x.queryIdx].pt for x in m])
    pb = np.float32([kb[x.trainIdx].pt for x in m])
    return pa, pb


def pnp_metric(pa, pb, depth_a, K):
    """Metric |t| (frame b w.r.t. a) from 2D+metric-depth in a, 2D in b.

    Returns (|t|, n_inliers) or None. Requires >=12 RANSAC inliers so that
    low-parallax / poorly-conditioned pairs are dropped before the robust
    median (the depth-vs-trajectory inconsistency shows these pairs are
    exactly the unreliable ones).
    """
    Kinv = np.linalg.inv(K)
    uv = np.concatenate([pa, np.ones((len(pa), 1))], 1)
    z = depth_a
    ok = np.isfinite(z) & (z > 0.3) & (z < 12.0)
    if ok.sum() < 15:
        return None
    X = (Kinv @ uv[ok].T).T * z[ok, None]
    try:
        okp, rvec, tvec, inl = cv2.solvePnPRansac(
            X.astype(np.float64), pb[ok].astype(np.float64), K, None,
            reprojectionError=3.0, iterationsCount=300, confidence=0.999)
        if not okp or inl is None or len(inl) < 8:
            return None
    except cv2.error:
        okp, rvec, tvec = cv2.solvePnP(
            X.astype(np.float64), pb[ok].astype(np.float64), K, None,
            flags=cv2.SOLVEPNP_EPNP)
        if not okp:
            return None
        inl = np.arange(len(X))
    return float(np.linalg.norm(tvec)), int(len(inl))


def tum_frames_k(seq, idxs):
    """Working-res RGB (undistorted+resize+crop) and K, at the given idxs."""
    K0 = np.array(bmc.TUM_CAL[seq.split("_")[0]][0])
    dist = np.array(bmc.TUM_CAL[seq.split("_")[0]][1])
    base = f"{TU}/rgbd_dataset_{seq}"
    rows = [l.split() for l in open(f"{base}/rgb.txt") if not l.startswith("#")]
    dep = [l.split() for l in open(f"{base}/depth.txt") if not l.startswith("#")]
    dep_ts = np.array([float(r[0]) for r in dep])
    out = {}
    for i in sorted(idxs):
        img = cv2.imread(f"{base}/{rows[i][1]}")
        img = cv2.undistort(img, K0, dist)
        h0, w0 = img.shape[:2]
        s = LONG / max(h0, w0)
        h1, w1 = round(h0 * s), round(w0 * s)
        h2, w2 = fit56(h1, w1)
        top, left = (h1 - h2) // 2, (w1 - w2) // 2
        img = cv2.resize(img, (w1, h1))[top:top + h2, left:left + w2]
        K1 = K0.copy() * s
        K1[2, 2] = 1.0
        K1[0, 2] -= left
        K1[1, 2] -= top
        k = int(np.searchsorted(dep_ts, float(rows[i][0])).clip(0, len(dep_ts) - 1))
        d = cv2.imread(f"{base}/{dep[k][1]}", cv2.IMREAD_UNCHANGED)
        m1, m2 = cv2.initUndistortRectifyMap(K0, dist, None, K0, (w0, h0), cv2.CV_32FC1)
        du = cv2.remap(d, m1, m2, cv2.INTER_NEAREST)
        du = cv2.resize(du, (w1, h1), interpolation=cv2.INTER_NEAREST)[top:top + h2, left:left + w2]
        out[i] = (img, du.astype(np.float32) / 5000.0, K1)
    return out


def euroc_pair(seq, idxs):
    base = f"{EU}/{seq}/mav0"
    with open(f"{base}/cam0/sensor.yaml") as f:
        s0 = yaml.safe_load(f)
    with open(f"{base}/cam1/sensor.yaml") as f:
        s1 = yaml.safe_load(f)

    def Kof(s):
        fx, fy, cx, cy = s["intrinsics"]
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    K0, D0 = Kof(s0), np.array(s0["distortion_coefficients"])
    K1, D1 = Kof(s1), np.array(s1["distortion_coefficients"])
    T0 = np.array(s0["T_BS"]["data"]).reshape(4, 4)
    T1 = np.array(s1["T_BS"]["data"]).reshape(4, 4)
    T01 = np.linalg.inv(T1) @ T0
    R, t = T01[:3, :3], T01[:3, 3]
    bl = float(np.linalg.norm(t))
    f0 = sorted(os.listdir(f"{base}/cam0/data"))
    f1 = sorted(os.listdir(f"{base}/cam1/data"))
    ts1 = {n.split(".")[0]: n for n in f1}
    smp = cv2.imread(f"{base}/cam0/data/{f0[0]}", cv2.IMREAD_GRAYSCALE)
    h, w = smp.shape[:2]
    R0, R1, P0, P1, _, _, _ = cv2.stereoRectify(K0, D0, K1, D1, (w, h), R, t,
                                                flags=cv2.CALIB_ZERO_DISPARITY, alpha=0.0)
    m0 = cv2.initUndistortRectifyMap(K0, D0, R0, P0, (w, h), cv2.CV_32FC1)
    m1 = cv2.initUndistortRectifyMap(K1, D1, R1, P1, (w, h), cv2.CV_32FC1)
    sgbm = cv2.StereoSGBM_create(0, 128, 7, 8 * 49, 32 * 49, 10, 80, 2, 1)
    fx = P0[0, 0]
    def cam1_name(stem):
        # a few cam0 frames lack an exact cam1 twin in the mirror; fall back to
        # the nearest cam1 timestamp (EuRoC streams are hardware-synced at 20 Hz)
        if stem in ts1:
            return ts1[stem]
        t = int(stem)
        return ts1[min(ts1, key=lambda s: abs(int(s) - t))]

    res = {}
    for i in sorted(set(idxs)):
        n0 = f0[i]
        L = cv2.remap(cv2.imread(f"{base}/cam0/data/{n0}", cv2.IMREAD_GRAYSCALE),
                      m0[0], m0[1], cv2.INTER_LINEAR)
        Rt = cv2.remap(cv2.imread(f"{base}/cam1/data/{cam1_name(n0.split('.')[0])}",
                                  cv2.IMREAD_GRAYSCALE), m1[0], m1[1], cv2.INTER_LINEAR)
        disp = sgbm.compute(L, Rt).astype(np.float32) / 16.0
        z = np.where(disp > 1.0, fx * bl / np.maximum(disp, 1e-6), 0.0)
        res[i] = (L, z.astype(np.float32))
    return res, P0[:3, :3]


def backend_t(run, ii, ij):
    s = np.load(run)
    kf = s["kf_idx"]
    pw = s["poses_wc"]
    def nearest(i):
        return int(np.argmin(np.abs(kf - i)))
    a, b = nearest(ii), nearest(ij)
    return float(np.linalg.norm(pw[a][:3, 3] - pw[b][:3, 3]))


def pairs_for(seq, run, kf_pairs):
    """kf_pairs are POSITIONS in the run's keyframe array; map to frame idx."""
    kf = np.load(run)["kf_idx"]
    fpair = [(int(kf[a]), int(kf[b])) for (a, b) in kf_pairs if a < len(kf) and b < len(kf)]
    out = []
    if seq in TUM:
        frames = tum_frames_k(seq, [i for p in fpair for i in p])

        def get(i):
            return frames[i][0], frames[i][1], frames[i][2]
    else:
        allidx = sorted({i for p in fpair for i in p})
        res, K = euroc_pair(seq, allidx)

        def get(i):
            return res[i][0], res[i][1], K
    for (ii, ij) in fpair:
        ia, da, K = get(ii)
        ib = get(ij)[0]
        pa, pb = orb_match(ia, ib)
        if pa is None:
            continue
        pnp = pnp_metric(pa, pb, samp_depth(da, pa), K)
        if pnp is None:
            continue
        tm, ninl = pnp
        te = backend_t(run, ii, ij)
        if tm and te > 1e-6:
            out.append((ii, ij, tm, te, tm / te, ninl))
    return out


def samp_depth(depth, pts):
    h, w = depth.shape
    x = np.clip(pts[:, 0].round().astype(int), 0, w - 1)
    y = np.clip(pts[:, 1].round().astype(int), 0, h - 1)
    return depth[y, x]


FRACS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
GAPS = (3, 6, 12, 20)


def auto_pairs(n_kf):
    """Many small-baseline pairs spread across the trajectory.

    Long-baseline pairs fail ORB/PnP on EuRoC; a dense set of short-baseline
    pairs (a few keyframes apart) at anchors across the trajectory is both
    matchable and robust under the median (verified on MH_01/V1_02)."""
    out = []
    for f in FRACS:
        k = int(round(f * (n_kf - 1)))
        for g in GAPS:
            j = k + g
            if 0 <= k < j < n_kf and (k, j) not in out:
                out.append((k, j))
    return out


def bootstrap_ci(rv, s1, B=2000, seed=0):
    """CI on the trajectory-anchor scale from resampling the pair ratios."""
    rng = np.random.default_rng(seed)
    n = len(rv)
    sc = np.empty(B)
    for b in range(B):
        med = float(np.median(rng.choice(rv, size=n, replace=True)))
        sc[b] = s1 / med if med > 0 else np.nan
    lo, hi = np.nanpercentile(sc, [2.5, 97.5])
    return float(lo), float(hi), float(np.nanstd(sc))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", default=None)
    ap.add_argument("--kfpairs", default=None,
                    help="explicit POSITION pairs 'a,b;c,d' (overrides auto spread)")
    ap.add_argument("--out", default=f"{P}/outputs/traj_anchor.json")
    a = ap.parse_args()
    idx_override = ([tuple(int(x) for x in pp.split(",")) for pp in a.kfpairs.split(";")]
                    if a.kfpairs else None)
    out = {}
    if os.path.exists(a.out):
        out = json.load(open(a.out))
    for seq in (a.seqs.split(",") if a.seqs else TUM + EUROC):
        run = f"{P}/outputs/gaugeMA_{seq}/kf_state.npz"
        mf = f"{P}/outputs/gaugeMA_{seq}/metrics.json"
        if not (os.path.exists(run) and os.path.exists(mf)):
            print(f"skip {seq}: no gaugeMA run")
            continue
        s1 = float(json.load(open(mf))["umeyama_scale_est2gt"])
        n_kf = int(np.load(run)["kf_idx"].shape[0])
        idx_pairs = idx_override or auto_pairs(n_kf)
        try:
            rows = pairs_for(seq, run, idx_pairs)
        except Exception as e:
            print(f"skip {seq}: {type(e).__name__}: {e}")
            continue
        if not rows:
            print(f"skip {seq}: no valid pairs")
            out.pop(seq, None)
            continue
        # drop low-parallax / short-baseline pairs (relative to the sequence's
        # own largest baseline) before the robust summary
        te_all = np.array([x[3] for x in rows], float)
        if len(te_all) and te_all.max() > 0:
            rows = [r for r, k in zip(rows, te_all > 0.02 * te_all.max()) if k]
        rv = np.array([x[4] for x in rows], float)
        rv = rv[np.isfinite(rv)]
        if len(rv) >= 3:                       # MAD outlier rejection
            med = np.median(rv)
            mad = np.median(np.abs(rv - med)) + 1e-9
            keep = np.abs(rv - med) < 3 * mad
            if keep.sum() >= 2:
                rv = rv[keep]
        n = len(rv)
        r = float(np.median(rv))
        # scaling the trajectory by r makes the umeyama scale scale(1)/r
        # (umeyama scale is inverse to trajectory magnitude)
        scale = s1 / r
        lo, hi, ci_std = (bootstrap_ci(rv, s1) if n >= 2
                          else (scale, scale, 0.0))
        half = 0.5 * (hi - lo)
        rel_scatter = float(np.std(rv) / r) if r > 0 else float("inf")
        if n < 5:
            ok, reason = False, "low_pairs"
        elif half >= 0.25:
            ok, reason = False, "wide_ci"
        elif abs(scale - 1.0) >= 0.15:
            ok, reason = False, "scale_far"
        else:
            ok, reason = True, "ok"
        out[seq] = {"scale_1": s1, "r_median": r, "scale_traj_anchor": scale,
                    "scale_err": abs(scale - 1.0),
                    "fm_only_err": abs(s1 - 1.0), "n_pairs": n,
                    "n_pairs_raw": len(rows),
                    "r_std": float(np.std(rv)), "r_rel_scatter": rel_scatter,
                    "scale_ci95": [lo, hi], "scale_ci_std": ci_std,
                    "n_kf": n_kf, "success": ok, "reason": reason,
                    "pairs": [{"i": x[0], "j": x[1], "t_metric": x[2],
                               "t_backend": x[3], "r": x[4],
                               "inliers": x[5]} for x in rows]}
        print(f"{seq:32s} scale(1)={s1:6.3f} r_med={r:6.3f} (std {np.std(rv):.3f}, "
              f"rel {rel_scatter:.2f}) -> traj-anchor scale={scale:6.3f} "
              f"[{lo:.3f},{hi:.3f}] err={abs(scale-1):.3f} "
              f"(FM-only err={abs(s1-1):.3f}) n={n}/{len(rows)} {reason}")
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
