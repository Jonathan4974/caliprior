"""Phase 2: JDSA variant with a single controllable scale state.

Why this exists: stock JDSA gives every keyframe its own
free 2x2 bilinear scale grid, so it re-absorbs ANY global scale change in the
served prior. An external adapter that only rewrites `prior_extractor`'s
output therefore cannot change the estimate. To run the intended ladder
(stock-JDSA < frozen-NLL < online-state) the scale mechanism itself must be
constrained. This module does that by replacing `depth_video.JDSA` with a
depth-only solve in which the prior is scaled by one scalar `b` that the
caller controls.

  mode="frozen": b fixed (frozen-NLL arm; b=1 == trust the prior)
  mode="stock" : defer to the original JDSA (hand-tuned arm)
  mode="online": b updated by a random-walk filter from the depth-residual
                 innovation after each solve (C1 arm)

Only the depth prior's Jacobian block is kept; the scale block is zeroed, so
`schur_solve_mono_prior` solves depth alone. The innovation `b_mle` is the
least-squares scalar mapping served prior -> current backend inverse depth.
"""
import torch
import droid_backends

import geom.ba as _ba
from geom.ba import safe_scatter_add_mat, safe_scatter_add_vec, disp_retr
from geom.chol import schur_solve_mono_prior

STATE = {
    "mode": "frozen",
    "b": 1.0,
    "K": 0.15,          # RW gain on b
    "b_min": 0.05,
    "b_max": 20.0,
    "n": 0,             # solve calls
    "update_every": 1,
    "trace": [],        # (call, b, b_mle, mean|rd|)
    "orig": None,
    # E1 metric-factor arm: per-pixel weighting from teacher confidence.
    # "const" reproduces the scalar-alpha behaviour; "conf" weights the prior
    # residual by the cached per-pixel confidence (downsampled to the 1/8 grid
    # that `disps_prior` lives on), normalised to mean 1 so `alpha` stays
    # comparable across arms.
    "weight": "const",
    "conf": {},         # video-slot -> (ht, wd) float32 conf tile
    # E2 sigma ladder: a scalar variance-scale `c` on the confidence-derived
    # sigma, adapted by covariance matching so the depth-channel NIS -> 1.
    # sigma_eff^2 = c * sigma0^2, weight w = 1/sigma_eff^2.
    "sigma_adapt": False,
    "c": 1.0,
    "scalar_fixed": False,       # apply c without adapting (transfer baseline)
    "Kc": 0.10,
    "c_min": 0.2,
    "c_max": 5.0,
    "nis_trace": [],    # (call, nis, c, weight)
    # E2 sigma-regime: replace the single scalar c with K confidence-quantile
    # regime scales c_k. Edges are calibrated from the valid prior pixels of the
    # first `regime_calib_solves` solves; c_k adapts each regime's NIS to 1 until
    # `freeze_at`, after which both edges and c_k are frozen (temporal hold-out).
    "sigma_regime": False,
    "n_regimes": 3,
    "c_regime": [],              # K scales (set once edges ready)
    "regime_edges": [],          # K-1 confidence-quantile edges
    "regime_ready": False,
    "regime_frozen": False,
    "regime_samples": [],        # 1-D valid-confidence tensors during calibration
    "regime_calib_solves": 60,
    "regime_fit_frames": 300,
    "regime_samp_cap": 5000,     # per-solve subsample
    "freeze_at": 0,              # >0: stop adapting scalar/regime at this solve
    "c_reg_lo": 0.05,
    "c_reg_hi": 20.0,
    "nis_regime": [],            # last per-regime NIS
    "nis_regime_trace": [],      # (call, [nis_k], [c_k], frozen)
    "conf_hit": 0,
    "conf_miss": 0,
    # kx in JDSA are *keyframe buffer* indices, but set_conf keys by *frame*
    # index (video.tstamp). idx_map() returns the live tstamp array so the two
    # can be reconciled; without it, conf lookups are misaligned.
    "idx_map": None,
}


def set_conf(slot, conf_down):
    """Stash a downsampled (ht, wd) confidence tile for one served frame.

    Called by the adapter (run_wired.prior_extractor) with the SAME 1/8
    subsampling depth_video uses for `disps_prior` (`[3::8, 3::8]`), so the
    tile aligns pixel-for-pixel with the prior residual.
    """
    STATE["conf"][int(slot)] = conf_down


def clear_conf():
    STATE["conf"] = {}


def set_regime(edges, c_regime):
    """Install frozen regime edges + scales (cross-sequence transfer)."""
    STATE["regime_edges"] = [float(e) for e in edges]
    STATE["c_regime"] = [float(c) for c in c_regime]
    STATE["regime_ready"] = True
    STATE["regime_frozen"] = True
    STATE["regime_samples"] = []


def set_idx_map(fn):
    """Install a callable returning the live keyframe-buffer -> frame tstamp map."""
    STATE["idx_map"] = fn


def _conf_tiles(kx, ht, wd, device, dtype):
    """(M, ht*wd) raw confidence for the served frames, or None if unavailable."""
    if STATE["weight"] != "conf" or not STATE["conf"]:
        return None
    imap = None
    if STATE["idx_map"] is not None:
        try:
            imap = STATE["idx_map"]()
        except Exception:
            imap = None
    tiles = []
    for k in kx.tolist():
        f = int(k)
        if imap is not None and 0 <= f < len(imap):
            f = int(imap[f])
        c = STATE["conf"].get(f)
        if c is None:
            STATE["conf_miss"] += 1
            tiles.append(torch.ones(ht, wd))
        else:
            STATE["conf_hit"] += 1
            tiles.append(torch.as_tensor(c))
    return torch.stack(tiles).to(device=device, dtype=dtype).reshape(-1, ht * wd)


def _finalize_regimes():
    """Set confidence-quantile edges from the calibration samples, then c_k=1."""
    if STATE["regime_ready"] or not STATE["regime_samples"]:
        return
    s = torch.cat(STATE["regime_samples"]).float()
    K = STATE["n_regimes"]
    qs = torch.linspace(0.0, 1.0, K + 1)[1:-1]
    edges = torch.quantile(s, qs).tolist()
    for i in range(1, len(edges)):  # enforce strictly increasing
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-6
    STATE["regime_edges"] = edges
    STATE["c_regime"] = [1.0] * K
    STATE["regime_ready"] = True
    STATE["regime_samples"] = []


def _regime_id(Ct):
    """Regime index per pixel from frozen edges (0..K-1). Always `pixel_space`."""
    if not STATE["regime_ready"]:
        return None
    edges = torch.as_tensor(STATE["regime_edges"], device=Ct.device, dtype=Ct.dtype)
    return torch.bucketize(Ct, edges)


def _pixel_weights(kx, ht, wd, device, dtype):
    """(M, ht*wd) confidence weights, mean-normalised; None falls back to const.

    Regime arms divide by the per-pixel regime scale `c_{r(i)}`; the scalar arm
    divides by the single `c`; the frozen arm divides by neither.
    """
    Ct = _conf_tiles(kx, ht, wd, device, dtype)
    if Ct is None:
        return None, None
    W = Ct / Ct.mean().clamp_min(1e-6)
    R = _regime_id(Ct)
    if STATE["sigma_regime"] and R is not None:
        cvec = torch.as_tensor(STATE["c_regime"], device=device, dtype=dtype)[R]
        W = W / cvec
    elif STATE["sigma_adapt"] or STATE["scalar_fixed"]:
        W = W / STATE["c"]
    return W, R

_ORIG_JDSA = None


def configure(mode="frozen", b0=1.0, K=0.15, update_every=1, weight="const",
              sigma_adapt=False, Kc=0.10, sigma_regime=False, n_regimes=3,
              regime_calib_solves=60, regime_fit_frames=300, freeze_at=0,
              c0=1.0, scalar_fixed=False, c_reg_lo=0.05, c_reg_hi=20.0):
    STATE["mode"] = mode
    STATE["b"] = float(b0)
    STATE["K"] = float(K)
    STATE["update_every"] = int(update_every)
    STATE["weight"] = weight
    STATE["sigma_adapt"] = bool(sigma_adapt)
    STATE["scalar_fixed"] = bool(scalar_fixed)
    STATE["Kc"] = float(Kc)
    STATE["n"] = 0
    STATE["trace"] = []
    STATE["conf"] = {}
    STATE["c"] = float(c0)
    STATE["nis_trace"] = []
    STATE["sigma_regime"] = bool(sigma_regime)
    STATE["n_regimes"] = int(n_regimes)
    STATE["c_regime"] = []
    STATE["regime_edges"] = []
    STATE["regime_ready"] = False
    STATE["regime_frozen"] = False
    STATE["regime_samples"] = []
    STATE["regime_calib_solves"] = int(regime_calib_solves)
    STATE["regime_fit_frames"] = int(regime_fit_frames)
    STATE["freeze_at"] = int(freeze_at)
    STATE["c_reg_lo"] = float(c_reg_lo)
    STATE["c_reg_hi"] = float(c_reg_hi)
    STATE["nis_regime"] = []
    STATE["nis_regime_trace"] = []
    STATE["conf_hit"] = 0
    STATE["conf_miss"] = 0


def _b_mle(disps_new, dp, kx, ht, wd):
    """LS scalar b with disps_new ~= b * dp over valid prior pixels."""
    d = disps_new[0, kx].reshape(-1, ht * wd)
    p = dp.reshape(-1, ht * wd)
    mask = p > 1e-6
    if int(mask.sum()) < 32:
        return None
    dd, pp = d[mask], p[mask]
    denom = float((pp * pp).sum())
    if denom < 1e-12:
        return None
    return float((dd * pp).sum() / denom)


def JDSA_scaled(target, weight, eta, poses, disps, intrinsics,
                disps_prior, dscales, ii, jj, alpha):
    B, P, ht, wd = disps.shape
    C, w = droid_backends.proj_trans(poses.data.squeeze(), disps[0],
                                     intrinsics[0], target, weight, ii, jj)
    kx, _ = torch.unique(ii, return_inverse=True)
    M = kx.shape[0]
    dp = disps_prior[kx]
    m = (dp > 0).to(torch.float).view(-1, ht * wd)

    b = torch.as_tensor(STATE["b"], device=disps.device, dtype=disps.dtype)
    rd = (disps[0, kx] - dp * b).view(-1, ht * wd)
    Jd = torch.ones_like(rd).view(1, -1, 1, ht * wd)
    a = torch.ones(M, ht * wd, 1, device=disps.device, dtype=disps.dtype) * alpha

    # depth-only normal equations: zero scale block (D = 1)
    Hs = torch.zeros(B, M, M, 1, 1, device=disps.device, dtype=disps.dtype)
    Es = torch.zeros(B, M, M, 1, ht * wd, device=disps.device, dtype=disps.dtype)
    vs = torch.zeros(B, M, 1, 1, device=disps.device, dtype=disps.dtype)

    Cmat = C[None] + m * a.squeeze() * (Jd * Jd).squeeze() + (1 - m) * eta.view(*C.shape)
    wvec = w[None] - m * a.squeeze() * rd * Jd.squeeze()
    _, dz, dzcov = schur_solve_mono_prior(Cmat, wvec, Hs, Es, vs, dzcov=True)

    disps_new = disp_retr(disps, dz.view(B, -1, ht, wd), kx)
    disps_new = torch.where(disps_new > 10, torch.zeros_like(disps_new), disps_new)
    disps_new = disps_new.clamp(min=0.001)

    # innovation -> RW update of b (log space for positivity/scale symmetry)
    mle = _b_mle(disps_new, dp, kx, ht, wd)
    STATE["n"] += 1
    if mle is not None and STATE["mode"] == "online" and \
            (STATE["n"] % STATE["update_every"] == 0):
        import math
        b_old = STATE["b"]
        lb = math.log(b_old) + STATE["K"] * (math.log(max(mle, 1e-6)) - math.log(b_old))
        STATE["b"] = float(min(max(math.exp(lb), STATE["b_min"]), STATE["b_max"]))
    if mle is not None and STATE["n"] % 50 == 0:
        STATE["trace"].append((STATE["n"], STATE["b"], mle, float(rd.abs().mean())))

    return disps_new, dscales, dzcov


def JDSA_metric(target, weight, eta, poses, disps, intrinsics,
                disps_prior, dscales, ii, jj, alpha):
    """E1 metric-depth factor arm (no scale state, optional per-pixel sigma).

    Differences from stock JDSA:
      * no per-keyframe scale grid: the scale block is zeroed and `dscales` is
        never updated, so the served prior enters at its own (metric) scale;
      * the residual is `disps - disps_prior` with no free `b`, i.e. the prior is
        taken at face value (b=1). This is the arm that tests whether a metric
        residual pins the gauge, and whether a deliberate local step survives;
      * optionally (`weight="conf"`) the residual is weighted per pixel by the
        teacher's confidence, mean-normalised so `alpha` is comparable.
    """
    B, P, ht, wd = disps.shape
    STATE["n"] += 1
    n = STATE["n"]
    C, w = droid_backends.proj_trans(poses.data.squeeze(), disps[0],
                                     intrinsics[0], target, weight, ii, jj)
    kx, _ = torch.unique(ii, return_inverse=True)
    M = kx.shape[0]
    dp = disps_prior[kx]
    m = (dp > 0).to(torch.float).view(-1, ht * wd)

    rd = (disps[0, kx] - dp).view(-1, ht * wd)
    Jd = torch.ones_like(rd).view(1, -1, 1, ht * wd)

    # E2 sigma-regime calibration: pool valid prior-pixel confidence over the
    # first `regime_calib_solves` solves, then freeze the quantile edges.
    if STATE["weight"] == "conf":
        if not STATE["regime_ready"]:
            if n <= STATE["regime_calib_solves"]:
                Ct = _conf_tiles(kx, ht, wd, disps.device, disps.dtype)
                if Ct is not None:
                    sv = Ct[m > 0]
                    if sv.numel() > STATE["regime_samp_cap"]:
                        idx = torch.randperm(sv.numel(), device=sv.device)
                        sv = sv[idx[:STATE["regime_samp_cap"]]]
                    if sv.numel() > 0:
                        STATE["regime_samples"].append(sv.detach().cpu())
            else:
                _finalize_regimes()

    W, R = _pixel_weights(kx, ht, wd, disps.device, disps.dtype)
    if W is None:
        a = torch.ones(M, ht * wd, 1, device=disps.device, dtype=disps.dtype) * alpha
        a = a.squeeze()
    else:
        a = alpha * W  # (M, ht*wd)

    # depth-only normal equations: zero scale block (D = 1)
    Hs = torch.zeros(B, M, M, 1, 1, device=disps.device, dtype=disps.dtype)
    Es = torch.zeros(B, M, M, 1, ht * wd, device=disps.device, dtype=disps.dtype)
    vs = torch.zeros(B, M, 1, 1, device=disps.device, dtype=disps.dtype)

    Cmat = C[None] + m * a * (Jd * Jd).squeeze() + (1 - m) * eta.view(*C.shape)
    wvec = w[None] - m * a * rd * Jd.squeeze()
    _, dz, dzcov = schur_solve_mono_prior(Cmat, wvec, Hs, Es, vs, dzcov=True)

    disps_new = disp_retr(disps, dz.view(B, -1, ht, wd), kx)
    disps_new = torch.where(disps_new > 10, torch.zeros_like(disps_new), disps_new)
    disps_new = disps_new.clamp(min=0.001)

    # E2 depth-channel NIS: mean normalised squared prior residual,
    # e = rd / sigma, sigma^2 = 1/a  =>  e^2 = a * rd^2.  Consistency: NIS ~ 1.
    frozen = STATE["freeze_at"] > 0 and n > STATE["freeze_at"]
    denom = float(m.sum())
    if denom > 0:
        nis = float((m * a * rd * rd).sum() / denom)
        STATE["nis_trace"].append((n, nis, float(STATE["c"]), STATE["weight"]))
        if STATE["sigma_adapt"] and not frozen:
            c_new = STATE["c"] * (1.0 + STATE["Kc"] * (nis - 1.0))
            STATE["c"] = float(min(max(c_new, STATE["c_min"]), STATE["c_max"]))

    # E2 sigma-regime: per-regime NIS, then covariance-matching update of c_k.
    if STATE["weight"] == "conf" and R is not None and denom > 0:
        sq = m * a * rd * rd
        nis_k = []
        for k in range(STATE["n_regimes"]):
            mk = (R == k) & (m > 0)
            dk = int(mk.sum())
            nis_k.append(float(sq[mk].sum() / dk) if dk >= 8 else float("nan"))
        STATE["nis_regime"] = nis_k
        if frozen and not STATE["regime_frozen"]:
            STATE["regime_frozen"] = True
        STATE["nis_regime_trace"].append(
            (n, list(nis_k), list(STATE["c_regime"]),
             bool(STATE["regime_frozen"])))
        if STATE["sigma_regime"] and not STATE["regime_frozen"]:
            for k in range(STATE["n_regimes"]):
                v = nis_k[k]
                if v == v:  # skip NaN (too few pixels in regime)
                    cnew = STATE["c_regime"][k] * (1.0 + STATE["Kc"] * (v - 1.0))
                    STATE["c_regime"][k] = float(
                        min(max(cnew, STATE["c_reg_lo"]), STATE["c_reg_hi"]))

    if n % 50 == 0:
        STATE["trace"].append((n, STATE["b"],
                               float(rd.abs().mean()), STATE["weight"]))
    return disps_new, dscales, dzcov


def install(depth_video_module):
    """Patch depth_video.JDSA according to STATE['mode']."""
    global _ORIG_JDSA
    if _ORIG_JDSA is None:
        _ORIG_JDSA = depth_video_module.JDSA

    def dispatch(*args, **kwargs):
        if STATE["mode"] == "stock":
            return _ORIG_JDSA(*args, **kwargs)
        if STATE["mode"] == "metric":
            return JDSA_metric(*args, **kwargs)
        return JDSA_scaled(*args, **kwargs)

    depth_video_module.JDSA = dispatch
