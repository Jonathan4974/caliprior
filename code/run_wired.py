#!/usr/bin/env python3
"""Phase 1 wired-in run: HI-SLAM2 tracking with MapAnything metric depth as the
prior, Omnidata normals kept for the GS path.

Injection is external (no upstream edits): MotionFilter.prior_extractor is
overridden per-instance to serve cached metric depth; the frame index rides
along via a wrapped track(). JDSA + alpha stay stock -- this is the frozen
prior arm; the online bias state (Phase 2) reads kf_state.npz + metrics.

Working resolution == cache dims exactly (per-dataset), so cached K applies
verbatim and cached depth needs no resampling (v4 stretch-vs-pad lesson).

GT/ATE per dataset:
  euroc: cam ns timestamps vs state_groundtruth_estimate0 (nearest)
  tum:   rgb.txt vs groundtruth.txt (nearest on timestamps)
  kitti: grey poses.txt (index-aligned)

Usage (hislam2 env, see env_setup.sh):
  python run_wired.py --seq MH_01_easy --cache outputs/ma_cache/MH_01_easy.npz ...
"""
import os, sys, argparse, json, types
import numpy as np
import torch
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (100000, rlimit[1]))

import caliprior_paths as _CP  # path resolution (env-overridable)
sys.path.insert(0, _CP.HISLAM2_ROOT)
sys.path.insert(0, _CP.HISLAM2_ROOT + "/hislam2")
sys.path.insert(0, _CP.PROJECT_ROOT)
from hi2 import Hi2
import lietorch
from build_ma_cache import euroc_frames, tum_frames, kitti_frames  # noqa (reuse exact preprocess)
from caliprior_common import (umeyama_ate, load_euroc_gt, load_tum_gt,
                              load_kitti_gt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True,
                    help="MH_01_easy | V1_02_medium | freiburg1_xyz | freiburg1_desk | kitti00_800")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--config", default=_CP.HISLAM2_ROOT + "/config/euroc_config.yaml")
    ap.add_argument("--weights", default=_CP.HISLAM2_ROOT + "/pretrained_models/droid.pth")
    ap.add_argument("--out", required=True)
    ap.add_argument("--buffer", type=int, default=-1)
    ap.add_argument("--mode", default="stock", choices=["stock", "frozen", "online", "metric"])
    ap.add_argument("--b0", type=float, default=1.0)
    ap.add_argument("--K", type=float, default=0.15)
    ap.add_argument("--update-every", type=int, default=1,
                    help="RW update period for the online b state")
    ap.add_argument("--weight", default="const", choices=["const", "conf"],
                    help="E1 metric arm: weight the prior residual by a scalar "
                         "alpha (const) or by per-pixel teacher confidence (conf)")
    ap.add_argument("--sigma-adapt", action="store_true",
                    help="E2: adapt a scalar variance-scale c on the confidence "
                         "sigma by covariance matching so depth-channel NIS -> 1")
    ap.add_argument("--Kc", type=float, default=0.10,
                    help="E2: gain for the innovation-adapted sigma scale")
    ap.add_argument("--sigma-regime", action="store_true",
                    help="E2-regime: adapt a per-confidence-regime variance "
                         "scale c_k so each regime's depth-channel NIS -> 1")
    ap.add_argument("--n-regimes", type=int, default=3,
                    help="E2-regime: number of confidence-quantile regimes")
    ap.add_argument("--regime-calib", type=int, default=60,
                    help="E2-regime: solves used to calibrate quantile edges")
    ap.add_argument("--regime-fit", type=int, default=300,
                    help="E2-regime: fit window; c_k/edges freeze after this")
    ap.add_argument("--freeze-at", type=int, default=0,
                    help="E2: freeze scalar/regime adaptation after this solve "
                         "(0 = never)")
    ap.add_argument("--regime-c-lo", type=float, default=0.05,
                    help="E2-regime: lower guard on c_k (stability guard)")
    ap.add_argument("--regime-c-hi", type=float, default=20.0,
                    help="E2-regime: upper guard on c_k")
    ap.add_argument("--regime-from", default=None,
                    help="E2-regime transfer: metrics.json whose frozen edges + "
                         "c_regime are installed and held fixed")
    ap.add_argument("--c0", type=float, default=1.0,
                    help="E2 transfer: initial scalar variance scale c")
    ap.add_argument("--scalar-fixed", action="store_true",
                    help="E2 transfer: apply c without adapting it")
    ap.add_argument("--no-gs", action="store_true",
                    help="skip the Gaussian backend (scale label comes from "
                         "tracking); works around GS CUDA crashes on long "
                         "EuRoC sequences")
    ap.add_argument("--limit", type=int, default=0, help="use first N frames only (smoke)")
    ap.add_argument("--alpha", type=float, default=None,
                    help="override mono_depth_alpha (prior strength)")
    ap.add_argument("--gauge", default="fixed", choices=["fixed", "ma"],
                    help="fixed: stock normalize (mean depth=2.0); ma: anchor "
                         "the gauge to MapAnything's mean metric depth")
    ap.add_argument("--b", type=float, default=1.0,
                    help="scale the served FM depth by b before the init "
                         "gauge anchor (trajectory scale ~ b * trajectory(1))")
    ap.add_argument("--b2", type=float, default=None,
                    help="P3.4: served-depth scale for frames >= --bswitch "
                         "(tests whether a local FM-scale step reaches local "
                         "reconstruction geometry)")
    ap.add_argument("--bswitch", type=int, default=0,
                    help="P3.4: frame index where --b2 takes over")
    ap.add_argument("--anchor", type=float, default=1.0,
                    help="scale ONLY the init gauge depth by `anchor` (P3.3). "
                         "scale(anchor)=scale(1)/anchor, so anchor=scale(1) "
                         "gives metric scale")
    ap.add_argument("--anchor-mean", type=float, default=0.0,
                    help="N2: set the init gauge mean depth (metres) directly "
                         "from a real sensor anchor, overriding the FM's mean")
    a = ap.parse_args()
    a.out = os.path.abspath(a.out)
    a.cache = os.path.abspath(a.cache)
    a.config = os.path.abspath(a.config)
    a.weights = os.path.abspath(a.weights)
    os.makedirs(a.out, exist_ok=True)
    # HI-SLAM2 loads its Omnidata checkpoints via paths relative to its root.
    os.chdir(_CP.HISLAM2_ROOT)
    C = np.load(a.cache)
    depth_cache = C["depth"]
    # E1: per-pixel teacher confidence, only loaded for the conf-weighted
    # metric arm (the tile is subsampled exactly as depth_video subsamples the
    # served depth, so it aligns with `disps_prior`).
    conf_cache = (C["conf"] if ("conf" in C and a.mode == "metric"
                                and a.weight == "conf") else None)
    # working dims MUST equal cache dims (K exact, no resampling). The
    # generator's LONG default (560) is wrong for caches built with
    # --longest 1120 (KITTI) -- derive it from the cache instead.
    import build_ma_cache as bmc
    bmc.LONG = int(max(depth_cache.shape[1], depth_cache.shape[2]))

    if a.seq.startswith(("MH_", "V1_", "V2_")):
        files, K0, gen = euroc_frames(a.seq)
        cam_ns = np.array([int(f.split(".")[0]) for f in files])
        gt_ns, gt_p = load_euroc_gt(a.seq)
        def gt_of(kf_ns):
            gi = np.searchsorted(gt_ns, kf_ns).clip(0, len(gt_ns) - 1)
            return gt_p[gi]
        key_of = lambda kf_idx: cam_ns[kf_idx]  # noqa
    elif a.seq.startswith("freiburg"):
        rows, K0, gen = tum_frames(a.seq)
        ts, gtp = load_tum_gt(a.seq)
        def gt_of(kf_idx):
            return gtp[kf_idx]
        key_of = lambda kf_idx: (ts[kf_idx] * 1e9).astype(np.int64)  # noqa
    elif a.seq == "kitti00_800":
        n, K0, gen = kitti_frames(800)
        gtp = load_kitti_gt()
        def gt_of(kf_idx):
            return gtp[kf_idx]
        key_of = lambda kf_idx: kf_idx  # noqa
    else:
        raise ValueError(a.seq)

    frames = list(gen())
    N = len(frames)
    if a.limit:
        frames = frames[:a.limit]
        N = len(frames)
    gh, gw = frames[0][1].shape[:2]
    ch, cw = depth_cache.shape[1], depth_cache.shape[2]
    assert (gh, gw) == (ch, cw), f"generator {(gw,gh)} != cache {(cw,ch)}"
    assert N <= len(depth_cache), f"{N} frames > {len(depth_cache)} cached"

    torch.multiprocessing.set_start_method("spawn")
    if a.gauge == "ma":
        import depth_video

        def normalize_metric(self):
            # stock normalize sets mean depth to config scale_multiplier (2.0),
            # discarding the FM's metric scale. Anchor the gauge to the FM's
            # mean metric depth instead: new mean depth = m_MA.
            with self.get_lock():
                n = self.counter.value
                dp = self.disps_prior[:n]
                v = dp > 1e-6
                m_ma = float((1.0 / dp[v]).mean()) if bool(v.any()) else 2.0
                if not (m_ma > 1e-3 and m_ma < 1e4):
                    m_ma = 2.0
                m_ma *= a.anchor  # P3.3: gauge-only anchor knob
                if a.anchor_mean > 0.0:  # N2: real sensor anchor (metres)
                    m_ma = a.anchor_mean
                s = float(self.disps[:n].mean().item()) * m_ma
                if not (s > 1e-6 and s < 1e6):
                    return
                self.poses[:n, :3] *= s
                self.disps[:n] /= s
                self.disps_up[:n] /= s
                self.dscales[:n] /= s
                self.doffset[:n] /= s
                self.dirty[:n] = True
                self._metric_scale = m_ma
        depth_video.DepthVideo.normalize = normalize_metric
    if a.mode != "stock":
        import depth_video
        import jdsa_modes
        jdsa_modes.configure(mode=a.mode, b0=a.b0, K=a.K,
                             update_every=a.update_every, weight=a.weight,
                             sigma_adapt=a.sigma_adapt, Kc=a.Kc,
                             sigma_regime=a.sigma_regime,
                             n_regimes=a.n_regimes,
                             regime_calib_solves=a.regime_calib,
                             regime_fit_frames=a.regime_fit,
                             freeze_at=a.freeze_at, c0=a.c0,
                             scalar_fixed=a.scalar_fixed,
                             c_reg_lo=a.regime_c_lo, c_reg_hi=a.regime_c_hi)
        if a.regime_from is not None:
            tj = json.load(open(a.regime_from))
            jdsa_modes.set_regime(tj["regime_edges"], tj["c_regime_final"])
        jdsa_modes.install(depth_video)
        # conf alignment: prior_extractor runs once immediately before every
        # keyframe append/insert, and video.tstamp[index] is that frame's index.
        # Queue each tile and register it under the frame index actually written
        # (item[0]) so conf keys always match video.tstamp (skip_blur and the
        # terminate-time traj_filler otherwise desync a buffer-index key).
        _PENDING = []
        if conf_cache is not None:
            import depth_video as _dv
            _orig_append = _dv.DepthVideo.append
            _orig_setitem = _dv.DepthVideo.__setitem__

            def _consume(item):
                if _PENDING:
                    jdsa_modes.set_conf(int(item[0]), _PENDING.pop(0))

            def _append(self, *item):
                _consume(item)
                return _orig_append(self, *item)

            def _setitem(self, index, item):
                _consume(item)
                return _orig_setitem(self, index, item)

            _dv.DepthVideo.append = _append
            _dv.DepthVideo.__setitem__ = _setitem
    args = type("A", (), {})()
    args.config, args.weights = a.config, a.weights
    args.buffer = 1200 if a.buffer < 0 else a.buffer  # terminate() inserts many kfs; 518 overflows
    args.output = a.out
    args.gsvis, args.droidvis, args.gtdepthdir = False, False, None

    hi2 = None
    for t, (fi, rgb, K) in enumerate(frames):
        image = torch.as_tensor(rgb).permute(2, 0, 1)
        intr = torch.tensor([K[0, 0], K[1, 1], K[0, 2], K[1, 2]])
        if hi2 is None:
            args.image_size = [image.shape[1], image.shape[2]]
            hi2 = Hi2(args)
            if a.mode != "stock":
                # conf is keyed by frame index; JDSA sees keyframe-buffer
                # indices. Reconcile via the live tstamp array.
                jdsa_modes.set_idx_map(
                    lambda: hi2.video.tstamp[:hi2.video.counter.value].cpu().numpy())
            if a.no_gs:
                # GS-free vehicle: the trajectory scale we need for the gate
                # label comes from tracking, not the Gaussian backend, which
                # crashes on some long EuRoC visuals. Stub the GS hooks.
                def _no_gs_finalize(*ar, **kw):
                    n = hi2.video.counter.value
                    P = hi2.video.poses[:n].detach().cpu().numpy()
                    return np.concatenate(
                        [np.arange(n)[:, None], P], axis=1)
                hi2.call_gs = lambda *ar, **kw: None
                hi2.gs.finalize = _no_gs_finalize
                hi2.gs.eval_rendering = lambda *ar, **kw: None
            if a.alpha is not None:
                hi2.video.mono_depth_alpha = a.alpha
            fx = hi2.filterx
            fx.orig_prior = fx.prior_extractor

            def prior_extractor(self, im_tensor):
                _, normal = self.orig_prior(im_tensor)
                idx = self.cur_idx
                d = torch.from_numpy(
                    depth_cache[idx].astype(np.float32)).cuda()
                if conf_cache is not None:
                    _PENDING.append(
                        conf_cache[idx][3::8, 3::8].astype(np.float32))
                bb = a.b
                if a.b2 is not None and idx >= a.bswitch:
                    bb = a.b2
                if bb != 1.0:
                    d = d * bb
                return d, normal
            fx.prior_extractor = types.MethodType(prior_extractor, fx)
            fx.orig_track = fx.track

            def track(self, tstamp, image, intrinsics=None, is_last=False):
                self.cur_idx = int(tstamp)
                return self.orig_track(tstamp, image, intrinsics, is_last)
            fx.track = types.MethodType(track, fx)
        hi2.track(t, image[None], intrinsics=intr[None], is_last=(t == N - 1))
        if t % 500 == 0:
            print(f"[{t}/{N}] kf={hi2.video.counter.value}", flush=True)

    try:
        if a.mode != "stock":
            import jdsa_modes
            n_track_solves = jdsa_modes.STATE["n"]
        else:
            n_track_solves = 0
        hi2.terminate()
    except Exception as e:  # GS-backend can be None-valued on short/OOD runs;
        print(f"WARN terminate failed ({type(e).__name__}: {e}); "
              f"dumping pre-terminate state", flush=True)
        n_track_solves = locals().get("n_track_solves", 0)
    n = hi2.video.counter.value
    kf_idx = hi2.video.tstamp[:n].cpu().numpy().astype(int)
    poses_wc = lietorch.SE3(hi2.video.poses[:n]).inv().matrix().data.cpu().numpy()
    keys = key_of(kf_idx)
    if a.seq.startswith(("MH_", "V1_", "V2_")):
        np.savetxt(f"{a.out}/traj_kf.txt",
                   np.concatenate([keys[:, None] / 1e9, poses_wc.reshape(n, -1)], axis=1))
        p_gt = gt_of(keys)
    else:
        np.savetxt(f"{a.out}/traj_kf.txt",
                   np.concatenate([keys[:, None].astype(float), poses_wc.reshape(n, -1)], axis=1))
        p_gt = gt_of(kf_idx)
    np.savez(f"{a.out}/kf_state.npz", kf_idx=kf_idx,
             poses_wc=poses_wc, dscales=hi2.video.dscales[:n].cpu().numpy(),
             disps=hi2.video.disps[:n].cpu().numpy(),
             msf=C["msf"][kf_idx], chunk=C["chunk"][kf_idx])
    rmse, s = umeyama_ate(poses_wc[:, :3, 3], p_gt)
    metrics = {"seq": a.seq, "n_frames": N, "n_kf": int(n),
               "ate_rmse_m": rmse, "umeyama_scale_est2gt": s,
               "scale_err_pct": abs(s - 1.0) * 100.0, "alpha": a.alpha,
               "gauge": a.gauge, "b_served": a.b, "anchor": a.anchor,
               "anchor_mean": a.anchor_mean,
               "cache": os.path.basename(a.cache), "mode": a.mode,
               "weight": a.weight}
    if a.mode != "stock":
        import jdsa_modes
        metrics["b_final"] = jdsa_modes.STATE["b"]
        metrics["jdsa_calls"] = jdsa_modes.STATE["n"]
        metrics["n_track_solves"] = n_track_solves
        metrics["b_trace"] = jdsa_modes.STATE["trace"]
        nt = jdsa_modes.STATE.get("nis_trace", [])
        metrics["sigma_adapt"] = a.sigma_adapt
        metrics["c_final"] = jdsa_modes.STATE.get("c", 1.0)
        if nt:
            nis_vals = [x[1] for x in nt]
            metrics["nis_mean"] = float(sum(nis_vals) / len(nis_vals))
            metrics["nis_first"] = float(nis_vals[0])
            metrics["nis_last"] = float(nis_vals[-1])
            hold = [x[1] for x in nt if a.freeze_at > 0 and x[0] > a.freeze_at]
            metrics["nis_holdout"] = (float(sum(hold) / len(hold))
                                      if hold else None)
            metrics["nis_holdout_n"] = len(hold)
            metrics["nis_trace"] = nt
        # E2 sigma-regime state + per-regime NIS (logged for every conf arm, so
        # the scalar/frozen arms also expose cross-regime spread).
        metrics["sigma_regime"] = a.sigma_regime
        metrics["n_regimes"] = a.n_regimes
        metrics["conf_hit"] = jdsa_modes.STATE.get("conf_hit", 0)
        metrics["conf_miss"] = jdsa_modes.STATE.get("conf_miss", 0)
        metrics["regime_edges"] = jdsa_modes.STATE.get("regime_edges", [])
        metrics["c_regime_final"] = jdsa_modes.STATE.get("c_regime", [])
        metrics["regime_ready"] = jdsa_modes.STATE.get("regime_ready", False)
        metrics["regime_frozen"] = jdsa_modes.STATE.get("regime_frozen", False)
        metrics["freeze_at"] = a.freeze_at
        nrt = jdsa_modes.STATE.get("nis_regime_trace", [])
        metrics["nis_regime_trace_n"] = len(nrt)

        def _flat(rows):
            return [v for (_, ks, _, _) in rows for v in ks if v == v]

        if nrt:
            allv = _flat(nrt)
            metrics["nis_regime_mean"] = (sum(allv) / len(allv)) if allv else None
            hold_r = [r for r in nrt if a.freeze_at > 0 and r[0] > a.freeze_at]
            hv = _flat(hold_r)
            metrics["nis_regime_holdout"] = (sum(hv) / len(hv)) if hv else None
            metrics["nis_regime_holdout_n"] = len(hold_r)
            metrics["nis_regime_last"] = [v for v in nrt[-1][1]]
            metrics["nis_regime_trace"] = nrt
    json.dump(metrics, open(f"{a.out}/metrics.json", "w"), indent=2)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
