#!/usr/bin/env python3
"""T1.4: end-to-end trust-gated protocol with metric trajectory error.

Arms per sequence (all offline from saved gaugeMA state; no re-runs):
  FM-only   : gauge anchored to MapAnything's mean metric depth.
  traj-anchor: pin trajectory scale from a sensor metric translation.
  protocol  : gate the FM scale on the init-window dispersion of r_init; trust
              the FM when tight, else use the trajectory anchor.

Reports metric SCALE error and rigid-aligned (SE(3), scale fixed 1) ATE in
metres, so a scale error shows up as trajectory error. Emits
outputs/protocol_ate.json. CPU only (rs3d venv).

NOTE: imports none of run_wired (which pulls in the torch/lietorch vehicle);
the GT loaders below replicate its exact conventions.
"""
import os, json, argparse
import numpy as np

import caliprior_paths as _CP  # path resolution (env-overridable)
P = _CP.PROJECT_ROOT
EU = _CP.EUROC_ROOT
TU = _CP.TUM_ROOT
ALL = ["freiburg1_xyz", "freiburg1_desk", "freiburg2_xyz", "freiburg2_desk",
       "freiburg3_long_office_household",
       "MH_01_easy", "MH_02_easy", "V1_01_easy", "V1_02_medium", "V2_01_easy"]


def rigid_ate(p_est, p_gt):
    """ATE after best rotation+translation, scale fixed to 1 (no Sim(3))."""
    mu_e, mu_g = p_est.mean(0), p_gt.mean(0)
    E, G = p_est - mu_e, p_gt - mu_g
    U, S, Vt = np.linalg.svd(E.T @ G)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    al = E @ R.T + mu_g
    return float(np.sqrt(((al - p_gt) ** 2).sum(1).mean()))


def gt_positions(seq, kf_idx):
    if seq.startswith(("MH_", "V1_", "V2_")):
        base = f"{EU}/{seq}/mav0"
        files = sorted(os.listdir(f"{base}/cam0/data"))
        cam_ns = np.array([int(f.split(".")[0]) for f in files])
        gt = np.loadtxt(f"{base}/state_groundtruth_estimate0/data.csv",
                        delimiter=",", skiprows=1)
        gt_ns, gt_p = gt[:, 0].astype(np.int64), gt[:, 1:4]
        gi = np.searchsorted(gt_ns, cam_ns[kf_idx]).clip(0, len(gt_ns) - 1)
        return gt_p[gi]
    base = f"{TU}/rgbd_dataset_{seq}"
    rgb = [l.split() for l in open(f"{base}/rgb.txt") if not l.startswith("#")]
    g = [l.split() for l in open(f"{base}/groundtruth.txt") if not l.startswith("#")]
    gts = np.array([float(r[0]) for r in g])
    gtp = np.array([[float(x) for x in r[1:4]] for r in g])
    ts = np.array([float(r[0]) for r in rgb])
    gi = np.searchsorted(gts, ts).clip(0, len(gts) - 1)
    return gtp[gi][kf_idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", default=f"{P}/outputs/gate_ood.json")
    ap.add_argument("--traj", default=f"{P}/outputs/traj_anchor.json")
    ap.add_argument("--seqs", default=None,
                    help="comma list; default = the original ten (ALL)")
    ap.add_argument("--tau", type=float, default=None,
                    help="dispersion threshold; default = the gate file's "
                         "best_thr_labelA (chosen 0.0625 on the 10-seq set)")
    ap.add_argument("--out", default=f"{P}/outputs/protocol_ate.json")
    a = ap.parse_args()
    Graw = json.load(open(a.gate))
    G = Graw["per_seq"]
    T = json.load(open(a.traj))
    tau = (a.tau if a.tau is not None
           else float(Graw.get("best_thr_labelA", 0.0625)))
    seqs = a.seqs.split(",") if a.seqs else ALL

    print(f"gate tau={tau:.5f} (log-std r_init)")
    print(f"{'seq':32s} {'logstd':>7s} {'gate':>5s} {'s_fm':>6s} {'s_anc':>6s} "
          f"{'ate_fm':>7s} {'ate_anc':>7s} {'protocol':>8s}")
    rows, prot_scale, prot_ate, fm_scale, fm_ate = {}, [], [], [], []
    for seq in seqs:
        if seq not in T or seq not in G:
            print(f"skip {seq}: missing")
            continue
        st = np.load(f"{P}/outputs/gaugeMA_{seq}/kf_state.npz")
        p = st["poses_wc"][:, :3, 3].astype(np.float64)
        gt = gt_positions(seq, st["kf_idx"])
        s_fm = float(T[seq]["scale_1"])
        r = float(T[seq]["r_median"])
        p_anc = p * r                       # convert backend units -> metres
        ate_fm = rigid_ate(p, gt)
        ate_anc = rigid_ate(p_anc, gt)
        trusted = G[seq]["log_std"] < tau
        if trusted:
            s_p, ate_p, src = s_fm, ate_fm, "FM"
        else:
            s_p, ate_p, src = s_fm / r, ate_anc, "anchor"
        rows[seq] = {"log_std": G[seq]["log_std"], "trusted": bool(trusted),
                     "source": src, "scale_fm": s_fm, "scale_anchor": s_fm / r,
                     "scale_protocol": s_p, "r_median": r, "n_pairs": T[seq]["n_pairs"],
                     "ate_fm": ate_fm, "ate_anchor": ate_anc, "ate_protocol": ate_p}
        prot_scale.append(abs(s_p - 1.0)); prot_ate.append(ate_p)
        fm_scale.append(abs(s_fm - 1.0)); fm_ate.append(ate_fm)
        print(f"{seq:32s} {G[seq]['log_std']:7.4f} {str(trusted):>5s} "
              f"{s_fm:6.3f} {s_fm/r:6.3f} {ate_fm:7.3f} {ate_anc:7.3f} "
              f"{src:>8s}")

    n = len(rows)
    out = {"tau": tau, "n_seq": n,
           "protocol_mean_scale_err": float(np.mean(prot_scale)),
           "protocol_mean_ate_m": float(np.mean(prot_ate)),
           "fm_only_mean_scale_err": float(np.mean(fm_scale)),
           "fm_only_mean_ate_m": float(np.mean(fm_ate)),
           "n_trusted": int(sum(r["trusted"] for r in rows.values())),
           "per_seq": rows}
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"\n{n} seqs: FM-only scale err {out['fm_only_mean_scale_err']:.3f} "
          f"/ ATE {out['fm_only_mean_ate_m']:.3f} m")
    print(f"          protocol scale err {out['protocol_mean_scale_err']:.3f} "
          f"/ ATE {out['protocol_mean_ate_m']:.3f} m "
          f"({out['n_trusted']} trusted)")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
