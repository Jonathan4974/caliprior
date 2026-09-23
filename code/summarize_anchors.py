#!/usr/bin/env python3
"""T1.2: consolidate the three anchor arms on all 10 sequences.

  FM-only        : gauge anchored to MapAnything's mean metric depth.
  depth-mean     : gauge mean depth set from the dataset sensor (TUM RGB-D /
                   EuRoC stereo) -- the "fails" arm, now measured on all 10.
  trajectory     : metric translation anchor matched to the backend.

Reads outputs/{anchors.json, traj_anchor.json, anchorN2_*/metrics.json,
gaugeMA_*/metrics.json}; emits outputs/anchor_compare.json. CPU only.
"""
import os, json
import numpy as np

import caliprior_paths as _CP  # path resolution (env-overridable)
from caliprior_common import ALL10 as ALL
P = _CP.PROJECT_ROOT


def main():
    anchors = json.load(open(f"{P}/outputs/anchors.json"))
    traj = json.load(open(f"{P}/outputs/traj_anchor.json"))
    rows = {}
    for seq in ALL:
        gm = f"{P}/outputs/gaugeMA_{seq}/metrics.json"
        s1 = float(json.load(open(gm))["umeyama_scale_est2gt"]) if os.path.exists(gm) else None
        dm = f"{P}/outputs/anchorN2_{seq}/metrics.json"
        s_d = float(json.load(open(dm))["umeyama_scale_est2gt"]) if os.path.exists(dm) else None
        s_t = traj.get(seq, {}).get("scale_traj_anchor")
        rows[seq] = {
            "modality": anchors.get(seq, {}).get("modality"),
            "sensor_mean_m": anchors.get(seq, {}).get("anchor_mean"),
            "fm_scale": s1, "fm_err": (abs(s1 - 1.0) if s1 is not None else None),
            "depth_anchor_scale": s_d,
            "depth_anchor_err": (abs(s_d - 1.0) if s_d is not None else None),
            "traj_anchor_scale": s_t,
            "traj_anchor_err": traj.get(seq, {}).get("scale_err"),
            "traj_n_pairs": traj.get(seq, {}).get("n_pairs")}
    json.dump(rows, open(f"{P}/outputs/anchor_compare.json", "w"), indent=2)

    print(f"{'seq':32s} {'sensor_m':>8s} {'FM_err':>7s} {'depth_err':>9s} "
          f"{'traj_err':>8s} {'n':>3s}")
    for seq, r in rows.items():
        de = "--" if r["depth_anchor_err"] is None else f"{r['depth_anchor_err']:.3f}"
        print(f"{seq:32s} {r['sensor_mean_m']:8.3f} {r['fm_err']:7.3f} "
              f"{de:>9s} {r['traj_anchor_err']:8.3f} {r['traj_n_pairs']:3d}")
    def agg(key):
        v = [r[key] for r in rows.values() if r[key] is not None]
        return float(np.mean(v)) if v else None
    print(f"\nmean scale error: FM {agg('fm_err'):.3f}  "
          f"depth-mean {agg('depth_anchor_err'):.3f}  "
          f"trajectory {agg('traj_anchor_err'):.3f}")
    print(f"wrote {P}/outputs/anchor_compare.json")


if __name__ == "__main__":
    main()
