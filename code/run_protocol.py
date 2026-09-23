#!/usr/bin/env python3
"""N1.4: end-to-end trust-gated metric protocol.

Protocol: at deployment, run a short prior-free bootstrap, compute the
init-window dispersion of r_init, and
  - if dispersion < tau_gate  -> trust the FM metric scale: anchor the gauge to
    the FM (`--gauge ma`), giving metric scale to the FM's own accuracy;
  - else                      -> reject the FM scale and require one metric
    anchor (N2); the FM is used for dense structure only.

This script reports, per sequence, the gate decision and the scale error the
protocol would incur, versus the FM-only error it avoids when it rejects.

Reads outputs/gate_ood.json. Optional outputs/anchors.json (N2) supplies a real
sensor anchor mean depth per sequence; when present the rejected sequences are
recovered with the anchor (`scale(anchor)=scale(1)/anchor`).
"""
import os, json, argparse
import numpy as np

import caliprior_paths as _CP  # path resolution (env-overridable)
P = _CP.PROJECT_ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", default=f"{P}/outputs/gate_ood.json")
    ap.add_argument("--anchors", default=f"{P}/outputs/anchors.json")
    ap.add_argument("--tau", type=float, default=0.05,
                    help="dispersion threshold (in-domain ~0.01-0.02, OOD >0.2)")
    ap.add_argument("--out", default=f"{P}/outputs/protocol.json")
    a = ap.parse_args()
    G = json.load(open(a.gate))
    anchors = json.load(open(a.anchors)) if os.path.exists(a.anchors) else {}
    rows = G["per_seq"]

    print(f"gate threshold tau = {a.tau} (log-std r_init)")
    print(f"{'seq':32s} {'logstd':>7s} {'gate':>6s} {'scale_ma':>9s} "
          f"{'FM-only err':>11s} {'anchor':>8s} {'protocol err':>12s}")
    prot_err, fm_only_rejected = [], []
    for k in sorted(rows):
        d = rows[k]
        sm = d.get("scale_ma")
        if sm is None:
            continue
        trusted = d["log_std"] < a.tau
        fm_err = abs(sm - 1.0)
        anc = anchors.get(k, {}).get("anchor_scale", None)
        if trusted:
            perr = fm_err
            src = "FM"
        elif anc is not None:
            # anchor supplies the metric scale; error = anchor's own error
            perr = abs(anc - 1.0)
            src = "anchor"
        else:
            perr = float("nan")   # rejected, no anchor available -> unresolved
            src = "none"
        prot_err.append(perr if not np.isnan(perr) else np.nan)
        if not trusted:
            fm_only_rejected.append(fm_err)
        print(f"{k:32s} {d['log_std']:7.4f} {str(trusted):>6s} {sm:9.3f} "
              f"{fm_err:11.3f} {str(src):>8s} {perr:12.3f}")

    valid = [e for e in prot_err if not np.isnan(e)]
    out = {"tau": a.tau, "n_seq": len(valid),
           "protocol_mean_scale_err": float(np.mean(valid)) if valid else None,
           "rejected_fm_only_mean_err": (float(np.mean(fm_only_rejected))
                                         if fm_only_rejected else None),
           "n_trusted": int(sum(1 for e in prot_err if not np.isnan(e)))}
    json.dump({"gate": G, "protocol": out, "anchors_used": bool(anchors)},
              open(a.out, "w"), indent=2)
    print(f"\nresolved {out['n_seq']} seqs; protocol mean scale err = "
          f"{out['protocol_mean_scale_err']}")
    if fm_only_rejected:
        print(f"rejected seqs: FM-only would have erred "
              f"{out['rejected_fm_only_mean_err']:.3f} (avoided by gating)")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
