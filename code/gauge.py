#!/usr/bin/env python3
"""P2A: calibrate the observable -> metric-correction map and test transfer.

The gauge is set once at warmup, so global scale is a single scalar per
sequence. MEASURED (P2B, gaugelin): scale(b) = scale(b=1) / b, i.e. serving a
larger FM depth SHRINKS the GT scale. So the oracle correction is
b_gt = scale(gaugeMA, b=1)  (NOT 1/scale). The GT-free observable is r_init,
the median D_MA/D_backend over the init window (observable_b.py).

Physical expectation: if the FM depth bias is beta (D_MA = beta*D_true) and the
prior-free backend gauge gamma is domain-independent (D_backend = gamma*D_true),
then r = beta/gamma and b_gt = 1/beta = (1/gamma)/r  ->  b_gt should be
MONOTONE DECREASING in r. This script tests that against measurement.

Protocol: fit log b = a + p log r on the *source* domain (TUM, has GT), then
apply to the *target* (EuRoC) with no target GT and compare b_obs to b_gt.

Outputs outputs/gauge_map.json.
"""
import os, json, argparse
import numpy as np

import caliprior_paths as _CP  # path resolution (env-overridable)
P = _CP.PROJECT_ROOT
SOURCES = ["freiburg1_xyz", "freiburg1_desk"]
TARGETS = ["MH_01_easy", "V1_02_medium"]


def load_data():
    obs = json.load(open(f"{P}/outputs/observable_b.json"))
    data = {}
    for seq in SOURCES + TARGETS:
        mf = f"{P}/outputs/gaugeMA_{seq}/metrics.json"
        if seq not in obs or not os.path.exists(mf):
            print(f"  (missing) {seq}")
            continue
        m = json.load(open(mf))
        s = float(m["umeyama_scale_est2gt"])
        data[seq] = {
            "r_init": float(obs[seq]["r_init"]),
            "r_init_log": float(obs[seq]["r_init_log"]),
            "r_std": float(obs[seq]["std"]),
            "scale_ma": s,
            "b_gt": s,
        }
    return data


def fit_loglog(items):
    x = np.array([np.log(d["r_init"]) for d in items])
    y = np.array([np.log(d["b_gt"]) for d in items])
    if len(items) == 1:
        return 0.0, float(y[0] - 0.0 * x[0])  # p=0 meaningless with 1 pt
    p, a = np.polyfit(x, y, 1)
    return float(a), float(p)


def pred_b(r, a, p):
    return float(np.exp(a + p * np.log(r)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{P}/outputs/gauge_map.json")
    a = ap.parse_args()
    D = load_data()
    src = [D[s] for s in SOURCES if s in D]
    tgt = [s for s in TARGETS if s in D]
    if len(src) < 1:
        print("no source points yet"); return

    print("\n== source (TUM) calibration points ==")
    for s in SOURCES:
        if s in D:
            d = D[s]
            print(f"  {s:16s} r_init={d['r_init']:6.3f} scale_ma={d['scale_ma']:6.3f} b_gt={d['b_gt']:7.3f}")

    fit_a, fit_p = fit_loglog(src)
    print(f"\nfit (source only): b = exp({fit_a:.3f}) * r^{fit_p:.3f}")

    # upper-bound reference: fit on ALL available seqs (uses target GT; NOT a
    # deployable map, only shows whether a monotone map exists at all)
    allpts = src + [D[s] for s in tgt]
    all_a, all_p = fit_loglog(allpts)

    out = {"fit_source": {"a": fit_a, "p": fit_p,
                          "sources": [s for s in SOURCES if s in D]},
           "fit_all_oracle": {"a": all_a, "p": all_p,
                              "seqs": [s for s in SOURCES + TARGETS if s in D]},
           "per_seq": {}}

    # monotonicity test across all available seqs: physical model says b_gt
    # DECREASES with r. Count pairwise sign agreement.
    seqs_all = [s for s in SOURCES + TARGETS if s in D]
    rr = np.array([D[s]["r_init"] for s in seqs_all])
    bb = np.array([D[s]["b_gt"] for s in seqs_all])
    conc = 0
    disc = 0
    for i in range(len(rr)):
        for j in range(i + 1, len(rr)):
            sr = np.sign(rr[i] - rr[j])
            sb = np.sign(bb[i] - bb[j])
            if sr * sb > 0:
                conc += 1      # b increases with r (violates physical model)
            else:
                disc += 1      # b decreases with r (consistent)
    print("\n== monotonicity across all 4 seqs (physical model: b_gt down vs r) ==")
    for s in seqs_all:
        print(f"  {s:16s} r_init={D[s]['r_init']:6.3f} b_gt={D[s]['b_gt']:7.3f}")
    print(f"  concordant(b up w/ r)={conc}  discordant(b down w/ r)={disc}")
    out["monotonicity"] = {"concordant": conc, "discordant": disc}

    print("\n== transfer test on target (EuRoC), no target GT ==")
    print(f"{'seq':16s} {'r_init':>7s} {'b_obs':>8s} {'b_gt':>8s} "
          f"{'log-err':>8s} {'scale_err_online':>16s} {'scale_err_frozen':>16s}")
    b_cal = pred_b(src[0]["r_init"], fit_a, fit_p) if fit_p != 0 else src[0]["b_gt"]
    for s in SOURCES + TARGETS:
        if s not in D:
            continue
        d = D[s]
        b_obs = pred_b(d["r_init"], fit_a, fit_p)
        s_online = b_obs * d["scale_ma"]
        s_frozen = b_cal * d["scale_ma"]
        d["b_obs"] = b_obs
        d["scale_err_online"] = abs(s_online - 1.0)
        d["scale_err_frozen"] = abs(s_frozen - 1.0)
        out["per_seq"][s] = d
        role = "src" if s in SOURCES else "TGT"
        print(f"{s:16s} {d['r_init']:7.3f} {b_obs:8.3f} {d['b_gt']:8.3f} "
              f"{np.log(b_obs / d['b_gt']):8.3f} {d['scale_err_online']:16.3f} "
              f"{d['scale_err_frozen']:16.3f}   {role}")

    # PASS/KILL: online beats frozen on target scale error
    win = 0
    for s in tgt:
        d = out["per_seq"][s]
        if d["scale_err_online"] < d["scale_err_frozen"]:
            win += 1
    out["target_online_wins"] = f"{win}/{len(tgt)}"
    print(f"\nPASS needs online < frozen on >=2/3 informative seqs "
          f"(informative = f1_xyz, f1_desk, MH_01): {win}/{len(tgt)} on targets")
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
