#!/usr/bin/env python3
"""P2C: assemble the 5-arm metric-scale ladder and apply the MVP PASS/KILL.

Given the gauge is set once at warmup, trajectory(b) ~= b * trajectory(1), so
each arm differs only by the scalar b served before the init anchor:
  down     : gauge fixed (no FM metric scale)          -> scale_down
  in-domain: gauge ma, b=1                              -> scale_ma
  frozen   : gauge ma, b=b_cal (source value)           -> b_cal*scale_ma
  online   : gauge ma, b=b_obs(r_init) (GT-free)        -> b_obs*scale_ma
  oracle   : gauge ma, b=b_gt=1/scale_ma                -> 1.0

Reads outputs/{wired_*,gaugeMA_*}/metrics.json + outputs/gauge_map.json.
"""
import os, json
import numpy as np

import caliprior_paths as _CP  # path resolution (env-overridable)
P = _CP.PROJECT_ROOT
SEQS = ["freiburg1_xyz", "freiburg1_desk", "MH_01_easy", "V1_02_medium"]
INFORMATIVE = ["freiburg1_xyz", "freiburg1_desk", "MH_01_easy"]


def met(path):
    return json.load(open(path)) if os.path.exists(path) else None


def main():
    gm = json.load(open(f"{P}/outputs/gauge_map.json")) if \
        os.path.exists(f"{P}/outputs/gauge_map.json") else {"per_seq": {}}
    ps = gm.get("per_seq", {})
    # b_cal = frozen source correction = mean b_gt over source (TUM) seqs
    src_b = [v["b_gt"] for k, v in ps.items() if k.startswith("freiburg")]
    b_cal = float(np.mean(src_b)) if src_b else 1.0
    print(f"b_cal (frozen, from TUM source) = {b_cal:.4f}\n")

    rows = []
    for seq in SEQS:
        down = met(f"{P}/outputs/wired_{seq}/metrics.json")
        ma1 = met(f"{P}/outputs/gaugeMA_{seq}/metrics.json")
        if ma1 is None:
            continue
        scale_ma = ma1["umeyama_scale_est2gt"]
        ate_ma = ma1["ate_rmse_m"]
        scale_down = down["umeyama_scale_est2gt"] if down else None
        ate_down = down["ate_rmse_m"] if down else None
        d = ps.get(seq, {})
        b_obs = d.get("b_obs", None)
        b_gt = 1.0 / scale_ma
        r = {}
        r["down"] = abs(scale_down - 1.0) if scale_down is not None else None
        r["in-domain"] = abs(scale_ma - 1.0)
        r["frozen"] = abs(b_cal * scale_ma - 1.0)
        r["online"] = abs(b_obs * scale_ma - 1.0) if b_obs is not None else None
        r["oracle"] = abs(b_gt * scale_ma - 1.0)
        rows.append({"seq": seq, "r_init": d.get("r_init"), "b_obs": b_obs,
                     "b_gt": b_gt, "scale_down": scale_down, "scale_ma": scale_ma,
                     "ate_down": ate_down, "ate_ma": ate_ma,
                     "scale_err": r})

    hdr = f"{'seq':16s} {'r_init':>7s} | {'down':>7s} {'in-dom':>7s} " \
          f"{'frozen':>7s} {'online':>7s} {'oracle':>7s} | b_obs/b_gt"
    print(hdr)
    print("-" * len(hdr))
    for x in rows:
        e = x["scale_err"]
        def f(v): return f"{v:7.3f}" if v is not None else "   n/a "
        oss = (x["b_obs"] / x["b_gt"]) if (x["b_obs"] and x["b_gt"]) else None
        print(f"{x['seq']:16s} {x['r_init']:7.3f} | {f(e['down'])} {f(e['in-domain'])} "
              f"{f(e['frozen'])} {f(e['online'])} {f(e['oracle'])} | "
              f"{oss:.2f}" if oss else f"{x['seq']:16s} {x['r_init']:7.3f} | "
              f"{f(e['down'])} {f(e['in-domain'])} {f(e['frozen'])} "
              f"{f(e['online'])} {f(e['oracle'])} | n/a")

    # PASS/KILL: online beats frozen on >=2 of 3 informative seqs, no ATE regression
    info = [x for x in rows if x["seq"] in INFORMATIVE and x["scale_err"]["online"] is not None]
    wins = [x["seq"] for x in info
            if x["scale_err"]["online"] < x["scale_err"]["frozen"]]
    ate_ok = all(abs(x["ate_ma"] - x["ate_down"]) < 0.05 * max(x["ate_down"], 1e-9)
                 for x in info if x["ate_down"] is not None)
    print(f"\ninformative seqs: {[x['seq'] for x in info]}")
    print(f"online beats frozen: {wins}  ({len(wins)}/{len(info)})")
    print(f"ATE not regressed (<=5%): {ate_ok}")
    verdict = "PASS" if (len(wins) >= 2 and ate_ok) else "KILL/INCOMPLETE"
    print(f"VERDICT: {verdict}")
    json.dump({"b_cal": b_cal, "rows": rows, "online_wins": wins,
               "ate_ok": ate_ok, "verdict": verdict},
              open(f"{P}/outputs/gauge_ladder.json", "w"), indent=2)
    print(f"wrote {P}/outputs/gauge_ladder.json")


if __name__ == "__main__":
    main()
