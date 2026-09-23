#!/usr/bin/env python3
"""Summarise the E1 metric-factor arms against the existing baselines.

Reads metrics.json from outputs/e1_<arm>_<seq>/ and outputs/gaugeMA_<seq>/, and
prints umeyama scale (est->GT) and its |error| per sequence. Baseline columns:
  stock        : the normalized vehicle (no prior metric scale)
  gaugeMA      : prior sets the gauge (the "adopt" result)
Arms:
  metricconst  : metric residual, scalar alpha
  metricconf   : metric residual, per-pixel teacher-confidence weighting
  metricadopt  : metric residual + gauge anchored to the teacher
"""
import json
import os

import caliprior_paths as _CP  # path resolution (env-overridable)
OUT = f"{_CP.PROJECT_ROOT}/outputs"

SEQS = ["freiburg1_xyz", "freiburg1_desk", "freiburg2_xyz", "freiburg2_desk",
        "freiburg3_long_office_household", "MH_01_easy", "MH_02_easy",
        "V1_01_easy", "V1_02_medium", "V2_01_easy"]


def load(d):
    f = os.path.join(OUT, d, "metrics.json")
    if not os.path.exists(f):
        return None
    try:
        return json.load(open(f))
    except Exception:
        return None


def scale(d):
    m = load(d)
    if not m:
        return None
    return m.get("umeyama_scale_est2gt")


def fmt(x):
    return "  --  " if x is None else f"{x:6.3f}"


def main():
    cols = [
        ("stock",       lambda s: f"e1_stock_{s}"),
        ("gaugeMA",     lambda s: f"gaugeMA_{s}"),
        ("metricconst", lambda s: f"e1_metricconst_{s}"),
        ("metricconf",  lambda s: f"e1_metricconf_{s}"),
        ("metricadopt", lambda s: f"e1_metricadopt_{s}"),
    ]
    hdr = f"{'sequence':<34}" + "".join(f"{c[0]:>12}" for c in cols)
    print(hdr)
    print("-" * len(hdr))
    for s in SEQS:
        row = f"{s:<34}" + "".join(f"{fmt(scale(c[1](s))):>12}" for c in cols)
        print(row)

    # |scale-1| mean over sequences where a column is present
    print("-" * len(hdr))
    means = {}
    for name, f in cols:
        vals = [abs(scale(f(s)) - 1.0) for s in SEQS if scale(f(s)) is not None]
        means[name] = (sum(vals) / len(vals)) if vals else None
    print(f"{'mean |scale-1|':<34}" +
          "".join(f"{('  --  ' if means[n] is None else f'{means[n]:6.3f}'):>12}"
                  for n, _ in cols))


if __name__ == "__main__":
    main()
