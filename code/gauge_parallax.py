#!/usr/bin/env python3
"""T2.6 (certificate): depth-scale reprojection sensitivity vs translation.

At zero translation, R X_i with X_i = z K^-1 u is invariant to z -> lam*z under
projection (both scale together), so a depth-scale-only perturbation is ALSO a
gauge: scale is completely unobservable. As |t| grows the depth-scale direction
leaves the null space and the residual grows. This quantifies the low-parallax
degeneracy that holds at exactly t=0.

Sweeps |t| and reports the max reprojection delta for a 10% depth-scale error
(lam=1.1), plus the parallax angle atan(|t|/mean_depth). Emits
outputs/gauge_parallax.json (+ png). CPU only.
"""
import json
import numpy as np
import gauge_null as GN

import caliprior_paths as _CP  # path resolution (env-overridable)
P = _CP.PROJECT_ROOT


def main():
    rng = np.random.default_rng(0)
    fx, fy, cx, cy = 460.0, 460.0, 320.0, 240.0
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    h, w = 480, 640
    u = np.stack([rng.uniform(20, w - 20, 400), rng.uniform(20, h - 20, 400)], 1)
    z = rng.uniform(0.5, 8.0, len(u))
    R = GN.rot([0.3, 1.0, -0.2], 0.25)
    d = np.array([0.4, -0.15, 0.08]); d = d / np.linalg.norm(d)

    u0, _ = GN.project(K, R, d * 0.5, u, z)
    lam = 1.1
    mags = [0.0, 1e-4, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0]
    rows = []
    for m in mags:
        t = d * m
        uref, _ = GN.project(K, R, t, u, z)        # same pose, unscaled depth
        ud, _ = GN.project(K, R, t, u, lam * z)    # same pose, 10% depth error
        dd = float(np.abs(ud - uref).max())
        par = float(np.arctan(m / z.mean()))
        rows.append({"baseline": m, "parallax_rad": par,
                     "depth_only_delta_px": dd})
        print(f"  |t|={m:7.4f}  parallax={par*180/np.pi:6.2f} deg  "
              f"depth-only 10% delta={dd:8.4f} px")

    xy = np.array([[r["baseline"], r["depth_only_delta_px"]] for r in rows])
    out = {"lambda": lam, "rows": rows,
           "zero_baseline_delta_px": rows[0]["depth_only_delta_px"],
           "baseline_075_delta_px": float(
               np.interp(0.75, xy[:, 0], xy[:, 1])),
           "monotone_increasing": bool(np.all(np.diff(xy[:, 1]) >= -1e-9))}
    json.dump(out, open(f"{P}/outputs/gauge_parallax.json", "w"), indent=2)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4.6, 3.2))
        ax.plot(xy[:, 0], xy[:, 1], "o-")
        ax.set_xscale("symlog", linthresh=1e-4)
        ax.set_xlabel("translation |t| (a.u.)")
        ax.set_ylabel("max reprojection delta, 10% depth scale (px)")
        ax.set_title("depth-scale observability vs parallax")
        fig.tight_layout(); fig.savefig(f"{P}/outputs/gauge_parallax.png", dpi=150)
        print(f"wrote {P}/outputs/gauge_parallax.png")
    except Exception as e:
        print(f"(no png: {e})")
    print(f"zero-baseline delta = {out['zero_baseline_delta_px']:.2e} px; "
          f"monotone = {out['monotone_increasing']}")
    print(f"wrote {P}/outputs/gauge_parallax.json")


if __name__ == "__main__":
    main()
