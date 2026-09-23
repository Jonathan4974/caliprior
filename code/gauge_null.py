#!/usr/bin/env python3
"""P3.1: the monocular similarity gauge is null -- analytic lemma + numeric check.

Lemma. Let u_i be a pixel in camera i with depth z_i > 0, K the intrinsics, and
(R, t) the relative pose i->j. Backproject X_i = z_i K^-1 [u_i,1] and project
X_j = R X_i + t. Apply the similarity rescale (t, z) -> (lambda t, lambda z)
with R fixed. Then X_j -> lambda X_j, so u_j = pi(K X_j) is UNCHANGED while the
induced depth scales by lambda. Hence the reprojection residual is invariant to
the joint (depth, translation) scale -- exactly the gauge `normalize()` fixes in
`depth_video.py:159-168`. A depth-only rescale (t fixed) is NOT invariant.

Controls: (i) joint rescale => projection delta ~ 0; (ii) depth-only rescale =>
delta > 0 (when t != 0); (iii) rotation perturbation => delta > 0.

This is the measured statement made rigorous. Emits outputs/gauge_null.json.
"""
import json
import numpy as np

import caliprior_paths as _CP  # path resolution (env-overridable)
P = _CP.PROJECT_ROOT


def project(K, R, t, u, z):
    Kinv = np.linalg.inv(K)
    uv = np.concatenate([u, np.ones((len(u), 1))], 1)      # (N,3)
    Xi = z[:, None] * (Kinv @ uv.T).T                       # (N,3)
    Xj = (R @ Xi.T).T + t[None, :]                          # (N,3)
    pj = (K @ Xj.T).T                                       # (N,3)
    return pj[:, :2] / pj[:, 2:3], pj[:, 2]


def rot(axis, ang):
    a = np.array(axis, float)
    a /= np.linalg.norm(a)
    Kx = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(ang) * Kx + (1 - np.cos(ang)) * (Kx @ Kx)


def main():
    rng = np.random.default_rng(0)
    fx, fy, cx, cy = 460.0, 460.0, 320.0, 240.0
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    h, w = 480, 640
    u = np.stack([rng.uniform(20, w - 20, 400), rng.uniform(20, h - 20, 400)], 1)
    z = rng.uniform(0.5, 8.0, len(u))
    R = rot([0.3, 1.0, -0.2], 0.25)
    t = np.array([0.4, -0.15, 0.08])

    u0, zj0 = project(K, R, t, u, z)

    lam = 3.7
    # (i) joint gauge: t -> lam t, z -> lam z  (the null direction)
    u_joint, zj_joint = project(K, R, lam * t, u, lam * z)
    d_joint = float(np.abs(u_joint - u0).max())

    # (ii) depth-only rescale (t fixed): NOT a gauge
    u_depth, _ = project(K, R, t, u, lam * z)
    d_depth = float(np.abs(u_depth - u0).max())

    # (iii) rotation perturbation control
    u_rot, _ = project(K, rot([1, 0, 0], 0.01) @ R, t, u, z)
    d_rot = float(np.abs(u_rot - u0).max())

    # residual impact: fixed correspondence (u0), reproject gauge-scaled geometry
    r_joint = float(np.abs((u_joint - u0)).mean())
    r_depth = float(np.abs((u_depth - u0)).mean())
    ok = d_joint < 1e-8 and d_depth > 1e-2 and d_rot > 1e-2

    out = {"lambda": lam, "n_points": int(len(u)),
           "joint_projection_delta_max_px": d_joint,
           "depth_only_projection_delta_max_px": d_depth,
           "rotation_perturbation_delta_max_px": d_rot,
           "joint_residual_mean_px": r_joint,
           "depth_only_residual_mean_px": r_depth,
           "gauge_is_null": bool(d_joint < 1e-8),
           "depth_only_is_NOT_null": bool(d_depth > 1e-2),
           "lemma_confirmed": bool(ok),
           "depth_scaling_under_gauge": float((zj_joint / zj0).mean())}
    json.dump(out, open(f"{P}/outputs/gauge_null.json", "w"), indent=2)

    print("== P3.1 gauge-null check (mono similarity gauge) ==")
    print(f"  joint (t,z)->lam*(t,z):   max proj delta = {d_joint:.2e} px  [null]")
    print(f"  depth-only z->lam*z:      max proj delta = {d_depth:.4f} px  [NOT null]")
    print(f"  rotation +0.01 rad:       max proj delta = {d_rot:.4f} px  [control]")
    print(f"  induced depth ratio under gauge = {out['depth_scaling_under_gauge']:.3f} (lam={lam})")
    print(f"  lemma_confirmed = {out['lemma_confirmed']}")
    print(f"  wrote {P}/outputs/gauge_null.json")


if __name__ == "__main__":
    main()
