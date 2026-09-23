# Run commands

Exact commands behind every row of `REPRO_INDEX.md`. The original runs used a
SLURM cluster; here each step is a plain command you can run directly (wrap in
your own scheduler if needed). Run everything from `PROJECT_ROOT` after

```bash
source setup.sh
source configs/env_setup.sh        # for the SLAM runs
```

Two interpreters are used:

- `$PY_SLAM` = `${CONDA_ROOT}/envs/hislam2/bin/python` (SLAM runs, GPU).
- `$PY_CPU`  = `${VENV_ROOT}/bin/python` (cache building / analysis, GPU for the
  depth models, CPU for the rest).

Cache building needs the depth-model packages and a GPU; see `docs/SETUP.md`.

## 1. Prior caches

```bash
$PY_CPU code/build_ma_cache.py   --out outputs/ma_cache
$PY_CPU code/build_dav2_cache.py --out outputs/dav2_cache
$PY_CPU code/build_da3m_cache.py --out outputs/da3m_cache

# KITTI odometry control at a different working resolution
$PY_CPU code/build_ma_cache.py --out outputs/ma_cache_1120 \
    --only kitti00_800 --longest 1120 --chunk 25

# gate blind-spot probe (perturbed intrinsics)
$PY_CPU code/build_ma_cache.py --out outputs/ma_cache_k120 \
    --only freiburg1_xyz,freiburg1_desk,MH_01_easy \
    --longest 560 --chunk 150 --k-scale 1.20
```

Set `HF_HOME` (see `setup.sh`) so model weights cache locally.

## 2. Baselines, gauge arms, anchors

```bash
# stock vehicle
$PY_SLAM code/run_stock.py --seq MH_01_easy --out outputs/stock_MH_01_easy

# wired-in frozen prior
$PY_SLAM code/run_wired.py --seq MH_01_easy \
    --cache outputs/ma_cache/MH_01_easy.npz --out outputs/wired_MH_01_easy

# gauge anchored to the prior
$PY_SLAM code/run_wired.py --seq MH_01_easy \
    --cache outputs/ma_cache/MH_01_easy.npz --gauge ma --mode stock \
    --out outputs/gaugeMA_MH_01_easy

# prior-free bootstrap (gate warm-up)
$PY_SLAM code/run_wired.py --seq MH_01_easy \
    --cache outputs/ma_cache/MH_01_easy.npz \
    --alpha 0 --gauge fixed --mode stock --out outputs/noprior_MH_01_easy

# sensor-derived mean-depth anchor
$PY_SLAM code/run_wired.py --seq MH_01_easy \
    --cache outputs/ma_cache/MH_01_easy.npz \
    --gauge ma --anchor-mean 2.2750 --mode stock --out outputs/anchorN_MH_01_easy

# local-scale step response
$PY_SLAM code/run_wired.py --seq freiburg1_xyz \
    --cache outputs/ma_cache/freiburg1_xyz.npz \
    --gauge ma --anchor 1.1 --b 1.0 --b2 1.25 --bswitch 400 --mode stock \
    --out outputs/step_freiburg1_xyz
```

KITTI control:

```bash
$PY_SLAM code/run_stock.py --seq kitti00_800 --dataset kitti \
    --workwh "1120 336" --buffer 1200 --out outputs/stock_kitti1120
$PY_SLAM code/run_wired.py --seq kitti00_800 \
    --cache outputs/ma_cache_1120/kitti00_800.npz --out outputs/wired_kitti1120
```

## 3. E1, metric-depth factor arms

```bash
# per sequence, four arms
$PY_SLAM code/run_wired.py --seq $S --cache outputs/ma_cache/$S.npz \
    --mode stock --out outputs/e1_stock_$S
$PY_SLAM code/run_wired.py --seq $S --cache outputs/ma_cache/$S.npz \
    --mode metric --weight const --alpha 1.0 --out outputs/e1_metricconst_$S
$PY_SLAM code/run_wired.py --seq $S --cache outputs/ma_cache/$S.npz \
    --mode metric --weight conf --alpha 1.0 --out outputs/e1_metricconf_$S
$PY_SLAM code/run_wired.py --seq $S --cache outputs/ma_cache/$S.npz \
    --mode metric --weight conf --alpha 1.0 --gauge ma \
    --out outputs/e1_metricadopt_$S
```

Alpha sweep: add `--alpha 0.01|1.0|10.0` under `--mode metric`. Summarise with
`code/summarize_e1.py`.

## 4. E2, sigma ladder and confidence regimes

```bash
# scalar ladder
$PY_SLAM code/run_wired.py --seq $S --cache outputs/ma_cache/$S.npz \
    --mode metric --alpha 1.0 --weight const --out outputs/e2_const_$S
$PY_SLAM code/run_wired.py --seq $S --cache outputs/ma_cache/$S.npz \
    --mode metric --alpha 1.0 --weight conf --out outputs/e2_frozen_$S
$PY_SLAM code/run_wired.py --seq $S --cache outputs/ma_cache/$S.npz \
    --mode metric --alpha 1.0 --weight conf --sigma-adapt --Kc 0.1 \
    --out outputs/e2_adapt_$S

# per-regime (K in 2,3,4)
$PY_SLAM code/run_wired.py --seq $S --cache outputs/ma_cache/$S.npz \
    --mode metric --alpha 1.0 --weight conf --sigma-regime --n-regimes 3 \
    --regime-calib 60 --regime-fit 300 --freeze-at 300 \
    --out outputs/e2reg_regimeK3_$S

# K-sensitivity guard (tight c range)
$PY_SLAM code/run_wired.py --seq $S --cache outputs/ma_cache/$S.npz \
    --mode metric --alpha 1.0 --weight conf --sigma-regime --n-regimes $K \
    --regime-calib 60 --regime-fit 300 --freeze-at 300 \
    --regime-c-lo 0.20 --regime-c-hi 5.0 \
    --out outputs/e2regG_regimeK${K}_$S
```

Tables/figures: `code/regime_nis.py`, `code/guard_compare.py`.

## 5. E3, gate expansion (16 sequences)

For each extra sequence `$S` (the six remaining EuRoC sequences plus the
remaining TUM sequences):

```bash
$PY_SLAM code/run_wired.py --seq $S --cache outputs/ma_cache/$S.npz \
    --alpha 0 --gauge fixed --mode stock --limit 900 --out outputs/noprior_$S
$PY_SLAM code/run_wired.py --seq $S --cache outputs/ma_cache/$S.npz \
    --gauge ma --out outputs/gaugeMA_$S
```

DA-V2 / DA3METRIC gate models use the same two commands with
`--cache outputs/{dav2,da3m}_cache/$S.npz` and outputs `dav2_*` / `da3m_*`.

## 6. Gate, anchors, protocols, figures

```bash
$PY_CPU code/gate_ood.py            # trust gate (dispersion)
$PY_CPU code/gate_robust.py         # gate robustness / bootstrap
$PY_CPU code/gate_model_calib.py    # per-model gate calibration
$PY_CPU code/gate_diag.py           # blind-spot diagnostics

$PY_CPU code/build_anchor.py        # sensor mean-depth anchors
$PY_CPU code/build_traj_anchor.py   # trajectory-scale anchor
$PY_CPU code/summarize_anchors.py

$PY_CPU code/run_gauge_ladder.py    # five-arm ladder
$PY_CPU code/run_protocol.py        # ten-sequence gate + protocol
$PY_CPU code/run_protocol_ate.py    # protocol ATE in metres
$PY_CPU code/observable_b.py        # warm-up-ratio observable
$PY_CPU code/gauge.py               # observable -> correction map
$PY_CPU code/gauge_null.py          # gauge blind spot
$PY_CPU code/gauge_parallax.py      # low-parallax certificate
$PY_CPU code/lowparallax.py
$PY_CPU code/local_drift.py

$PY_CPU code/make_figs.py           # -> figures/caliprior_summary.png
$PY_CPU code/make_figs2.py          # -> figures/caliprior_tier12.png
$PY_CPU code/make_figs3.py          # -> figures/caliprior_mechanisms.png
```
