# E1, metric-depth factor arm (build brief)

Status: **built, run, and analysed**. The paper's \S5.9 carries the table and the
underlying numbers are in `results/`. Summary: the metric
residual does **not** pin the global gauge (metric-const/metric-conf ≈ stock;
metric+anchor ≈ gaugeMA), per-pixel confidence changes shape not scale, and the
local lever is **live** under the metric factor (25% step pass-through 0.065 vs
stock's 0.0016, ~40x). This separates the universal gauge-null result from the
stock-vehicle normalization artifact.

Why E1 exists: the RA-L upgrade plan's biggest exposure is that the "no free
signal" and "no local lever" results were measured on the *stock vehicle*, which
median-normalizes the prior at every keyframe. E1 wires the prior as a true
metric-depth factor so the universal finding (gauge null direction) can be
separated from the vehicle artifact (median normalization discards scale) **by
experiment instead of by rewritten prose**.

## What changed

| file | change |
|---|---|
| `jdsa_modes.py` | new `mode="metric"` → `JDSA_metric()`; new `weight="const" | "conf"`; `set_conf()`, `_pixel_weights()`; `configure(..., weight=)` |
| `run_wired.py` | `--mode metric`, `--weight {const,conf}`; adapter stashes the cached confidence tile per served frame; `weight` logged in `metrics.json` |

`JDSA_metric` differs from stock `JDSA` in three ways:

1. **No scale state.** The scale block is zeroed and `dscales` is never
   updated, so the served prior enters at its own (metric) scale.
2. **No free `b`.** The residual is `disps - disps_prior` at face value
   (`b=1`), i.e. the prior's metric scale is trusted, not re-fitted. This is the
   arm that tests whether a metric residual pins the gauge.
3. **Optional per-pixel σ** (`--weight conf`): the residual is weighted by the
   teacher's cached confidence, subsampled `[3::8, 3::8]` to match the grid
   `disps_prior` lives on, mean-normalised so `alpha` stays comparable to the
   `const` arm.

## Smoke verification (not a result)

```
run_wired.py --seq freiburg1_xyz --cache outputs/ma_cache/freiburg1_xyz.npz \
  --config <euroc config with position_lr_max_steps=150> \
  --mode metric --weight conf --limit 120 --out outputs/smoke_metric_conf_xyz
```

Completed: `n_kf=17`, `jdsa_calls=52`, `ate_rmse_m=0.0138`,
`umeyama_scale=0.477`, `b_trace` records `mean|rd|=0.474, weight="conf"`.
No traceback. (Scale stays ≈0.48 because `mono_depth_alpha` is still the stock
`0.01` and the gauge is still fixed at 2.0 over 120 frames, see confounds.)

## Arm matrix (full run)

For each of the ten sequences in `outputs/ma_cache/`:

| arm | command sketch | what it isolates |
|---|---|---|
| stock | `--mode stock` (exists) | median-normalized vehicle |
| metric, const | `--mode metric --weight const` | metric factor alone |
| metric, conf | `--mode metric --weight conf` | + per-pixel teacher σ |
| metric + adopt | `--mode metric --weight conf --gauge ma` | prior sets the gauge |
| trajectory anchor | `traj_anchor.json` (exists) | external metric reference |

Expected: the metric factor breaks the joint gauge, so the reconstruction adopts
the teacher's scale (TUM ≈ 0.90, matching the `--gauge ma` results); the
`conf`-weighted arm should change *shape/consistency*, not the global scale.

## Kill gate and the §0 trap

The plan's warning is load-bearing: **do not read an improved NIS as recovered
scale.** With a metric factor, depth-channel innovations stay small even at a
30% scale error, because the bias is absorbed into the gauge (Proposition 1 /
Corollary 1). E1 is a *validity* test of the metric factor; the σ-ladder and NIS
accounting are E2 and are certified only on the observable half.

## Confounds to hold fixed

- `mono_depth_alpha` sets the prior's leverage. The smoke used the
  stock `0.01`, which is too weak to move scale in 120 frames, the full run
  must sweep `--alpha` (e.g. 0.01 / 1 / 10) and report it, or a flat result is
  uninterpretable.
- `--gauge ma` vs stock `normalize()` (mean depth 2.0) is a separate axis; do
  not conflate "the factor pins the gauge" with "we anchored the gauge."
- The confidence tile must use the *same* `[3::8, 3::8]` subsampling as
  `depth_video`; changing it silently misaligns σ with the residual.

## Status of the E1 runs

1. **Done**: full ten-sequence run across the arm matrix (one GS-backend OOM on
   MH_02 adopt, re-run with `--mem=160G`). See `scripts/run_pipeline.sh`.
2. **Done**: local-step test under `--mode metric`: pass-through 0.065 vs stock
   0.0016.
3. **Done**: the outcome is reported in \S5.9 "What is universal, and what
   belongs to the stock vehicle."
4. Optional: a metric arm that bypasses the initialization `normalize()` entirely
   (rather than anchoring to the teacher via `--gauge ma`) to confirm that the
   init gauge constant, not only the per-keyframe grid, is what blocks the
   metric residual. `metric + anchor` already largely covers this; treat as
   confirmation, not required.
