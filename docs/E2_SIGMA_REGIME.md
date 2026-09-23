# E2-sigma-regime, pre-registration

Status: **written before any run.** Do not edit hypotheses/kill criteria after
the first job is submitted; append results separately.

## Motivation (paper \S5.10)

The scalar variance-scale `c` on the confidence-derived precision is adapted by
covariance matching so the depth-channel NIS -> 1. It works under moderate shift
but **saturates**: on the extreme-OOD MH_01 the adapted `c_final = 0.200 =
c_min` while NIS only reaches 0.146. A single global temperature cannot certify
consistency when the miscalibration is regime-dependent.

Question: **is the FM teacher's confidence miscalibration a single global
temperature, or is it regime-structured?**

## Hypotheses

- **H1 (regime-structured).** A per-confidence-regime variance scale `c_k`
  reaches post-freeze held-out NIS closer to 1 than the scalar on >= 2 of 3
  shifted sequences (EuRoC is a domain shift for the FM prior).
- **H2 (reduced spread).** Per-regime adaptation reduces the cross-regime
  spread of `NIS_k` relative to the scalar treatment: the scalar flattens one
  aggregate number while regimes disagree.
- **H3 (transfer).** Scales `c_k` fit on TUM, frozen and applied to EuRoC,
  give lower held-out NIS than the scalar's single `c` transferred the same way.

Null / kill: if H1 and H2 both fail, the scalar is adequate; we report that and
keep the clamp as a documented limit. **No re-framing after seeing numbers.**

## Model (frozen)

- Regimes: K confidence-quantile bins, edges computed once from the valid prior
  pixels of the first `F` frames (calibration window), then frozen. Default
  K=3, F=300 solves. Sensitivity K in {2,4}.
- Weights: `W_i = conf_i / c_{r(i)}`, mean-normalised over valid prior pixels
  (keeps `alpha` comparable), `a = alpha * W`. Higher confidence => lower sigma.
- Update: `c_k <- c_k * (1 + Kc*(NIS_k - 1))`, shared `Kc=0.1`.
- Guard: `c_k` in [0.05, 20], wide and **reported** (never silently clamped).

## Validity guard (why this is not vacuous)

With K scales fit to K regime-NIS values, aggregate NIS -> 1 is trivially
achievable. Therefore the claim rests **only** on:

1. **Temporal hold-out.** Adapt on the fit window, freeze `c_k`, then measure
   per-regime and aggregate NIS on all later frames.
2. **Cross-sequence transfer.** Fit `c_k` on TUM, freeze, apply to EuRoC.
3. **Cross-regime spread** of `NIS_k` (scalar vs regime).

Aggregate in-sample NIS is reported for completeness and is **not** evidence.

## Arms x sequences

- Arms: `const`, `frozen` (conf, no adapt), `scalar-adapt`, `regime-K2`,
  `regime-K3`, `regime-K4`.
- Sequences: `freiburg1_xyz`, `freiburg1_desk`, `MH_01_easy`, `V1_02_medium`
  (the four E2 sequences, for comparability).
- Transfer runs: apply the frozen TUM `c_k` to `MH_01_easy` and `V1_02_medium`
  (and scalar `c` for the baseline), via `--regime-from`.

## Kill criterion (pre-registered)

Adopt per-regime **iff** BOTH:

- (a) post-freeze held-out aggregate NIS is closer to 1 than the scalar arm on
  at least 2 of the 3 shifted sequences (f1_desk, MH_01, V1_02; f1_xyz is the
  in-domain anchor), AND
- (b) mean cross-regime spread of `NIS_k` is lower than the scalar arm.

Otherwise: scalar adequate; clamp limit stands.

## Deliverables

`outputs/e2reg_*` metrics + `regime_nis.py` figure/table -> `results/regime_nis.txt`
-> paper \S5.10.
