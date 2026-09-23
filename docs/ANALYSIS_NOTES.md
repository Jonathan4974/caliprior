# Additional analysis notes

Per-sequence and secondary results kept with the code. This file is repository
documentation, not a submission supplement: the letter is self-contained, and
the five-arm ladder, the full metric-factor arm ladder, the per-regime sigma
table, and the constant-bias table are also in the paper. What remains here is
supporting detail (warm-up-ratio correlation, gate labels) plus the same tables
for completeness.

## S1. Warm-up ratio does not predict the anchor

The candidate free signal is the warm-up ratio
`r_init = median(D_teacher / D_backend)` from a prior-free run at the fixed gauge.
Its average versus the oracle anchor:

| sequence | r_init | oracle anchor | domain |
|---|---|---|---|
| f1_xyz | 0.478 | 0.907 | TUM |
| f1_desk | 0.512 | 1.063 | TUM |
| MH_01 | 0.734 | 0.132 | EuRoC |
| V1_02 | 0.189 | 0.210 | EuRoC |

The simple physical model (`r = beta/gamma`, `anchor = 1/beta`) predicts the
anchor should fall as `r` rises, but only if the reference gauge is the same in
every domain. It is not: the pairwise signs agree three times and disagree three
times, so there is no usable monotone relation, and a fit calibrated on one
domain extrapolates 18x wrong on MH_01.

## S2. Five-arm ladder (scale error)

Arms: trivial fixed gauge, in-domain, frozen calibration, online adaptation,
oracle.

| sequence | fixed gauge | in-domain | frozen | online | oracle |
|---|---|---|---|---|---|
| f1_xyz | 0.496 | 0.093 | 0.107 | 0.177 | 0.000 |
| f1_desk | 0.403 | 0.063 | 0.047 | 0.130 | 0.000 |
| MH_01 | 0.661 | 0.868 | 0.870 | 0.682 | 0.000 |
| V1_02 | 0.808 | 0.790 | 0.794 | 0.977 | 0.000 |

The online arm beats the frozen arm on only one of three informative sequences.
We rejected the online-transfer route.

## S3. Full metric-factor arm ladder (umeyama scale per sequence)

| arm | f1_xyz | f1_desk | fr2_xyz | fr2_desk | fr3_long | MH_01 | MH_02 | V1_01 | V1_02 | V2_01 | mean |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stock | 0.505 | 0.597 | 0.680 | 1.000 | 1.132 | 0.339 | 0.231 | 1.065 | 0.192 | 1.439 | 0.409 |
| metric, scalar | 0.505 | 0.577 | 0.748 | 1.088 | 1.450 | 0.270 | 0.271 | 1.333 | 0.212 | 1.881 | 0.517 |
| metric, confidence | 0.499 | 0.538 | 0.750 | 1.090 | 1.432 | 0.353 | 0.265 | 1.351 | 0.242 | 2.008 | 0.523 |
| metric + anchor | 0.896 | 0.964 | 0.708 | 0.823 | 1.288 | 0.133 | 0.125 | 1.403 | 0.036 | 1.607 | 0.462 |
| anchor only | 0.907 | 1.063 | 0.641 | 0.808 | 1.005 | 0.132 | 0.097 | 1.107 | 0.210 | 1.203 | 0.358 |

Last column is mean `|scale - 1|`. The metric residual does not make the global
gauge observable: the scalar and confidence-weighted arms sit at or slightly
worse than stock and never approach the anchored result. Raising the prior weight
`alpha` from 0.01 to 10 reproduces the `0.44 -> 0.50` shift but never reaches the
teacher's own 0.90.

## S4. Per-regime sigma is fragile

Scalar versus `K=3` confidence-quantile regimes, depth-channel NIS (target 1),
median over solves.

| sequence | scalar med. | regime med. | scalar NIS_k | regime NIS_k |
|---|---|---|---|---|
| f1_xyz | 1.380 | 1.315 | [0.60, 2.00, 2.78] | [1.33, 1.34, 1.24] |
| f1_desk | 0.469 | 0.707 | [0.12, 0.37, 1.05] | [0.50, 0.36, 0.67] |
| MH_01 | 0.033 | 0.504 | [0.01, 0.01, 0.04] | [0.04, 0.07, 0.67] |
| V1_02 | 0.025 | 0.615 | [0.01, 0.01, 0.03] | [0.16, 0.17, 0.64] |

Per-regime sigma is closer to 1 than the scalar on all three shifted sequences
and reduces the mean per-regime deviation on all four. It is **not robust**: the
pre-registered K-sensitivity (`K in {2,4}`) fails. Unguarded, K2/K4 hung in the
backend on both shifted sequences, and the surviving K3 pins the guard floor
(`c_k = 0.05`) and diverges on 11% of held-out MH_01 solves. Re-running under a
stability guard (`c_lo = 0.2`, time-capped) makes every arm converge but erases
the calibration gain (MH_01 K3 median 0.032, V1_02 K3 median 0.019). Only the
scalar adaptation is reported as robust. The calibration is also domain-specific:
freezing the TUM-fitted scalar or per-regime sigma and applying it to EuRoC
leaves held-out NIS at 0.03-0.09, far from 1.

## S5. The gate is blind to a constant bias

Perturbing the teacher intrinsics (`fx, fy x 1.20`) induces a constant ~5%
teacher depth bias (the model partly compensates the focal change).

| sequence | gate | dispersion (real -> perturbed) | adopted scale (real -> perturbed) |
|---|---|---|---|
| f1_xyz | trust | 0.012 -> 0.019 | 0.907 -> 0.877 |
| f1_desk | trust | 0.022 -> 0.026 | 1.063 -> 1.022 |
| MH_01 | reject | 0.240 -> 0.260 | 0.132 -> 0.120 |

`r_init` is a ratio against the backend, so a constant multiplicative teacher
bias rescales it uniformly and leaves its dispersion (the gate statistic)
unchanged. The gate detects teacher inconsistency/OOD, not a globally biased
teacher; a constant bias must be caught by the anchor. Stated as a designed
limitation.

## S6. Gate labels, sixteen sequences

Out-of-distribution label defined purely from metric truth
(`|scale - 1| > 0.30`), seven gate-positive:

- **trust (9):** f1_xyz, f1_desk, fr2_desk, fr3_long_office, V1_01, V2_01,
  V1_03, V2_02, V2_03
- **out of distribution (7):** fr2_xyz, MH_01, MH_02, V1_02, MH_03, MH_04, MH_05

The label is not the domain: V1_01 and V2_01 are EuRoC yet trustworthy, and
fr2_xyz is TUM yet out of distribution.
