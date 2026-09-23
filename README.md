# CaliPrior: code and evidence release

Code, data, figures, and paper sources accompanying the paper *"What Metric
Scale Can a Frozen Foundation-Model Depth Prior Give Monocular SLAM?"*
(RA-L submission). It is a code-and-evidence release, not a supplement: the
paper is self-contained without it.

## What the paper claims

A frozen foundation-model (FM) metric-depth prior does **not** transfer metric
scale to monocular SLAM on its own. The global depth rescale a metric prior would
supply is an exact blind spot of the monocular factor graph, so no residual and
no online estimator can observe it. The fix is a GT-free **trust gate** on the
dispersion of a short prior-free warm-up ratio plus a one-shot **trajectory-scale
anchor** from an external metric sensor.

## Layout

```
code/          analysis and experiment scripts (Python)
configs/       environment and vehicle configuration
docs/          setup, upstream pin, dataset layout, run commands, repro index
figures/       paper figures (caliprior_{summary,tier12,mechanisms}.png)
results/       headline result JSONs used by the paper
paper/         LaTeX sources and prebuilt PDFs for the three build variants
data/          dataset download helpers
scripts/       end-to-end pipeline and smoke test
```

## Quick start

```bash
source setup.sh          # exports PROJECT_ROOT, HISLAM2_ROOT, ... (see docs/SETUP.md)
bash scripts/smoke_test.sh
bash scripts/run_pipeline.sh MH_01_easy   # minimal end-to-end, needs GPU
```

Full installation (HI-SLAM2 backend at the pinned commit, conda + CPU envs,
CUDA extensions) is in `docs/SETUP.md`. Dataset layout is in
`docs/DATASETS.md`; the exact command behind every result is in `docs/RUNS.md`;
the result-to-script mapping is in `docs/REPRO_INDEX.md`.

## Headline numbers

| quantity | value |
|---|---|
| gauge blind spot (reprojection change under 10% depth rescale) | `3.4e-13` px |
| trust gate AUROC, MapAnything (16 sequences) | 0.968 |
| network-internal confidence AUROC | 0.44 |
| trajectory anchor mean scale error (10 sequences) | 0.024 |
| depth-mean anchor mean scale error | 0.306 |
| protocol ATE, teacher-only -> gate+anchor (10 sequences) | 5.94 m -> 0.11 m |
| protocol ATE, teacher-only -> gate+anchor (16 sequences) | 14.7 m -> 0.33 m |

## Corrections and negative results

The release documents three corrections found during the work, all reflected in
the reported numbers: a confidence-tile keying misalignment (fixed before the
reported runs), a withdrawn "local lever is live" claim, and a per-regime sigma
result reported as fragile after its pre-registered sensitivity check failed.
See `docs/REPRO_INDEX.md`.

## License and attribution

BSD 3-Clause (`LICENSE`). The release depends on, but does not redistribute,
HI-SLAM2 (BSD 3-Clause) and the foundation-model depth priors; see `NOTICE`.
Citation metadata is in `CITATION.cff` (placeholders to be completed).
