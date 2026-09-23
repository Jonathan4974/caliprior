# Upstream dependencies

Exact versions / identifiers behind the release.

## HI-SLAM2 (SLAM backend)

| item | value |
|---|---|
| repository | https://github.com/Willyzw/HI-SLAM2 |
| pinned commit | `76c833c` |
| license | BSD 3-Clause (Princeton Vision & Learning Lab) |
| imported modules | `hi2`, `depth_video`, `lietorch`, `droid_backends`, `geom.ba`, `geom.chol` |

Build notes: see `docs/SETUP.md` and `configs/env_setup.sh`.

## Foundation-model depth priors

| model | package / hub id | used by |
|---|---|---|
| MapAnything | `mapanything`; `facebook/map-anything-apache` | `code/build_ma_cache.py` |
| Depth Anything V2 | `transformers.AutoModelForDepthEstimation` | `code/build_dav2_cache.py` |
| Depth Anything 3 | `depth_anything_3` | `code/build_da3m_cache.py` |

The cache builders call `from_pretrained(...)`; weights download into `$HF_HOME`
on first use and are not redistributed.

## Python / CUDA

- PyTorch 2.1.2, CUDA 11.8 (`requirements/environment-hislam2.yaml`).
- `torch-scatter`, `simple-knn`, `diff-gaussian-rasterization`,
  `evaluate_3d_reconstruction_lib` (`requirements/requirements-hislam2.txt`).
- Analysis venv: `numpy`, `opencv-python`, `matplotlib`, `pyyaml`
  (`requirements/requirements-analysis.txt`).

## Reproducibility caveats

The paper reports three corrections found during the work: a confidence-tile
keying misalignment (fixed before the reported numbers), a withdrawn
"local lever is live" claim, and a per-regime sigma result reported as fragile
after its pre-registered sensitivity check failed. See `docs/REPRO_INDEX.md`.
