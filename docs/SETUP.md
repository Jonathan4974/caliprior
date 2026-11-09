# Setup

Tested on Linux with an NVIDIA GPU (CUDA 11.8). Three pieces are needed: the
HI-SLAM2 backend, a Python environment for the SLAM runs, and a CPU environment
for cache building and analysis.

## 1. HI-SLAM2 backend

The scripts import the backend from a local checkout; nothing from it is
redistributed here.

```bash
git clone --recursive https://github.com/Willyzw/HI-SLAM2.git "$HISLAM2_ROOT"
cd "$HISLAM2_ROOT"
git checkout 76c833c
```

Then create the `hislam2` conda environment and build the CUDA extensions:

```bash
conda env create -f /path/to/caliprior/requirements/environment-hislam2.yaml
conda activate hislam2
pip install -r /path/to/caliprior/requirements/requirements-hislam2.txt --no-build-isolation
```

The `setup.py` extensions (`droid_backends`, `lietorch_backends`) have no rpath,
so add the torch library directory to `LD_LIBRARY_PATH` (see
`configs/env_setup.sh`). Rebuild the extensions with `python setup.py build` and
copy the resulting `.so` from `build/lib.*` into `site-packages` by hand; do
**not** `pip install` them, because pip reuses its wheel cache and can keep
stale-architecture builds.

## 2. CPU environment (cache building / analysis)

```bash
python -m venv "$VENV_ROOT"
. "$VENV_ROOT/bin/activate"
pip install -r requirements/requirements-analysis.txt
```

Cache building additionally needs the depth-model packages
(`mapanything`, `transformers` for DA-V2, `depth_anything_3`) and a GPU; see
`docs/UPSTREAM.md`.

## 3. Configure paths

```bash
source setup.sh
```

`setup.sh` exports `PROJECT_ROOT`, `HISLAM2_ROOT`, `CONDA_ROOT`, `VENV_ROOT`,
`DATA_ROOT`, and `HF_HOME`, and derives the per-dataset roots. Override any of
them before sourcing if your layout differs. `code/caliprior_paths.py` reads the
same variables, so the scripts work whether or not you source the file (the
defaults assume this release's layout).

For the SLAM runs, also source the backend environment:

```bash
source configs/env_setup.sh
```

## 4. Data and models

See `docs/DATASETS.md` for the expected dataset layout and `data/` for download
helpers. Model weights are fetched on first use into `$HF_HOME`.

## 5. Verify

```bash
bash scripts/smoke_test.sh
```

This checks imports, path resolution, and that the recorded result files are
present, without running the full pipeline. `scripts/run_pipeline.sh` runs the
end-to-end order.
