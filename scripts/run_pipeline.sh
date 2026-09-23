#!/usr/bin/env bash
# Minimal end-to-end run for one sequence, to confirm the release works.
# The full pipeline (all arms/sequences) is in docs/RUNS.md.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"
source setup.sh
source configs/env_setup.sh

SEQ="${1:-MH_01_easy}"
PY_SLAM="${CONDA_ROOT}/envs/hislam2/bin/python"
PY_CPU="${VENV_ROOT}/bin/python"
CACHE="outputs/ma_cache/${SEQ}.npz"

echo "== 1. build prior cache ($SEQ) =="
$PY_CPU code/build_ma_cache.py --out outputs/ma_cache --only "$SEQ"

echo "== 2. stock baseline =="
$PY_SLAM code/run_stock.py --seq "$SEQ" --out "outputs/stock_${SEQ}"

echo "== 3. wired-in prior, gauge anchored =="
$PY_SLAM code/run_wired.py --seq "$SEQ" --cache "$CACHE" \
  --gauge ma --mode stock --out "outputs/gaugeMA_${SEQ}"

echo "== 4. prior-free bootstrap =="
$PY_SLAM code/run_wired.py --seq "$SEQ" --cache "$CACHE" \
  --alpha 0 --gauge fixed --mode stock --out "outputs/noprior_${SEQ}"

echo "== 5. trust gate =="
$PY_CPU code/gate_ood.py || true

echo "== 6. figures =="
$PY_CPU code/make_figs.py
$PY_CPU code/make_figs2.py
$PY_CPU code/make_figs3.py

echo "done: outputs under $HERE/outputs, figures under $HERE/figures"
