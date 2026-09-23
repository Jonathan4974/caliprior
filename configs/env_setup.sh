#!/usr/bin/env bash
# Shared environment for the SLAM runs (the `hislam2` conda environment).
#
#   source setup.sh && source configs/env_setup.sh
#
# Cluster-specific module loads (lmod/module) from the original runs are not
# included; load your own compiler/CUDA toolchain before the extension build if
# needed (see docs/SETUP.md).

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate hislam2

# The setup.py-built extensions (droid_backends, lietorch_backends) lack an
# rpath, so the torch lib directory must be on LD_LIBRARY_PATH.
export LD_LIBRARY_PATH="${CONDA_ROOT}/envs/hislam2/lib/python3.11/site-packages/torch/lib:${LD_LIBRARY_PATH}"

# SASS coverage sm_61 (P6000/Titan) .. sm_86 + PTX JIT for 89/90/120.
export TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX"
export PYTHONUNBUFFERED=1

# NOTE: never `pip install` the CUDA extensions -- pip reuses its wheel cache
# and can silently keep stale-arch builds. Rebuild with `python setup.py build`
# in the extension directory and copy the resulting .so from build/lib.* into
# site-packages by hand.
