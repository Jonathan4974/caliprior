#!/usr/bin/env bash
# CaliPrior release - environment configuration.
#
# Source this file (do not execute it):
#     source setup.sh
#
# Every variable below can be overridden before sourcing, e.g.
#     HISLAM2_ROOT=/opt/HI-SLAM2 DATA_ROOT=/data source setup.sh
# The defaults assume the recommended layout (see docs/SETUP.md).

_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${PROJECT_ROOT:=$_HERE}"
: "${HISLAM2_ROOT:=$HOME/HI-SLAM2}"
: "${CONDA_ROOT:=$HOME/miniconda3}"
: "${VENV_ROOT:=$PROJECT_ROOT/.venv}"
: "${DATA_ROOT:=$HOME/datasets}"
: "${HF_HOME:=$HOME/.cache/huggingface}"

# Dataset roots derive from DATA_ROOT unless set explicitly.
: "${EUROC_ROOT:=$DATA_ROOT/euroc}"
: "${TUM_ROOT:=$DATA_ROOT/tum_rgbd_benchmark}"
: "${KITTI_ROOT:=$DATA_ROOT/kitti_odom_color/data_odometry_color/dataset}"
: "${KITTI_GREY_ROOT:=$DATA_ROOT/kitti_odom_grey/sequences}"

export PROJECT_ROOT HISLAM2_ROOT CONDA_ROOT VENV_ROOT DATA_ROOT HF_HOME
export EUROC_ROOT TUM_ROOT KITTI_ROOT KITTI_GREY_ROOT

echo "CaliPrior environment"
echo "  PROJECT_ROOT   = $PROJECT_ROOT"
echo "  HISLAM2_ROOT   = $HISLAM2_ROOT"
echo "  CONDA_ROOT     = $CONDA_ROOT"
echo "  VENV_ROOT      = $VENV_ROOT"
echo "  DATA_ROOT      = $DATA_ROOT"
echo "  HF_HOME        = $HF_HOME"
