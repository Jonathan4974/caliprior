#!/usr/bin/env bash
# KITTI odometry requires registration and cannot be fetched automatically.
set -euo pipefail
cat <<'EOF'
KITTI odometry is distributed after registration:
  https://www.cvlibs.net/datasets/kitti/eval_odometry.php

Download the odometry ground-truth poses and the color/grey image archives,
then lay them out as:

  $KITTI_ROOT/sequences/00/calib.txt
  $KITTI_ROOT/sequences/00/image_2/*.png        (color)
  $KITTI_GREY_ROOT/00/poses.txt                 (ground truth)
  $KITTI_GREY_ROOT/00/image_0/*.png             (grey)

where $KITTI_ROOT and $KITTI_GREY_ROOT come from setup.sh.
The paper uses sequence 00, first 800 frames.
EOF
