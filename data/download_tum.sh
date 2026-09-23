#!/usr/bin/env bash
# Download and extract the TUM RGB-D sequences used in the paper.
set -euo pipefail
: "${TUM_ROOT:?source setup.sh first (TUM_ROOT unset)}"
mkdir -p "$TUM_ROOT"
BASE="https://vision.in.tum.de/rgbd/dataset"

SEQS=(
  "freiburg1/rgbd_dataset_freiburg1_xyz"
  "freiburg1/rgbd_dataset_freiburg1_desk"
  "freiburg2/rgbd_dataset_freiburg2_xyz"
  "freiburg2/rgbd_dataset_freiburg2_desk"
  "freiburg3/rgbd_dataset_freiburg3_long_office_household"
)

for s in "${SEQS[@]}"; do
  name="$(basename "$s")"
  if [ -d "$TUM_ROOT/$name" ]; then
    echo "have $name"
    continue
  fi
  echo "downloading $name"
  curl -L --fail -o "$TUM_ROOT/$name.tgz" "$BASE/$s.tgz"
  tar -xzf "$TUM_ROOT/$name.tgz" -C "$TUM_ROOT"
  rm -f "$TUM_ROOT/$name.tgz"
done
echo "TUM RGB-D ready under $TUM_ROOT"
