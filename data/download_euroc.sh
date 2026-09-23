#!/usr/bin/env bash
# Download and extract the EuRoC MAV sequences used in the paper.
set -euo pipefail
: "${EUROC_ROOT:?source setup.sh first (EUROC_ROOT unset)}"
mkdir -p "$EUROC_ROOT"
BASE="http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset"

SEQS=(
  "machine_hall/MH_01_easy"
  "machine_hall/MH_02_easy"
  "machine_hall/MH_03_medium"
  "machine_hall/MH_04_difficult"
  "machine_hall/MH_05_difficult"
  "vicon_room1/V1_01_easy"
  "vicon_room1/V1_02_medium"
  "vicon_room1/V1_03_difficult"
  "vicon_room2/V2_01_easy"
  "vicon_room2/V2_02_medium"
  "vicon_room2/V2_03_difficult"
)

for s in "${SEQS[@]}"; do
  name="$(basename "$s")"
  if [ -d "$EUROC_ROOT/$name" ]; then
    echo "have $name"
    continue
  fi
  echo "downloading $name"
  curl -L --fail -o "$EUROC_ROOT/$name.zip" "$BASE/$s/$name.zip"
  unzip -q "$EUROC_ROOT/$name.zip" -d "$EUROC_ROOT"
  rm -f "$EUROC_ROOT/$name.zip"
done
echo "EuRoC MAV ready under $EUROC_ROOT"
