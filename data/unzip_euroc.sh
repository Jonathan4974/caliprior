#!/usr/bin/env bash
# Extract EuRoC MAV sequences from pre-downloaded room-level ZIPs.
# Inner <seq>.zip contains mav0/ at its root; extract it into $EUROC_ROOT/<seq>/.
set -euo pipefail
: "${EUROC_ROOT:?source setup.sh first (EUROC_ROOT unset)}"
mkdir -p "$EUROC_ROOT"

ROOM_ZIPS=(
  "machine_hall"
  "vicon_room1"
  "vicon_room2"
)

declare -A ROOM_SEQS=(
  ["machine_hall"]="MH_01_easy MH_02_easy MH_03_medium MH_04_difficult MH_05_difficult"
  ["vicon_room1"]="V1_01_easy V1_02_medium V1_03_difficult"
  ["vicon_room2"]="V2_01_easy V2_02_medium V2_03_difficult"
)

for room in "${ROOM_ZIPS[@]}"; do
  room_zip="$EUROC_ROOT/$room.zip"
  if [ ! -f "$room_zip" ]; then
    echo "WARNING: $room_zip not found, skipping $room"
    continue
  fi

  echo "extracting outer $room.zip"
  unzip -q -o "$room_zip" -d "$EUROC_ROOT"

  for seq in ${ROOM_SEQS[$room]}; do
    if [ -d "$EUROC_ROOT/$seq/mav0" ]; then
      echo "have $seq (mav0 present)"
      continue
    fi
    inner_zip="$EUROC_ROOT/$room/$seq/$seq.zip"
    if [ ! -f "$inner_zip" ]; then
      echo "WARNING: $inner_zip not found, skipping $seq"
      continue
    fi

    echo "extracting inner $seq.zip -> $EUROC_ROOT/$seq/"
    mkdir -p "$EUROC_ROOT/$seq"
    unzip -q -o "$inner_zip" -d "$EUROC_ROOT/$seq"
    rm -rf "$EUROC_ROOT/$seq/__MACOSX"
  done

  for seq in ${ROOM_SEQS[$room]}; do
    rm -f  "$EUROC_ROOT/$room/$seq/$seq.bag"
    rm -f  "$EUROC_ROOT/$room/$seq/$seq.zip"
    rm -rf "$EUROC_ROOT/$room/$seq/__MACOSX"
    rm -f  "$EUROC_ROOT/$room/$seq/.DS_Store" "$EUROC_ROOT/$room/$seq/._.DS_Store"
  done
  rm -rf "$EUROC_ROOT/$room/__MACOSX"
  rm -f  "$EUROC_ROOT/$room/.DS_Store" "$EUROC_ROOT/$room/._.DS_Store"
  rmdir  "$EUROC_ROOT/$room"/* 2>/dev/null || true
  rmdir  "$EUROC_ROOT/$room"   2>/dev/null || true
done

echo "EuRoC MAV ready under $EUROC_ROOT"