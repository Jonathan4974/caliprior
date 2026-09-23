#!/usr/bin/env bash
# Fast, non-GPU checks that the release is wired correctly.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"
source setup.sh >/dev/null

PY="${VENV_ROOT}/bin/python"
[ -x "$PY" ] || PY="$(command -v python3)"
echo "python: $PY"

fail=0
check () { if eval "$2"; then echo "ok   $1"; else echo "FAIL $1"; fail=1; fi; }

echo "-- path resolution --"
"$PY" - <<'PY'
import sys, os
sys.path.insert(0, os.path.join("code"))
import caliprior_paths as P
for k in ("PROJECT_ROOT", "HISLAM2_ROOT", "DATA_ROOT", "EUROC_ROOT",
          "TUM_ROOT", "KITTI_ROOT", "KITTI_GREY_ROOT", "HF_HOME"):
    print(f"  {k} = {getattr(P, k)}")
assert os.path.isdir(P.PROJECT_ROOT)
PY
check "caliprior_paths resolves" "[ $? -eq 0 ]"

echo "-- syntax --"
check "code compiles" "$PY -m py_compile code/*.py"

echo "-- recorded results --"
for f in results/protocol_ate_16.json results/traj_anchor.json \
         results/gate_ood_16.json results/gate_robust_16.json \
         results/gauge_null.json results/regime_nis.txt; do
  check "results/$(basename "$f")" "[ -e '$f' ]"
done

echo "-- figures / paper --"
for f in figures/caliprior_summary.png figures/caliprior_tier12.png \
         figures/caliprior_mechanisms.png paper/main_ral_8p.pdf; do
  check "$f" "[ -e '$f' ]"
done

echo "-- required docs --"
for f in README.md LICENSE NOTICE CITATION.cff setup.sh \
         docs/SETUP.md docs/UPSTREAM.md docs/DATASETS.md docs/RUNS.md \
         docs/REPRO_INDEX.md; do
  check "$f" "[ -e '$f' ]"
done

if [ "$fail" -eq 0 ]; then
  echo "SMOKE TEST PASSED"
else
  echo "SMOKE TEST FAILED"
fi
exit "$fail"
