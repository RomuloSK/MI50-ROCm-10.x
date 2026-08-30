#!/usr/bin/env bash
set -euo pipefail

# Compile/run the HIP graph path for native gfx906.  Graph execution is kept
# separate from the baseline runtime smoke so a graph-only runtime limitation
# is visible without weakening allocation/copy/kernel coverage.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python3 "${ROOT_DIR}/scripts/mi50_features.py" --check-environment
ROCM_PATH="${ROCM_PATH:-/opt/rocm-mi50}"
source "${ROOT_DIR}/scripts/mi50_rocm_environment.sh"
mi50_export_rocm_environment "${ROCM_PATH}"
HIPCC="${HIPCC:-${ROCM_PATH}/bin/hipcc}"
BUILD_DIR="${MI50_GRAPH_BUILD_DIR:-${ROOT_DIR}/out/graph-smoke-gfx906}"

for key in HSA_OVERRIDE_GFX_VERSION ROCR_OVERRIDE_GFX_VERSION; do
  if [[ -n "${!key:-}" ]]; then
    echo "ISA override is not allowed: ${key}" >&2
    exit 6
  fi
done
if [[ ! -x "${HIPCC}" ]]; then
  echo "HIP graph smoke: GPU-test-pending (hipcc unavailable at ${HIPCC})" >&2
  exit 77
fi

mkdir -p "${BUILD_DIR}"
OUTPUT="${BUILD_DIR}/mi50_graph_smoke"
"${HIPCC}" --offload-arch=gfx906 -O3 -std=c++17 \
  "${ROOT_DIR}/tests/hip/mi50_runtime_smoke.hip" -o "${OUTPUT}"
if [[ ! -e /dev/kfd ]]; then
  echo "HIP graph smoke: GPU-test-pending (/dev/kfd unavailable); native binary compiled" >&2
  exit 77
fi
"${OUTPUT}" --graph
