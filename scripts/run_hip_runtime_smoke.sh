#!/usr/bin/env bash
set -euo pipefail

# Compile and run the first native MI50 HIP runtime tier. Exit 77 is reserved
# for the expected pre-hardware state; any compiler or runtime failure is real.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python3 "${ROOT_DIR}/scripts/mi50_features.py" --check-environment

for key in HSA_OVERRIDE_GFX_VERSION ROCR_OVERRIDE_GFX_VERSION; do
  if [[ -n "${!key:-}" ]]; then
    echo "ISA override is not allowed: ${key}" >&2
    exit 2
  fi
done

ROCM_PATH="${ROCM_PATH:-/opt/rocm-mi50}"
HIPCC="${HIPCC:-${ROCM_PATH}/bin/hipcc}"
if [[ ! -x "${HIPCC}" ]]; then
  echo "HIP runtime smoke: GPU-test-pending (hipcc unavailable at ${HIPCC})" >&2
  exit 77
fi

BUILD_DIR="${MI50_RUNTIME_SMOKE_BUILD_DIR:-${ROOT_DIR}/out/runtime-smoke-gfx906}"
mkdir -p "${BUILD_DIR}"
OUTPUT="${BUILD_DIR}/mi50_runtime_smoke"
"${HIPCC}" --offload-arch=gfx906 -O3 -std=c++17 \
  "${ROOT_DIR}/tests/hip/mi50_runtime_smoke.hip" -o "${OUTPUT}"
if [[ ! -e /dev/kfd ]]; then
  echo "HIP runtime smoke: GPU-test-pending (/dev/kfd unavailable); native binary compiled" >&2
  exit 77
fi
"${OUTPUT}"
