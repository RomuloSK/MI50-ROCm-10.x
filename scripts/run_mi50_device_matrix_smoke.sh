#!/usr/bin/env bash
set -euo pipefail

# Compile/run the native gfx906 precision and peer-access smoke. Exit 77 is
# reserved for the expected pre-hardware state; a real device error is fatal.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python3 "${ROOT_DIR}/scripts/mi50_features.py" --check-environment

for key in HSA_OVERRIDE_GFX_VERSION ROCR_OVERRIDE_GFX_VERSION; do
  if [[ -n "${!key:-}" ]]; then
    echo "ISA override is not allowed: ${key}" >&2
    exit 2
  fi
done

if [[ ! -e /dev/kfd ]]; then
  echo "MI50 device matrix smoke: GPU-test-pending (/dev/kfd unavailable)" >&2
  exit 77
fi

ROCM_PATH="${ROCM_PATH:-/opt/rocm-mi50}"
HIPCC="${HIPCC:-${ROCM_PATH}/bin/hipcc}"
if [[ ! -x "${HIPCC}" ]]; then
  echo "MI50 device matrix smoke: hipcc unavailable at ${HIPCC}" >&2
  exit 77
fi

BUILD_DIR="${MI50_DEVICE_MATRIX_BUILD_DIR:-${ROOT_DIR}/out/device-matrix-smoke-gfx906}"
mkdir -p "${BUILD_DIR}"
OUTPUT="${BUILD_DIR}/mi50_device_matrix_smoke"
"${HIPCC}" --offload-arch=gfx906 -O3 -std=c++17 \
  "${ROOT_DIR}/tests/hip/mi50_device_matrix_smoke.hip" -o "${OUTPUT}"
"${OUTPUT}"
