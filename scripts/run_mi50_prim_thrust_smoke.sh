#!/usr/bin/env bash
set -euo pipefail

# Compile/run native rocPRIM and rocThrust reduction checks for gfx906.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python3 "${ROOT_DIR}/scripts/mi50_features.py" --check-environment
ROCM_PATH="${ROCM_PATH:-/opt/rocm-mi50}"
source "${ROOT_DIR}/scripts/mi50_rocm_environment.sh"
mi50_export_rocm_environment "${ROCM_PATH}"
HIPCC="${HIPCC:-${ROCM_PATH}/bin/hipcc}"
BUILD_DIR="${MI50_PRIM_THRUST_BUILD_DIR:-${ROOT_DIR}/out/prim-thrust-smoke-gfx906}"

for key in HSA_OVERRIDE_GFX_VERSION ROCR_OVERRIDE_GFX_VERSION; do
  if [[ -n "${!key:-}" ]]; then
    echo "PRIM/Thrust smoke: ISA override is not allowed: ${key}" >&2
    exit 6
  fi
done
if [[ ! -x "${HIPCC}" ]]; then
  echo "PRIM/Thrust smoke: GPU-test-pending (hipcc unavailable at ${HIPCC})" >&2
  exit 77
fi
for header in rocprim/rocprim.hpp thrust/reduce.h; do
  if [[ ! -f "${ROCM_PATH}/include/${header}" ]]; then
    echo "PRIM/Thrust smoke: missing ${ROCM_PATH}/include/${header}" >&2
    exit 77
  fi
done
mkdir -p "${BUILD_DIR}"
OUTPUT="${BUILD_DIR}/mi50_prim_thrust_smoke"
"${HIPCC}" --offload-arch=gfx906 -O3 -std=c++17 \
  "${ROOT_DIR}/tests/hip/mi50_prim_thrust_smoke.hip" \
  -I"${ROCM_PATH}/include" -o "${OUTPUT}"
if [[ ! -e /dev/kfd ]]; then
  echo "PRIM/Thrust smoke: GPU-test-pending (/dev/kfd unavailable); native binary compiled" >&2
  exit 77
fi
"${OUTPUT}"
