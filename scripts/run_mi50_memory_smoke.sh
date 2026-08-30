#!/usr/bin/env bash
set -euo pipefail

# Compile/run the bounded per-device memory smoke for native gfx906.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python3 "${ROOT_DIR}/scripts/mi50_features.py" --check-environment
ROCM_PATH="${ROCM_PATH:-/opt/rocm-mi50}"
source "${ROOT_DIR}/scripts/mi50_rocm_environment.sh"
mi50_export_rocm_environment "${ROCM_PATH}"
HIPCC="${ROCM_PATH}/bin/hipcc"
BUILD_DIR="${MI50_MEMORY_BUILD_DIR:-${ROOT_DIR}/out/memory-smoke-gfx906}"

for key in HSA_OVERRIDE_GFX_VERSION ROCR_OVERRIDE_GFX_VERSION; do
  if [[ -n "${!key:-}" ]]; then
    echo "memory smoke: ISA override is not allowed: ${key}" >&2
    exit 6
  fi
done
if [[ ! -x "${HIPCC}" ]]; then
  echo "memory smoke: GPU-test-pending (hipcc unavailable at ${HIPCC})" >&2
  exit 77
fi
if [[ ! -f "${ROCM_PATH}/include/hip/hip_runtime.h" ]]; then
  echo "memory smoke: HIP headers are missing under ${ROCM_PATH}" >&2
  exit 77
fi
mkdir -p "${BUILD_DIR}"
OUTPUT="${BUILD_DIR}/mi50_memory_smoke"
"${HIPCC}" --offload-arch=gfx906 -O3 -std=c++17 \
  "${ROOT_DIR}/tests/hip/mi50_memory_smoke.hip" -o "${OUTPUT}"
if [[ ! -e /dev/kfd ]]; then
  echo "memory smoke: GPU-test-pending (/dev/kfd unavailable); native binary compiled" >&2
  exit 77
fi
"${OUTPUT}"
