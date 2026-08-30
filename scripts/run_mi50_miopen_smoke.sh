#!/usr/bin/env bash
set -euo pipefail

# Compile/run a small native MIOpen FP32 convolution for gfx906.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROCM_PATH="${ROCM_PATH:-/opt/rocm-mi50}"
source "${ROOT_DIR}/scripts/mi50_rocm_environment.sh"
mi50_export_rocm_environment "${ROCM_PATH}"
HIPCC="${HIPCC:-${ROCM_PATH}/bin/hipcc}"
BUILD_DIR="${MI50_MIOPEN_BUILD_DIR:-${ROOT_DIR}/out/miopen-smoke-gfx906}"

for key in HSA_OVERRIDE_GFX_VERSION ROCR_OVERRIDE_GFX_VERSION; do
  if [[ -n "${!key:-}" ]]; then
    echo "MIOpen smoke: ISA override is not allowed: ${key}" >&2
    exit 6
  fi
done
if [[ ! -x "${HIPCC}" ]]; then
  echo "MIOpen smoke: GPU-test-pending (hipcc unavailable at ${HIPCC})" >&2
  exit 77
fi
if [[ ! -f "${ROCM_PATH}/include/miopen/miopen.h" || ! -f "${ROCM_PATH}/lib/libMIOpen.so" ]]; then
  echo "MIOpen smoke: development files are missing under ${ROCM_PATH}" >&2
  exit 77
fi

mkdir -p "${BUILD_DIR}"
OUTPUT="${BUILD_DIR}/mi50_miopen_convolution_smoke"
"${HIPCC}" --offload-arch=gfx906 -O3 -std=c++17 \
  "${ROOT_DIR}/tests/miopen/mi50_miopen_convolution_smoke.cpp" \
  -I"${ROCM_PATH}/include" -L"${ROCM_PATH}/lib" -lMIOpen \
  -Wl,-rpath,"${ROCM_PATH}/lib" -o "${OUTPUT}"
if [[ ! -e /dev/kfd ]]; then
  echo "MIOpen smoke: GPU-test-pending (/dev/kfd unavailable); native binary compiled" >&2
  exit 77
fi
"${OUTPUT}"
