#!/usr/bin/env bash
set -euo pipefail

# Compile/run the supported native rocBLAS Tensile path for gfx906.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROCM_PATH="${ROCM_PATH:-/opt/rocm-mi50}"
HIPCC="${HIPCC:-${ROCM_PATH}/bin/hipcc}"
BUILD_DIR="${MI50_ROCBLAS_BUILD_DIR:-${ROOT_DIR}/out/rocblas-smoke-gfx906}"

for key in HSA_OVERRIDE_GFX_VERSION ROCR_OVERRIDE_GFX_VERSION; do
  if [[ -n "${!key:-}" ]]; then
    echo "ISA override is not allowed: ${key}" >&2
    exit 6
  fi
done
if [[ ! -x "${HIPCC}" ]]; then
  echo "rocBLAS smoke: GPU-test-pending (hipcc unavailable at ${HIPCC})" >&2
  exit 77
fi
if [[ ! -f "${ROCM_PATH}/include/rocblas/rocblas.h" || ! -f "${ROCM_PATH}/lib/librocblas.so" ]]; then
  echo "rocBLAS smoke: development files are missing under ${ROCM_PATH}" >&2
  exit 77
fi

mkdir -p "${BUILD_DIR}"
OUTPUT="${BUILD_DIR}/mi50_rocblas_smoke"
"${HIPCC}" --offload-arch=gfx906 -O3 -std=c++17 \
  "${ROOT_DIR}/tests/rocblas/mi50_rocblas_smoke.cpp" \
  -I"${ROCM_PATH}/include" -L"${ROCM_PATH}/lib" -lrocblas \
  -Wl,-rpath,"${ROCM_PATH}/lib" -o "${OUTPUT}"
if [[ ! -e /dev/kfd ]]; then
  echo "rocBLAS smoke: GPU-test-pending (/dev/kfd unavailable); native binary compiled" >&2
  exit 77
fi
"${OUTPUT}"
