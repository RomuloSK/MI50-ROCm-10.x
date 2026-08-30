#!/usr/bin/env bash
set -euo pipefail

# Compile/run the optional native INT8 -> INT32 rocBLAS GEMM path for gfx906.
# INT8 is deliberately separate from the FP16/FP32/FP64 release smoke: ROCm
# exposes the generic API on more targets than it provides tuned Vega20
# kernels, so this gate must report the result per card rather than silently
# treating an unsupported kernel as a successful fallback.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python3 "${ROOT_DIR}/scripts/mi50_features.py" --check-environment
ROCM_PATH="${ROCM_PATH:-/opt/rocm-mi50}"
source "${ROOT_DIR}/scripts/mi50_rocm_environment.sh"
mi50_export_rocm_environment "${ROCM_PATH}"
HIPCC="${ROCM_PATH}/bin/hipcc"
BUILD_DIR="${MI50_INT8_BUILD_DIR:-${ROOT_DIR}/out/rocblas-int8-smoke-gfx906}"

for key in HSA_OVERRIDE_GFX_VERSION ROCR_OVERRIDE_GFX_VERSION; do
  if [[ -n "${!key:-}" ]]; then
    echo "ISA override is not allowed: ${key}" >&2
    exit 6
  fi
done
if [[ ! -x "${HIPCC}" ]]; then
  echo "rocBLAS INT8 smoke: GPU-test-pending (hipcc unavailable at ${HIPCC})" >&2
  exit 77
fi
if [[ ! -f "${ROCM_PATH}/include/rocblas/rocblas.h" || ! -f "${ROCM_PATH}/lib/librocblas.so" ]]; then
  echo "rocBLAS INT8 smoke: development files are missing under ${ROCM_PATH}" >&2
  exit 77
fi

mkdir -p "${BUILD_DIR}"
OUTPUT="${BUILD_DIR}/mi50_rocblas_int8_smoke"
"${HIPCC}" --offload-arch=gfx906 -O3 -std=c++17 \
  "${ROOT_DIR}/tests/rocblas/mi50_rocblas_smoke.cpp" \
  -I"${ROCM_PATH}/include" -L"${ROCM_PATH}/lib" -lrocblas \
  -Wl,-rpath,"${ROCM_PATH}/lib" -o "${OUTPUT}"
if [[ ! -e /dev/kfd ]]; then
  echo "rocBLAS INT8 smoke: GPU-test-pending (/dev/kfd unavailable); native binary compiled" >&2
  exit 77
fi
"${OUTPUT}" --int8
