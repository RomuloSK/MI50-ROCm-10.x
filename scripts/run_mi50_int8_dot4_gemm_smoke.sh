#!/usr/bin/env bash
set -euo pipefail

# Compile/run the end-to-end packed-dot4 INT8 GEMM fallback for gfx906.
# rocBLAS remains preferred; this path is an explicit quantized-inference
# fallback candidate until hardware benchmarking proves it useful.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python3 "${ROOT_DIR}/scripts/mi50_features.py" --check-environment
ROCM_PATH="${ROCM_PATH:-/opt/rocm-mi50}"
HIPCC="${HIPCC:-${ROCM_PATH}/bin/hipcc}"
BUILD_DIR="${MI50_INT8_DOT4_GEMM_BUILD_DIR:-${ROOT_DIR}/out/int8-dot4-gemm-smoke-gfx906}"

for key in HSA_OVERRIDE_GFX_VERSION ROCR_OVERRIDE_GFX_VERSION; do
  if [[ -n "${!key:-}" ]]; then
    echo "ISA override is not allowed: ${key}" >&2
    exit 6
  fi
done
if [[ ! -x "${HIPCC}" ]]; then
  echo "INT8 dot4 GEMM smoke: GPU-test-pending (hipcc unavailable at ${HIPCC})" >&2
  exit 77
fi

mkdir -p "${BUILD_DIR}"
OUTPUT="${BUILD_DIR}/mi50_int8_dot4_gemm_smoke"
"${HIPCC}" --offload-arch=gfx906 -O3 -std=c++17 \
  "${ROOT_DIR}/tests/hip/mi50_int8_dot4_gemm_smoke.hip" -o "${OUTPUT}"
if [[ ! -e /dev/kfd ]]; then
  echo "INT8 dot4 GEMM smoke: GPU-test-pending (/dev/kfd unavailable); native binary compiled" >&2
  exit 77
fi
"${OUTPUT}"
