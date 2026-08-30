#!/usr/bin/env bash
set -euo pipefail

# Compile/run the native GCN5.1 packed INT8 dot4 primitive for gfx906.  This
# is a low-level fallback candidate, not a claim that rocBLAS INT8 GEMM is
# available or that quantized LLM kernels are production-ready.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python3 "${ROOT_DIR}/scripts/mi50_features.py" --check-environment
ROCM_PATH="${ROCM_PATH:-/opt/rocm-mi50}"
source "${ROOT_DIR}/scripts/mi50_rocm_environment.sh"
mi50_export_rocm_environment "${ROCM_PATH}"
HIPCC="${ROCM_PATH}/bin/hipcc"
BUILD_DIR="${MI50_INT8_DOT4_BUILD_DIR:-${ROOT_DIR}/out/int8-dot4-smoke-gfx906}"

for key in HSA_OVERRIDE_GFX_VERSION ROCR_OVERRIDE_GFX_VERSION; do
  if [[ -n "${!key:-}" ]]; then
    echo "ISA override is not allowed: ${key}" >&2
    exit 6
  fi
done
if [[ ! -x "${HIPCC}" ]]; then
  echo "INT8 dot4 smoke: GPU-test-pending (hipcc unavailable at ${HIPCC})" >&2
  exit 77
fi

mkdir -p "${BUILD_DIR}"
OUTPUT="${BUILD_DIR}/mi50_int8_dot4_smoke"
"${HIPCC}" --offload-arch=gfx906 -O3 -std=c++17 \
  "${ROOT_DIR}/tests/hip/mi50_int8_dot4_smoke.hip" -o "${OUTPUT}"
if [[ ! -e /dev/kfd ]]; then
  echo "INT8 dot4 smoke: GPU-test-pending (/dev/kfd unavailable); native binary compiled" >&2
  exit 77
fi
"${OUTPUT}"
