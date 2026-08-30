#!/usr/bin/env bash
set -euo pipefail

# Compile/run native rocSPARSE CSR SpMV and rocSOLVER LU checks for gfx906.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROCM_PATH="${ROCM_PATH:-/opt/rocm-mi50}"
source "${ROOT_DIR}/scripts/mi50_rocm_environment.sh"
mi50_export_rocm_environment "${ROCM_PATH}"
HIPCC="${HIPCC:-${ROCM_PATH}/bin/hipcc}"
BUILD_DIR="${MI50_SPARSE_SOLVER_BUILD_DIR:-${ROOT_DIR}/out/sparse-solver-smoke-gfx906}"

for key in HSA_OVERRIDE_GFX_VERSION ROCR_OVERRIDE_GFX_VERSION; do
  if [[ -n "${!key:-}" ]]; then
    echo "SPARSE/SOLVER smoke: ISA override is not allowed: ${key}" >&2
    exit 6
  fi
done
if [[ ! -x "${HIPCC}" ]]; then
  echo "SPARSE/SOLVER smoke: GPU-test-pending (hipcc unavailable at ${HIPCC})" >&2
  exit 77
fi
for header in rocsparse/rocsparse.h rocsolver/rocsolver.h; do
  if [[ ! -f "${ROCM_PATH}/include/${header}" ]]; then
    echo "SPARSE/SOLVER smoke: missing ${ROCM_PATH}/include/${header}" >&2
    exit 77
  fi
done
mkdir -p "${BUILD_DIR}"
OUTPUT="${BUILD_DIR}/mi50_sparse_solver_smoke"
"${HIPCC}" --offload-arch=gfx906 -O3 -std=c++17 \
  "${ROOT_DIR}/tests/hip/mi50_sparse_solver_smoke.cpp" \
  -I"${ROCM_PATH}/include" -L"${ROCM_PATH}/lib" -lrocsparse -lrocsolver -lrocblas \
  -Wl,-rpath,"${ROCM_PATH}/lib" -o "${OUTPUT}"
if [[ ! -e /dev/kfd ]]; then
  echo "SPARSE/SOLVER smoke: GPU-test-pending (/dev/kfd unavailable); native binary compiled" >&2
  exit 77
fi
"${OUTPUT}"
