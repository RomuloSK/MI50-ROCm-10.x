#!/usr/bin/env bash
set -euo pipefail

# Compile/run small ABI checks for the supported ROCm math libraries.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROCM_PATH="${ROCM_PATH:-/opt/rocm-mi50}"
source "${ROOT_DIR}/scripts/mi50_rocm_environment.sh"
mi50_export_rocm_environment "${ROCM_PATH}"
HIPCC="${HIPCC:-${ROCM_PATH}/bin/hipcc}"
BUILD_DIR="${MI50_LIBRARY_ABI_BUILD_DIR:-${ROOT_DIR}/out/library-abi-smoke-gfx906}"

for key in HSA_OVERRIDE_GFX_VERSION ROCR_OVERRIDE_GFX_VERSION; do
  if [[ -n "${!key:-}" ]]; then
    echo "library ABI smoke: ISA override is not allowed: ${key}" >&2
    exit 6
  fi
done
if [[ ! -x "${HIPCC}" ]]; then
  echo "library ABI smoke: GPU-test-pending (hipcc unavailable at ${HIPCC})" >&2
  exit 77
fi
for header in miopen/miopen.h rocblas/rocblas.h rocfft/rocfft.h rocrand/rocrand.h \
              rocsolver/rocsolver.h rocsparse/rocsparse.h; do
  if [[ ! -f "${ROCM_PATH}/include/${header}" ]]; then
    echo "library ABI smoke: missing ${ROCM_PATH}/include/${header}" >&2
    exit 77
  fi
done

mkdir -p "${BUILD_DIR}"
OUTPUT="${BUILD_DIR}/mi50_library_abi_smoke"
"${HIPCC}" --offload-arch=gfx906 -O2 -std=c++17 \
  "${ROOT_DIR}/tests/hip/mi50_library_abi_smoke.cpp" \
  -I"${ROCM_PATH}/include" -L"${ROCM_PATH}/lib" \
  -lMIOpen -lrocfft -lrocrand -lrocsparse -lrocsolver -lrocblas \
  -Wl,-rpath,"${ROCM_PATH}/lib" -o "${OUTPUT}"
if [[ ! -e /dev/kfd ]]; then
  echo "library ABI smoke: GPU-test-pending (/dev/kfd unavailable); native binary compiled" >&2
  exit 77
fi
"${OUTPUT}"
