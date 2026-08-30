#!/usr/bin/env bash
set -euo pipefail

# Compile/run the HIPRTC path for native gfx906.  The binary performs the
# host-side COMGR compilation before checking /dev/kfd, so a pre-hardware host
# still verifies that runtime compilation emits a non-empty target object.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python3 "${ROOT_DIR}/scripts/mi50_features.py" --check-environment
ROCM_PATH="${ROCM_PATH:-/opt/rocm-mi50}"
HIPCC="${HIPCC:-${ROCM_PATH}/bin/hipcc}"
BUILD_DIR="${MI50_HIPRTC_BUILD_DIR:-${ROOT_DIR}/out/hiprtc-smoke-gfx906}"

for key in HSA_OVERRIDE_GFX_VERSION ROCR_OVERRIDE_GFX_VERSION; do
  if [[ -n "${!key:-}" ]]; then
    echo "ISA override is not allowed: ${key}" >&2
    exit 6
  fi
done
if [[ ! -x "${HIPCC}" ]]; then
  echo "HIPRTC smoke: GPU-test-pending (hipcc unavailable at ${HIPCC})" >&2
  exit 77
fi
if [[ ! -f "${ROCM_PATH}/include/hip/hiprtc.h" || ! -f "${ROCM_PATH}/lib/libhiprtc.so" ]]; then
  echo "HIPRTC smoke: runtime compiler development files are missing under ${ROCM_PATH}" >&2
  exit 77
fi

mkdir -p "${BUILD_DIR}"
OUTPUT="${BUILD_DIR}/mi50_hiprtc_smoke"
"${HIPCC}" --offload-arch=gfx906 -O3 -std=c++17 \
  "${ROOT_DIR}/tests/hip/mi50_hiprtc_smoke.cpp" \
  -I"${ROCM_PATH}/include" -L"${ROCM_PATH}/lib" -lhiprtc \
  -Wl,-rpath,"${ROCM_PATH}/lib" -o "${OUTPUT}"
"${OUTPUT}"
