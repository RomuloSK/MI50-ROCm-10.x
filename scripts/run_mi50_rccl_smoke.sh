#!/usr/bin/env bash
set -euo pipefail

# Compile/run the native dual-card RCCL smoke.  This is intentionally a
# separate tier from the single-device HIP smoke because a one-card system
# cannot prove the peer path required by split-model inference.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROCM_ROOT="${ROCM_PATH:-}"
BUILD_DIR="${MI50_RCCL_BUILD_DIR:-${ROOT_DIR}/out/rccl-smoke}"

if [[ -n "${HSA_OVERRIDE_GFX_VERSION:-}" || -n "${ROCR_OVERRIDE_GFX_VERSION:-}" ]]; then
  echo "RCCL smoke: ISA override variables are forbidden" >&2
  exit 6
fi
if [[ -z "$ROCM_ROOT" ]]; then
  echo "set ROCM_PATH to the MI50 ROCm installation" >&2
  exit 2
fi
ROCM_ROOT="$(cd "$ROCM_ROOT" && pwd)"
HIPCC="${ROCM_ROOT}/bin/hipcc"
if [[ ! -x "$HIPCC" ]]; then
  echo "RCCL smoke: hipcc unavailable at ${HIPCC}" >&2
  exit 77
fi
if [[ ! -f "${ROCM_ROOT}/include/rccl/rccl.h" || ! -f "${ROCM_ROOT}/lib/librccl.so" ]]; then
  echo "RCCL smoke: RCCL development files are missing under ${ROCM_ROOT}" >&2
  exit 77
fi

mkdir -p "$BUILD_DIR"
OBJECT="${BUILD_DIR}/mi50_rccl_smoke"
"$HIPCC" --offload-arch=gfx906 \
  -std=c++17 -O2 \
  "${ROOT_DIR}/tests/rccl/mi50_rccl_smoke.cpp" \
  -I"${ROCM_ROOT}/include" -L"${ROCM_ROOT}/lib" -lrccl \
  -Wl,-rpath,"${ROCM_ROOT}/lib" -o "$OBJECT"
if [[ ! -e /dev/kfd ]]; then
  echo "RCCL smoke: GPU-test-pending (/dev/kfd unavailable); native binary compiled" >&2
  exit 77
fi
"$OBJECT"
