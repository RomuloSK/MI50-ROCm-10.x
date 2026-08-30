#!/usr/bin/env bash
set -euo pipefail

# Build a PyTorch source checkout against the locally built MI50 ROCm stack.
# This is deliberately a wrapper around PyTorch's own build system: it does
# not patch PyTorch source and it never masquerades as a newer GPU.  The
# resulting wheel is a host-side artifact until it passes real MI50 tests.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
python3 "${ROOT_DIR}/scripts/mi50_features.py" --check-environment
SOURCE_ROOT=""
ROCM_ROOT="${ROCM_PATH:-}"
BUILD_DIR=""
WHEEL_DIR=""
JOBS="${MAX_JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)}"
CLEAN=0
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: build_pytorch_gfx906.sh PYTORCH_SOURCE [options]

Build PyTorch against a locally installed ROCm 10.x gfx906 stack.

Options:
  --rocm DIR       ROCm installation (default: ROCM_PATH).
  --build-dir DIR  Out-of-tree setuptools/CMake build directory.
  --wheel-dir DIR  Destination for the wheel (default: build/pytorch-wheels).
  --jobs N         Parallel build jobs (default: MAX_JOBS/CPU count).
  --clean          Remove only the selected PyTorch build directory first.
  --dry-run        Print the exact environment/command without building.
  --help           Show this help.

The wrapper sets PYTORCH_ROCM_ARCH=gfx906 and keeps AOTriton, flash
attention, memory-efficient attention, Triton and hipBLASLt opt-in paths
disabled.  Those paths need separate gfx906 code-generation and hardware
validation before they can be advertised.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --rocm) ROCM_ROOT="$2"; shift 2 ;;
    --build-dir) BUILD_DIR="$2"; shift 2 ;;
    --wheel-dir) WHEEL_DIR="$2"; shift 2 ;;
    --jobs) JOBS="$2"; shift 2 ;;
    --clean) CLEAN=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --help|-h) usage; exit 0 ;;
    --) shift; break ;;
    -*) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    *)
      if [[ -n "$SOURCE_ROOT" ]]; then
        echo "only one PyTorch source directory may be provided" >&2
        exit 2
      fi
      SOURCE_ROOT="$1"
      shift
      ;;
  esac
done

if [[ -z "$SOURCE_ROOT" ]]; then
  echo "a PyTorch source directory is required" >&2
  usage >&2
  exit 2
fi
if [[ ! -d "$SOURCE_ROOT" ]]; then
  echo "PyTorch source directory does not exist: ${SOURCE_ROOT}" >&2
  exit 2
fi
SOURCE_ROOT="$(cd "$SOURCE_ROOT" && pwd)"
if [[ ! -f "${SOURCE_ROOT}/setup.py" ]]; then
  echo "not a PyTorch source tree (setup.py missing): ${SOURCE_ROOT}" >&2
  exit 2
fi
if [[ -z "$ROCM_ROOT" ]]; then
  echo "set ROCM_PATH or pass --rocm pointing at the MI50 installation" >&2
  exit 2
fi
if [[ ! -d "$ROCM_ROOT" ]]; then
  echo "ROCm installation does not exist: ${ROCM_ROOT}" >&2
  exit 2
fi
ROCM_ROOT="$(cd "$ROCM_ROOT" && pwd)"
if [[ -n "${HSA_OVERRIDE_GFX_VERSION:-}" ]]; then
  echo "HSA_OVERRIDE_GFX_VERSION is forbidden; build against native gfx906" >&2
  exit 6
fi
if ! [[ "$JOBS" =~ ^[1-9][0-9]*$ ]]; then
  echo "--jobs must be a positive integer (got ${JOBS})" >&2
  exit 2
fi

BUILD_DIR="${BUILD_DIR:-${SOURCE_ROOT}/build-mi50-gfx906}"
WHEEL_DIR="${WHEEL_DIR:-${ROOT_DIR}/out/pytorch-wheels/gfx906}"
BUILD_DIR="$(mkdir -p "$BUILD_DIR" && cd "$BUILD_DIR" && pwd)"
WHEEL_DIR="$(mkdir -p "$WHEEL_DIR" && cd "$WHEEL_DIR" && pwd)"
if [[ "$CLEAN" -eq 1 ]]; then
  # The path was resolved above; only this user-selected build directory is
  # removed, never the source tree or repository workspace.
  if [[ "$BUILD_DIR" == "$SOURCE_ROOT" || "$BUILD_DIR" == "/" ]]; then
    echo "refusing to clean the source tree or filesystem root: ${BUILD_DIR}" >&2
    exit 6
  fi
  cmake -E rm -rf "$BUILD_DIR"
  mkdir -p "$BUILD_DIR"
fi

export ROCM_PATH="$ROCM_ROOT"
export ROCM_HOME="$ROCM_ROOT"
export HIP_PATH="$ROCM_ROOT"
export PATH="${ROCM_ROOT}/bin:${PATH}"
export CMAKE_PREFIX_PATH="${ROCM_ROOT}${CMAKE_PREFIX_PATH:+:${CMAKE_PREFIX_PATH}}"
export PYTORCH_ROCM_ARCH="gfx906"
export USE_ROCM="1"
export USE_CUDA="0"
export USE_NCCL="1"
export USE_SYSTEM_NCCL="1"
export BUILD_TEST="0"
export BUILD_CAFFE2="0"
export USE_FLASH_ATTENTION="0"
export USE_MEM_EFF_ATTENTION="0"
export USE_AOTRITON="0"
export USE_TRITON="0"
export ROCBLAS_USE_HIPBLASLT="${ROCBLAS_USE_HIPBLASLT:-0}"
export MAX_JOBS="$JOBS"
export PYTORCH_BUILD_DIR="$BUILD_DIR"
export TORCH_CUDA_ARCH_LIST=""

if [[ -n "${PYTORCH_BUILD_VERSION:-}" ]]; then
  export PYTORCH_BUILD_VERSION
fi
if [[ -n "${PYTORCH_BUILD_NUMBER:-}" ]]; then
  export PYTORCH_BUILD_NUMBER
fi

cmd=(python3 setup.py bdist_wheel --dist-dir "$WHEEL_DIR" --build-base "$BUILD_DIR")
printf 'PyTorch gfx906 build environment:\n'
printf '  ROCM_PATH=%q\n  PYTORCH_ROCM_ARCH=%q\n  MAX_JOBS=%q\n' "$ROCM_PATH" "$PYTORCH_ROCM_ARCH" "$MAX_JOBS"
printf '  USE_AOTRITON=%q USE_FLASH_ATTENTION=%q USE_MEM_EFF_ATTENTION=%q USE_TRITON=%q\n' \
  "$USE_AOTRITON" "$USE_FLASH_ATTENTION" "$USE_MEM_EFF_ATTENTION" "$USE_TRITON"
printf '  command:'
printf ' %q' "${cmd[@]}"
printf '\n'

if [[ "$DRY_RUN" -eq 1 ]]; then
  exit 0
fi

cd "$SOURCE_ROOT"
"${cmd[@]}"

# Keep provenance next to the wheel.  A missing /dev/kfd is expected during
# pre-hardware development and is recorded rather than treated as success.
python3 - "$WHEEL_DIR" "$SOURCE_ROOT" "$ROCM_ROOT" "$JOBS" <<'PY'
import json
import os
import platform
import sys
from pathlib import Path

out = Path(sys.argv[1])
source = Path(sys.argv[2])
rocm = Path(sys.argv[3])
jobs = int(sys.argv[4])
metadata = {
    "schema_version": 1,
    "project": "pytorch",
    "target": "gfx906",
    "source": str(source.resolve()),
    "rocm_path": str(rocm.resolve()),
    "build_options": {
        "PYTORCH_ROCM_ARCH": "gfx906",
        "USE_ROCM": "1",
        "USE_CUDA": "0",
        "USE_NCCL": "1",
        "USE_SYSTEM_NCCL": "1",
        "USE_AOTRITON": "0",
        "USE_FLASH_ATTENTION": "0",
        "USE_MEM_EFF_ATTENTION": "0",
        "USE_TRITON": "0",
        "ROCBLAS_USE_HIPBLASLT": "0",
    },
    "jobs": jobs,
    "platform": {"system": platform.system(), "release": platform.release()},
    "runtime_status": "GPU-test-pending" if not Path("/dev/kfd").exists() else "hardware-validation-required",
    "hsa_override_used": bool(os.environ.get("HSA_OVERRIDE_GFX_VERSION")),
    "wheels": sorted(p.name for p in out.glob("*.whl")),
}
(out / "mi50-build-metadata.json").write_text(
    json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY

echo "PyTorch gfx906 wheel(s) written to ${WHEEL_DIR}"
