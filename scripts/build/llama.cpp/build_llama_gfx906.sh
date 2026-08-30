#!/usr/bin/env bash
set -euo pipefail

# Configure/build llama.cpp with native HIP gfx906 targeting.  The project is
# independent of PyTorch and is the primary production-style GGUF path for
# MI50 inference.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
python3 "${ROOT_DIR}/scripts/mi50_features.py" --check-environment
SOURCE_ROOT=""
ROCM_ROOT="${ROCM_PATH:-}"
BUILD_DIR=""
INSTALL_DIR=""
JOBS="${MAX_JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)}"
CLEAN=0
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: build_llama_gfx906.sh LLAMA_CPP_SOURCE [options]

Options:
  --rocm DIR       ROCm installation (default: ROCM_PATH).
  --build-dir DIR  CMake build directory.
  --install-dir DIR Installation prefix (default: out/llama-gfx906).
  --jobs N         Parallel build jobs (default: MAX_JOBS/CPU count).
  --clean          Remove only the selected build directory first.
  --dry-run        Print the CMake configuration without building.
  --help           Show this help.

The wrapper uses GGML_HIP=ON, GGML_NATIVE=OFF, AMDGPU_TARGETS=gfx906 and
CMAKE_HIP_ARCHITECTURES=gfx906.  It keeps rocBLAS on the mature Tensile
backend (ROCBLAS_USE_HIPBLASLT=0) and never sets HSA_OVERRIDE_GFX_VERSION.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --rocm) ROCM_ROOT="$2"; shift 2 ;;
    --build-dir) BUILD_DIR="$2"; shift 2 ;;
    --install-dir) INSTALL_DIR="$2"; shift 2 ;;
    --jobs) JOBS="$2"; shift 2 ;;
    --clean) CLEAN=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --help|-h) usage; exit 0 ;;
    --) shift; break ;;
    -*) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    *)
      if [[ -n "$SOURCE_ROOT" ]]; then
        echo "only one llama.cpp source directory may be provided" >&2
        exit 2
      fi
      SOURCE_ROOT="$1"
      shift
      ;;
  esac
done

if [[ -z "$SOURCE_ROOT" ]]; then
  echo "a llama.cpp source directory is required" >&2
  usage >&2
  exit 2
fi
if [[ ! -d "$SOURCE_ROOT" ]]; then
  echo "llama.cpp source directory does not exist: ${SOURCE_ROOT}" >&2
  exit 2
fi
SOURCE_ROOT="$(cd "$SOURCE_ROOT" && pwd)"
if [[ ! -f "${SOURCE_ROOT}/CMakeLists.txt" ]]; then
  echo "not a llama.cpp source tree (CMakeLists.txt missing): ${SOURCE_ROOT}" >&2
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
INSTALL_DIR="${INSTALL_DIR:-${ROOT_DIR}/out/llama-gfx906}"
BUILD_DIR="$(mkdir -p "$BUILD_DIR" && cd "$BUILD_DIR" && pwd)"
INSTALL_DIR="$(mkdir -p "$INSTALL_DIR" && cd "$INSTALL_DIR" && pwd)"
if [[ "$CLEAN" -eq 1 ]]; then
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
export GPU_TARGETS="gfx906"
export AMDGPU_TARGETS="gfx906"
export CMAKE_HIP_ARCHITECTURES="gfx906"
export ROCBLAS_USE_HIPBLASLT="${ROCBLAS_USE_HIPBLASLT:-0}"

cmake_args=(
  -S "$SOURCE_ROOT"
  -B "$BUILD_DIR"
  -G Ninja
  -DCMAKE_BUILD_TYPE=Release
  -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR"
  -DCMAKE_PREFIX_PATH="$ROCM_ROOT"
  -DGGML_HIP=ON
  -DGGML_NATIVE=OFF
  -DLLAMA_BUILD_TOOLS=ON
  -DLLAMA_BUILD_TESTS=ON
  -DAMDGPU_TARGETS=gfx906
  -DCMAKE_HIP_ARCHITECTURES=gfx906
)
printf 'llama.cpp gfx906 CMake configuration:'
printf ' %q' cmake "${cmake_args[@]}"
printf '\n'

if [[ "$DRY_RUN" -eq 1 ]]; then
  exit 0
fi

cmake "${cmake_args[@]}"
cmake --build "$BUILD_DIR" --parallel "$JOBS"
cmake --install "$BUILD_DIR"

python3 - "$INSTALL_DIR" "$SOURCE_ROOT" "$ROCM_ROOT" "$JOBS" <<'PY'
import json
import os
import platform
import sys
from pathlib import Path

install, source, rocm, jobs = sys.argv[1:]
install_path = Path(install)
metadata = {
    "schema_version": 1,
    "target": "gfx906",
    "project": "llama.cpp",
    "source": str(Path(source).resolve()),
    "rocm_path": str(Path(rocm).resolve()),
    "cmake_options": {
        "GGML_HIP": "ON",
        "GGML_NATIVE": "OFF",
        "AMDGPU_TARGETS": "gfx906",
        "CMAKE_HIP_ARCHITECTURES": "gfx906",
        "GPU_TARGETS": "gfx906",
        "LLAMA_BUILD_TOOLS": "ON",
        "LLAMA_BUILD_TESTS": "ON",
        "ROCBLAS_USE_HIPBLASLT": "0",
    },
    "jobs": int(jobs),
    "platform": {"system": platform.system(), "release": platform.release()},
    "runtime_status": "GPU-test-pending" if not Path("/dev/kfd").exists() else "hardware-validation-required",
    "hsa_override_used": bool(os.environ.get("HSA_OVERRIDE_GFX_VERSION")),
}
(install_path / "mi50-build-metadata.json").write_text(
    json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY

echo "llama.cpp gfx906 installed to ${INSTALL_DIR}"
