#!/usr/bin/env bash
set -euo pipefail

# Build a PyTorch source checkout against the locally built MI50 ROCm stack.
# This is deliberately a wrapper around PyTorch's own build system: it applies
# only the reviewed downstream policy patches and never masquerades as a
# newer GPU. The resulting wheel is a host-side artifact until it passes real
# MI50 tests.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
python3 "${ROOT_DIR}/scripts/mi50_features.py" --check-environment
SOURCE_ROOT=""
ROCM_ROOT="${ROCM_PATH:-}"
HIPBLASLT_HOST_ROOT="${PYTORCH_HIPBLASLT_HOST:-}"
PATCH_DIR="${PYTORCH_MI50_PATCH_DIR:-${ROOT_DIR}/patches/downstream/pytorch}"
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
  --hipblaslt-host DIR  Host-only hipBLASLt compatibility package used only
                        to satisfy PyTorch's compile-time interface.
  PYTORCH_MI50_PATCH_DIR may override the reviewed downstream patch directory.
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
    --hipblaslt-host) HIPBLASLT_HOST_ROOT="$2"; shift 2 ;;
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
if [[ ! -d "$PATCH_DIR" ]]; then
  echo "downstream patch directory does not exist: ${PATCH_DIR}" >&2
  exit 2
fi
PATCH_DIR="$(cd "$PATCH_DIR" && pwd)"
if [[ -z "$ROCM_ROOT" ]]; then
  echo "set ROCM_PATH or pass --rocm pointing at the MI50 installation" >&2
  exit 2
fi
if [[ ! -d "$ROCM_ROOT" ]]; then
  echo "ROCm installation does not exist: ${ROCM_ROOT}" >&2
  exit 2
fi
ROCM_ROOT="$(cd "$ROCM_ROOT" && pwd)"
if [[ -n "$HIPBLASLT_HOST_ROOT" ]]; then
  if [[ ! -d "$HIPBLASLT_HOST_ROOT" ]]; then
    echo "host-only hipBLASLt package does not exist: ${HIPBLASLT_HOST_ROOT}" >&2
    exit 2
  fi
  HIPBLASLT_HOST_ROOT="$(cd "$HIPBLASLT_HOST_ROOT" && pwd)"
fi
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

# PyTorch's setup_helpers/env.py intentionally uses a source-relative
# `build/` directory and v2.13's setup parser rejects distutils' historical
# `--build-base` option.  Keep the requested out-of-tree build directory by
# linking that fixed entry point to it.  Refuse to replace a real checkout
# directory: an existing source build may contain user data or a prior config
# that must be removed explicitly by the caller.
BUILD_LINK_CREATED=0
SOURCE_BUILD_LINK="${SOURCE_ROOT}/build"
if [[ "$BUILD_DIR" != "$SOURCE_BUILD_LINK" ]]; then
  if [[ -L "$SOURCE_BUILD_LINK" ]]; then
    LINK_TARGET="$(readlink -f "$SOURCE_BUILD_LINK")"
    if [[ "$LINK_TARGET" != "$BUILD_DIR" ]]; then
      echo "source build symlink points at ${LINK_TARGET}, expected ${BUILD_DIR}" >&2
      exit 6
    fi
  elif [[ -e "$SOURCE_BUILD_LINK" ]]; then
    echo "${SOURCE_BUILD_LINK} already exists; remove it or use --build-dir ${SOURCE_BUILD_LINK}" >&2
    exit 6
  else
    ln -s "$BUILD_DIR" "$SOURCE_BUILD_LINK"
    BUILD_LINK_CREATED=1
  fi
fi
cleanup_build_link() {
  if [[ "$BUILD_LINK_CREATED" -eq 1 ]]; then
    rm -f "$SOURCE_BUILD_LINK"
  fi
}
trap cleanup_build_link EXIT

export ROCM_PATH="$ROCM_ROOT"
export ROCM_HOME="$ROCM_ROOT"
export HIP_PATH="$ROCM_ROOT"
export PATH="${ROCM_ROOT}/bin:${ROCM_ROOT}/lib/llvm/bin:${PATH}"
export CMAKE_PREFIX_PATH="${ROCM_ROOT}${CMAKE_PREFIX_PATH:+:${CMAKE_PREFIX_PATH}}"
MI50_PYTORCH_LIB_PATH="${ROCM_ROOT}/lib"
for MI50_PYTORCH_EXTRA_LIB_PATH in \
  "${ROCM_ROOT}/lib/rocm_sysdeps/lib" \
  "${ROCM_ROOT}/lib/llvm/lib"; do
  if [[ -d "${MI50_PYTORCH_EXTRA_LIB_PATH}" ]]; then
    MI50_PYTORCH_LIB_PATH="${MI50_PYTORCH_LIB_PATH}:${MI50_PYTORCH_EXTRA_LIB_PATH}"
  fi
done
export LD_LIBRARY_PATH="${MI50_PYTORCH_LIB_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
unset MI50_PYTORCH_EXTRA_LIB_PATH MI50_PYTORCH_LIB_PATH
if [[ -n "$HIPBLASLT_HOST_ROOT" ]]; then
  export CMAKE_PREFIX_PATH="${HIPBLASLT_HOST_ROOT}:${CMAKE_PREFIX_PATH}"
  export LD_LIBRARY_PATH="${HIPBLASLT_HOST_ROOT}/lib:${LD_LIBRARY_PATH}"
  # Preserve the resolved package location in provenance even when the path
  # was supplied through --hipblaslt-host rather than the caller's shell.
  export PYTORCH_HIPBLASLT_HOST="$HIPBLASLT_HOST_ROOT"
fi
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
export USE_ROCM_CK_GEMM="0"
export USE_ROCM_CK_SDPA="0"
export ROCBLAS_USE_HIPBLASLT="${ROCBLAS_USE_HIPBLASLT:-0}"
export PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED="0"
export MAX_JOBS="$JOBS"
export PYTORCH_BUILD_DIR="$BUILD_DIR"
export TORCH_CUDA_ARCH_LIST=""

apply_downstream_patches() {
  local patch_file
  shopt -s nullglob
  local patch_files=("${PATCH_DIR}"/*.patch)
  shopt -u nullglob
  for patch_file in "${patch_files[@]}"; do
    if git -C "$SOURCE_ROOT" apply --ignore-space-change --unidiff-zero --check "$patch_file" >/dev/null 2>&1; then
      git -C "$SOURCE_ROOT" apply --ignore-space-change --unidiff-zero "$patch_file"
      echo "applied downstream patch $(basename "$patch_file")"
    elif git -C "$SOURCE_ROOT" apply --ignore-space-change --unidiff-zero --reverse --check "$patch_file" >/dev/null 2>&1; then
      echo "downstream patch already applied $(basename "$patch_file")"
    else
      echo "downstream patch does not apply cleanly: ${patch_file}" >&2
      exit 6
    fi
  done
}

if [[ -n "${PYTORCH_BUILD_VERSION:-}" ]]; then
  export PYTORCH_BUILD_VERSION
fi
if [[ -n "${PYTORCH_BUILD_NUMBER:-}" ]]; then
  export PYTORCH_BUILD_NUMBER
fi

# The ROCm build is generated from PyTorch's CUDA sources.  CI invokes this
# same AMD hipification pass before setup.py; doing it here makes the wrapper
# self-contained and produces the c10/hip and ATen/hip CMake inputs.  It is
# deliberately skipped for --dry-run so inspection never mutates a checkout.
if [[ "$DRY_RUN" -eq 1 ]]; then
  cmd=(python3 setup.py bdist_wheel --dist-dir "$WHEEL_DIR")
  printf 'PyTorch gfx906 build environment:\n'
  printf '  ROCM_PATH=%q\n  PYTORCH_ROCM_ARCH=%q\n  MAX_JOBS=%q\n' "$ROCM_PATH" "$PYTORCH_ROCM_ARCH" "$MAX_JOBS"
  printf '  USE_AOTRITON=%q USE_FLASH_ATTENTION=%q USE_MEM_EFF_ATTENTION=%q USE_TRITON=%q\n' \
    "$USE_AOTRITON" "$USE_FLASH_ATTENTION" "$USE_MEM_EFF_ATTENTION" "$USE_TRITON"
  printf '  USE_ROCM_CK_GEMM=%q USE_ROCM_CK_SDPA=%q PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED=%q\n' \
    "$USE_ROCM_CK_GEMM" "$USE_ROCM_CK_SDPA" "$PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED"
  printf '  PYTORCH_MI50_PATCH_DIR=%q\n' "$PATCH_DIR"
  printf '  command:'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  exit 0
fi

apply_downstream_patches

if [[ -f "${SOURCE_ROOT}/tools/amd_build/build_amd.py" ]]; then
  echo "hipifying PyTorch CUDA sources for the ROCm build"
  (cd "$SOURCE_ROOT" && python3 tools/amd_build/build_amd.py)
else
  echo "PyTorch AMD hipify entry point is missing: ${SOURCE_ROOT}/tools/amd_build/build_amd.py" >&2
  exit 2
fi

# v2.13's setup.py accepts --dist-dir but deliberately does not expose
# distutils' --build-base.  The source-relative build/ symlink above controls
# the CMake build location instead.
cmd=(python3 setup.py bdist_wheel --dist-dir "$WHEEL_DIR")
printf 'PyTorch gfx906 build environment:\n'
printf '  ROCM_PATH=%q\n  PYTORCH_ROCM_ARCH=%q\n  MAX_JOBS=%q\n' "$ROCM_PATH" "$PYTORCH_ROCM_ARCH" "$MAX_JOBS"
printf '  USE_AOTRITON=%q USE_FLASH_ATTENTION=%q USE_MEM_EFF_ATTENTION=%q USE_TRITON=%q\n' \
  "$USE_AOTRITON" "$USE_FLASH_ATTENTION" "$USE_MEM_EFF_ATTENTION" "$USE_TRITON"
printf '  command:'
printf ' %q' "${cmd[@]}"
printf '\n'

cd "$SOURCE_ROOT"
"${cmd[@]}"

# Keep provenance next to the wheel.  A missing /dev/kfd is expected during
# pre-hardware development and is recorded rather than treated as success.
metadata_cmd=(python3 "${ROOT_DIR}/scripts/build/pytorch/write_pytorch_metadata.py"
  --wheel-dir "$WHEEL_DIR" --source "$SOURCE_ROOT" --rocm "$ROCM_ROOT"
  --build-dir "$BUILD_DIR" --jobs "$JOBS" --patch-dir "$PATCH_DIR")
if [[ -n "$HIPBLASLT_HOST_ROOT" ]]; then
  metadata_cmd+=(--hipblaslt-host "$HIPBLASLT_HOST_ROOT")
fi
"${metadata_cmd[@]}"

echo "PyTorch gfx906 wheel(s) written to ${WHEEL_DIR}"
