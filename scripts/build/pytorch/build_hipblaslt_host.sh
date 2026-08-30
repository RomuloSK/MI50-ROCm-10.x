#!/usr/bin/env bash
set -euo pipefail

# Build only the hipBLASLt host interface needed by PyTorch. MI50 production
# GEMM is routed through hipBLAS/rocBLAS; this package contains no hipBLASLt
# device kernels or newer-ISA payloads.
SOURCE_ROOT=""
ROCM_ROOT="${ROCM_PATH:-}"
BUILD_DIR=""
INSTALL_DIR=""
JOBS="${MAX_JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)}"
CLEAN=0

usage() {
  cat <<'EOF'
Usage: build_hipblaslt_host.sh HIPBLASLT_SOURCE [options]

Options:
  --rocm DIR       MI50 ROCm installation (default: ROCM_PATH).
  --build-dir DIR  CMake build directory.
  --install-dir DIR Installation prefix (default: build/hipblaslt-host-install).
  --jobs N         Ninja parallel jobs (default: MAX_JOBS/CPU count).
  --clean          Remove only the selected build directory first.
  --help           Show this help.

Device compilation is disabled. amdclang/amdclang++ from the selected ROCm
installation are still used so HIP's offload flags are understood.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --rocm) ROCM_ROOT="$2"; shift 2 ;;
    --build-dir) BUILD_DIR="$2"; shift 2 ;;
    --install-dir) INSTALL_DIR="$2"; shift 2 ;;
    --jobs) JOBS="$2"; shift 2 ;;
    --clean) CLEAN=1; shift ;;
    --help|-h) usage; exit 0 ;;
    -*) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    *)
      [[ -z "$SOURCE_ROOT" ]] || { echo "only one source directory may be provided" >&2; exit 2; }
      SOURCE_ROOT="$1"
      shift
      ;;
  esac
done

[[ -n "$SOURCE_ROOT" ]] || { echo "a hipBLASLt source directory is required" >&2; exit 2; }
[[ -d "$SOURCE_ROOT" ]] || { echo "source directory does not exist: $SOURCE_ROOT" >&2; exit 2; }
[[ -n "$ROCM_ROOT" && -d "$ROCM_ROOT" ]] || { echo "set ROCM_PATH or pass --rocm" >&2; exit 2; }
if [[ -d "${ROCM_ROOT}/rocm" && ! -d "${ROCM_ROOT}/bin" ]]; then
  ROCM_ROOT="${ROCM_ROOT}/rocm"
fi
[[ -x "$ROCM_ROOT/lib/llvm/bin/amdclang" && -x "$ROCM_ROOT/lib/llvm/bin/amdclang++" ]] || {
  echo "amdclang/amdclang++ are missing from $ROCM_ROOT" >&2
  exit 2
}
[[ "$JOBS" =~ ^[1-9][0-9]*$ ]] || { echo "--jobs must be a positive integer" >&2; exit 2; }
if [[ -n "${HSA_OVERRIDE_GFX_VERSION:-}" ]]; then
  echo "HSA_OVERRIDE_GFX_VERSION is forbidden" >&2
  exit 6
fi

SOURCE_ROOT="$(cd "$SOURCE_ROOT" && pwd)"
ROCM_ROOT="$(cd "$ROCM_ROOT" && pwd)"
BUILD_DIR="${BUILD_DIR:-${SOURCE_ROOT}/build-mi50-host}"
INSTALL_DIR="${INSTALL_DIR:-${BUILD_DIR}/install}"
BUILD_DIR="$(mkdir -p "$BUILD_DIR" && cd "$BUILD_DIR" && pwd)"
INSTALL_DIR="$(mkdir -p "$INSTALL_DIR" && cd "$INSTALL_DIR" && pwd)"
if [[ "$CLEAN" -eq 1 ]]; then
  [[ "$BUILD_DIR" != "/" && "$BUILD_DIR" != "$SOURCE_ROOT" ]] || { echo "refusing to clean unsafe path" >&2; exit 6; }
  cmake -E rm -rf "$BUILD_DIR"
  mkdir -p "$BUILD_DIR"
fi

export ROCM_PATH="$ROCM_ROOT"
export ROCM_HOME="$ROCM_ROOT"
export HIP_PATH="$ROCM_ROOT"
export PATH="$ROCM_ROOT/bin:$ROCM_ROOT/lib/llvm/bin:$PATH"
MI50_HIPBLASLT_LIB_PATH="$ROCM_ROOT/lib"
for MI50_HIPBLASLT_EXTRA_LIB_PATH in \
  "$ROCM_ROOT/lib/rocm_sysdeps/lib" \
  "$ROCM_ROOT/lib/llvm/lib"; do
  if [[ -d "$MI50_HIPBLASLT_EXTRA_LIB_PATH" ]]; then
    MI50_HIPBLASLT_LIB_PATH="$MI50_HIPBLASLT_LIB_PATH:$MI50_HIPBLASLT_EXTRA_LIB_PATH"
  fi
done
export LD_LIBRARY_PATH="$MI50_HIPBLASLT_LIB_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
unset MI50_HIPBLASLT_EXTRA_LIB_PATH MI50_HIPBLASLT_LIB_PATH

cmake -S "$SOURCE_ROOT" -B "$BUILD_DIR" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
  -DCMAKE_C_COMPILER="$ROCM_ROOT/lib/llvm/bin/amdclang" \
  -DCMAKE_CXX_COMPILER="$ROCM_ROOT/lib/llvm/bin/amdclang++" \
  -DCMAKE_PREFIX_PATH="$ROCM_ROOT" \
  -DROCM_PATH="$ROCM_ROOT" \
  -DGPU_TARGETS=gfx906 \
  -DHIPBLASLT_ENABLE_DEVICE=OFF \
  -DHIPBLASLT_ENABLE_EXTOPS=OFF \
  -DHIPBLASLT_ENABLE_MATRIX_TRANSFORM=OFF \
  -DHIPBLASLT_ENABLE_CLIENT=OFF \
  -DHIPBLASLT_ENABLE_HOST=ON \
  -DBUILD_TESTING=OFF \
  -DTENSILELITE_ENABLE_HOST=ON
cmake --build "$BUILD_DIR" --parallel "$JOBS"
cmake --install "$BUILD_DIR"

if find "$INSTALL_DIR" -type f \( -name '*.co' -o -name '*.hsaco' -o -name '*gfx*.o' \) -print -quit | grep -q .; then
  echo "device code unexpectedly present in host-only hipBLASLt install" >&2
  exit 7
fi
echo "host-only hipBLASLt installed at $INSTALL_DIR"
