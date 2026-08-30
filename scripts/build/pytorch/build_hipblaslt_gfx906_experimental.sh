#!/usr/bin/env bash
set -euo pipefail

# Experimental only: try the ROCm 10 hipBLASLt device path for gfx906 in an
# isolated prefix. The stable MI50 distribution never consumes this prefix.
# Exit 78 means the source configured but did not produce a usable kernel
# catalog; callers may treat that as an unavailable forward-port candidate.
SOURCE_ROOT=""
ROCM_ROOT="${ROCM_PATH:-}"
BUILD_DIR=""
INSTALL_DIR=""
JOBS="${MAX_JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)}"
CLEAN=0

usage() {
  cat <<'EOF'
Usage: build_hipblaslt_gfx906_experimental.sh HIPBLASLT_SOURCE [options]

Options:
  --rocm DIR       MI50 ROCm installation (default: ROCM_PATH).
  --build-dir DIR  Isolated CMake build directory.
  --install-dir DIR Isolated installation prefix.
  --jobs N         Ninja parallel jobs (default: MAX_JOBS/CPU count).
  --clean          Remove only the selected build directory first.
  --help           Show this help.

The command enables only the hipBLASLt host/device core for gfx906. ExtOps,
MatrixTransform, client programs and tests stay disabled. This is not part of
the supported ROCm package until a non-empty gfx906 kernel catalog is emitted
and passes hardware correctness tests.
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
[[ -x "$ROCM_ROOT/bin/hipcc" && -x "$ROCM_ROOT/bin/hipconfig" ]] || {
  echo "hipcc/hipconfig are missing from $ROCM_ROOT" >&2
  exit 2
}
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
BUILD_DIR="${BUILD_DIR:-${SOURCE_ROOT}/build-mi50-gfx906-experimental}"
INSTALL_DIR="${INSTALL_DIR:-${BUILD_DIR}/install}"
BUILD_DIR="$(mkdir -p "$BUILD_DIR" && cd "$BUILD_DIR" && pwd)"
INSTALL_DIR="$(mkdir -p "$INSTALL_DIR" && cd "$INSTALL_DIR" && pwd)"
if [[ "$CLEAN" -eq 1 ]]; then
  [[ "$BUILD_DIR" != "/" && "$BUILD_DIR" != "$SOURCE_ROOT" ]] || { echo "refusing to clean unsafe path" >&2; exit 6; }
  cmake -E rm -rf "$BUILD_DIR"
  mkdir -p "$BUILD_DIR"
fi

# The generated Tensile validation command captures PATH at configure time.
# Put the selected installation first so it cannot silently resolve /opt/rocm
# from a host environment.
export ROCM_PATH="$ROCM_ROOT"
export HIP_PATH="$ROCM_ROOT"
export PATH="$ROCM_ROOT/bin:$ROCM_ROOT/lib/llvm/bin:/usr/local/bin:/usr/bin:/bin"

cmake -S "$SOURCE_ROOT" -B "$BUILD_DIR" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
  -DCMAKE_C_COMPILER="$ROCM_ROOT/lib/llvm/bin/amdclang" \
  -DCMAKE_CXX_COMPILER="$ROCM_ROOT/lib/llvm/bin/amdclang++" \
  -DCMAKE_PREFIX_PATH="$ROCM_ROOT" \
  -DROCM_PATH="$ROCM_ROOT" \
  -DGPU_TARGETS=gfx906 \
  -DHIPBLASLT_ENABLE_DEVICE=ON \
  -DHIPBLASLT_ENABLE_EXTOPS=OFF \
  -DHIPBLASLT_ENABLE_MATRIX_TRANSFORM=OFF \
  -DHIPBLASLT_ENABLE_CLIENT=OFF \
  -DHIPBLASLT_ENABLE_HOST=ON \
  -DBUILD_TESTING=OFF \
  -DTENSILELITE_ENABLE_HOST=ON
cmake --build "$BUILD_DIR" --parallel "$JOBS"
cmake --install "$BUILD_DIR"

mapfile -t CODE_OBJECTS < <(find "$INSTALL_DIR" -type f \( -name '*.co' -o -name '*.hsaco' \) -print | sort)
if [[ "${#CODE_OBJECTS[@]}" -eq 0 ]]; then
  echo "hipBLASLt experimental build emitted no gfx906 code object" >&2
  exit 78
fi
for object in "${CODE_OBJECTS[@]}"; do
  header="$("$ROCM_ROOT/lib/llvm/bin/llvm-readelf" --file-header "$object" 2>/dev/null)"
  if ! grep -q 'gfx906' <<<"$header"; then
    echo "code object is not marked gfx906: $object" >&2
    exit 7
  fi
  # llvm-readelf renders the index as two fields (`[ 7]`), so the section
  # name is field 3 and the size is field 7 in its stable GNU-like format.
  text_size="$("$ROCM_ROOT/lib/llvm/bin/llvm-readelf" --sections "$object" 2>/dev/null | awk '$3 == ".text" {print $7; exit}')"
  if [[ -z "$text_size" || "$text_size" =~ ^0+$ ]]; then
    echo "gfx906 code object has no executable .text section: $object" >&2
    exit 78
  fi
done
echo "experimental hipBLASLt gfx906 build produced ${#CODE_OBJECTS[@]} non-empty code object(s) in $INSTALL_DIR"
