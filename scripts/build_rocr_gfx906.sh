#!/usr/bin/env bash
set -euo pipefail

# Build the patched ROCr runtime directly. This is the smallest useful Linux
# artifact and is intentionally separate from the full TheRock distribution:
# it gives us a fast, deterministic compile/link gate before MI50 hardware is
# available.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python3 "${ROOT_DIR}/scripts/mi50_features.py" --check-environment
SOURCE_ROOT=""
SOURCE_REPO_ROOT="${ROCR_SOURCE_REPO_ROOT:-}"
BUILD_DIR="${ROOT_DIR}/out/build/rocr-gfx906"
INSTALL_PREFIX="${ROOT_DIR}/out/install/rocr-gfx906"
JOBS="${JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)}"
DO_INSTALL=1

usage() {
  cat <<'EOF'
Usage: build_rocr_gfx906.sh --source-root DIR [options]

  --source-root DIR     Patched rocm-systems/projects/rocr-runtime checkout.
  --source-repo-root DIR Parent rocm-systems checkout (for commit verification).
  --build-dir DIR       CMake/Ninja build directory.
  --install-prefix DIR  Install prefix (default: out/install/rocr-gfx906).
  --device-lib-path DIR Optional AMDGPU bitcode directory for image kernels.
  --jobs N              Parallel build jobs.
  --no-install          Build and validate without installing.
  --help                Show this help.

The source must contain the MI50 patches, including ROCR_TARGET_DEVICES and
ROCR_TARGET_DEVICES_GFX906. No ISA override is set or accepted.
EOF
}

DEVICE_LIB_PATH="${ROCR_DEVICE_LIB_PATH:-}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --source-root) SOURCE_ROOT="$2"; shift 2 ;;
    --source-repo-root) SOURCE_REPO_ROOT="$2"; shift 2 ;;
    --build-dir) BUILD_DIR="$2"; shift 2 ;;
    --install-prefix) INSTALL_PREFIX="$2"; shift 2 ;;
    --device-lib-path) DEVICE_LIB_PATH="$2"; shift 2 ;;
    --jobs) JOBS="$2"; shift 2 ;;
    --no-install) DO_INSTALL=0; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "this direct ROCr build is Linux-only; use WSL or a Linux host" >&2
  exit 2
fi
if [[ -z "$SOURCE_ROOT" ]]; then
  echo "--source-root is required" >&2
  exit 2
fi
SOURCE_ROOT="$(cd "$SOURCE_ROOT" && pwd)"
if [[ -z "$SOURCE_REPO_ROOT" ]]; then
  SOURCE_REPO_ROOT="$(git -C "$SOURCE_ROOT" rev-parse --show-toplevel 2>/dev/null || true)"
fi
if [[ -z "$SOURCE_REPO_ROOT" || ! -d "$SOURCE_REPO_ROOT" ]]; then
  echo "could not determine the parent rocm-systems checkout; pass --source-repo-root" >&2
  exit 2
fi
SOURCE_REPO_ROOT="$(cd "$SOURCE_REPO_ROOT" && pwd)"
if [[ ! -f "${SOURCE_ROOT}/CMakeLists.txt" ]]; then
  echo "not a ROCr source tree: ${SOURCE_ROOT}" >&2
  exit 2
fi
export SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH:-$(git -C "$SOURCE_ROOT" show -s --format=%ct HEAD)}"
export TZ=UTC
export LC_ALL=C
if [[ "$(uname -s)" == "Linux" ]]; then
  while IFS= read -r -d '' helper; do
    if LC_ALL=C grep -q $'\r' "$helper"; then
      sed -i 's/\r$//' "$helper"
    fi
  done < <(find "$SOURCE_ROOT" -type f \( -name '*.sh' -o -name '*.py' \) -print0)
fi
if ! grep -q "ROCR_TARGET_DEVICES" "${SOURCE_ROOT}/CMakeLists.txt"; then
  echo "source is missing the gfx906 target-scoping patch: ${SOURCE_ROOT}" >&2
  exit 6
fi
if ! grep -Rqs "ROCR_TARGET_DEVICES_GFX906" "${SOURCE_ROOT}/runtime"; then
  echo "source is missing the gfx906 shader-table safety patch: ${SOURCE_ROOT}" >&2
  exit 6
fi
for tool in cmake ninja python3; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "${tool} is required but was not found in PATH" >&2
    exit 5
  fi
done

cmake_args=(
  -S "$SOURCE_ROOT"
  -B "$BUILD_DIR"
  -G Ninja
  -DCMAKE_BUILD_TYPE=Release
  -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX"
  -DBUILD_SHARED_LIBS=ON
  -DBUILD_ROCR=ON
  -DBUILD_THUNK_VIRTIO=OFF
  -DIMAGE_SUPPORT=ON
  -DROCR_TARGET_DEVICES=gfx906
)
if [[ -n "$DEVICE_LIB_PATH" ]]; then
  cmake_args+=("-DROCR_DEVICE_LIB_PATH=${DEVICE_LIB_PATH}")
fi
if [[ -n "${Clang_DIR:-}" ]]; then
  cmake_args+=("-DClang_DIR=${Clang_DIR}")
fi
if [[ -n "${LLVM_DIR:-}" ]]; then
  cmake_args+=("-DLLVM_DIR=${LLVM_DIR}")
fi

mkdir -p "$BUILD_DIR"
python3 "${ROOT_DIR}/scripts/verify_patch_lock.py" \
  --repository-root "${ROOT_DIR}" \
  --lock-file "${ROOT_DIR}/sources.lock.json" \
  --json-out "${ROOT_DIR}/out/patch-lock-verification.json" \
  --strict
python3 "${ROOT_DIR}/scripts/verify_source_lock.py" \
  --source-root "$SOURCE_REPO_ROOT" \
  --repository-name rocm-systems \
  --repository-path "$SOURCE_REPO_ROOT" \
  --lock-file "${ROOT_DIR}/sources.lock.json" \
  --json-out "${BUILD_DIR}/source-lock-verification.json" \
  --strict
python3 "${ROOT_DIR}/scripts/write_build_provenance.py" \
  --repository-root "$ROOT_DIR" \
  --source-root "$SOURCE_ROOT" \
  --build-root "$BUILD_DIR" \
  --output "${BUILD_DIR}/build-provenance.json"
echo "configuring ROCr for gfx906"
cmake "${cmake_args[@]}"
echo "building ROCr with ${JOBS} jobs"
cmake --build "$BUILD_DIR" --parallel "$JOBS"
python3 "${ROOT_DIR}/scripts/validate_rocr_build.py" \
  --build-root "$BUILD_DIR" \
  --json-out "${BUILD_DIR}/gfx906-rocr-validation.json" \
  --strict

if [[ "$DO_INSTALL" -eq 1 ]]; then
  cmake --install "$BUILD_DIR"
  cp -f "${BUILD_DIR}/build-provenance.json" "${INSTALL_PREFIX}/build-provenance.json"
  cp -f "${BUILD_DIR}/source-lock-verification.json" "${INSTALL_PREFIX}/source-lock-verification.json"
  cp -f "${ROOT_DIR}/out/patch-lock-verification.json" "${INSTALL_PREFIX}/patch-lock-verification.json"
  cp -f "${BUILD_DIR}/gfx906-rocr-validation.json" "${INSTALL_PREFIX}/gfx906-rocr-validation.json"
  python3 "${ROOT_DIR}/scripts/rocr_host_smoke.py" \
    --library "${INSTALL_PREFIX}/lib/libhsa-runtime64.so" \
    --json-out "${BUILD_DIR}/gfx906-rocr-host-smoke.json" \
    --strict
  python3 "${ROOT_DIR}/scripts/mi50_features.py" --json > "${INSTALL_PREFIX}/mi50_features.json"
  echo "installed target-scoped ROCr at ${INSTALL_PREFIX}"
fi
echo "ROCr gfx906 artifact is host-validated; GPU execution remains pending-hardware"
