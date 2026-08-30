#!/usr/bin/env bash
set -euo pipefail

# Compile a tiny HIP translation unit for the real MI50 ISA.  This is a
# host-only check; no runtime API is called and no GPU is required.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python3 "${ROOT_DIR}/scripts/mi50_features.py" --check-environment
ROCM_PATH="${ROCM_PATH:-/opt/rocm-mi50}"
source "${ROOT_DIR}/scripts/mi50_rocm_environment.sh"
mi50_export_rocm_environment "${ROCM_PATH}"
HIPCC="${ROCM_PATH}/bin/hipcc"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/out/hip-smoke}"
OUTPUT="${OUTPUT_DIR}/gfx906_compile_smoke"

if [[ ! -x "${HIPCC}" ]]; then
  echo "hipcc unavailable at ${HIPCC}; HIP compile smoke is pending-build" >&2
  exit 77
fi

mkdir -p "${OUTPUT_DIR}"
"${HIPCC}" \
  --offload-arch=gfx906 \
  -O3 \
  -std=c++17 \
  "${ROOT_DIR}/tests/hip/gfx906_compile_smoke.hip" \
  -o "${OUTPUT}"

# An ELF note or embedded metadata must retain the native target.  `strings`
# is a conservative fallback when llvm-readelf is not installed; it does not
# replace the stricter code-object validator used for a packaged artifact.
if command -v llvm-readelf >/dev/null 2>&1; then
  if ! {
    llvm-readelf --notes "${OUTPUT}"
    llvm-readelf --sections "${OUTPUT}"
    strings "${OUTPUT}"
  } 2>/dev/null | grep -qi gfx906; then
    echo "compiled HIP binary has no gfx906 metadata" >&2
    exit 1
  fi
elif ! strings "${OUTPUT}" | grep -qi gfx906; then
  echo "compiled HIP binary has no gfx906 marker" >&2
  exit 1
fi

echo "native gfx906 HIP compile smoke passed: ${OUTPUT}"
