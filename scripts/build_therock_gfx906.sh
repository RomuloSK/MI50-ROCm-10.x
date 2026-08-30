#!/usr/bin/env bash
set -euo pipefail

# Configure/build the pinned TheRock ROCm 10.0 source for gfx906.
# This script only builds artifacts. It never enables HSA_OVERRIDE_GFX_VERSION.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python3 "${ROOT_DIR}/scripts/mi50_features.py" --check-environment
SOURCE_ROOT=""
FETCH_ROOT=""
BUILD_DIR="${ROOT_DIR}/out/build/gfx906"
ARTIFACT_DIR="${ROOT_DIR}/out/artifacts/gfx906"
JOBS="${JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)}"
CONFIGURE_ONLY=0
SKIP_AUDIT=0
MIN_FREE_GIB="${MI50_MIN_FREE_GIB:-32}"
BUILD_PROFILE="${MI50_BUILD_PROFILE:-full}"
BUILD_PYTHON_PACKAGES="${MI50_BUILD_PYTHON_PACKAGES:-ON}"
EXPERIMENTAL_NEW_ISA_PORTS="${MI50_ENABLE_EXPERIMENTAL_NEW_ISA_PORTS:-OFF}"
if ! [[ "${MIN_FREE_GIB}" =~ ^[0-9]+$ ]] || [[ "${MIN_FREE_GIB}" -lt 1 ]]; then
  echo "MI50_MIN_FREE_GIB must be a positive integer (got ${MIN_FREE_GIB})" >&2
  exit 2
fi
case "${BUILD_PROFILE}" in
  full|inference) ;;
  *)
    echo "MI50_BUILD_PROFILE must be full or inference (got ${BUILD_PROFILE})" >&2
    exit 2
    ;;
esac
case "${BUILD_PYTHON_PACKAGES^^}" in
  ON|OFF) ;;
  *)
    echo "MI50_BUILD_PYTHON_PACKAGES must be ON or OFF (got ${BUILD_PYTHON_PACKAGES})" >&2
    exit 2
    ;;
esac
case "${EXPERIMENTAL_NEW_ISA_PORTS^^}" in
  ON|OFF) ;;
  *)
    echo "MI50_ENABLE_EXPERIMENTAL_NEW_ISA_PORTS must be ON or OFF (got ${EXPERIMENTAL_NEW_ISA_PORTS})" >&2
    exit 2
    ;;
esac

usage() {
  cat <<'EOF'
Usage: build_therock_gfx906.sh [options]

  --source-root DIR   Existing TheRock checkout at therock-10.0.
  --fetch-root DIR    Fetch all pinned repositories below DIR.
  --build-dir DIR     CMake build directory.
  --artifact-dir DIR  Artifact output directory.
  --jobs N            Parallel build jobs (default: CPU count).
  --configure-only    Stop after configure and gfx906 metadata validation.
  --skip-audit        Reuse an existing source audit (useful for incremental
                      resumes after an infrastructure interruption).
  --help              Show this help.

The build uses THEROCK_AMDGPU_FAMILIES=gfx906 and produces an unofficial
community artifact. GPU execution remains pending until real MI50 hardware is
available.

MI50_BUILD_PROFILE=full (default) builds the complete enabled TheRock graph.
MI50_BUILD_PROFILE=inference keeps core, communication, math and ML libraries
but disables debug, profiler, data-center, media, emulation and storage groups
to produce a smaller llama.cpp/PyTorch bring-up artifact.
MI50_BUILD_PYTHON_PACKAGES=ON (default) builds the split ROCm SDK wheels and
gfx906 device wheel after the native artifact graph; set it to OFF for a
faster native-only iteration.
MI50_ENABLE_EXPERIMENTAL_NEW_ISA_PORTS=OFF (default) keeps newer-ISA
forward-port candidates disabled in the stable artifact; set it to ON only
for a deliberately isolated experimental build and hardware test.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source-root) SOURCE_ROOT="$2"; shift 2 ;;
    --fetch-root) FETCH_ROOT="$2"; shift 2 ;;
    --build-dir) BUILD_DIR="$2"; shift 2 ;;
    --artifact-dir) ARTIFACT_DIR="$2"; shift 2 ;;
    --jobs) JOBS="$2"; shift 2 ;;
    --configure-only) CONFIGURE_ONLY=1; shift ;;
    --skip-audit) SKIP_AUDIT=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -n "$SOURCE_ROOT" && -n "$FETCH_ROOT" ]]; then
  echo "choose either --source-root or --fetch-root, not both" >&2
  exit 2
fi

clone_pinned() {
  local name="$1" url="$2" ref="$3" commit="$4" destination="${5:-${FETCH_ROOT}/${name}}"
  mkdir -p "$(dirname "$destination")"
  # A normal clone has a .git directory, while a TheRock submodule has a .git
  # file pointing into the superproject's modules directory. Accept either.
  if [[ ! -e "${destination}/.git" ]]; then
    # TheRock's 10.0 baseline is an annotated tag, while the matching ROCm
    # repositories use release branches.  `git clone --branch` only accepts a
    # branch/tag that exists under the advertised name; probe branches first
    # and fall back to the default checkout for tag-only refs. The exact commit
    # below remains the authority for reproducibility.
    if git ls-remote --exit-code --heads "$url" "refs/heads/${ref}" >/dev/null 2>&1; then
      git clone --branch "$ref" --no-tags "$url" "$destination"
    else
      git clone --no-tags "$url" "$destination"
    fi
  fi
  # Keep Linux shebangs and helper scripts in LF form even when this script is
  # launched from Git for Windows with core.autocrlf enabled globally.
  git -C "$destination" config core.autocrlf false
  if ! git -C "$destination" fetch --no-tags origin "$ref"; then
    # A tag-only ref may not be fetched by name from a no-tags clone; fetching
    # the locked object id works for both annotated tags and branch commits.
    git -C "$destination" fetch --no-tags origin "$commit"
  fi
  git -C "$destination" checkout --detach "$commit"
  git -C "$destination" submodule update --init --recursive
  echo "pinned ${name} at $(git -C "$destination" rev-parse HEAD)"
}

if [[ -n "$FETCH_ROOT" ]]; then
  THE_ROCK_ROOT="${FETCH_ROOT}/TheRock"
  clone_pinned "TheRock" "https://github.com/ROCm/TheRock.git" "therock-10.0" "16adc4d875fd4f65ea23c7c84e1c66706fde3047" "$THE_ROCK_ROOT"

  # TheRock's build graph consumes these repositories as submodules. Keep the
  # exact lock-file commits, but place them at the paths CMake expects instead
  # of leaving sibling checkouts that are never discovered.
  clone_pinned "rocm-libraries" "https://github.com/ROCm/rocm-libraries.git" "release/therock-10.0" "8d1ae90eff7d022f26019ec55b2ec6a7674b3112" "$THE_ROCK_ROOT/rocm-libraries"
  clone_pinned "rocm-systems" "https://github.com/ROCm/rocm-systems.git" "release/therock-10.0" "6b0e43f341195e203754e08f850e437ff2fc09f9" "$THE_ROCK_ROOT/rocm-systems"
  clone_pinned "llvm-project" "https://github.com/ROCm/llvm-project.git" "release/therock-10.0" "8f497e0992fb7513f7f78a6f6b6f1056c375e961" "$THE_ROCK_ROOT/compiler/amd-llvm"
  SOURCE_ROOT="${FETCH_ROOT}/TheRock"
fi

if [[ -z "$SOURCE_ROOT" ]]; then
  echo "provide --source-root or --fetch-root" >&2
  exit 2
fi
python3 "${ROOT_DIR}/scripts/verify_patch_lock.py" \
  --repository-root "${ROOT_DIR}" \
  --lock-file "${ROOT_DIR}/sources.lock.json" \
  --json-out "${ROOT_DIR}/out/patch-lock-verification.json" \
  --strict
SOURCE_ROOT="$(cd "$SOURCE_ROOT" && pwd)"
if [[ ! -f "${SOURCE_ROOT}/CMakeLists.txt" ]]; then
  echo "not a TheRock source tree: ${SOURCE_ROOT}" >&2
  exit 2
fi
# Keep generated archives and compiler metadata stable across runs.  The
# source commit timestamp is the default; callers may provide an audited
# SOURCE_DATE_EPOCH explicitly for a hermetic release build.
export SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH:-$(git -C "$SOURCE_ROOT" show -s --format=%ct HEAD)}"
export TZ=UTC
export LC_ALL=C

normalize_linux_helpers() {
  # A checkout created by Git for Windows can retain CRLF in executable
  # helpers even after core.autocrlf is changed.  WSL's /bin/bash interprets a
  # CR at the end of a shebang or option as part of the token (for example,
  # `-r\r`), so normalize only helper scripts before CMake invokes them.
  if [[ "$(uname -s)" != "Linux" ]]; then
    return
  fi
  local helper
  while IFS= read -r -d '' helper; do
    if LC_ALL=C grep -q $'\r' "$helper"; then
      sed -i 's/\r$//' "$helper"
    fi
  done < <(find "$SOURCE_ROOT" -type f \( -name '*.sh' -o -name '*.py' \) -print0)
}

normalize_linux_helpers

apply_patch_queue() {
  local patch_file patch_root
  shopt -s nullglob
  local patch_files=("${ROOT_DIR}/patches"/*.patch)
  shopt -u nullglob
  for patch_file in "${patch_files[@]}"; do
    # TheRock keeps ROCr in the rocm-systems submodule.  Applying every patch
    # from the superproject silently fails because git treats submodule files
    # as a single gitlink.  Probe the superproject and known submodules and
    # apply at the repository that owns the paths in the patch.
    local applied=0
    local -a patch_roots=("$SOURCE_ROOT")
    for patch_root in \
      "$SOURCE_ROOT/rocm-systems" \
      "$SOURCE_ROOT/rocm-libraries" \
      "$SOURCE_ROOT/compiler/amd-llvm"; do
      if [[ -d "$patch_root" ]]; then
        patch_roots+=("$patch_root")
      fi
    done

    for patch_root in "${patch_roots[@]}"; do
      if git -C "$patch_root" apply --ignore-space-change --unidiff-zero --check "$patch_file" >/dev/null 2>&1; then
        git -C "$patch_root" apply --ignore-space-change --unidiff-zero "$patch_file"
        echo "applied $(basename "$patch_file") in ${patch_root#$SOURCE_ROOT/}"
        applied=1
        break
      fi
      if git -C "$patch_root" apply --ignore-space-change --reverse --check --unidiff-zero "$patch_file" >/dev/null 2>&1; then
        echo "already applied $(basename "$patch_file") in ${patch_root#$SOURCE_ROOT/}"
        applied=1
        break
      fi
    done

    # A later policy patch can intentionally narrow an earlier opt-in patch.
    # Treat the earlier patch as satisfied when that superseding patch is
    # already applied; this keeps reruns idempotent after a source audit adds
    # a hardware-specific fallback classification.
    if [[ "$(basename "$patch_file")" == "0001-gfx906-forward-port-target-policy.patch" ]]; then
      local superseding_patch="${ROOT_DIR}/patches/0007-hipsparselt-gfx906-fallback-policy.patch"
      if [[ -f "$superseding_patch" ]] && \
         git -C "$SOURCE_ROOT" apply --ignore-space-change --reverse --check --unidiff-zero "$superseding_patch" >/dev/null 2>&1; then
        echo "superseded $(basename "$patch_file") by $(basename "$superseding_patch")"
        applied=1
      fi
    fi

    # Patch 0011 intentionally extends the 0010 ROCclr hunk. Once both are
    # present, git cannot reverse-check 0010 in isolation because its original
    # post-image has been extended by the environment override. Treat the
    # exact newer patch as satisfying the earlier one, just as above.
    if [[ "$(basename "$patch_file")" == "0010-rocclr-optional-opengl-for-hip.patch" ]]; then
      local superseding_patch="${ROOT_DIR}/patches/0011-rocclr-opengl-env-override.patch"
      if [[ -f "$superseding_patch" ]]; then
        for patch_root in "${patch_roots[@]}"; do
          if git -C "$patch_root" apply --ignore-space-change --reverse --check --unidiff-zero "$superseding_patch" >/dev/null 2>&1; then
            echo "superseded $(basename "$patch_file") by $(basename "$superseding_patch")"
            applied=1
            break
          fi
        done
      fi
    fi

    # Patch 0021 adds the stable rocBLAS hipBLASLt switch immediately after
    # context introduced by 0018. Once 0021 is applied, Git cannot
    # reverse-check 0018 as an isolated patch even though all of its changes
    # remain present; treat the exact newer patch as satisfying 0018.
    if [[ "$(basename "$patch_file")" == "0018-therock-skip-unsupported-hipblaslt-activation.patch" ]]; then
      local superseding_patch="${ROOT_DIR}/patches/0021-therock-disable-rocblas-hipblaslt-linkage.patch"
      if [[ -f "$superseding_patch" ]] && \
         git -C "$SOURCE_ROOT" apply --ignore-space-change --reverse --check --unidiff-zero "$superseding_patch" >/dev/null 2>&1; then
        echo "superseded $(basename "$patch_file") by $(basename "$superseding_patch")"
        applied=1
      fi
    fi

    # Patch 0026 extends the MIOpen dependency-policy block introduced by
    # 0020. Once the nested-configure forwarding patch is present, Git cannot
    # reverse-check 0020 as an isolated hunk even though its guards remain;
    # treat the exact newer patch as satisfying 0020 on reruns.
    if [[ "$(basename "$patch_file")" == "0020-therock-guard-hipblaslt-dnn-dependencies.patch" ]]; then
      local superseding_patch="${ROOT_DIR}/patches/0026-therock-pass-miopen-hipblaslt-policy.patch"
      if [[ -f "$superseding_patch" ]]; then
        for patch_root in "${patch_roots[@]}"; do
          if git -C "$patch_root" apply --ignore-space-change --reverse --check --unidiff-zero "$superseding_patch" >/dev/null 2>&1; then
            echo "superseded $(basename "$patch_file") by $(basename "$superseding_patch")"
            applied=1
            break
          fi
        done
      fi
    fi

    if [[ "$applied" -eq 0 ]]; then
      echo "patch cannot be applied cleanly to TheRock or its source submodules: $patch_file" >&2
      exit 6
    fi
  done
}

apply_patch_queue

# CMake otherwise reports this several minutes into dependency discovery. Give
# pre-hardware Windows/WSL users an actionable failure at the entry point.
if ! command -v cmake >/dev/null 2>&1; then
  echo "cmake is required but was not found in PATH" >&2
  exit 5
fi
if ! command -v ninja >/dev/null 2>&1; then
  echo "ninja is required but was not found in PATH" >&2
  exit 5
fi
if [[ -z "${CC:-}" ]] && ! command -v cc >/dev/null 2>&1 && \
   ! command -v gcc >/dev/null 2>&1 && ! command -v clang >/dev/null 2>&1 && \
   ! command -v cl >/dev/null 2>&1; then
  echo "no C compiler found; install clang/gcc or enter a Visual Studio toolchain" >&2
  exit 5
fi
if [[ -z "${CXX:-}" ]] && ! command -v c++ >/dev/null 2>&1 && \
   ! command -v g++ >/dev/null 2>&1 && ! command -v clang++ >/dev/null 2>&1 && \
   ! command -v cl >/dev/null 2>&1; then
  echo "no C++ compiler found; install clang++/g++ or enter a Visual Studio toolchain" >&2
  exit 5
fi
if ! python3 -c 'import CppHeaderParser' >/dev/null 2>&1; then
  echo "CppHeaderParser is required for HIP profiling-header generation; install CppHeaderParser==2.7.4" >&2
  exit 5
fi
if ! python3 -c 'import msgpack, zstandard; import joblib' >/dev/null 2>&1; then
  echo "joblib==1.5.1, msgpack==1.1.1 and zstandard==0.25.0 are required by TheRock artifact/device generation" >&2
  exit 5
fi
if ! python3 -c 'import magic' >/dev/null 2>&1; then
  echo "python3-magic is required to build ROCm SDK wheels; install python3-magic" >&2
  exit 5
fi
if ! command -v tclsh >/dev/null 2>&1; then
  echo "tclsh is required by rocprofiler-systems SQLite amalgamation generation; install tcl" >&2
  exit 5
fi

mkdir -p "${BUILD_DIR}" "${ARTIFACT_DIR}"
if [[ "$(uname -s)" == "Linux" ]]; then
  free_kib="$(df -Pk "${BUILD_DIR}" | awk 'NR == 2 {print $4}')"
  if [[ -n "${free_kib}" && "${free_kib}" -lt $((MIN_FREE_GIB * 1024 * 1024)) ]]; then
    echo "insufficient free space for a full TheRock build: ${free_kib} KiB available;" >&2
    echo "free at least ${MIN_FREE_GIB} GiB or set MI50_MIN_FREE_GIB to a justified lower value" >&2
    exit 5
  fi
fi
python3 "${ROOT_DIR}/scripts/verify_source_lock.py" \
  --source-root "$SOURCE_ROOT" \
  --lock-file "${ROOT_DIR}/sources.lock.json" \
  --json-out "${BUILD_DIR}/source-lock-verification.json" \
  --strict
if [[ "$SKIP_AUDIT" -eq 0 ]]; then
  python3 "${ROOT_DIR}/scripts/audit_gfx906.py" --root "$SOURCE_ROOT" --json-out "${BUILD_DIR}/gfx906-audit.json"
else
  echo "skipping source audit by request; existing audit metadata is retained"
fi

ENABLE_OPENCL="${MI50_ENABLE_OPENCL:-OFF}"
case "${ENABLE_OPENCL^^}" in
  ON|OFF) ;;
  *)
    echo "MI50_ENABLE_OPENCL must be ON or OFF (got ${ENABLE_OPENCL})" >&2
    exit 2
    ;;
esac

BUILD_TESTING="${MI50_BUILD_TESTING:-ON}"
case "${BUILD_TESTING^^}" in
  ON|OFF) ;;
  *)
    echo "MI50_BUILD_TESTING must be ON or OFF (got ${BUILD_TESTING})" >&2
    exit 2
    ;;
esac
BUILD_TESTING="${BUILD_TESTING^^}"
ENABLE_OPENCL="${ENABLE_OPENCL^^}"
# Record the normalized, actually-used settings in build provenance.  Merely
# assigning shell locals would leave these fields empty in the child Python
# process even though CMake received the intended values.
export MI50_BUILD_PROFILE="${BUILD_PROFILE}"
export MI50_BUILD_TESTING="${BUILD_TESTING}"
export MI50_BUILD_PYTHON_PACKAGES="${BUILD_PYTHON_PACKAGES^^}"
export MI50_ENABLE_EXPERIMENTAL_NEW_ISA_PORTS="${EXPERIMENTAL_NEW_ISA_PORTS^^}"
export MI50_ENABLE_OPENCL="${ENABLE_OPENCL}"
export MI50_MIN_FREE_GIB="${MIN_FREE_GIB}"

# TheRock configures ROCclr in a nested CMake invocation, so pass the optional
# OpenGL switch through the environment as well as the top-level cache.  This
# keeps HIP-only builds independent of host GL development packages.
export ROCCLR_ENABLE_OPENGL="${ROCCLR_ENABLE_OPENGL:-OFF}"

python3 "${ROOT_DIR}/scripts/write_build_provenance.py" \
  --repository-root "$ROOT_DIR" \
  --source-root "$SOURCE_ROOT" \
  --build-root "$BUILD_DIR" \
  --artifact-root "$ARTIFACT_DIR" \
  --output "${BUILD_DIR}/build-provenance.json"

cmake_args=(
  -S "$SOURCE_ROOT"
  -B "$BUILD_DIR"
  -G Ninja
  -DCMAKE_BUILD_TYPE=Release
  -DTHEROCK_AMDGPU_FAMILIES=gfx906
  -DTHEROCK_DIST_AMDGPU_FAMILIES=gfx906
  -DTHEROCK_TEST_AMDGPU_FAMILIES=gfx906
  -DMI50_ENABLE_FORWARD_PORTS=ON
  # The default expansion is the stable value `-DMI50_ENABLE_EXPERIMENTAL_NEW_ISA_PORTS=OFF`;
  # callers must opt in explicitly to newer-ISA forward-port experiments.
  -DMI50_ENABLE_EXPERIMENTAL_NEW_ISA_PORTS=${EXPERIMENTAL_NEW_ISA_PORTS^^}
  -DTHEROCK_BUILD_TESTING=${BUILD_TESTING}
  -DTHEROCK_ENABLE_HIP_RUNTIME=ON
  # OpenCL is optional in the Linux-first MI50 deliverable.  The TheRock
  # opencl-runtime artifact pulls in ocl-clr, whose optional host OpenGL
  # dependency is not required by ROCr/HIP/LLM inference and is not available
  # in the minimal Ubuntu builder.  Keep this explicit so a cached build
  # cannot silently re-enable the optional path; set MI50_ENABLE_OPENCL=ON
  # when building in an image that provides the OpenGL development packages.
  -DTHEROCK_ENABLE_OCL_RUNTIME=${ENABLE_OPENCL}
  -DTHEROCK_ENABLE_OCL_ICD=${ENABLE_OPENCL}
  # ROCclr otherwise requires host OpenGL development files even for HIP-only
  # compute builds.  Keep GL interop opt-in for the minimal Linux artifact;
  # users who need graphics sharing can set this back to ON in their build.
  -DROCCLR_ENABLE_OPENGL=${ROCCLR_ENABLE_OPENGL}
  -DTHEROCK_ENABLE_CORE_RUNTIME_TESTS=OFF
  -DTHEROCK_ENABLE_RCCL=ON
  -DTHEROCK_ENABLE_BLAS=ON
  -DTHEROCK_ENABLE_RAND=ON
  -DTHEROCK_ENABLE_SOLVER=ON
  -DTHEROCK_ENABLE_SPARSE=ON
  -DTHEROCK_ENABLE_FFT=ON
  -DTHEROCK_ENABLE_PRIM=ON
  -DTHEROCK_ENABLE_MIOPEN=ON
  -DTHEROCK_ENABLE_ROCWMMA=OFF
  -DTHEROCK_ENABLE_HIPSPARSELT=OFF
  -DTHEROCK_ENABLE_ROCPROFILER_COMPUTE=OFF
  -DROCROLLER_BUILD_TESTING=OFF
)

# The experimental switch must control the feature gates as well as the
# target-policy exclusion list. Without this conditional, setting
# MI50_ENABLE_EXPERIMENTAL_NEW_ISA_PORTS=ON would remove the target filters
# but still leave every candidate disabled, making the switch misleading.
if [[ "${EXPERIMENTAL_NEW_ISA_PORTS^^}" == "ON" ]]; then
  cmake_args+=(
    -DTHEROCK_ENABLE_HIPBLASLTPROVIDER=ON
    -DTHEROCK_ENABLE_HIPTENSOR=ON
    -DTHEROCK_ENABLE_COMPOSABLE_KERNEL=ON
  )
else
  cmake_args+=(
    # Stable release path: rocBLAS/Tensile and MIOpen legacy/assembly are the
    # only admitted MI50 backends until candidate catalogs pass validation.
    -DTHEROCK_ENABLE_HIPBLASLTPROVIDER=OFF
    -DTHEROCK_ENABLE_HIPTENSOR=OFF
    -DTHEROCK_ENABLE_COMPOSABLE_KERNEL=OFF
  )
fi

# A full graph is the release target.  The inference profile is an explicit,
# reproducible recovery mode for pre-hardware bring-up on hosts with limited
# storage; it does not change target policy or silently remove core math/ML
# dependencies used by PyTorch and llama.cpp.
if [[ "${BUILD_PROFILE}" == "inference" ]]; then
  cmake_args+=(
    -DTHEROCK_ENABLE_STORAGE_LIBS=OFF
    -DTHEROCK_ENABLE_DEBUG_TOOLS=OFF
    -DTHEROCK_ENABLE_PROFILER=OFF
    -DTHEROCK_ENABLE_DC_TOOLS=OFF
    -DTHEROCK_ENABLE_MEDIA_LIBS=OFF
    -DTHEROCK_ENABLE_EMULATION=OFF
  )
fi

if [[ -n "${ROCR_DEVICE_LIB_PATH:-}" ]]; then
  cmake_args+=("-DROCR_DEVICE_LIB_PATH=${ROCR_DEVICE_LIB_PATH}")
fi

echo "configuring TheRock for gfx906"
cmake "${cmake_args[@]}"
python3 "${ROOT_DIR}/scripts/verify_configure_gfx906.py" \
  --build-root "$BUILD_DIR" \
  --json-out "${BUILD_DIR}/gfx906-configure-verification.json" \
  --strict

if [[ "$CONFIGURE_ONLY" -eq 1 ]]; then
  echo "configure-only validation passed; build was not started"
  exit 0
fi

# An interrupted run can leave artifact slices whose payload files are 0-byte
# truncations of perfectly good files in the `<project>/stage` trees (observed
# for amd-llvm run/lib and hipify run, which is exactly the HIP compiler).
# Removing those slices before the build makes ninja re-run their Populate
# step from the intact stage trees instead of flattening empty files.
python3 "${ROOT_DIR}/scripts/repair_artifact_manifests.py" \
  --artifacts-dir "${BUILD_DIR}/artifacts" \
  --subprojects-json "${BUILD_DIR}/artifact_subprojects.json" \
  --heal --build-root "$BUILD_DIR" \
  --json-out "${BUILD_DIR}/artifact-manifest-heal.json"

echo "building TheRock with ${JOBS} jobs"
cmake --build "$BUILD_DIR" --parallel "$JOBS"

# TheRock's `artifact-flatten` is manifest driven: it copies only the prefixes
# listed in each slice's artifact_manifest.txt.  Observed in this gfx906 build,
# the amd-llvm run/lib and hipify run slices keep their payload on disk but end
# with a 0-byte manifest, which silently removes the LLVM/AMDGPU compiler,
# device bitcode, libamd_comgr, hipcc, hipconfig and hipify from dist/rocm and
# from the Python wheels.  Rebuild the manifests from the bytes that exist
# before anything consumes them.
python3 "${ROOT_DIR}/scripts/repair_artifact_manifests.py" \
  --artifacts-dir "${BUILD_DIR}/artifacts" \
  --subprojects-json "${BUILD_DIR}/artifact_subprojects.json" \
  --json-out "${BUILD_DIR}/artifact-manifest-repair.json" \
  --strict

# Fail before packaging if ROCr's target-scoped device objects were not
# produced.  This is deliberately independent of GPU execution: a successful
# host build is useful pre-hardware evidence, but it is not runtime support.
python3 "${ROOT_DIR}/scripts/validate_rocr_build.py" \
  --build-root "$BUILD_DIR" \
  --json-out "${BUILD_DIR}/gfx906-rocr-validation.json" \
  --strict

# The exact archive target can differ between TheRock revisions. Build it when
# available, but do not hide a successful normal build behind a target mismatch.
if cmake --build "$BUILD_DIR" --target help 2>/dev/null | grep -qE '(^|[[:space:]])therock-(dist|archives)($|[[:space:]])'; then
  # Once CMake advertises the distribution target, a failure here is a real
  # packaging failure.  Do not turn it into a misleading host-only success.
  if cmake --build "$BUILD_DIR" --target help 2>/dev/null | grep -qE '(^|[[:space:]])dist-rocm\+expunge($|[[:space:]])'; then
    # Remove a previous flattened tree first.  This prevents a changed
    # configuration (for example, a newly disabled forward-port candidate)
    # from leaking stale files into the next release archive.
    cmake --build "$BUILD_DIR" --target dist-rocm+expunge --parallel "$JOBS"
  fi
  cmake --build "$BUILD_DIR" --target therock-dist --parallel "$JOBS"
fi

# TheRock revisions are not consistent about publishing archives: some only
# leave a flattened dist/rocm tree.  Copy native archives/wheels when present
# and package that tree below when they are not.
found=0
while IFS= read -r -d '' file; do
  case "$(basename "$file")" in
    therock-dist-*|rocm-*|rocm_*|pytorch*|llama*) ;;
    *) continue ;;
  esac
  # The auxiliary overlay's dist_info.json describes an internal build
  # component, not the final MI50 distribution.  Generate our own metadata
  # after packaging instead of allowing this file to overwrite it.
  case "$(basename "$file")" in
    dist_info.json) continue ;;
  esac
  cp -f "$file" "$ARTIFACT_DIR/"
  found=1
done < <(find "$BUILD_DIR" -type f \( -name '*.tar.gz' -o -name '*.whl' -o -name 'dist_info.json' \) -print0 2>/dev/null)

DIST_ROOT="${BUILD_DIR}/dist/rocm"
if [[ -d "$DIST_ROOT" ]]; then
  # With KPACK split enabled, therock-dist is the target-specific flattened
  # tree and generic core runtime/HIP files remain in split artifact slices.
  # Merge run/lib/dev slices into the same install prefix before packaging so
  # the tarball is usable on a fresh host (and contains the gfx906 device
  # slices as well as the generic headers and runtime).
  mapfile -t package_artifacts < <(
    python3 - "${BUILD_DIR}" <<'PY'
import json
import re
import sys
from pathlib import Path

build_root = Path(sys.argv[1])
manifest = build_root / "artifact_subprojects.json"
active = set(json.loads(manifest.read_text(encoding="utf-8"))) if manifest.is_file() else None
pattern = re.compile(r"^(.+)_(run|lib|dev)_(.+)$")
for candidate in sorted((build_root / "artifacts").iterdir()):
    if not candidate.is_dir() or not (candidate / "artifact_manifest.txt").is_file():
        continue
    match = pattern.match(candidate.name)
    if not match:
        continue
    if active is not None and match.group(1) not in active:
        continue
    print(candidate)
PY
  )
  if [[ "${#package_artifacts[@]}" -gt 0 ]]; then
    python3 "${SOURCE_ROOT}/build_tools/fileset_tool.py" artifact-flatten \
      -o "$DIST_ROOT" "${package_artifacts[@]}"
  fi
  # The RDC test binary is installed beside its FetchContent-built GTest
  # shared libraries.  Some TheRock CMake/RPATH normalization drops the bare
  # $ORIGIN entry, leaving a binary that cannot find those co-located test
  # libraries on a clean install.  Restore the relative entry before running
  # the dynamic-link gate so the archive remains self-contained.
  RDC_TEST_DIR="${DIST_ROOT}/share/rdc/rdctst_tests"
  RDC_TEST_BIN="${RDC_TEST_DIR}/rdctst"
  if [[ -x "$RDC_TEST_BIN" && -f "${RDC_TEST_DIR}/libgtest.so.1.14.0" && -f "${RDC_TEST_DIR}/libgtest_main.so.1.14.0" ]]; then
    if ! command -v patchelf >/dev/null 2>&1; then
      echo "patchelf is required to repair the RDC test RUNPATH" >&2
      exit 4
    fi
    RDC_TEST_RPATH="$(patchelf --print-rpath "$RDC_TEST_BIN" 2>/dev/null || true)"
    case ":${RDC_TEST_RPATH}:" in
      *':$ORIGIN:'*) ;;
      *)
        if [[ -n "$RDC_TEST_RPATH" ]]; then
          RDC_TEST_RPATH="\$ORIGIN:${RDC_TEST_RPATH}"
        else
          RDC_TEST_RPATH="\$ORIGIN"
        fi
        patchelf --set-rpath "$RDC_TEST_RPATH" "$RDC_TEST_BIN"
        ;;
    esac
  fi
  # Gate the merged tree on the files that make a ROCm distribution usable.
  # This is host-only evidence: it proves the package is complete, not that an
  # MI50 can execute it.
  python3 "${ROOT_DIR}/scripts/validate_dist_contents.py" \
    --dist-dir "$DIST_ROOT" \
    --target gfx906 \
    --json-out "${BUILD_DIR}/gfx906-dist-contents.json" \
    --strict
  python3 "${ROOT_DIR}/scripts/validate_elf_dependencies.py" \
    --root "$DIST_ROOT" \
    --json-out "${BUILD_DIR}/gfx906-elf-dependencies.json" \
    --strict
  python3 "${ROOT_DIR}/scripts/package_rocm_gfx906.py" \
    --source-dir "$DIST_ROOT" \
    --output "${ARTIFACT_DIR}/rocm-10.0.0+mi50.5-mi50-gfx906-linux.tar.gz" \
    --metadata-output "${ARTIFACT_DIR}/dist_info.json" \
    --version "10.0.0+mi50.5" \
    --target gfx906
  found=1
fi

if [[ "${BUILD_PYTHON_PACKAGES^^}" == "ON" && -f "${SOURCE_ROOT}/build_tools/build_python_packages.py" ]]; then
  PYTHON_PACKAGE_ROOT="${ARTIFACT_DIR}/python"
  cmake -E rm -rf "$PYTHON_PACKAGE_ROOT"
  mkdir -p "$PYTHON_PACKAGE_ROOT"
  python3 "${SOURCE_ROOT}/build_tools/build_python_packages.py" \
    --artifact-dir "${BUILD_DIR}/artifacts" \
    --dest-dir "$PYTHON_PACKAGE_ROOT" \
    --version "10.0.0+mi50.5" \
    --linux-amdgpu-families gfx906 \
    --no-wheel-compression
  python3 "${ROOT_DIR}/scripts/filter_rocm_wheels.py" \
    --package-dir "${PYTHON_PACKAGE_ROOT}/dist" \
    > "${PYTHON_PACKAGE_ROOT}/wheel-filter.json"
  python3 "${ROOT_DIR}/scripts/validate_python_packages.py" \
    --package-dir "${PYTHON_PACKAGE_ROOT}/dist" \
    --target gfx906 \
    --version "10.0.0+mi50.5" \
    --json-out "${PYTHON_PACKAGE_ROOT}/python-package-validation.json" \
    --strict
  found=1
fi

if [[ "$found" -eq 0 ]]; then
  echo "no distributable artifacts found below ${BUILD_DIR}" >&2
  echo "inspect ${BUILD_DIR} and the CMake target list before retrying" >&2
  exit 4
fi

cp -f "${BUILD_DIR}/build-provenance.json" "${ARTIFACT_DIR}/build-provenance.json"
cp -f "${BUILD_DIR}/source-lock-verification.json" "${ARTIFACT_DIR}/source-lock-verification.json"
cp -f "${ROOT_DIR}/out/patch-lock-verification.json" "${ARTIFACT_DIR}/patch-lock-verification.json"
cp -f "${BUILD_DIR}/gfx906-rocr-validation.json" "${ARTIFACT_DIR}/gfx906-rocr-validation.json"

python3 "${ROOT_DIR}/scripts/validate_artifacts.py" \
  --artifact-root "$ARTIFACT_DIR" \
  --json-out "${ARTIFACT_DIR}/artifact-validation.json" \
  --strict
python3 "${ROOT_DIR}/scripts/mi50_features.py" --json > "${ARTIFACT_DIR}/mi50_features.json"

echo "artifacts written to ${ARTIFACT_DIR}"
