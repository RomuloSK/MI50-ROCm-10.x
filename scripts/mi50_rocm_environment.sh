#!/usr/bin/env bash

# Export a complete user-space ROCm environment for standalone MI50 smokes.
# The function intentionally does not change shell options; callers may source
# this file from a shell that owns its own error-handling policy.
mi50_export_rocm_environment() {
  local mi50_root="${1:?ROCm root is required}"
  local mi50_override
  for mi50_override in HSA_OVERRIDE_GFX_VERSION ROCR_OVERRIDE_GFX_VERSION; do
    if [[ -n "${!mi50_override:-}" ]]; then
      echo "ISA override is not allowed: ${mi50_override}" >&2
      return 6
    fi
  done
  # The installer publishes an archive as PREFIX/rocm, while direct builds
  # commonly use the rocm directory itself.  Accept both layouts, but do not
  # mistake a direct tree that happens to contain an unrelated rocm/ folder.
  if [[ -d "${mi50_root}/rocm" && ! -d "${mi50_root}/bin" ]]; then
    mi50_root="${mi50_root}/rocm"
  fi
  export ROCM_PATH="${mi50_root}"
  export ROCM_HOME="${mi50_root}"
  export HIP_PATH="${mi50_root}"
  local mi50_path="${mi50_root}/bin:${mi50_root}/lib/llvm/bin"
  local mi50_path_entry
  local -a mi50_path_entries=()
  if [[ -n "${PATH:-}" ]]; then
    IFS=: read -r -a mi50_path_entries <<< "${PATH}"
    for mi50_path_entry in "${mi50_path_entries[@]}"; do
      if [[ -n "${mi50_path_entry}" && ":${mi50_path}:" != *":${mi50_path_entry}:"* ]]; then
        mi50_path="${mi50_path}:${mi50_path_entry}"
      fi
    done
  fi
  export PATH="${mi50_path}"

  local mi50_library_path="${mi50_root}/lib"
  local mi50_extra_library_path
  for mi50_extra_library_path in \
    "${mi50_root}/lib/rocm_sysdeps/lib" \
    "${mi50_root}/lib/llvm/lib"; do
    if [[ -d "${mi50_extra_library_path}" ]]; then
      mi50_library_path="${mi50_library_path}:${mi50_extra_library_path}"
    fi
  done
  local mi50_ld_library_path="${mi50_library_path}"
  local mi50_ld_entry
  local -a mi50_ld_entries=()
  if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
    IFS=: read -r -a mi50_ld_entries <<< "${LD_LIBRARY_PATH}"
    for mi50_ld_entry in "${mi50_ld_entries[@]}"; do
      if [[ -n "${mi50_ld_entry}" && ":${mi50_ld_library_path}:" != *":${mi50_ld_entry}:"* ]]; then
        mi50_ld_library_path="${mi50_ld_library_path}:${mi50_ld_entry}"
      fi
    done
  fi
  export LD_LIBRARY_PATH="${mi50_ld_library_path}"
}
