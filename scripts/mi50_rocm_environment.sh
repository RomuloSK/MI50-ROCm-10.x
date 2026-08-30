#!/usr/bin/env bash

# Export a complete user-space ROCm environment for standalone MI50 smokes.
# The function intentionally does not change shell options; callers may source
# this file from a shell that owns its own error-handling policy.
mi50_export_rocm_environment() {
  local mi50_root="${1:?ROCm root is required}"
  export ROCM_PATH="${mi50_root}"
  export ROCM_HOME="${mi50_root}"
  export HIP_PATH="${mi50_root}"
  export PATH="${mi50_root}/bin:${mi50_root}/lib/llvm/bin:${PATH}"

  local mi50_library_path="${mi50_root}/lib"
  local mi50_extra_library_path
  for mi50_extra_library_path in \
    "${mi50_root}/lib/rocm_sysdeps/lib" \
    "${mi50_root}/lib/llvm/lib"; do
    if [[ -d "${mi50_extra_library_path}" ]]; then
      mi50_library_path="${mi50_library_path}:${mi50_extra_library_path}"
    fi
  done
  export LD_LIBRARY_PATH="${mi50_library_path}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
}
