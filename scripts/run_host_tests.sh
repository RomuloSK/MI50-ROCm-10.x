#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python3 -m unittest discover -s "${ROOT_DIR}/tests" -v
python3 "${ROOT_DIR}/scripts/mi50_features.py" --check-environment
python3 "${ROOT_DIR}/scripts/mi50_doctor.py"
python3 "${ROOT_DIR}/scripts/mi50_hardware_gate.py"
python3 "${ROOT_DIR}/scripts/mi50_runtime_validation.py"

# If a custom ROCm install is already present, compile a native gfx906 HIP
# object.  Exit 77 means the compiler is not installed yet and is an expected
# pre-hardware state; any real compiler failure remains fatal.
if [[ -f "${ROOT_DIR}/scripts/hip_compile_smoke.sh" ]]; then
  set +e
  bash "${ROOT_DIR}/scripts/hip_compile_smoke.sh"
  hip_smoke_status=$?
  set -e
  if [[ "${hip_smoke_status}" -ne 0 && "${hip_smoke_status}" -ne 77 ]]; then
    exit "${hip_smoke_status}"
  fi
fi

# The runtime tier is GPU-dependent. Exit 77 is an explicit pending state;
# compiler/runtime errors remain fatal once a card and ROCm install exist.
if [[ -f "${ROOT_DIR}/scripts/run_hip_runtime_smoke.sh" ]]; then
  set +e
  bash "${ROOT_DIR}/scripts/run_hip_runtime_smoke.sh"
  hip_runtime_status=$?
  set -e
  if [[ "${hip_runtime_status}" -ne 0 && "${hip_runtime_status}" -ne 77 ]]; then
    exit "${hip_runtime_status}"
  fi
fi

if [[ -f "${ROOT_DIR}/scripts/run_mi50_device_matrix_smoke.sh" ]]; then
  set +e
  bash "${ROOT_DIR}/scripts/run_mi50_device_matrix_smoke.sh"
  device_matrix_status=$?
  set -e
  if [[ "${device_matrix_status}" -ne 0 && "${device_matrix_status}" -ne 77 ]]; then
    exit "${device_matrix_status}"
  fi
fi

if [[ -f "${ROOT_DIR}/scripts/run_mi50_rccl_smoke.sh" ]]; then
  set +e
  bash "${ROOT_DIR}/scripts/run_mi50_rccl_smoke.sh"
  rccl_status=$?
  set -e
  if [[ "${rccl_status}" -ne 0 && "${rccl_status}" -ne 77 ]]; then
    exit "${rccl_status}"
  fi
fi

if [[ -f "${ROOT_DIR}/scripts/run_mi50_rocblas_smoke.sh" ]]; then
  set +e
  bash "${ROOT_DIR}/scripts/run_mi50_rocblas_smoke.sh"
  rocblas_status=$?
  set -e
  if [[ "${rocblas_status}" -ne 0 && "${rocblas_status}" -ne 77 ]]; then
    exit "${rocblas_status}"
  fi
fi

if [[ -f "${ROOT_DIR}/scripts/run_mi50_library_abi_smoke.sh" ]]; then
  set +e
  bash "${ROOT_DIR}/scripts/run_mi50_library_abi_smoke.sh"
  library_abi_status=$?
  set -e
  if [[ "${library_abi_status}" -ne 0 && "${library_abi_status}" -ne 77 ]]; then
    exit "${library_abi_status}"
  fi
fi

if [[ -f "${ROOT_DIR}/scripts/run_mi50_miopen_smoke.sh" ]]; then
  set +e
  bash "${ROOT_DIR}/scripts/run_mi50_miopen_smoke.sh"
  miopen_status=$?
  set -e
  if [[ "${miopen_status}" -ne 0 && "${miopen_status}" -ne 77 ]]; then
    exit "${miopen_status}"
  fi
fi

if [[ -f "${ROOT_DIR}/scripts/run_mi50_fft_rand_smoke.sh" ]]; then
  set +e
  bash "${ROOT_DIR}/scripts/run_mi50_fft_rand_smoke.sh"
  fft_rand_status=$?
  set -e
  if [[ "${fft_rand_status}" -ne 0 && "${fft_rand_status}" -ne 77 ]]; then
    exit "${fft_rand_status}"
  fi
fi

if [[ -f "${ROOT_DIR}/scripts/run_mi50_sparse_solver_smoke.sh" ]]; then
  set +e
  bash "${ROOT_DIR}/scripts/run_mi50_sparse_solver_smoke.sh"
  sparse_solver_status=$?
  set -e
  if [[ "${sparse_solver_status}" -ne 0 && "${sparse_solver_status}" -ne 77 ]]; then
    exit "${sparse_solver_status}"
  fi
fi

if [[ -f "${ROOT_DIR}/scripts/run_mi50_prim_thrust_smoke.sh" ]]; then
  set +e
  bash "${ROOT_DIR}/scripts/run_mi50_prim_thrust_smoke.sh"
  prim_thrust_status=$?
  set -e
  if [[ "${prim_thrust_status}" -ne 0 && "${prim_thrust_status}" -ne 77 ]]; then
    exit "${prim_thrust_status}"
  fi
fi

if [[ -f "${ROOT_DIR}/scripts/run_mi50_memory_smoke.sh" ]]; then
  set +e
  bash "${ROOT_DIR}/scripts/run_mi50_memory_smoke.sh"
  memory_status=$?
  set -e
  if [[ "${memory_status}" -ne 0 && "${memory_status}" -ne 77 ]]; then
    exit "${memory_status}"
  fi
fi

if [[ -d "${ROOT_DIR}/out/artifacts/gfx906" ]]; then
  python3 "${ROOT_DIR}/scripts/validate_artifacts.py" \
    --artifact-root "${ROOT_DIR}/out/artifacts/gfx906" \
    --strict
  if [[ -d "${ROOT_DIR}/out/artifacts/gfx906/python/dist" ]]; then
    python3 "${ROOT_DIR}/scripts/validate_python_packages.py" \
      --package-dir "${ROOT_DIR}/out/artifacts/gfx906/python/dist" \
      --target gfx906 --version "10.0.0+mi50.5" --strict
  fi
else
  echo "artifact validation: pending-build"
fi

echo "host-only tests passed; GPU execution remains pending-hardware"
