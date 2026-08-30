# MI50 ROCm 10.x forward-port

This repository is the pre-hardware implementation scaffold for an unofficial,
Linux-first ROCm 10.x distribution targeting AMD Instinct MI50 (Vega20,
GCN5.1, `gfx906`). The objective is not merely to make an old ROCm release
run: it is to forward-port every ROCm 10 feature that can work correctly on
gfx906, optimize those paths for Vega20, and provide explicit fallbacks only
where the hardware or ISA makes a feature impossible.

The project deliberately does **not** replace the Linux kernel driver. Current
Linux `amdgpu`/KFD, LLVM, and Vega20 firmware already provide the low-level
pieces. The work here restores architecture-specific ROCm user space, ports
newer runtime and library behavior to gfx906, adds Vega20-specific tuning,
packages it coherently, and makes genuinely impossible paths fail safely.

## Current status

The repository can be used before MI50 hardware is available to:

- pin the ROCm/TheRock 10.0 source revisions;
- audit source trees for `gfx906`, deny-lists, BF16/FP8 and unsupported library paths;
- configure and build a gfx906-only Linux ROCm distribution;
- identify ROCm 10 features that are merely disabled by target policy and
  forward-port or optimize them where gfx906 semantics permit;
- validate package contents and device-code metadata;
- build PyTorch and llama.cpp against the custom stack;
- run host-only and CPU-side tests;
- report GPU-dependent tests as pending instead of guessing.

The machine-readable support contract is in [`support-matrix.json`](support-matrix.json).
Source provenance is in [`sources.lock.json`](sources.lock.json).
Its `optimization_profile` records the conservative Vega20 dispatch choices
(Tensile GEMM, legacy/assembly MIOpen, math SDPA and FP16-first inference).
The upstream-versus-forward-port delta and “latest compatible” acceptance
rules are in [`docs/COMPATIBILITY.md`](docs/COMPATIBILITY.md).
Experimental candidate builds and their current gfx906 evidence are tracked in
[`docs/EXPERIMENTAL_PORTS.md`](docs/EXPERIMENTAL_PORTS.md).
The ordered post-acquisition checks are in
[`docs/HARDWARE_VALIDATION.md`](docs/HARDWARE_VALIDATION.md).
Windows/WSL feasibility and its evidence boundary are in
[`docs/WINDOWS_WSL.md`](docs/WINDOWS_WSL.md).

## Target support contract

The supported inference path is:

- native `gfx906` LLVM/ROCr/HIP code generation;
- rocBLAS/Tensile with Vega20 kernels;
- compatible rocFFT, rocRAND, rocSPARSE, rocSOLVER, rocPRIM and rocThrust;
- MIOpen legacy/assembly gfx906 paths where the retained database is usable;
- PyTorch built locally with `PYTORCH_ROCM_ARCH=gfx906`;
- llama.cpp built with `GGML_HIP=ON` and `GPU_TARGETS=gfx906`;
- FP16/FP32 and validated quantized inference.

The support matrix distinguishes three different outcomes: (1) a feature that
can be enabled natively after a gfx906 port or optimization, (2) a feature that
needs a documented fallback, and (3) a feature that is fundamentally tied to
hardware absent from MI50. An upstream deny-list is therefore a porting input,
not automatically a permanent product decision. Do not use
`HSA_OVERRIDE_GFX_VERSION` to masquerade as a newer GPU.

The default Linux builder leaves the optional OpenCL/ocl-clr artifact disabled:
it is not needed by ROCr/HIP or the supported LLM paths and its host build pulls
in OpenGL development packages. Set `MI50_ENABLE_OPENCL=ON` in a builder that
provides those dependencies; this does not change the gfx906 policy or add a
GPU-runtime claim.

## Source baselines

The primary source baseline is the `therock-10.0` TheRock tag and matching
`release/therock-10.0` branches of the ROCm super-repositories. The lock file
contains the exact commits used by the build scripts. ROCm 10.1 changes may be
cherry-picked for either a reproducible gfx906 failure or a demonstrably
compatible feature/optimization, with provenance and regression coverage
recorded in the patch queue.

## Quick start (Linux)

The scripts are intentionally safe before hardware exists. They build and
inspect artifacts; actual device execution is reported as `pending-hardware`.

```bash
python3 scripts/audit_gfx906.py --root /path/to/source --json-out audit.json
python3 scripts/verify_source_lock.py --source-root /path/to/TheRock --strict
python3 scripts/verify_patch_lock.py --strict
python3 scripts/verify_configure_gfx906.py --build-root /path/to/build --strict
python3 scripts/validate_artifacts.py --artifact-root /path/to/artifact --strict
python3 scripts/mi50_features.py --json --check-environment
python3 -m unittest discover -s tests -v
# host-only HIP code-object check (returns 77 if ROCm is not installed yet)
bash scripts/hip_compile_smoke.sh
# native allocation/stream/event/kernel smoke (returns 77 before hardware)
bash scripts/run_hip_runtime_smoke.sh
# FP32/FP64 and dual-device peer-access smoke (returns 77 before hardware)
bash scripts/run_mi50_device_matrix_smoke.sh
# llama.cpp throughput/telemetry gate (returns 77 before hardware)
python3 scripts/mi50_llm_benchmark.py --model /path/to/model.gguf --dry-run
```

The reproducible Ubuntu builder installs `CppHeaderParser==2.7.4`,
`joblib==1.5.1`, `msgpack==1.1.1`, `zstandard==0.25.0`, `pytest==8.4.2`,
`pytest-subtests==0.15.0`, `tcl`, `texinfo`, the SQLite development headers required by
rocprofiler-sdk, `python3-magic` for SDK wheel generation, and the GL/X11/EGL development headers required by ROCclr's compile-time
interop declarations.  Runtime OpenGL discovery is disabled for the default
HIP-only artifact; set `ROCCLR_ENABLE_OPENGL=ON` to retain graphics interop.

The default `MI50_BUILD_PROFILE=full` keeps the complete enabled TheRock graph.
When storage or build time is constrained, set
`MI50_BUILD_PROFILE=inference` to disable debug/profiler/data-center/media/
emulation/storage groups while retaining the core runtime, communication,
math and ML libraries required for the supported inference path. This is a
smaller bring-up artifact, not a different architecture or a runtime support
claim.

To configure the provisional newer-library ports in an isolated build, set
`MI50_ENABLE_EXPERIMENTAL_NEW_ISA_PORTS=ON`. This enables the corresponding
hipBLASLt-provider, hipTensor and Composable Kernel feature gates in addition
to removing their `gfx906` target exclusions. It does not change the stable
artifact or promote a candidate to supported status; candidate device catalogs
must still be non-empty, target-pure, and pass MI50 hardware tests.

To configure a TheRock checkout already present on disk:

```bash
scripts/build_therock_gfx906.sh --source-root /path/to/TheRock
# after an interrupted incremental build (the existing audit is reused)
scripts/build_therock_gfx906.sh --source-root /path/to/TheRock --skip-audit
# constrained-storage inference bring-up (core + math/ML/communication only)
# If using a measured smaller filesystem, lower the preflight explicitly;
# otherwise the safe 32-GiB default still applies.
MI50_BUILD_PROFILE=inference MI50_MIN_FREE_GIB=8 \
  scripts/build_therock_gfx906.sh \
  --source-root /path/to/TheRock --skip-audit
```

For a fast runtime-only gate, point the standalone builder at the patched
ROCr project. It produces a shared `libhsa-runtime64.so` plus gfx906 trap,
blit and image objects before the rest of ROCm is built:

```bash
scripts/build_rocr_gfx906.sh \
  --source-root /path/to/rocm-systems/projects/rocr-runtime \
  --source-repo-root /path/to/rocm-systems \
  --device-lib-path /opt/rocm/llvm/lib/clang/18/amdgcn/bitcode
```

To fetch the pinned repositories automatically, run the same script with
`--fetch-root /path/to/sources`. A full build requires a Linux host with CMake,
Ninja, a C++ toolchain, Python, Git, and substantial disk/RAM. The build entry
point checks for 32 GiB of free space before configuring; override that
preflight only for a measured incremental build with `MI50_MIN_FREE_GIB`.
When TheRock does not emit release archives, the builder merges the active
split slices and writes a deterministic
`rocm-10.0.0+mi50.5-mi50-gfx906-linux.tar.gz` alongside `dist_info.json`,
build provenance and strict validation reports.

Build the downstream inference projects after `ROCM_PATH` points at the custom
installation:

```bash
ROCM_PATH=/opt/rocm-mi50 scripts/build/pytorch/build_pytorch_gfx906.sh /path/to/pytorch
ROCM_PATH=/opt/rocm-mi50 scripts/build/llama.cpp/build_llama_gfx906.sh /path/to/llama.cpp
```

PyTorch 2.13's ROCm build still requires the hipBLASLt host headers and link
interface even when its gfx906 device path is disabled. Build that small,
device-free shim from the pinned ROCm libraries checkout, then pass it to the
PyTorch wrapper:

```bash
ROCM_PATH=/opt/rocm-mi50 \
  scripts/build/pytorch/build_hipblaslt_host.sh \
  /path/to/TheRock/rocm-libraries/projects/hipblaslt \
  --build-dir /tmp/hipblaslt-host-build \
  --install-dir /tmp/hipblaslt-host
ROCM_PATH=/opt/rocm-mi50 \
  scripts/build/pytorch/build_pytorch_gfx906.sh /path/to/pytorch \
  --hipblaslt-host /tmp/hipblaslt-host
```

The shim has no `.co`, `.hsaco` or architecture-specific object files. It is
only a compile-time interface; production MI50 GEMM remains hipBLAS/rocBLAS
and the PyTorch wheel records this policy in `mi50-build-metadata.json`.

The PyTorch wrapper invokes the source checkout's `setup.py bdist_wheel` with
`PYTORCH_ROCM_ARCH=gfx906`, `USE_ROCM=1`, and AOTriton/flash/memory-efficient
attention/Triton disabled. The llama.cpp wrapper configures CMake with
`GGML_HIP=ON`, `GGML_NATIVE=OFF`, `AMDGPU_TARGETS=gfx906` and
`CMAKE_HIP_ARCHITECTURES=gfx906`. Both write a small `mi50-build-metadata.json`
next to their output and default `ROCBLAS_USE_HIPBLASLT=0`, keeping production
inference on the mature gfx906 rocBLAS/Tensile path. Set that variable to `1`
only for a separately measured hardware experiment; the wrappers reject
`HSA_OVERRIDE_GFX_VERSION` and never turn it on themselves.

The main rocBLAS smoke covers native FP16/FP32/FP64 GEMM. INT8 is intentionally
separate because MI50 support is a per-kernel question: compile and run
`scripts/run_mi50_int8_smoke.sh` to exercise `rocblas_gemm_ex` with INT8 inputs
and INT32 accumulation. It reports `GPU-test-pending` without `/dev/kfd` and
never treats a missing INT8 kernel as a silent software fallback.

The PyTorch wrapper also applies the reviewed downstream precision-policy patches
from `patches/downstream/pytorch/`. It makes the native capability contract
explicit: gfx906 BF16 and FP8 are unsupported, while callers that deliberately
choose software conversion remain responsible for their own emulation policy.
Patch digests are recorded in `mi50-build-metadata.json` and
[`downstream.lock.json`](downstream.lock.json).

Pinned downstream revisions and the host-shim policy are recorded in
[`downstream.lock.json`](downstream.lock.json). The current pre-hardware
artifacts are intentionally kept outside this source repository; reproduce
them with the scripts above and retain their SHA-256 values in build records.

Install a generated Linux archive into an isolated prefix (the installer does
not replace the kernel's `amdgpu`/KFD driver or an existing ROCm tree):

```bash
python3 scripts/install_rocm_mi50.py \
  --archive /path/to/rocm-10.0.0+mi50.5-mi50-gfx906-linux.tar.gz \
  --prefix /opt/rocm-mi50-10.0.0-mi50.5
source /opt/rocm-mi50-10.0.0-mi50.5/mi50-env.sh
```

The installer validates non-empty `hipcc`, LLVM device bitcode, ROCr,
rocBLAS/Tensile gfx906 data and MIOpen gfx906 data before publishing the
prefix. `--dry-run` performs only those checks; `--force` moves an existing
prefix to a timestamped `.previous-*` backup. The generated environment file
rejects `HSA_OVERRIDE_GFX_VERSION` and `ROCR_OVERRIDE_GFX_VERSION` so a native
MI50 run cannot be confused with ISA masquerading.

Run the diagnostic before a hardware test:

```bash
python3 scripts/mi50_doctor.py --artifact-root /opt/rocm-mi50
python3 scripts/mi50_kernel_readiness.py --output kernel-readiness.json
python3 scripts/mi50_hardware_gate.py --output hardware-gate.json
```

`mi50_kernel_readiness.py` is read-only: it reports whether Linux `amdgpu`,
KFD, `/dev/kfd` and the expected Vega20 firmware are present. It remains
`GPU-test-pending` when no card is bound and never installs or replaces the
kernel driver.

After a card is installed, run the benchmark without `--dry-run`; it requires
native `gfx906`/wave64 discovery, records `rocminfo` and `amd-smi` output, and
can compare throughput against a prior JSON report with `--baseline`.

## Hardware gate

No GPU runtime claim is made until two stock-VBIOS MI50 16GB cards are tested
on Ubuntu 24.04.4/kernel 6.8 with adequate passive-card airflow and power.
The hardware test backlog covers KFD enumeration, HIP, memory, rocBLAS/MIOpen,
RCCL, PyTorch, llama.cpp, ECC/RAS and 24-hour inference soak tests.

## Evidence

- [AMD MI50 specifications](https://www.amd.com/en/support/downloads/drivers.html/accelerators/instinct/instinct-mi-series/instinct-mi50.html)
- [ROCm 10.0 release notes and hardware matrix](https://rocm.docs.amd.com/en/develop/about/release-notes.html)
- [TheRock gfx906 target definition](https://raw.githubusercontent.com/ROCm/TheRock/therock-10.0/cmake/therock_amdgpu_targets.cmake)
- [TheRock GPU readiness matrix](https://github.com/ROCm/TheRock/blob/main/SUPPORTED_GPUS.md)
- [LLVM AMDGPU gfx906 documentation](https://rocm.docs.amd.com/projects/llvm-project/en/latest/LLVM/llvm/html/AMDGPUUsage.html)
- [Linux KFD Vega20 support](https://github.com/torvalds/linux/blob/master/drivers/gpu/drm/amd/amdkfd/kfd_device.c)

This is community-maintained software. It must not be represented as an AMD
officially supported ROCm release.
