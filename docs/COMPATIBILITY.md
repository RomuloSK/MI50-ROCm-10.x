# MI50 compatibility delta

This project is an unofficial ROCm 10.x user-space forward-port for `gfx906`.
It is not an AMD-supported ROCm release and it does not replace Linux
`amdgpu`/KFD or the required Vega20 firmware.

## Baselines

TheRock's current publication layout identifies ROCm 10.0 as the stable
release stream and ROCm 10.1 as a nightly stream. This project therefore keeps
the reproducible 10.0 lock as its release base; a 10.1 change is admitted only
as a commit-pinned cherry-pick after gfx906 code-generation, dispatch and
hardware evidence exists. A mutable nightly package or branch is never used as
an implicit dependency. See TheRock's [release-stream layout](https://github.com/ROCm/TheRock/blob/main/docs/development/s3_buckets.md).

| Area | Last upstream MI50-oriented baseline | This project |
| --- | --- | --- |
| ROCm user space | ROCm 5.7 was the last broadly supported `gfx906` release; later releases kept varying legacy pieces but removed MI50 from the official matrix | ROCm/TheRock 10.0.0 sources, pinned in `sources.lock.json`, with a gfx906-only build and a small patch queue |
| Kernel/runtime | Linux `amdgpu`/KFD and LLVM still understand Vega20/gfx906 | Reuses the host driver and builds target-scoped ROCr, COMGR, HIP and device libraries; no `HSA_OVERRIDE_GFX_VERSION` |
| GEMM | Mature gfx906 rocBLAS/Tensile data in older releases | Rebuilds/validates Vega20 assets, returns `NOT_IMPLEMENTED` instead of aborting if data is missing, and keeps rocBLAS as the correctness fallback while newer BLAS front ends are audited |
| Deep learning | MIOpen legacy/assembly paths were the practical MI50 route | Retains those paths where the ROCm 10 source data exists; newer kernels must pass gfx906 code-object and hardware tests |
| PyTorch | Older ROCm/PyTorch combinations are the known compatibility route | PyTorch is built against this stack with `PYTORCH_ROCM_ARCH=gfx906`, eager/math SDPA defaults, and flash/AOTriton disabled until ported |
| LLM inference | No official ROCm 10 MI50 package | llama.cpp HIP with native gfx906 is the primary pre-hardware packaging target; FP16/FP32 and tested quantized GGUF are supported paths |
| OpenCL | Optional ROCm runtime component; not required for HIP/LLM inference | Disabled by default in the minimal Linux builder because ocl-clr requires host OpenGL development packages; opt in with `MI50_ENABLE_OPENCL=ON` when those dependencies are provisioned |
| Windows | No qualified native Windows MI50 ROCm product path | Deferred. Linux is v1; WSL is useful for builds but is not a GPU-support claim |

## Forward-port policy

ROCm 10 target exclusions are treated as hypotheses, not automatically as
hardware limitations. The stable build keeps hipBLASLt, Composable Kernel and
hipTensor excluded because the current generated kernels either require newer
ISA features or have no gfx906 data. An explicit
`MI50_ENABLE_EXPERIMENTAL_NEW_ISA_PORTS=ON` enables deliberate attempts for
porting work and turns on the corresponding TheRock feature gates for the
hipBLASLt-provider, hipTensor and Composable Kernel; such builds are not the
supported MI50 path. hipSPARSELt is
deliberately retained on the fallback path because its current
structured-sparse implementation has only newer MFMA logic and no gfx906 data.
rocWMMA, current tensor-core-only paths, upstream vLLM and native Windows ROCm
remain excluded because they depend on ISA or platform features MI50 does not
provide.

Even when hipBLASLt is built, rocBLAS dispatches gfx906 to the mature Tensile
backend by default. `ROCBLAS_USE_HIPBLASLT=1` is an explicit opt-in for a later
MI50 experiment, not a production dependency.

The important non-negotiable difference is feature honesty: MI50 has wave64
GCN5.1 execution but no matrix cores and no native BF16/FP8 instructions. The
stack therefore reports BF16/FP8/WMMA as unsupported rather than silently
emulating or pretending that a newer ISA is present.

## What “latest compatible” means here

The target is not to copy every ROCm 10 label onto MI50. A feature is accepted
only when all of the following are true:

1. Its compiler/device library code can emit a real `gfx906` code object.
2. Runtime dispatch does not select a newer-ISA object or assume matrix cores.
3. Package validation finds the required target data and no forbidden override.
4. A real MI50 correctness test passes when hardware is available.
5. Performance is compared with the closest working MI50 baseline; a regression
   greater than 5% needs a documented correctness/stability justification.

Until item 4 is complete, reports and artifacts carry the explicit
`GPU-test-pending`/artifact-only status.

References: [ROCm release notes](https://rocm.docs.amd.com/en/develop/about/release-notes.html),
[TheRock GPU readiness](https://github.com/ROCm/TheRock/blob/main/SUPPORTED_GPUS.md),
[LLVM AMDGPU usage](https://rocm.docs.amd.com/projects/llvm-project/en/latest/LLVM/llvm/html/AMDGPUUsage.html).
