# Experimental gfx906 forward ports

The stable distribution is deliberately conservative: it routes MI50 GEMM
through rocBLAS/Tensile and does not ship a newer-ISA library merely because a
host compiler can build it. Candidate components are built in an isolated
prefix, inspected for native `gfx906` code objects, and admitted to the stable
package only after runtime correctness and performance tests on a real card.

## hipBLASLt device path

The pinned ROCm 10.0 `hipblaslt` source accepts `GPU_TARGETS=gfx906` and the
host/device project configures successfully with the following policy:

```text
HIPBLASLT_ENABLE_DEVICE=ON
HIPBLASLT_ENABLE_EXTOPS=OFF
HIPBLASLT_ENABLE_MATRIX_TRANSFORM=OFF
HIPBLASLT_ENABLE_CLIENT=OFF
HIPBLASLT_ENABLE_HOST=ON
GPU_TARGETS=gfx906
```

The isolated build reached Tensile validation and checked all 574,620 logic
solutions. It emitted an ELF container marked `gfx906` at
`lib/hipblaslt/library/gfx906/Kernels.so-000-gfx906.hsaco`, but the source
checkout's `library/` directory contains no gfx906 solution YAML files. Tensile
therefore parsed zero solutions and the emitted object has a zero-byte
`.text` section. It is not a usable GEMM implementation and is rejected by
`build_hipblaslt_gfx906_experimental.sh` with exit status 78.

Reproduce the experiment with:

```bash
ROCM_PATH=/opt/rocm-mi50 \
  scripts/build/pytorch/build_hipblaslt_gfx906_experimental.sh \
  /path/to/TheRock/rocm-libraries/projects/hipblaslt \
  --build-dir /tmp/hipblaslt-gfx906-experimental-build \
  --install-dir /tmp/hipblaslt-gfx906-experimental-install
```

The script fixes the toolchain `PATH` at configure time, prevents
`HSA_OVERRIDE_GFX_VERSION`, keeps the build isolated from the release prefix,
and rejects code objects without a non-empty executable section. A future port
must add or generate a reviewed gfx906 solution catalog, then pass native
MI50 correctness, dispatch, and performance gates before the candidate can
replace rocBLAS/Tensile.

The top-level TheRock builder has the same opt-in boundary. Setting
`MI50_ENABLE_EXPERIMENTAL_NEW_ISA_PORTS=ON` enables the hipBLASLt-provider,
hipTensor and Composable Kernel feature flags in the isolated graph; the
stable profile leaves them disabled. A configure-only graph was verified to
include all three projects with `gfx906` while continuing to exclude
hipSPARSELt, rocWMMA and rocprofiler-compute.

## Other candidates

Composable Kernel, hipTensor, AOTriton/flash attention and hipSPARSELt remain
separate investigations. Their current ROCm 10 source paths either require
matrix-core instructions absent on Vega20 or have no gfx906 kernel catalogs.
The supported PyTorch build consequently uses eager/math SDPA, while llama.cpp
uses HIP kernels compiled directly for `gfx906`.

This document records source/build evidence only. It does not establish MI50
runtime support until the hardware validation suite passes.
