# Windows and WSL status

The v1 deliverable is Linux user space. MI50 support requires the Linux
`amdgpu`/KFD device path and Vega20 firmware; a native Windows ROCm runtime is
therefore not a release target in this repository.

WSL is useful for building and running host-only checks, but WSL without GPU
passthrough must remain `GPU-test-pending`. Do not use
`HSA_OVERRIDE_GFX_VERSION` or `ROCR_OVERRIDE_GFX_VERSION` to turn a WSL CPU
environment into a false MI50 result.

## Build in WSL

Use Ubuntu 24.04 with a writable Linux filesystem and at least 32 GiB free for
the full TheRock graph. A smaller pre-hardware bring-up can use the inference
profile:

```bash
MI50_BUILD_PROFILE=inference MI50_BUILD_TESTING=OFF MI50_MIN_FREE_GIB=8 \
  scripts/build_therock_gfx906.sh \
  --source-root /sources/TheRock \
  --build-dir /workspace/out/build-inference \
  --artifact-dir /workspace/out/artifacts-inference \
  --skip-audit
```

If Ninja reports `Read-only file system`, stop the build, run `wsl --shutdown`
from PowerShell, free space on the Windows volume containing the distribution
VHDX, and verify `df -h /` plus a writable temporary file before resuming.
The build tree is reusable; pass `--skip-audit` after the audit already exists.

## What WSL can prove

- source lock and patch application;
- LLVM/Clang HIP compilation for real `gfx906` code objects;
- package, ELF, COMGR and host-runtime checks;
- CPU-side PyTorch/llama.cpp builds and model-format checks.

## What WSL cannot prove without GPU passthrough

- `/dev/kfd` enumeration, firmware loading or native `rocminfo` agents;
- HIP allocation/kernel correctness, rocBLAS/MIOpen/RCCL execution;
- MI50 clocks, ECC/RAS, reset behavior, thermals or performance;
- single- or dual-card LLM inference.

Those checks stay explicitly pending until stock-VBIOS MI50 hardware is tested
on Linux. Native Windows/WSL production support can be reconsidered only after
an independently validated KFD/driver path exists.
