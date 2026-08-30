# MI50 hardware validation gate

Run these checks only on a Linux host with the stock-VBIOS card installed.
The repository's `mi50_hardware_gate.py` intentionally remains
`GPU-test-pending` when `/dev/kfd` is absent.

## Bring-up order

1. Confirm passive-card airflow, four 8-pin power connectors, two usable PCIe
   x16 slots, current `linux-firmware` and the inbox `amdgpu`/KFD driver. Run
   `python3 scripts/mi50_kernel_readiness.py --output kernel-readiness.json`
   and resolve any explicit firmware/module failures before starting ROCr.
2. Run `python3 scripts/mi50_hardware_gate.py --require-gpu --output
   hardware-gate.json`, then `python3 scripts/mi50_runtime_validation.py
   --require-gpu --output runtime-validation.json`; require native `gfx906` in
   `rocminfo` before proceeding.
3. Run `rocminfo`, `hipconfig --full`, then
   `bash scripts/run_hip_runtime_smoke.sh` for allocation, memcpy,
   stream/event, kernel-launch and result checks. Run
   `bash scripts/run_mi50_device_matrix_smoke.sh` for FP32/FP64 and peer
   access; follow with graph, RTC and multi-process tests.
4. Check ECC/RAS, clocks, power, temperature and reset behavior with `amd-smi`.
5. Run `bash scripts/run_mi50_rocblas_smoke.sh`, then cover FP16/FP32/FP64
   and representative LLM GEMM shapes through rocBLAS. Run
   `bash scripts/run_mi50_library_abi_smoke.sh` before the individual
   rocFFT/rocRAND/rocSPARSE/rocSOLVER/MIOpen correctness suites. The native
   `bash scripts/run_mi50_fft_rand_smoke.sh` tier covers rocFFT and rocRAND
   output correctness first.
   then MIOpen, rocFFT, rocRAND, rocSPARSE and rocSOLVER tests.
6. Run `bash scripts/run_mi50_rccl_smoke.sh` to validate RCCL all-reduce,
   then cover all-gather and peer access over PCIe. Test an Infinity Fabric
   bridge separately if one is installed.
7. Run PyTorch GEMM/convolution/math-SDPA/Transformers and llama.cpp single-
card inference, then dual-card tensor splitting.
8. Soak repeated model load/unload and inference for 24 hours while recording
   throughput, memory, power, temperature and error/reset counts.

The first two gates also parse `rocminfo` and record the native agent contract:
at least one `gfx906` agent and a reported wavefront size of 64. A contradictory
wavefront report fails the gate; if a vendor tool omits the field, the report is
kept for manual review rather than inventing a result.

No test may set `HSA_OVERRIDE_GFX_VERSION`. A missing or mismatched device
object is a failure, not a reason to masquerade as a newer GPU.

## Release gates

Compare with ROCm 6.3.3 and the closest working ROCm 10.x gfx906 build. Record
prompt throughput, decode tokens/s, time-to-first-token, memory use, power and
error rate. Reject regressions over 5% unless a correctness or stability fix
justifies them. Do not claim 32GB support until a real 32GB MI50 is tested.
