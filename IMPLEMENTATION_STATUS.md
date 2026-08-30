# Implementation status

## Product objective

This is a ROCm 10.x forward-port for MI50/gfx906, not a frozen legacy-runtime
compatibility package. The implementation should recover the newest ROCm
features whose semantics can be made correct on Vega20, then tune them for
wave64, HBM2 and MI50's available execution units. A fallback is considered
the supported path only until a forward-port is validated, or when the feature
depends on hardware that MI50 fundamentally does not have.

## Completed before MI50 hardware is available

- Pinned ROCm 10.0/TheRock source metadata and project policy.
- Machine-readable gfx906 component support and fallback matrix.
- Deterministic source audit for architecture gates and forbidden ISA overrides.
- Artifact validation for gfx906 markers, rocBLAS/Tensile data and MIOpen data.
- Host ELF dependency gate for the flattened distribution, with an installer-side
  recheck before an archive is published into a prefix. The RDC test binary's
  co-located GTest libraries are restored through a relative `$ORIGIN` RUNPATH.
- The ordered validation suite now scopes `PATH` and `LD_LIBRARY_PATH` to the
  requested ROCm prefix, ensuring its diagnostics cannot silently probe a
  different system installation.
- rocBLAS missing-Tensile handling patch: an absent or unloadable gfx906
  library now returns `rocblas_status_not_implemented` to its caller rather
  than aborting the process, while valid target data follows the normal path.
- Native rocBLAS precision smokes now cover FP16, FP32 and FP64 in the main
  release gate, with an isolated INT8-to-INT32 `rocblas_gemm_ex` gate. INT8
  remains explicitly `validate-per-kernel` because the generic ROCm 10 API can
  be present without a tuned Vega20 code path; no fallback silently converts
  or advertises it as native until MI50 hardware evidence exists.
- The rocSPARSE smoke now exercises ROCm 10's generic two-stage SpMV-v2 API
  first (analysis plus compute) and falls back to the deprecated v1 API only
  on an explicit `rocsparse_status_not_implemented` result. This exposes the
  newest compatible dispatch path without hiding a real gfx906 runtime error.
- HIP runtime coverage now includes a separate graph-capture/instantiate/
  launch smoke. The baseline stream/event test remains independent, while
  `run_mi50_graph_smoke.sh` compiles the same native gfx906 kernel and marks
  graph execution pending until a real device is available.
- A HIPRTC smoke now compiles a kernel at runtime with
  `--gpu-architecture=gfx906`, verifies a non-empty code object, and (when
  `/dev/kfd` exists) loads and launches it through the HIP module API. This
  closes the runtime-compilation gap without using an ISA override.
- A dedicated hipBLAS SGEMM smoke now validates the public hipBLAS dispatch
  layer against the stable rocBLAS route with `ROCBLAS_USE_HIPBLASLT=0`;
  compile evidence is available before hardware and execution remains gated.
- `mi50_validation_suite.py` now runs the ordered bring-up smokes as one
  machine-readable report, preserving per-step `pass`, `GPU-test-pending`,
  `unsupported-on-gfx906`, and `fail` states. `--require-gpu` turns pending
  evidence into a release-gate failure instead of weakening the report.
- A low-level packed INT8 dot4 smoke validates the GCN5.1 `amd_mixed_dot`
  primitive as a possible quantized-inference building block. It is kept
  separate from rocBLAS INT8 because a primitive passing does not prove a
  complete GEMM catalog or end-to-end LLM kernel.
- An end-to-end native INT8→INT32 GEMM fallback smoke now uses that dot4
  primitive, validates non-multiple-of-four K tails, and compares the full
  matrix against a host reference. It remains an experimental fallback until
  it is benchmarked against rocBLAS on an actual MI50.
- Strict ROCr target-scoped code-object validator (trap handler, blit shaders,
  OpenCL image metadata, newer-ISA exclusion and shared-library symbol audit).
- ROCr loader fallback for WSL hosts that expose `/dev/dxg` without the optional
  `librocdxg.so`: failed shared-thunk loads now clear the DXG mode and use the
  statically linked Linux/KFD thunk, avoiding null-handle symbol resolution.
- ROCr also recovers when `librocdxg.so` exists but lacks a required export:
  the incomplete thunk is closed and the native Linux KFD thunk is retried,
  preventing a half-initialized WSL runtime from blocking normal Linux probing.
- Host diagnostic that reports `/dev/kfd`, `/dev/dri`, ROCm tools and pending status.
- Linux TheRock build/fetch entry point with gfx906-only CMake configuration.
- Standalone ROCr gfx906 build/install entry point for a fast Linux runtime
  compile/link gate.
- Correct submodule-path materialization for the pinned `rocm-systems`,
  `rocm-libraries` and LLVM checkouts.
- Opt-in TheRock target policy that attempts provisional gfx906 forward-port
  candidates (hipBLASLt, Composable Kernel and hipTensor), while preserving
  rocSPARSE as the fallback for hipSPARSELt's matrix-only implementation.
- Ubuntu 24.04 build-container definition.
- PyTorch and llama.cpp native gfx906 build entry points.
- Host-only unit tests and Python syntax checks.
- Canonical MI50 feature-policy module/CLI that rejects ISA masquerading,
  mixed-target builds and silent BF16/FP8/matrix-core claims.
- Per-build provenance JSON with locked commits, observed patch digests,
  toolchain versions and host/container context.
- Reusable hardware gate that transitions from `GPU-test-pending` to native
  gfx906 discovery only when `/dev/kfd` and rocminfo prove the card is present.
- Native HIP compile smoke and first-tier runtime validation commands that
  remain host-only/pending until a real `/dev/kfd` device is available.
- A native HIP runtime smoke binary covering discovery, allocation, async
  copies, streams, events, kernel launch and result verification; it is wired
  into the host test runner as exit-77 `GPU-test-pending` until hardware exists.
- The default Linux superbuild disables the optional OpenCL/ocl-clr artifact so
  a minimal builder does not fail on unrelated host OpenGL dependencies;
  `MI50_ENABLE_OPENCL=ON` remains available for a fully provisioned builder.
- ROCclr's Linux OpenGL discovery is now an explicit opt-in for the HIP-only
  build (`ROCCLR_ENABLE_OPENGL=OFF`); the reproducible image still installs
  GL/X11/EGL headers so graphics interop can be rebuilt when requested.
- The builder now pins `CppHeaderParser==2.7.4` and `texinfo`, which are needed
  by HIP profiling-header generation and ROCgdb documentation respectively.
- The builder now installs `libsqlite3-dev`, required for rocprofiler-sdk's
  SQLite-backed output support in the full ROCm 10.x profile.
- The builder now installs `gfortran`, required by hipBLASLt's configure-time
  Fortran toolchain probe even when rocBLAS remains the MI50 production path.
- The builder now pins `msgpack==1.1.1`, required by TheRock's artifact
  splitter during multi-architecture package generation.
- The builder now pins `joblib==1.5.1`, required by hipBLASLt/TensileLite's
  gfx906 device-library generation scripts.
- The builder now pins `zstandard==0.25.0`, required to unpack compressed
  code-object bundles during artifact splitting.
- The builder now installs `tcl` so rocprofiler-systems can regenerate its
  embedded SQLite amalgamation deterministically.
- The builder now pins `pytest==8.4.2` and `pytest-subtests==0.15.0` for the
  rocprofiler-systems host test/CTest generation path.
- The gfx906 profile now explicitly disables hipTensor, Composable Kernel,
  rocWMMA, hipSPARSELt, and rocprofiler-compute; these components have no
  Vega20 target in ROCm 10.x and otherwise attempt to configure newer GPUs.
- The stable profile now keeps hipBLASLt, Composable Kernel and hipTensor
  excluded even though the general forward-port switch remains enabled; an
  explicit `MI50_ENABLE_EXPERIMENTAL_NEW_ISA_PORTS=ON` is required to attempt
  those paths after a deliberate newer-ISA audit.
- The stable graph now skips hipBLASLt activation entirely, avoiding TheRock's
  no-target fallback to gfx1100; hipBLAS/rocBLAS therefore cannot publish a
  mixed-target artifact while the experimental path remains opt-in.
- The stable builder now exposes `MI50_ENABLE_EXPERIMENTAL_NEW_ISA_PORTS=ON`
  as an explicit, provenance-recorded escape hatch for isolated newer-ISA
  forward-port experiments; the release default remains `OFF`.
- rocRoller remains built for rocBLAS support, but its optional CMake/Pytest
  test suite is disabled because it expects a Pytest CMake package rather than
  the Python test executable used by the host validation image.
- The reproducible builder installs the pinned Rust 1.98.0 toolchain through
  `rustup`, which is required by the full-profile mirage emulation subproject.
- TheRock artifact flattening path is patched to publish files atomically;
  parallel component flatteners can no longer leave truncated or zero-byte
  ELF files in the shared `dist/rocm` tree.
- rocprofiler-sdk's TheRock integration is patched to use only the configured
  gfx906 distribution targets and to omit newer-GPU test bundles; profiler
  libraries remain available while incompatible test fixtures stay excluded.
- The Ubuntu builder now pins its 24.04 base image by OCI digest and records
  the daemon-resolved image ID/repository digest in
  `out/container-image-provenance.json`.
- The builder now creates a deterministic installable Linux tarball when a
  TheRock revision exposes only a flattened `dist/rocm` tree. It merges the
  active generic and gfx906 `run/lib/dev` slices first, expunges stale
  distributions between incremental runs, records archive SHA-256 metadata,
  and rejects packages missing HIP headers, ROCr, rocBLAS gfx906 data or the
  retained MIOpen gfx906 database.
- The builder can now produce and validate TheRock's split Python SDK wheels
  (`rocm-sdk-core`, `rocm-sdk-libraries`, `rocm-sdk-device-gfx906` and
  `rocm-sdk-devel`) plus the ROCm source distribution. Unsupported profiler
  compute/hipSPARSELt/rocWMMA/hipTensor/Composable Kernel payloads are removed
  from wheel archives with RECORD hashes recomputed before strict validation.
- The downstream PyTorch and llama.cpp wrappers now emit reproducible
  gfx906-only build commands and metadata, disable unqualified attention and
  Triton paths, reject ISA overrides, and record `GPU-test-pending` until a
  real `/dev/kfd` device is exercised.
- A PyTorch 2.13.0 wheel has now been built from pinned commit
  `cf30153c4c131c8164ee7798e5022d810682e2cb` with 3,281 native/C++/HIP
  targets, `PYTORCH_ROCM_ARCH=gfx906`, and math-only attention policy. The
  wheel is `torch-2.13.0+mi50.5-cp312-cp312-linux_x86_64.whl` (SHA-256
  `12b595e056ee0f79b12a4bf969477059f085e1e86450e85a367ce29622efd4cc`,
  179,844,209 bytes). The build applies the reviewed downstream patches
  `patches/downstream/pytorch/0001-mi50-accurate-rocm-precision-policy.patch`,
  and `patches/downstream/pytorch/0002-mi50-safe-tf32-capability-query.patch`.
  The wheel reports false for native gfx906 BF16 and TF32, and returns false
  instead of throwing when either capability is queried without a GPU. Host
  import reports ROCm 10.0.0/HIP 7.15 and zero devices on this WSL host, so its
  runtime status remains `GPU-test-pending`.
- PyTorch's compile-time hipBLASLt dependency is handled by a reproducible
  host-only build entry point. `build_hipblaslt_host.sh` uses the pinned
  ROCm 10.0 source with device, ExtOp, MatrixTransform and client targets
  disabled; the resulting package contains headers and the host library only.
  It is not a claim that hipBLASLt kernels work on MI50.
- An isolated experimental hipBLASLt device build was also exercised with
  `GPU_TARGETS=gfx906`: CMake and the 574,620-solution Tensile validation pass,
  but the pinned source has no gfx906 solution YAML and emits a zero-text
  helper object. `build_hipblaslt_gfx906_experimental.sh` now rejects that
  artifact (exit 78) instead of allowing an empty kernel catalog into release
  packages; the evidence and next porting step are recorded in
  `docs/EXPERIMENTAL_PORTS.md`.
- The top-level experimental switch is now functional: an isolated
  configure-only TheRock graph with `MI50_ENABLE_EXPERIMENTAL_NEW_ISA_PORTS=ON`
  enabled hipBLASLt, Composable Kernel and hipTensor for `gfx906` while keeping
  hipSPARSELt, rocWMMA and rocprofiler-compute excluded. The candidate graph
  was not merged into the stable artifact; its full LLVM rebuild was stopped
  after configure evidence to avoid duplicating the release toolchain.
- The Docker build context excludes generated source/build/artifact trees via
  `.dockerignore`, preventing a prior interrupted build from being copied into
  the builder image.
- The TheRock entry point now verifies every source checkout against
  `sources.lock.json` before applying patches or configuring CMake; arbitrary
  unpinned `--source-root` trees fail with a machine-readable report.
- The builder also verifies every patch SHA-256 digest from the lock before
  applying the queue, preventing local patch drift from entering an artifact.
- The TheRock entry point now has an explicit `MI50_BUILD_PROFILE=inference`
  recovery mode that disables optional storage/debug/profiler/data-center/
  media/emulation groups while retaining core, communication, math and ML
  libraries; `full` remains the release profile.
- rocBLAS solution enumeration now returns `rocblas_status_not_implemented`
  when the gfx906 Tensile library is unavailable, matching the execution-path
  fallback instead of dereferencing a null library/device description.
- rocBLAS Tensile host initialization no longer aborts merely because a
  CPU-only packaging/diagnostic host has no HIP device; real device calls still
  require native gfx906 validation.
- Compatibility delta and “latest compatible” acceptance rules documented in
  `docs/COMPATIBILITY.md`.
- Ordered bring-up, performance and soak gates documented in
  `docs/HARDWARE_VALIDATION.md`.
- Pre-hardware llama.cpp benchmark harness with native gfx906/wave64 gating,
  `amd-smi` telemetry capture and a 5% baseline-regression check.
- Native HIP device-matrix smoke for FP32/FP64 correctness and optional
  dual-device peer access, with a strict `MI50_REQUIRE_PEER=1` mode.
 - Distribution-completeness gate and artifact heal: `artifact-flatten` is
   manifest driven, so a 0-byte `artifact_manifest.txt` or a slice of 0-byte
   truncations silently drops the HIP compiler from `dist/rocm` and the wheels.
   `scripts/repair_artifact_manifests.py` rebuilds manifests from the payload
   that exists and, with `--heal --build-root`, deletes only slices whose
   empty files have non-empty counterparts in the intact `<project>/stage`
   trees, so legitimately empty upstream files are never mistaken for damage.
   `scripts/validate_dist_contents.py --strict` then fails the build if any
   required compiler/runtime path is absent or empty, gfx906 rocBLAS/MIOpen
   data is missing, or foreign-architecture device code leaked in. Both are
   covered by `tests/test_artifact_gates.py` (39 host tests pass).
 - `docs/RESUME.md` records the interrupted repackage state and the exact
   detached command that finishes it without a GPU.

## Source-graph verification

On WSL Ubuntu 24.04, the pinned TheRock 10.0 source graph configured
successfully with:

```text
THEROCK_AMDGPU_FAMILIES=gfx906
THEROCK_DIST_AMDGPU_FAMILIES=gfx906
THEROCK_TEST_AMDGPU_FAMILIES=gfx906
MI50_ENABLE_FORWARD_PORTS=ON
```

The stable generated artifact manifest contains rocBLAS and MIOpen; hipBLASLt,
Composable Kernel and hipTensor are not activated because their current ROCm 10
device paths require newer ISA features. hipSPARSELt may remain listed as a
host artifact, but its gfx906 device target is intentionally filtered because
its ROCm 10 logic is MFMA-only; rocSPARSE is the fallback. This proves
source-level inclusion only; it does not prove compilation or GPU runtime
correctness.

The standalone ROCr path has also completed an end-to-end configure, shared
build, install, target-object validation and host-load smoke test. It emitted
only the gfx906 trap handler, three GFX9 blit shaders and the gfx906 OpenCL
image object. `hsa_init` correctly remains a no-device result in WSL because
`/dev/kfd` is absent.

The current repository does not check compiled ROCm binaries into source
control. A WSL Ubuntu 24.04 host build has nevertheless compiled and linked a
shared target-scoped ROCr runtime (including gfx906 device objects); the
result remains artifact-only until it is run on a real MI50.

The broader TheRock superbuild has now configured and compiled LLVM, COMGR,
HIP/ROCclr, ROCr, diagnostics and ROCgdb with real gfx906 offload flags. It is
continuing through the upstream HIP test and ROCm library graph. OpenCL remains
disabled by default; a provisioned build can enable it with
`MI50_ENABLE_OPENCL=ON`. This is source/build evidence only and does not
upgrade any component to runtime-supported status.

The full Linux profile has also completed the reproducible packaging path. The
generated `rocm-10.0.0+mi50.5-mi50-gfx906-linux.tar.gz` archive contains the
merged HIP/ROCr install prefix, 211 rocBLAS gfx906 device objects and the three
retained MIOpen gfx906 database files. The latest rebuilt archive SHA-256 is
`f65d0d544506e2e4f3b86f6d190c834b0b2867817266a711b1885c7add3b8290` (2,307,831,941
bytes); strict artifact, ROCr, Python-package and ELF dependency validators pass.
The ELF audit scans 1,287 binaries and the build repairs the RDC test binary's
`$ORIGIN` RUNPATH so its bundled GTest libraries resolve after installation.
The archive excludes the
unsupported rocprofiler-compute, hipSPARSELt, rocWMMA, hipTensor and Composable
Kernel payloads. It remains artifact-only until exercised on an actual MI50.

A clean end-to-end full-profile rebuild (fresh `--fetch-root` source
checkout, `MI50_BUILD_TESTING=ON`, `MI50_BUILD_PYTHON_PACKAGES=ON`) has since
completed the same path with the ROCr WSL-thunk fallback patches and produced
the Python SDK artifacts:

```text
rocm-10.0.0+mi50.5-mi50-gfx906-linux.tar.gz  sha256 f65d0d544506e2e4f3b86f6d190c834b0b2867817266a711b1885c7add3b8290 (2,307,831,941 bytes)
rocm_profiler-10.0.0+mi50.5-...whl           sha256 fc023d6831731b07145df3c0332d1bb308f734a7c60fe11e6ab9dd2402c7ab62 (178,733,073 bytes)
rocm_sdk_core-10.0.0+mi50.5-...whl           sha256 392d3358658f6869eb382b2d7221b035a50649878ef793e742ce90eeba2f26a7 (1,286,786,853 bytes)
rocm_sdk_libraries-10.0.0+mi50.5-...whl      sha256 7b547b09de2c56f1d68e6e3c810b8a9f2128dc2c2d96bec2310995479ff6a0df (1,165,322,998 bytes)
rocm_sdk_device_gfx906-10.0.0+mi50.5-...whl  sha256 ea6ecb5600eaf26278f4f1d41fed53d8222be9e16ea8916b5bdf27dd4697f72e (154,831,770 bytes)
rocm_sdk_devel-10.0.0+mi50.5-...whl          sha256 6e3d45e3bfd2507287c8be7133d7d312ee48579b15fdbaa5472db51be9317724 (2,296,225,958 bytes)
rocm-10.0.0+mi50.5.tar.gz (sdist)             sha256 1e708242110cdfc18321eaf5adf32963f304d88e15dddf01eb2491a048609ccc (23,590 bytes)
```

The flattened tree carries 226 `gfx906`-named device files, of which 211 are
rocBLAS/Tensile code objects and 3 are retained MIOpen Vega20 database files.
Strict ROCr, source-lock, patch-lock, artifact and Python-package validators
all pass on this fresh output, and the artifact validator reports
`no_isa_override_instruction` pass across 15,949 checked files. The corrected
package includes a real 316,488-byte `bin/hipcc`, LLVM/AMDGPU compiler binaries,
`libamd_comgr.so` and non-empty HIP/OCML device bitcode; the previous archive's
0-byte compiler payload was rejected by the new distribution-completeness gate.

The current upstream llama.cpp checkout (`ggml-org/llama.cpp` commit
`c589f0ed10c643678c4707dd160c21ac7633ebc0`) also builds successfully against
the extracted package with `GGML_HIP=ON`, `AMDGPU_TARGETS=gfx906` and
`CMAKE_HIP_ARCHITECTURES=gfx906`. The installed `llama-cli`, `llama-bench` and
HIP backend are host-validated; startup correctly reports no ROCm-capable device
on this pre-hardware host and remains `GPU-test-pending`.

The ISA-override gate was refined while producing that evidence: upstream ROCm
helpers (notably `rocm_agent_enumerator`) legitimately *read*
`HSA_OVERRIDE_GFX_VERSION` to warn users, so a bare substring match made every
correct distribution fail the gate. The check now fails only when a file
*assigns* a concrete value to the override variable, which is the actual
masquerading hazard. Both the artifact validator and the source audit share one
detector (`mi50_policy.isa_override_findings`), and the regression tests pin
the enabling forms (`export …=9.0.8`, `-D…=`, `os.environ[…] = …`,
`putenv("…", "…")`, JSON mapping entries) against the harmless forms (reads,
comparisons, help strings, `unset`, empty assignment).

## Pending a real MI50 test rig

- KFD/firmware/device enumeration.
- HIP kernel launch, memory, graph and runtime-compilation tests.
- rocBLAS/MIOpen/RCCL execution tests.
- PyTorch and llama.cpp GPU correctness tests.
- Dual-card PCIe/Infinity Fabric behavior.
- ECC/RAS, reset and 24-hour inference soak.
- Performance comparison against ROCm 6.3.3 and ROCm 10.x gfx906 artifacts.

## Forward-port backlog

- Audit each ROCm 10 target exclusion and classify it as portable, requiring a
  gfx906-specific implementation, or hardware-limited.
- Port and benchmark compatible kernels/operators before accepting a fallback
  as permanent.
- Add Vega20-specific tuning and regression benchmarks for rocBLAS, MIOpen,
  attention, collective communication and LLM GEMM shapes.

The pending list is intentional. Passing host-only tests never upgrades a
component to runtime-supported status.
