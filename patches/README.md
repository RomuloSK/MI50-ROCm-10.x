# Patch queue policy

The queue is intentionally small and reviewable. It contains only changes
needed to make a gfx906-only ROCm 10.x graph configure and compile; it does
not change the reported ISA or claim that a newer GPU is an MI50.

Current patches:

- `0001-gfx906-forward-port-target-policy.patch` enables opt-in build attempts
  for ROCm 10 components whose upstream gfx906 exclusion is a portability
  policy rather than a proven hardware impossibility.
- `0002-rocr-gfx906-targeted-device-code.patch` limits ROCr generated device
  objects to gfx906 and accepts an explicit device-library path.
- `0003-rocr-gfx906-targeted-shader-table.patch` keeps the reduced shader table
  linkable and rejects non-gfx906 agents instead of silently reusing an object
  for the wrong ISA.
- `0004-rocr-gfx906-image-dispatch.patch` removes unresolved newer-ISA image
  objects from the shared runtime and makes non-gfx906 image dispatch fail
  explicitly.
- `0005-therock-forward-rocr-target-args.patch` passes the target-scoped
  gfx906 setting through TheRock's background ROCr subproject; a top-level
  cache variable alone is otherwise ignored.
- `0006-hipblaslt-register-gfx906-target.patch` registers gfx906 as an
  explicit hipBLASLt validation target without adding it to the default
  `all` list. rocBLAS remains the production fallback until hipBLASLt device
  data and MI50 correctness/performance are proven.
- `0007-hipsparselt-gfx906-fallback-policy.patch` keeps hipSPARSELt excluded:
  its ROCm 10 structured-sparse implementation has only newer MFMA logic and
  no gfx906 data, so rocSPARSE is the honest fallback.
- `0008-rocblas-no-abort-missing-tensile.patch` converts missing or unloadable
  target Tensile data from a process-wide abort into
  `rocblas_status_not_implemented`, allowing callers to select a fallback.
- `0009-rocblas-gfx906-hipblaslt-fallback.patch` keeps gfx906 on the mature
  rocBLAS/Tensile backend by default; `ROCBLAS_USE_HIPBLASLT=1` is reserved for
  explicit hardware experiments.
- `0010-rocclr-optional-opengl-for-hip.patch` makes Linux OpenGL discovery an
  explicit ROCclr option; the default is unchanged, but HIP-only builders can
  disable the host GL dependency.
- `0011-rocclr-opengl-env-override.patch` carries that option through TheRock's
  nested ROCclr configure step using `ROCCLR_ENABLE_OPENGL` in the environment.
- `0012-rocblas-no-abort-get-solutions.patch` makes the solution-list API
  return `rocblas_status_not_implemented` when gfx906 Tensile data is absent,
  matching the execution-path fallback instead of dereferencing null data.
- `0013-rocblas-no-abort-no-device.patch` lets rocBLAS load on a CPU-only host
  without aborting during Tensile host construction; device calls still return
  a normal unsupported/error status.
- `0014-therock-atomic-artifact-copies.patch` publishes flattened files through
  same-directory temporary files, preventing parallel artifact assembly from
  leaving truncated or zero-byte ELF files in `dist/rocm`.
- `0015-rocprofiler-gfx906-dist-target.patch` limits rocprofiler-sdk's test
  target set to the configured gfx906 distribution and keeps newer-GPU test
  bundles out of the MI50 build.
- `0016-rocroller-disable-pytest-gemm-tests.patch` keeps rocRoller available
  for rocBLAS while disabling its nested pytest/CMake GEMM test discovery.
- `0017-therock-disable-unsupported-gfx906-forward-ports.patch` makes the
  stable profile keep hipBLASLt, Composable Kernel and hipTensor excluded even
  when the general forward-port switch is on; an explicit experimental option
  is required to attempt their newer-ISA paths.
- `0018-therock-skip-unsupported-hipblaslt-activation.patch` prevents
  hipBLASLt's no-target fallback from silently selecting gfx1100; stable builds
  omit its activation and route rocBLAS through its native gfx906 Tensile data.
- `0019-therock-remove-hipblaslt-test-edge.patch` removes rocRoller's nested
  hipBLASLt test dependency so disabled hipBLASLt cannot re-enter the stable
  MIOpen/compile-commands dependency graph through a stage-stamp edge.
- `0020-therock-guard-hipblaslt-dnn-dependencies.patch` removes hipBLASLt from
  MIOpen and hipDNN provider dependencies in stable mode, retaining them only
  behind the explicit experimental-new-ISA switch.
- `0021-therock-disable-rocblas-hipblaslt-linkage.patch` configures stable
  rocBLAS without searching for or linking the unavailable hipBLASLt package;
  the experimental mode restores that linkage for later testing.
- `0022-therock-declare-amdsmi-for-rocblas.patch` declares rocBLAS's Linux
  `amd_smi` client dependency directly now that hipBLASLt no longer supplies it
  transitively, and closes the matching conditional provider block cleanly.
- `0023-therock-fix-amdsmi-package-prefix.patch` points TheRock's amdsmi
  package trampoline at its installed `lib/cmake/amd_smi` directory, allowing
  rocBLAS host clients to resolve the package without a non-hermetic search.
- `0024-therock-atomic-artifact-flatten.patch` makes artifact flattening use
  atomic, copy-on-write publication rather than hard-linking staged files;
  this prevents concurrent stage/flatten work from exposing partial host or
  device ELF files.
- `0025-miopen-disable-hipblaslt-stable.patch` forces MIOpen's stable profile
  to use its rocBLAS/legacy gfx906 path without an undeclared hipBLASLt lookup;
  the original hipBLASLt option remains available in experimental mode.
- `0026-therock-pass-miopen-hipblaslt-policy.patch` forwards the same stable vs
  experimental hipBLASLt decision into MIOpen's nested CMake configure, where
  top-level cache variables are otherwise not inherited.
- `0027-rocr-fallback-to-static-kfd.patch` fixes ROCr's failed optional WSL/DXG
  thunk load path: a missing `librocdxg.so` no longer counts as a loaded thunk,
  and Linux falls back to the statically linked KFD thunk instead of resolving
  symbols through a null handle.
- `0028-rocr-fallback-on-incomplete-dxg-api.patch` extends that protection to a
  present-but-incomplete `librocdxg.so`: missing DXG exports now close the
  optional thunk, clear WSL mode and retry the native Linux KFD thunk rather
  than leaving ROCr half-initialized.

The build entry point applies each patch to the repository that owns its paths:
TheRock itself for `0001`, and the pinned `rocm-systems` submodule for `0002`
and `0003`. This is important because a submodule is a gitlink from the
superproject's point of view.

When a patch becomes necessary, add it to a subsystem-specific directory and
record:

1. the exact source commit and failing command;
2. a minimal reproducer or static assertion;
3. why the change is gfx906-specific or architecture-generic;
4. expected fallback behavior;
5. host-only and (when hardware exists) MI50 test coverage;
6. upstream issue/PR provenance.

Never solve an MI50 problem by changing the reported ISA to gfx908 or another
newer target.
