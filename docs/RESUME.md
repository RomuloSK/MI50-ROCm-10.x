# Resuming the gfx906 pre-hardware build after a power loss

This records the exact state after recovery. Everything described here is
host-only and can be resumed without an MI50 card.

## Where the build stands

* Pinned source: `/root/mi50-full-sources2/TheRock` (TheRock `therock-10.0`,
  commit recorded in `sources.lock.json`).
* Build tree: `/root/mi50-full-build2` -- the full native gfx906 compile is
  already complete (LLVM, COMGR, HIP, ROCr, RCCL, rocBLAS with 211 gfx906
  objects, rocSPARSE, rocSOLVER, hipBLAS, hipSOLVER, MIOpen, hipDNN).
* Artifacts: `/root/mi50-full-artifacts3` (tarball + Python wheels).
* The corrected ROCm package has been repackaged, extracted and host-validated.
* llama.cpp is built and installed from its pinned gfx906 checkout.
* PyTorch 2.13.0 is built from its pinned checkout; the wheel and metadata are
  under `/root/mi50-downstream-artifacts/pytorch`.
* Remaining work is host-side validation/documentation and, when the cards
  arrive, the pending GPU test matrix.

## The defect that was being fixed

`dist/rocm` shipped a HIP compiler made of 0-byte files. Three artifact slices
(`amd-llvm_run_generic`, `amd-llvm_lib_generic`, `hipify_run_generic`) were
truncated copies of intact `<project>/stage` trees, almost certainly by a
previously interrupted build. Because `fileset_tool.py artifact-flatten` is
manifest driven, the empty payloads were flattened verbatim and the install
could not compile HIP.

Fixed by two host-only gates plus a heal pass:

1. `scripts/repair_artifact_manifests.py --heal --build-root <build>` compares
   every 0-byte payload file against its build-tree counterpart, so a file that
   is legitimately empty upstream (gdb fixtures, hipify shell helpers) is never
   mistaken for corruption, and only genuinely truncated slices are removed for
   regeneration. It still repairs an empty manifest from the payload that does
   exist.
2. `scripts/validate_dist_contents.py --strict` fails the build if any required
   compiler/runtime path is absent *or empty*, if gfx906 rocBLAS/MIOpen data is
   missing, or if another architecture's device code leaked in.

`scripts/build_therock_gfx906.sh` now runs the heal before the compile and the
repair + validation before packaging. Confirmed against the real tree: the heal
identified exactly the three corrupt slices, 212 slices clean.

Those three slice directories have already been deleted from
`/root/mi50-full-build2/artifacts`, so ninja regenerates them from the intact
stage trees on the next run. The flattened `dist/rocm` still holds the old
empty files; the builder expunges and rebuilds it.

## Rebuild command (WSL Ubuntu 24.04)

```bash
export PATH=/root/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/bin:/usr/bin
export RUSTUP_TOOLCHAIN=1.98.0 MI50_BUILD_PROFILE=full MI50_BUILD_TESTING=ON
export MI50_BUILD_PYTHON_PACKAGES=ON MI50_MIN_FREE_GIB=32
export MI50_ENABLE_OPENCL=OFF ROCCLR_ENABLE_OPENGL=OFF
setsid bash -c "bash /mnt/c/Users/Romulo/Desktop/rocm10.0-mi50/scripts/build_therock_gfx906.sh \
  --source-root /root/mi50-full-sources2/TheRock \
  --build-dir /root/mi50-full-build2 \
  --artifact-dir /root/mi50-full-artifacts3 \
  --skip-audit --jobs 4 > /root/mi50-full-build2/repackage.log 2>&1" \
  </dev/null >/dev/null 2>&1 & disown
```

`nohup` is not enough here: the child dies with the WSL session, hence
`setsid` plus detached stdio. Poll `/root/mi50-full-build2/repackage.log`.

## Acceptance after the run

```bash
python3 /mnt/c/Users/Romulo/Desktop/rocm10.0-mi50/scripts/validate_dist_contents.py \
  --dist-dir /root/mi50-full-build2/dist/rocm --target gfx906 --strict
python3 /mnt/c/Users/Romulo/Desktop/rocm10.0-mi50/scripts/validate_artifacts.py \
  --artifact-root /root/mi50-full-artifacts3 --strict
python3 /mnt/c/Users/Romulo/Desktop/rocm10.0-mi50/scripts/validate_python_packages.py \
  --package-dir /root/mi50-full-artifacts3/python/dist --strict
```

Then re-extract the tarball over `/opt/rocm-mi50` and compile the host-only
smoke with the *installed* compiler, which is the point of this fix:

```bash
/opt/rocm-mi50/bin/hipcc --offload-arch=gfx906 -c \
  /mnt/c/Users/Romulo/Desktop/rocm10.0-mi50/tests/hip/mi50_runtime_smoke.hip \
  -o /tmp/mi50_runtime_smoke.o
```

A link step needs the runtime libraries only, not a GPU. If the downstream
artifacts need to be regenerated, use the pinned revisions in
`downstream.lock.json`; build the host-only hipBLASLt shim first, then invoke
the PyTorch wrapper with `--hipblaslt-host`. The wrappers preserve incremental
build directories, so an interrupted build can be rerun without discarding
completed HIP objects.

None of the above is evidence of MI50 runtime support; every GPU test stays
`pending-hardware`.
