# Build recovery

The full TheRock build is large and WSL stores its Linux filesystem in a
growing VHDX. Keep substantial free space on the Windows volume that contains
the VHDX; if it reaches the host-volume limit, WSL can remount the filesystem
read-only and Ninja will report `Read-only file system` while writing logs or
dependency files.

Recovery is non-destructive:

1. Stop the build and run `wsl --shutdown` from PowerShell.
2. Free host disk space (at least the build preflight default of 32 GiB).
3. Start the same distribution and verify `df -h /` plus a writable temporary
   file before resuming.
4. The build verifies the TheRock checkout against `sources.lock.json` before
   touching the build tree. Re-run it with `--skip-audit`; the existing source audit and completed
   Ninja stages are reused:

   ```bash
   scripts/build_therock_gfx906.sh \
     --source-root /path/to/TheRock \
     --build-dir /path/to/build \
     --artifact-dir /path/to/artifacts \
     --skip-audit
   ```

For the first artifact-producing resume, upstream GPU test binaries can be
left out to reduce disk/time pressure by adding `MI50_BUILD_TESTING=OFF`; the
repository's host-only tests and pending-hardware smoke gates remain available
separately.

If the host volume cannot provide the full graph's working space, use the
explicit inference profile for a smaller PyTorch/llama.cpp bring-up:

```bash
MI50_BUILD_PROFILE=inference MI50_BUILD_TESTING=OFF MI50_MIN_FREE_GIB=8 \
  scripts/build_therock_gfx906.sh \
  --source-root /path/to/TheRock \
  --build-dir /path/to/build-inference \
  --artifact-dir /path/to/artifacts-inference \
  --skip-audit
```

The `8` GiB value only lowers the early guard; it is not a promise that every
host can complete the build with that amount of free space. Increase it when
the measured working set or Docker/WSL overhead requires more room.

The profile disables optional storage, debug, profiler, data-center, media and
emulation groups only. It still builds the core runtime, communication,
math and ML groups; the full profile remains the release target.

Do not delete the source checkout or build tree as a first recovery step. The
build is reproducible from `sources.lock.json`, but retaining completed stages
avoids repeating the compiler and runtime build.

For WSL VHD maintenance, follow Microsoft's [WSL disk-space and recovery
guidance](https://learn.microsoft.com/en-us/windows/wsl/disk-space) after making
the host volume writable again.
