# Ubuntu 24.04.4 build environment

This image contains the host-side toolchain used to build the pinned TheRock
ROCm 10.0 gfx906 artifacts. It intentionally does not include a kernel driver
or firmware and does not claim GPU support.

The Dockerfile pins the Ubuntu 24.04 multi-architecture index by digest. To
use a deliberately different mirror or refreshed digest, pass
`MI50_BASE_IMAGE=registry.example/ubuntu:24.04@sha256:...`; the selected base
and the daemon-resolved image identity are written to
`out/container-image-provenance.json` by `scripts/build_container.sh`.

Build it from the repository root:

```bash
scripts/build_container.sh
```

Run a source build by mounting a TheRock checkout and the output directory:

```bash
docker run --rm -it \
  -v /path/to/TheRock:/sources/TheRock \
  -v "$PWD/out:/workspace/out" \
  mi50-rocm10-builder:10.0.0-mi50.5 \
  bash -lc 'scripts/build_therock_gfx906.sh --source-root /sources/TheRock'
```
