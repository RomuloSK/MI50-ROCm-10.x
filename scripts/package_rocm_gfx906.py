#!/usr/bin/env python3
"""Create a reproducible Linux ROCm gfx906 distribution archive.

Some TheRock revisions materialize the distribution as a flattened
``dist/rocm`` tree but do not emit a tarball.  This helper turns that tree
into the installable artifact used by the MI50 compatibility build.  It
intentionally does not rewrite binaries or add an ISA override: the archive
contains the exact files produced by TheRock.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile


SCHEMA_VERSION = 1
DEFAULT_VERSION = "10.0.0+mi50.5"
DEFAULT_TARGET = "gfx906"
EXCLUDED_COMPONENTS = (
    "rocprofiler-compute",
    "hipsparselt",
    "rocwmma",
    "hiptensor",
    "composable_kernel",
    "composable-kernel",
)


def _relative_files(root: Path) -> list[str]:
    """Return archive members in stable order, including symlinks."""

    members: list[str] = []
    for path in root.rglob("*"):
        # rglob follows neither symlinked files nor symlinked directories for
        # traversal, while the link itself is still a valid archive member.
        if path.is_file() or path.is_symlink():
            members.append(path.relative_to(root).as_posix())
        elif path.is_dir():
            members.append(path.relative_to(root).as_posix() + "/")
    return sorted(set(members))


def validate_source(source: Path, target: str = DEFAULT_TARGET) -> dict:
    """Validate required target-specific assets without reading large blobs."""

    source = source.resolve()
    if not source.is_dir():
        raise ValueError(f"distribution tree does not exist: {source}")

    relative = _relative_files(source)
    lower = [name.lower() for name in relative]
    rocblas = [
        name
        for name in relative
        if "/rocblas/library/" in f"/{name.lower()}"
        and target.lower() in name.lower()
    ]
    miopen = [
        name
        for name in relative
        if "/miopen/" in f"/{name.lower()}"
        and target.lower() in name.lower()
    ]
    required = {
        "target": target,
        "rocblas_device_data": len(rocblas),
        "miopen_device_data": len(miopen),
        "has_hip_headers": (source / "include" / "hip").is_dir(),
        "has_llvm_llc": (source / "lib" / "llvm" / "bin" / "llc").is_file(),
        "has_rocr_runtime": any(
            "libhsa-runtime64" in name or "hsa-runtime" in name
            for name in lower
        ),
    }
    if not rocblas:
        raise ValueError(f"no {target} rocBLAS device data under {source}")
    if not miopen:
        raise ValueError(f"no {target} MIOpen device data under {source}")
    if not required["has_hip_headers"]:
        raise ValueError(f"HIP headers are missing under {source / 'include'}")
    if not required["has_llvm_llc"]:
        raise ValueError(f"LLVM llc is missing under {source / 'lib' / 'llvm' / 'bin'}")
    if not required["has_rocr_runtime"]:
        raise ValueError(f"ROCr runtime library is missing under {source / 'lib'}")
    return required


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_tree(
    source: Path,
    output: Path,
    metadata_output: Path | None = None,
    *,
    version: str = DEFAULT_VERSION,
    target: str = DEFAULT_TARGET,
) -> dict:
    """Package *source* with GNU tar/gzip and return the emitted metadata."""

    source = source.resolve()
    output = output.resolve()
    try:
        output.relative_to(source)
    except ValueError:
        pass
    else:
        raise ValueError("archive output must be outside the distribution source tree")
    checks = validate_source(source, target)
    output.parent.mkdir(parents=True, exist_ok=True)

    # GNU tar's sort/mtime/owner switches and gzip -n remove filesystem,
    # username and timestamp variation.  The temporary file keeps a failed
    # compression from leaving a seemingly valid partial artifact.
    with tempfile.NamedTemporaryFile(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent, delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        tar_command = [
            "tar",
            "--wildcards",
            *(
                option
                for component in EXCLUDED_COMPONENTS
                for option in (
                    f"--exclude={source.name}/**/{component}*",
                    f"--exclude={source.name}/**/lib{component}*",
                )
            ),
            "--sort=name",
            "--mtime=@0",
            "--owner=0",
            "--group=0",
            "--numeric-owner",
            "-C",
            str(source.parent),
            "-cf",
            "-",
            source.name,
        ]
        gzip_command = ["gzip", "-n", "-6"]
        with temporary_path.open("wb") as stream:
            tar_process = subprocess.Popen(tar_command, stdout=subprocess.PIPE)
            assert tar_process.stdout is not None
            gzip_process = subprocess.Popen(gzip_command, stdin=tar_process.stdout, stdout=stream)
            tar_process.stdout.close()
            gzip_status = gzip_process.wait()
            tar_status = tar_process.wait()
        if tar_status != 0 or gzip_status != 0:
            raise RuntimeError(
                f"tar/gzip failed (tar={tar_status}, gzip={gzip_status})"
            )
        os.replace(temporary_path, output)
        # NamedTemporaryFile defaults to owner-only permissions.  The release
        # archive is an installation input, so make it readable by the host
        # user while retaining a non-writable package artifact.
        output.chmod(0o644)
    finally:
        temporary_path.unlink(missing_ok=True)

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "package": "rocm-mi50-gfx906-linux",
        "version": version,
        "target": target,
        "format": "tar.gz",
        "archive": output.name,
        "archive_sha256": _sha256(output),
        "archive_bytes": output.stat().st_size,
        "source_tree": str(source),
        "source_checks": checks,
        "rocblas_marker": f"TensileLibrary*_{target}",
        "miopen_marker": f"{target}_60.HIP.fdb.txt",
        "excluded_components": list(EXCLUDED_COMPONENTS),
        "runtime_claim": "artifact-only; GPU execution remains pending-hardware",
        "isa_override": "disabled; no ISA masquerading is permitted",
        "status": "pass",
    }
    if metadata_output is not None:
        metadata_output = metadata_output.resolve()
        metadata_output.parent.mkdir(parents=True, exist_ok=True)
        temporary_metadata = metadata_output.with_suffix(metadata_output.suffix + ".tmp")
        temporary_metadata.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary_metadata, metadata_output)
    return metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path)
    parser.add_argument("--version", default=DEFAULT_VERSION)
    parser.add_argument("--target", default=DEFAULT_TARGET)
    args = parser.parse_args(argv)
    try:
        metadata = package_tree(
            args.source_dir,
            args.output,
            args.metadata_output,
            version=args.version,
            target=args.target,
        )
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
