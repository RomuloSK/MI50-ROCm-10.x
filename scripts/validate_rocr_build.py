#!/usr/bin/env python3
"""Validate a target-scoped ROCr build without requiring a GPU.

The ROCr generators use the historical ``*9`` filenames for GFX9 device
objects, while the OpenCL image object carries an explicit ``gfx906`` target
in its AMDGPU metadata.  This validator checks both forms and rejects a build
that accidentally generated newer-ISA objects alongside the MI50 objects.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path


NEWER_OBJECT_RE = re.compile(
    r"(?:gfx(?:10|11|12)|(?:10|11|12))\.hsaco$", re.IGNORECASE
)

# LLVM's ROCr runtime build bootstraps a compiler-side test/runtime tree under
# ``compiler/amd-llvm/build/runtimes``.  That tree intentionally emits the
# generic blit/trap objects for every ISA known to the compiler (gfx10--gfx12
# included), even when the final ROCr distribution is target-scoped to gfx906.
# Those objects are never installed or packaged.  Keep the newer-ISA check
# focused on files that can escape into the MI50 runtime/artifacts while still
# rejecting a newer object placed in the validated output root itself.
NON_DISTRIBUTABLE_PREFIXES = (
    "compiler/amd-llvm/build/runtimes/",
)


def _all_files(root: Path) -> list[Path]:
    return [path for path in root.rglob("*") if path.is_file() and not path.is_symlink()]


def _is_distributable(path: Path, root: Path) -> bool:
    relative = path.relative_to(root).as_posix()
    return not any(relative.startswith(prefix) for prefix in NON_DISTRIBUTABLE_PREFIXES)


def validate(root: Path) -> dict:
    all_files = _all_files(root)
    names = {path.name for path in all_files}
    hsaco_files = [
        path
        for path in all_files
        if path.suffix.lower() == ".hsaco" and _is_distributable(path, root)
    ]

    runtime_library = any(
        path.name == "libhsa-runtime64.a"
        or path.name == "libhsa-runtime64.so"
        or re.fullmatch(r"libhsa-runtime64\.so\.\d+(?:\.\d+)*", path.name)
        for path in all_files
    )
    trap_handler = any(
        path.name == "kCodeTrapHandlerV2_9.hsaco"
        for path in hsaco_files
    )
    blit_shaders = {
        "kCodeCopyAligned9.hsaco",
        "kCodeCopyMisaligned9.hsaco",
        "kCodeFill9.hsaco",
    }.issubset(names)
    image_objects = [
        path for path in all_files if path.name == "ocl_blit_object_gfx906"
    ]
    image_metadata = False
    for path in image_objects:
        try:
            data = path.read_bytes()
        except OSError:
            continue
        image_metadata |= b"amdhsa.target" in data and b"gfx906" in data

    newer_objects = sorted(
        str(path.relative_to(root))
        for path in hsaco_files
        if NEWER_OBJECT_RE.search(path.name)
    )

    generated_undefined: list[str] = []
    nm_path = shutil.which("nm")
    runtime_elfs = [
        path
        for path in all_files
        if path.name == "libhsa-runtime64.so"
        or re.fullmatch(r"libhsa-runtime64\.so\.\d+(?:\.\d+)*", path.name)
    ]
    if nm_path and runtime_elfs:
        for runtime_elf in runtime_elfs:
            result = subprocess.run(
                [nm_path, "-D", "--undefined-only", str(runtime_elf)],
                capture_output=True,
                text=True,
                check=False,
            )
            for line in result.stdout.splitlines():
                symbol = line.rsplit(maxsplit=1)[-1] if line.split() else ""
                if "ocl_blit_object_" in symbol or "kCodeTrapHandler" in symbol:
                    generated_undefined.append(symbol)

    checks = {
        "runtime_library": runtime_library,
        "gfx906_trap_handler": trap_handler,
        "gfx906_blit_shaders": blit_shaders,
        "gfx906_opencl_image_object": bool(image_objects),
        "gfx906_opencl_metadata": image_metadata,
        "no_newer_isa_objects": not newer_objects,
        "no_unresolved_generated_objects": not generated_undefined,
    }
    missing = [name for name, passed in checks.items() if not passed]
    return {
        "schema_version": 1,
        "build_root": str(root.resolve()),
        "files_checked": len(all_files),
        "hsaco_files": sorted(str(path.relative_to(root)) for path in hsaco_files),
        "newer_isa_objects": newer_objects,
        "generated_undefined_symbols": sorted(set(generated_undefined)),
        "nm_available": bool(nm_path),
        "checks": checks,
        "missing": missing,
        "status": "pass" if not missing else "fail",
        "runtime_claim": "artifact-only; GPU execution remains pending-hardware",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-root", type=Path, required=True)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)

    root = args.build_root.resolve()
    if not root.is_dir():
        parser.error(f"build root does not exist or is not a directory: {root}")

    report = validate(root)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(payload, encoding="utf-8")
    else:
        sys.stdout.write(payload)
    return 1 if args.strict and report["status"] != "pass" else 0


if __name__ == "__main__":
    raise SystemExit(main())
