#!/usr/bin/env python3
"""Fail when the flattened gfx906 distribution is missing load-bearing files.

TheRock populates ``dist/rocm`` from manifest-driven artifact slices, so a slice
whose ``artifact_manifest.txt`` is empty contributes nothing and the tree still
"succeeds".  That produced a distribution without a HIP compiler.  This gate
asserts the installed tree actually contains the pieces that make a ROCm
distribution usable, independently of any GPU.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


REQUIRED_PATHS = (
    "bin/hipcc",
    "bin/hipconfig",
    "bin/rocminfo",
    "include/hip/hip_runtime.h",
    "include/hsa/hsa.h",
    "lib/libamd_comgr.so",
    "lib/libamdhip64.so",
    "lib/libhsa-runtime64.so",
    "lib/llvm/bin/amdclang++",
    "lib/llvm/bin/clang++",
    "lib/llvm/bin/llc",
    "lib/llvm/bin/amdlld",
    "lib/llvm/amdgcn/bitcode/hip.bc",
    "lib/llvm/amdgcn/bitcode/ocml.bc",
)


DEVICE_CODE_SUFFIXES = (
    ".hsaco",
    ".co",
    ".kdb",
    ".dat",
    ".o",
    ".out",
    ".bc",
)


def _has_target_data(root: Path, relative: Path, target: str) -> int:
    directory = root / relative
    if not directory.is_dir():
        return 0
    return sum(
        1
        for path in directory.rglob("*")
        if target in path.name and (path.is_file() or path.is_symlink())
    )


def validate(dist_dir: Path, target: str = "gfx906") -> dict:
    dist_dir = dist_dir.resolve()
    checks: dict[str, bool] = {}
    details: dict[str, object] = {}

    if not dist_dir.is_dir():
        return {
            "schema_version": 1,
            "dist_dir": str(dist_dir),
            "target": target,
            "checks": {"distribution_present": False},
            "missing": ["distribution_present"],
            "status": "fail",
            "runtime_claim": "host-only inspection; GPU execution remains pending-hardware",
        }

    for relative in REQUIRED_PATHS:
        path = dist_dir / relative
        present = path.is_file() or path.is_symlink()
        # A zero-byte file is the signature of a truncated concurrent copy, not
        # a usable payload.
        nonempty = True
        if present and path.is_file():
            nonempty = path.stat().st_size > 0
        checks[f"required:{relative}"] = present and nonempty
        if not nonempty:
            details[f"empty:{relative}"] = True

    rocblas_objects = _has_target_data(dist_dir, Path("lib/rocblas/library"), target)
    miopen_assets = _has_target_data(dist_dir, Path("share/miopen"), target)
    checks[f"target_rocblas_data:{target}"] = rocblas_objects > 0
    checks[f"target_miopen_data:{target}"] = miopen_assets > 0
    details["rocblas_target_files"] = rocblas_objects
    details["miopen_target_files"] = miopen_assets

    # No newer-ISA *device code* may leak into a gfx906-only distribution.
    # Upstream ships architecture-named headers, profiler configs and tuner
    # tables for every GPU, so a name-only match would be a permanent false
    # positive; restrict the invariant to loadable code-object payloads and keep
    # the informational list separate from the gate.
    target_token = re.compile(r"gfx[0-9]{2,4}")
    foreign_code = sorted(
        path.relative_to(dist_dir).as_posix()
        for path in dist_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in DEVICE_CODE_SUFFIXES
        and any(
            token != f"{target}" for token in target_token.findall(path.name.lower())
        )
    )
    checks[f"single_target_device_code:{target}"] = not foreign_code
    details["foreign_device_code_files"] = foreign_code
    details["foreign_targets_in_file_names"] = sorted(
        {
            token
            for path in dist_dir.rglob("*")
            if path.is_file()
            for token in target_token.findall(path.name.lower())
            if token != target
        }
    )

    missing = [name for name, passed in checks.items() if not passed]
    return {
        "schema_version": 1,
        "dist_dir": str(dist_dir),
        "target": target,
        "checks": checks,
        "details": details,
        "missing": missing,
        "status": "pass" if not missing else "fail",
        "runtime_claim": "host-only inspection; GPU execution remains pending-hardware",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist-dir", type=Path, required=True)
    parser.add_argument("--target", default="gfx906")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)

    report = validate(args.dist_dir, args.target)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    for name in report["missing"]:
        print(f"distribution check failed: {name}", flush=True)
    return 1 if args.strict and report["status"] != "pass" else 0


if __name__ == "__main__":
    raise SystemExit(main())
