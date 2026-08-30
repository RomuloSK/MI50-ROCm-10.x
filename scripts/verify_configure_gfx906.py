#!/usr/bin/env python3
"""Verify that a TheRock configure actually selected the MI50 forward-port.

This check is intentionally CPU-only. It validates generated CMake metadata and
the artifact-subproject manifest; it never claims that any GPU kernel runs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REQUIRED_CACHE = {
    "MI50_ENABLE_FORWARD_PORTS": "ON",
    "THEROCK_AMDGPU_FAMILIES": "gfx906",
    "THEROCK_DIST_AMDGPU_FAMILIES": "gfx906",
    "THEROCK_TEST_AMDGPU_FAMILIES": "gfx906",
}

REQUIRED_ARTIFACT_SUBPROJECTS = {
    # hipBLASLt is an explicit forward-port candidate and is required only
    # when the experimental newer-ISA option is enabled. Stable gfx906 builds
    # must contain rocBLAS but must not activate hipBLASLt's gfx1100 fallback.
    "blas": {"rocBLAS"},
    "miopen": {"MIOpen"},
}

# These projects are optional compatibility candidates.  A gfx906 build may
# explicitly disable them when ROCm 10.x has no usable Vega20 implementation;
# if a caller leaves the cache unset, retain the historical requirement so
# older configure reports remain meaningful.
CONDITIONAL_ARTIFACT_SUBPROJECTS = {
    "hipblaslt": ("MI50_ENABLE_EXPERIMENTAL_NEW_ISA_PORTS", {"hipBLASLt"}),
    "composable-kernel": ("THEROCK_ENABLE_COMPOSABLE_KERNEL", {"composable_kernel"}),
    "hiptensor": ("THEROCK_ENABLE_HIPTENSOR", {"hipTensor"}),
}


def read_cache(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line or raw_line.startswith("//") or raw_line.startswith("#"):
            continue
        if ":" not in raw_line or "=" not in raw_line:
            continue
        key_type, value = raw_line.split("=", 1)
        key = key_type.split(":", 1)[0]
        values[key] = value
    return values


def verify(build_root: Path) -> dict:
    cache_path = build_root / "CMakeCache.txt"
    manifest_path = build_root / "artifact_subprojects.json"
    errors: list[str] = []

    if not cache_path.is_file():
        errors.append(f"missing {cache_path}")
        cache = {}
    else:
        cache = read_cache(cache_path)
        for key, expected in REQUIRED_CACHE.items():
            actual = cache.get(key)
            if actual != expected:
                errors.append(f"{key}={actual!r}; expected {expected!r}")

    if not manifest_path.is_file():
        errors.append(f"missing {manifest_path}")
        manifest = {}
    else:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            errors.append(f"invalid {manifest_path}: {exc}")
            manifest = {}

    for artifact, required in REQUIRED_ARTIFACT_SUBPROJECTS.items():
        actual = set(manifest.get(artifact, []))
        for project in sorted(required - actual):
            errors.append(f"artifact {artifact!r} is missing subproject {project!r}")

    for artifact, (cache_key, required) in CONDITIONAL_ARTIFACT_SUBPROJECTS.items():
        enabled = cache.get(cache_key, "ON")
        if enabled.upper() in {"OFF", "0", "FALSE", "NO"}:
            continue
        actual = set(manifest.get(artifact, []))
        for project in sorted(required - actual):
            errors.append(f"artifact {artifact!r} is missing subproject {project!r}")

    return {
        "schema_version": 1,
        "build_root": str(build_root.resolve()),
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "required_cache": REQUIRED_CACHE,
        "required_artifact_subprojects": {
            key: sorted(value) for key, value in REQUIRED_ARTIFACT_SUBPROJECTS.items()
        },
        "conditional_artifact_subprojects": {
            key: {"cache_key": cache_key, "required": sorted(value)}
            for key, (cache_key, value) in CONDITIONAL_ARTIFACT_SUBPROJECTS.items()
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-root", type=Path, required=True)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)

    report = verify(args.build_root.resolve())
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(payload, encoding="utf-8")
    else:
        sys.stdout.write(payload)

    return 1 if args.strict and report["status"] != "pass" else 0


if __name__ == "__main__":
    raise SystemExit(main())
