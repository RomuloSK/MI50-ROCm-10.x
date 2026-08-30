#!/usr/bin/env python3
"""Write reproducibility metadata for a MI50 ROCm build."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any


def command_output(*args: str) -> str | None:
    executable = shutil.which(args[0])
    if not executable:
        return None
    try:
        result = subprocess.run(
            [executable, *args[1:]], capture_output=True, text=True, timeout=15, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    output = (result.stdout or result.stderr).strip()
    return output.splitlines()[0] if output else None


def git_revision(path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    value = result.stdout.strip()
    return value if result.returncode == 0 else None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_provenance(
    output: Path,
    *,
    repository_root: Path,
    source_root: Path | None = None,
    build_root: Path | None = None,
    artifact_root: Path | None = None,
) -> dict[str, Any]:
    lock_path = repository_root / "sources.lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    patches = []
    for patch in lock.get("patches", []):
        patch_path = repository_root / patch["file"]
        patches.append(
            {
                "file": patch["file"],
                "sha256": sha256(patch_path) if patch_path.is_file() else None,
            }
        )
    source_commits = {}
    if source_root:
        for candidate in (
            source_root,
            source_root / "rocm-systems",
            source_root / "rocm-libraries",
            source_root / "compiler" / "amd-llvm",
        ):
            if candidate.is_dir():
                source_commits[str(candidate.relative_to(source_root)) or "."] = git_revision(candidate)

    payload: dict[str, Any] = {
        "schema_version": 1,
        "project_version": lock["project_version"],
        "rocm_version": lock["rocm_version"],
        "target": lock["target"],
        "container": lock.get("container"),
        "repositories_locked": lock["repositories"],
        "source_commits_observed": source_commits,
        "patches_observed": patches,
        "toolchain": {
            "python": command_output("python3", "--version") or command_output("python", "--version"),
            "cmake": command_output("cmake", "--version"),
            "ninja": command_output("ninja", "--version"),
            "cc": command_output(os.environ.get("CC", "cc"), "--version"),
            "cxx": command_output(os.environ.get("CXX", "c++"), "--version"),
            "meson": command_output("meson", "--version"),
        },
        "host": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "container_digest": os.environ.get("MI50_BUILD_CONTAINER_DIGEST"),
        },
        "environment": {
            key: os.environ.get(key)
            for key in (
                "THEROCK_AMDGPU_FAMILIES",
                "THEROCK_DIST_AMDGPU_FAMILIES",
                "THEROCK_TEST_AMDGPU_FAMILIES",
                "MI50_ENABLE_FORWARD_PORTS",
                "MI50_ENABLE_OPENCL",
                "MI50_BUILD_TESTING",
                "MI50_BUILD_PROFILE",
                "MI50_BUILD_PYTHON_PACKAGES",
                "MI50_ENABLE_EXPERIMENTAL_NEW_ISA_PORTS",
                "MI50_MIN_FREE_GIB",
                "ROCCLR_ENABLE_OPENGL",
                "ROCR_TARGET_DEVICES",
                "SOURCE_DATE_EPOCH",
                "TZ",
                "LC_ALL",
            )
        },
        "paths": {
            "source_root": str(source_root.resolve()) if source_root else None,
            "build_root": str(build_root.resolve()) if build_root else None,
            "artifact_root": str(artifact_root.resolve()) if artifact_root else None,
        },
        "runtime_claim": "artifact-only; GPU execution remains pending-hardware",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--build-root", type=Path)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    write_provenance(
        args.output.resolve(),
        repository_root=args.repository_root.resolve(),
        source_root=args.source_root.resolve() if args.source_root else None,
        build_root=args.build_root.resolve() if args.build_root else None,
        artifact_root=args.artifact_root.resolve() if args.artifact_root else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
