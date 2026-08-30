#!/usr/bin/env python3
"""Report host/device prerequisites without making unsupported claims."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

try:  # Support both module and direct-script execution.
    from .mi50_policy import feature_contract
except ImportError:  # pragma: no cover - exercised by the shell entry point.
    from mi50_policy import feature_contract

try:  # Keep every diagnostic on the same normalized ROCm prefix.
    from .mi50_runtime_paths import normalize_rocm_root, runtime_environment as scoped_runtime_environment
except ImportError:  # pragma: no cover - exercised by the shell entry point.
    from mi50_runtime_paths import normalize_rocm_root, runtime_environment as scoped_runtime_environment

try:  # Works both as ``python -m scripts...`` and as a file entry point.
    from .mi50_kernel_readiness import collect_readiness
except ImportError:  # pragma: no cover - exercised by the shell entry point.
    from mi50_kernel_readiness import collect_readiness


def resolve_rocm_root(artifact_root: Path | None) -> Path | None:
    """Resolve an installed prefix or a direct ``rocm/`` tree."""

    if artifact_root is None:
        return None
    return normalize_rocm_root(artifact_root)


def runtime_environment(rocm_root: Path | None) -> dict[str, str]:
    return scoped_runtime_environment(rocm_root)


def command_version(
    command: str,
    *,
    rocm_root: Path | None = None,
    environment: dict[str, str] | None = None,
) -> str | None:
    if rocm_root is not None:
        candidate = rocm_root / "bin" / command
        executable = str(candidate) if candidate.is_file() and os.access(candidate, os.X_OK) else None
    else:
        search_environment = environment if environment is not None else os.environ
        search_path = search_environment.get("PATH", "") if environment is not None else None
        executable = shutil.which(command, path=search_path)
    if not executable:
        return None
    try:
        result = subprocess.run(
            [executable, "--version"],
            capture_output=True,
            text=True,
            timeout=5,
            env=environment,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "present (version query failed)"
    if result.returncode != 0:
        return f"present (version query failed, exit {result.returncode})"
    line = (result.stdout or result.stderr).strip().splitlines()
    return line[0] if line else "present"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--rocm", type=Path, help="ROCm installation or archive prefix")
    args = parser.parse_args(argv)

    requested_root = args.rocm or args.artifact_root
    if requested_root is None and os.environ.get("ROCM_PATH"):
        requested_root = Path(os.environ["ROCM_PATH"])
    rocm_root = resolve_rocm_root(requested_root)
    environment = runtime_environment(rocm_root)
    paths = {"/dev/kfd": Path("/dev/kfd"), "/dev/dri": Path("/dev/dri")}
    devices = {name: path.exists() for name, path in paths.items()}
    tools = {
        name: command_version(name, rocm_root=rocm_root, environment=environment)
        for name in ("rocminfo", "hipconfig", "amd-smi")
    }
    kernel_readiness = collect_readiness()
    report = {
        "schema_version": 1,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "devices": devices,
        "kernel_readiness": kernel_readiness,
        "tools": tools,
        "artifact_root": str(args.artifact_root.resolve()) if args.artifact_root else None,
        "rocm_root": str(rocm_root) if rocm_root else None,
        "feature_contract": feature_contract(),
        "status": (
            "fail"
            if kernel_readiness["status"] == "fail"
            else "GPU-test-pending"
            if not devices["/dev/kfd"]
            else "requires-rocminfo"
        ),
        "policy": {
            "gfx_override_allowed": False,
            "runtime_support_claim": "Do not claim MI50 runtime support until hardware tests pass.",
        },
    }
    json.dump(report, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 1 if report["status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
