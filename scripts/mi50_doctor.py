#!/usr/bin/env python3
"""Report host/device prerequisites without making unsupported claims."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from mi50_policy import feature_contract

try:  # Works both as ``python -m scripts...`` and as a file entry point.
    from .mi50_kernel_readiness import collect_readiness
except ImportError:  # pragma: no cover - exercised by the shell entry point.
    from mi50_kernel_readiness import collect_readiness


def command_version(command: str) -> str | None:
    executable = shutil.which(command)
    if not executable:
        return None
    try:
        result = subprocess.run(
            [executable, "--version"], capture_output=True, text=True, timeout=5, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return "present (version query failed)"
    line = (result.stdout or result.stderr).strip().splitlines()
    return line[0] if line else "present"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path)
    args = parser.parse_args(argv)

    paths = {"/dev/kfd": Path("/dev/kfd"), "/dev/dri": Path("/dev/dri")}
    devices = {name: path.exists() for name, path in paths.items()}
    tools = {name: command_version(name) for name in ("rocminfo", "hipconfig", "amd-smi")}
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
