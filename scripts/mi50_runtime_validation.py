#!/usr/bin/env python3
"""Execute the first native MI50 runtime validation tier.

The command is safe to run before the cards arrive: without ``/dev/kfd`` it
returns ``GPU-test-pending``.  On a real Linux host it checks that the
diagnostic tools discover a native ``gfx906`` agent and records their output;
it never treats an HSA ISA override as support.  Longer library, collective,
and inference tests are intentionally separate release-gate commands so a
bring-up run cannot accidentally become a destructive stress test.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

try:  # Works both as ``python -m scripts...`` and as a file entry point.
    from .rocminfo_parser import parse_rocminfo
    from .mi50_kernel_readiness import collect_readiness
except ImportError:  # pragma: no cover - exercised by the shell entry point.
    from rocminfo_parser import parse_rocminfo
    from mi50_kernel_readiness import collect_readiness


def runtime_environment(rocm_path: str | None = None) -> dict[str, str]:
    environment = dict(os.environ)
    selected = rocm_path or environment.get("ROCM_PATH")
    if not selected:
        return environment
    root = Path(selected).expanduser().resolve()
    nested = root / "rocm"
    if nested.is_dir():
        root = nested
    environment["ROCM_PATH"] = str(root)
    environment["PATH"] = os.pathsep.join(
        [str(root / "bin"), str(root / "lib" / "llvm" / "bin"), environment.get("PATH", "")]
    )
    environment["LD_LIBRARY_PATH"] = os.pathsep.join(
        [
            str(root / "lib"),
            str(root / "lib" / "rocm_sysdeps" / "lib"),
            str(root / "lib" / "llvm" / "lib"),
            environment.get("LD_LIBRARY_PATH", ""),
        ]
    )
    return environment


def run_command(
    command: list[str], *, timeout: int = 60, environment: dict[str, str] | None = None
) -> dict[str, object]:
    search_path = environment.get("PATH") if environment is not None else None
    executable = shutil.which(command[0], path=search_path)
    if executable is None:
        return {"command": command, "status": "missing"}
    try:
        result = subprocess.run(
            [executable, *command[1:]],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=environment,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "command": command,
            "status": "timeout",
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
        }
    return {
        "command": command,
        "status": "pass" if result.returncode == 0 else "fail",
        "returncode": result.returncode,
        "stdout": result.stdout[-20000:],
        "stderr": result.stderr[-20000:],
    }


def validate_runtime(*, require_gpu: bool = False, rocm_path: str | None = None) -> dict[str, object]:
    errors: list[str] = []
    overrides = [
        key
        for key in ("HSA_OVERRIDE_GFX_VERSION", "ROCR_OVERRIDE_GFX_VERSION")
        if os.environ.get(key)
    ]
    report: dict[str, object] = {
        "schema_version": 1,
        "target": "gfx906",
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "runtime_claim": "native runtime validation; performance remains a separate release gate",
        "commands": [],
    }
    environment = runtime_environment(rocm_path)
    report["rocm_path"] = environment.get("ROCM_PATH")
    if overrides:
        report["status"] = "fail"
        report["errors"] = ["ISA override is set: " + ", ".join(overrides)]
        return report

    kfd = Path("/dev/kfd").exists()
    kernel_readiness = collect_readiness()
    report["devices"] = {"/dev/kfd": kfd, "/dev/dri": Path("/dev/dri").exists()}
    report["kernel_readiness"] = kernel_readiness
    if kernel_readiness["status"] == "fail":
        report["status"] = "fail"
        report["errors"] = list(kernel_readiness["errors"])
        return report
    if not kfd:
        report["status"] = "GPU-test-pending" if not require_gpu else "fail"
        report["errors"] = [] if not require_gpu else ["/dev/kfd is unavailable"]
        return report

    commands = [
        run_command(["rocminfo"], environment=environment),
        run_command(["hipconfig", "--full"], environment=environment),
        run_command(["amd-smi", "list"], environment=environment),
    ]
    report["commands"] = commands
    rocminfo = "\n".join(
        str(item.get("stdout", "")) for item in commands if item["command"][0] == "rocminfo"
    )
    parsed_rocminfo = parse_rocminfo(rocminfo)
    report["rocminfo_contract"] = parsed_rocminfo
    if not parsed_rocminfo["has_native_gfx906"]:
        errors.append("rocminfo did not report a native gfx906 agent")
    if parsed_rocminfo["wavefront_sizes"] and not parsed_rocminfo["wavefront64_observed"]:
        errors.append("rocminfo reported no wavefront-size 64 GPU agent")
    for item in commands:
        if item["status"] == "missing":
            errors.append(f"missing required command: {item['command'][0]}")
        elif item["status"] != "pass":
            errors.append(f"command failed: {' '.join(item['command'])}")
    report["errors"] = errors
    report["status"] = "pass" if not errors else "fail"
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--rocm", help="ROCm prefix used for diagnostics")
    args = parser.parse_args(argv)
    report = validate_runtime(require_gpu=args.require_gpu, rocm_path=args.rocm)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    else:
        sys.stdout.write(payload)
    return 1 if report["status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
