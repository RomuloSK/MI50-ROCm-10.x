#!/usr/bin/env python3
"""Run the first MI50 hardware gate when Linux GPU access is available.

With no card (or no `/dev/kfd`) this intentionally emits a successful
``GPU-test-pending`` report so CI can run before hardware acquisition.  It
never treats an HSA override as a substitute for native gfx906 discovery.
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
    from .mi50_runtime_paths import runtime_environment as scoped_runtime_environment
except ImportError:  # pragma: no cover - exercised by the shell entry point.
    from rocminfo_parser import parse_rocminfo
    from mi50_kernel_readiness import collect_readiness
    from mi50_runtime_paths import runtime_environment as scoped_runtime_environment


def runtime_environment(rocm_path: str | None = None) -> dict[str, str]:
    return scoped_runtime_environment(rocm_path)


def run_command(
    command: list[str], *, timeout: int = 30, environment: dict[str, str] | None = None
) -> dict[str, object]:
    if environment is not None and environment.get("ROCM_PATH"):
        candidate = Path(environment["ROCM_PATH"]) / "bin" / command[0]
        executable = str(candidate) if candidate.is_file() and os.access(candidate, os.X_OK) else None
    else:
        search_path = environment.get("PATH", "") if environment is not None else None
        executable = shutil.which(command[0], path=search_path)
    if not executable:
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
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "command": command,
            "status": "timeout" if isinstance(exc, subprocess.TimeoutExpired) else "fail",
            "stdout": str(getattr(exc, "stdout", "") or ""),
            "stderr": str(getattr(exc, "stderr", "") or exc),
        }
    return {
        "command": command,
        "status": "pass" if result.returncode == 0 else "fail",
        "returncode": result.returncode,
        "stdout": result.stdout[-12000:],
        "stderr": result.stderr[-12000:],
    }


def run_gate(*, require_gpu: bool = False, rocm_path: str | None = None) -> dict[str, object]:
    errors: list[str] = []
    override_keys = [key for key in ("HSA_OVERRIDE_GFX_VERSION", "ROCR_OVERRIDE_GFX_VERSION") if os.environ.get(key)]
    if override_keys:
        errors.append("ISA override is set: " + ", ".join(override_keys))
    kfd = Path("/dev/kfd").exists()
    dri = Path("/dev/dri").exists()
    kernel_readiness = collect_readiness()
    report: dict[str, object] = {
        "schema_version": 1,
        "target": "gfx906",
        "platform": {"system": platform.system(), "release": platform.release(), "machine": platform.machine()},
        "devices": {"/dev/kfd": kfd, "/dev/dri": dri},
        "kernel_readiness": kernel_readiness,
        "runtime_claim": "hardware validation only; this gate does not certify performance",
    }
    environment = runtime_environment(rocm_path)
    report["rocm_path"] = environment.get("ROCM_PATH")
    if errors:
        report["status"] = "fail"
        report["errors"] = errors
        return report
    if kernel_readiness["status"] == "fail":
        report["status"] = "fail"
        report["errors"] = list(kernel_readiness["errors"])
        return report
    if not kfd:
        report["status"] = "GPU-test-pending"
        report["errors"] = []
        if require_gpu:
            report["errors"] = ["/dev/kfd is unavailable"]
            report["status"] = "fail"
        return report

    commands = [
        run_command([name, *args], environment=environment)
        for name, args in (("rocminfo", ()), ("hipconfig", ("--full",)), ("amd-smi", ("list",)))
    ]
    report["commands"] = commands
    rocminfo_text = "\n".join(str(item.get("stdout", "")) for item in commands if item["command"][0] == "rocminfo")
    parsed_rocminfo = parse_rocminfo(rocminfo_text)
    report["rocminfo_contract"] = parsed_rocminfo
    if not parsed_rocminfo["has_native_gfx906"]:
        errors.append("rocminfo did not report native gfx906")
    if parsed_rocminfo["wavefront_sizes"] and not parsed_rocminfo["wavefront64_observed"]:
        errors.append("rocminfo reported no wavefront-size 64 GPU agent")
    for item in commands:
        if item["status"] == "missing":
            errors.append(f"missing diagnostic command: {item['command'][0]}")
        elif item["status"] != "pass":
            errors.append(f"diagnostic command failed: {' '.join(item['command'])}")
    report["errors"] = errors
    report["status"] = "pass" if not errors else "fail"
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--rocm", help="ROCm prefix used for diagnostics")
    args = parser.parse_args(argv)
    report = run_gate(require_gpu=args.require_gpu, rocm_path=args.rocm)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    else:
        sys.stdout.write(payload)
    return 0 if report["status"] != "fail" else 1


if __name__ == "__main__":
    raise SystemExit(main())
