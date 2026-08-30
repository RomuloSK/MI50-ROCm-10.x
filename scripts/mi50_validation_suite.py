#!/usr/bin/env python3
"""Run the ordered MI50 validation suite and emit one machine-readable report.

Every GPU-dependent smoke keeps its own exit semantics: 77 means
``GPU-test-pending`` and 78 means an optional capability is explicitly
unsupported.  The suite never upgrades a pre-hardware run to runtime support.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

# Keep the order aligned with docs/HARDWARE_VALIDATION.md.  Python readiness
# gates run before native smoke binaries, and the optional INT8 path is kept
# beside the other BLAS checks rather than being folded into the release gate.
STEPS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("kernel-readiness", "python", ("scripts/mi50_kernel_readiness.py",)),
    ("hardware-gate", "python", ("scripts/mi50_hardware_gate.py",)),
    ("runtime-validation", "python", ("scripts/mi50_runtime_validation.py",)),
    ("hip-runtime", "bash", ("scripts/run_hip_runtime_smoke.sh",)),
    ("hip-graph", "bash", ("scripts/run_mi50_graph_smoke.sh",)),
    ("hiprtc", "bash", ("scripts/run_mi50_hiprtc_smoke.sh",)),
    ("device-matrix", "bash", ("scripts/run_mi50_device_matrix_smoke.sh",)),
    ("hipBLAS", "bash", ("scripts/run_mi50_hipblas_smoke.sh",)),
    ("rocBLAS", "bash", ("scripts/run_mi50_rocblas_smoke.sh",)),
    ("rocBLAS-INT8", "bash", ("scripts/run_mi50_int8_smoke.sh",)),
    ("INT8-dot4", "bash", ("scripts/run_mi50_int8_dot4_smoke.sh",)),
    ("library-abi", "bash", ("scripts/run_mi50_library_abi_smoke.sh",)),
    ("MIOpen", "bash", ("scripts/run_mi50_miopen_smoke.sh",)),
    ("rocFFT-rocRAND", "bash", ("scripts/run_mi50_fft_rand_smoke.sh",)),
    ("rocSPARSE-rocSOLVER", "bash", ("scripts/run_mi50_sparse_solver_smoke.sh",)),
    ("rocPRIM-rocThrust", "bash", ("scripts/run_mi50_prim_thrust_smoke.sh",)),
    ("memory", "bash", ("scripts/run_mi50_memory_smoke.sh",)),
    ("RCCL", "bash", ("scripts/run_mi50_rccl_smoke.sh",)),
)


def run_step(command: list[str], *, environment: dict[str, str], timeout: int) -> dict[str, Any]:
    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        return {
            "status": "fail",
            "returncode": None,
            "stdout": "",
            "stderr": str(exc),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "status": "fail",
            "returncode": None,
            "stdout": str(exc.stdout or "")[-20000:],
            "stderr": f"timeout after {timeout}s\n{str(exc.stderr or '')[-20000:]}",
        }
    status = {0: "pass", 77: "GPU-test-pending", 78: "unsupported-on-gfx906"}.get(
        result.returncode, "fail"
    )
    # Readiness Python gates intentionally return 0 for a non-required
    # pre-hardware run, so their JSON status—not just their process exit code—
    # is authoritative. Native smoke wrappers use exit 77/78 and are covered
    # by the mapping above.
    try:
        payload = json.loads(result.stdout)
    except (json.JSONDecodeError, TypeError):
        payload = None
    if isinstance(payload, dict) and payload.get("status") in {
        "pass",
        "GPU-test-pending",
        "unsupported-on-gfx906",
        "fail",
    }:
        status = str(payload["status"])
    return {
        "status": status,
        "returncode": result.returncode,
        "stdout": result.stdout[-20000:],
        "stderr": result.stderr[-20000:],
    }


def run_suite(*, rocm_path: str | None, require_gpu: bool, timeout: int) -> dict[str, Any]:
    environment = dict(os.environ)
    if rocm_path:
        environment["ROCM_PATH"] = str(Path(rocm_path).expanduser().resolve())
    reports: list[dict[str, Any]] = []
    for name, kind, relative_command in STEPS:
        executable = [sys.executable, *relative_command] if kind == "python" else ["bash", *relative_command]
        path = ROOT / relative_command[0]
        if not path.is_file():
            reports.append(
                {
                    "name": name,
                    "command": executable,
                    "status": "fail",
                    "returncode": None,
                    "stdout": "",
                    "stderr": f"missing suite step: {path}",
                }
            )
            continue
        report = run_step(executable, environment=environment, timeout=timeout)
        report["name"] = name
        report["command"] = executable
        reports.append(report)

    statuses = [str(report["status"]) for report in reports]
    if "fail" in statuses:
        overall = "fail"
    elif require_gpu and "GPU-test-pending" in statuses:
        overall = "fail"
    elif "GPU-test-pending" in statuses:
        overall = "GPU-test-pending"
    elif "unsupported-on-gfx906" in statuses:
        overall = "partial"
    else:
        overall = "pass"
    return {
        "schema_version": 1,
        "target": "gfx906",
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "rocm_path": environment.get("ROCM_PATH"),
        "require_gpu": require_gpu,
        "runtime_claim": (
            "native MI50 execution evidence only; no certification"
            if overall == "pass"
            else "pre-hardware or partial evidence; do not claim MI50 runtime support"
        ),
        "status": overall,
        "steps": reports,
        "summary": {
            "pass": statuses.count("pass"),
            "pending": statuses.count("GPU-test-pending"),
            "unsupported": statuses.count("unsupported-on-gfx906"),
            "fail": statuses.count("fail"),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rocm", help="ROCm prefix passed to native smoke steps")
    parser.add_argument("--output", type=Path, help="write the report JSON to this path")
    parser.add_argument("--require-gpu", action="store_true", help="turn pending steps into a failure")
    parser.add_argument("--timeout", type=int, default=3600, help="per-step timeout in seconds")
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    report = run_suite(rocm_path=args.rocm, require_gpu=args.require_gpu, timeout=args.timeout)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    else:
        sys.stdout.write(payload)
    return 0 if report["status"] != "fail" else 1


if __name__ == "__main__":
    raise SystemExit(main())
