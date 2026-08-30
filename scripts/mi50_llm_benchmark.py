#!/usr/bin/env python3
"""Run and record an MI50 llama.cpp benchmark without making ISA claims.

The harness is intentionally separate from the build.  It executes
``llama-bench`` only after ``/dev/kfd`` and native ``gfx906``/wave64 evidence
are present, captures diagnostic output, and optionally compares parsed
throughput with a JSON baseline.  Without hardware it emits an explicit
``GPU-test-pending`` report (or fails when ``--require-gpu`` is requested).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

try:  # Support both module and direct-script execution.
    from .mi50_policy import validate_environment
    from .rocminfo_parser import parse_rocminfo
    from .mi50_runtime_paths import runtime_environment as scoped_runtime_environment
except ImportError:  # pragma: no cover
    from mi50_policy import validate_environment
    from rocminfo_parser import parse_rocminfo
    from mi50_runtime_paths import runtime_environment as scoped_runtime_environment


THROUGHPUT_PATTERNS = {
    "prompt_tokens_per_second": re.compile(
        r"(?:prompt|pp)[^\n]*?([0-9]+(?:\.[0-9]+)?)\s*t/s", re.IGNORECASE
    ),
    "decode_tokens_per_second": re.compile(
        r"(?:eval|decode|tg)[^\n]*?([0-9]+(?:\.[0-9]+)?)\s*t/s", re.IGNORECASE
    ),
}


def runtime_environment(rocm_path: str | None = None) -> dict[str, str]:
    """Return a child environment scoped to one ROCm installation."""

    return scoped_runtime_environment(rocm_path)


def command_path(
    command: str,
    *,
    environment: dict[str, str] | None = None,
    strict_rocm: bool = False,
) -> str | None:
    """Resolve a command or explicit executable path."""

    if os.path.sep in command or (os.path.altsep and os.path.altsep in command):
        path = Path(command).expanduser()
        return str(path) if path.is_file() and os.access(path, os.X_OK) else None
    if strict_rocm and environment is not None and environment.get("ROCM_PATH"):
        candidate = Path(environment["ROCM_PATH"]) / "bin" / command
        return str(candidate) if candidate.is_file() and os.access(candidate, os.X_OK) else None
    search_path = environment.get("PATH", "") if environment is not None else None
    return shutil.which(command, path=search_path)


def run_command(
    command: Sequence[str],
    *,
    timeout: int = 120,
    environment: dict[str, str] | None = None,
    strict_rocm: bool = False,
) -> dict[str, Any]:
    """Run one diagnostic/benchmark command with bounded captured output."""

    executable = command_path(command[0], environment=environment, strict_rocm=strict_rocm)
    if executable is None:
        return {"command": list(command), "status": "missing"}
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
            "command": list(command),
            "status": "timeout" if isinstance(exc, subprocess.TimeoutExpired) else "fail",
            "stdout": str(getattr(exc, "stdout", "") or "")[-20000:],
            "stderr": str(getattr(exc, "stderr", "") or exc)[-20000:],
        }
    return {
        "command": list(command),
        "status": "pass" if result.returncode == 0 else "fail",
        "returncode": result.returncode,
        "stdout": (result.stdout or "")[-20000:],
        "stderr": (result.stderr or "")[-20000:],
    }


def parse_throughput(text: str) -> dict[str, float]:
    """Extract stable prompt/decode throughput fields from llama-bench text."""

    metrics: dict[str, float] = {}
    values_by_name: dict[str, list[float]] = {name: [] for name in THROUGHPUT_PATTERNS}
    # Parse line-by-line so the word "eval" in llama.cpp's combined
    # "prompt eval time" label cannot be mistaken for decode throughput.
    for line in text.splitlines():
        lowered = line.lower()
        if "prompt" in lowered or re.search(r"(?:^|[\s,])pp(?:[\s,]|$)", lowered):
            match = THROUGHPUT_PATTERNS["prompt_tokens_per_second"].search(line)
            if match:
                values_by_name["prompt_tokens_per_second"].append(float(match.group(1)))
            continue
        if re.search(r"(?:^|[\s,])(?:eval|decode|tg)(?:[\s,]|$)", lowered):
            match = THROUGHPUT_PATTERNS["decode_tokens_per_second"].search(line)
            if match:
                values_by_name["decode_tokens_per_second"].append(float(match.group(1)))
    for name, values in values_by_name.items():
        if values:
            # llama-bench can print one row per test shape.  Preserve the
            # fastest observed row while retaining the complete raw output.
            metrics[name] = max(values)
    return metrics


def compare_baseline(
    current: dict[str, float], baseline: dict[str, float], *, tolerance: float = 0.05
) -> list[dict[str, float | str]]:
    """Return regressions exceeding the allowed relative tolerance."""

    regressions: list[dict[str, float | str]] = []
    for metric, reference in sorted(baseline.items()):
        if metric not in current or reference <= 0:
            continue
        observed = current[metric]
        minimum = reference * (1.0 - tolerance)
        if observed < minimum:
            regressions.append(
                {
                    "metric": metric,
                    "baseline": reference,
                    "observed": observed,
                    "minimum": minimum,
                    "relative_change": observed / reference - 1.0,
                }
            )
    return regressions


def load_baseline(path: Path | None) -> dict[str, float]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload.get("metrics", payload)
    if not isinstance(metrics, dict):
        raise ValueError("benchmark baseline must contain an object named 'metrics'")
    result: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float)) and value > 0:
            result[str(key)] = float(value)
    return result


def _snapshot(environment: dict[str, str] | None = None) -> dict[str, dict[str, Any]]:
    return {
        "rocminfo": run_command(["rocminfo"], environment=environment, strict_rocm=True),
        "amd_smi_list": run_command(["amd-smi", "list"], environment=environment, strict_rocm=True),
        "amd_smi_metric": run_command(["amd-smi", "metric"], environment=environment, strict_rocm=True),
    }


def run_benchmark(
    *,
    model: Path,
    llama_bench: str = "llama-bench",
    gpu_layers: int = 999,
    repetitions: int = 3,
    extra_args: Sequence[str] = (),
    baseline: Path | None = None,
    tolerance: float = 0.05,
    require_gpu: bool = False,
    dry_run: bool = False,
    rocm_path: str | None = None,
) -> dict[str, Any]:
    """Run the benchmark gate and return a JSON-serializable report."""

    report: dict[str, Any] = {
        "schema_version": 1,
        "target": "gfx906",
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "model": str(model.resolve()),
        "runtime_claim": "hardware benchmark evidence only; not AMD certification",
        "commands": {},
        "metrics": {},
        "errors": [],
    }

    environment = runtime_environment(rocm_path)
    report["rocm_path"] = environment.get("ROCM_PATH")
    violations = validate_environment(environment)
    if violations:
        report["status"] = "fail"
        report["errors"] = violations
        return report
    if not model.is_file():
        report["status"] = "fail"
        report["errors"] = [f"model does not exist: {model}"]
        return report

    command = [llama_bench, "-m", str(model), "-ngl", str(gpu_layers), "-r", str(repetitions)]
    command.extend(extra_args)
    report["benchmark_command"] = command
    if dry_run:
        report["status"] = "GPU-test-pending"
        report["errors"] = []
        return report

    if not Path("/dev/kfd").exists():
        report["status"] = "fail" if require_gpu else "GPU-test-pending"
        if require_gpu:
            report["errors"] = ["/dev/kfd is unavailable"]
        return report

    before = _snapshot(environment)
    report["commands"]["before"] = before
    parsed = parse_rocminfo(str(before["rocminfo"].get("stdout", "")))
    report["rocminfo_contract"] = parsed
    if not parsed["has_native_gfx906"]:
        report["errors"].append("rocminfo did not report native gfx906")
    if parsed["wavefront_sizes"] and not parsed["wavefront64_observed"]:
        report["errors"].append("rocminfo reported no wavefront-size 64 GPU agent")
    if report["errors"]:
        report["status"] = "fail"
        return report

    result = run_command(command, timeout=3600, environment=environment)
    report["commands"]["llama_bench"] = result
    output = "\n".join((str(result.get("stdout", "")), str(result.get("stderr", ""))))
    report["raw_benchmark_output"] = output[-40000:]
    report["metrics"] = parse_throughput(output)

    after = _snapshot(environment)
    report["commands"]["after"] = after
    if result["status"] != "pass":
        report["errors"].append("llama-bench failed")

    try:
        reference = load_baseline(baseline)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        report["errors"].append(f"invalid baseline: {exc}")
        reference = {}
    report["baseline"] = reference
    report["regressions"] = compare_baseline(report["metrics"], reference, tolerance=tolerance)
    if report["regressions"]:
        report["errors"].append(f"throughput regression exceeds {tolerance:.1%} tolerance")
    report["status"] = "pass" if not report["errors"] else "fail"
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--rocm", help="ROCm prefix used for llama-bench and diagnostics")
    parser.add_argument("--llama-bench", default="llama-bench")
    parser.add_argument("--gpu-layers", type=int, default=999)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--tolerance", type=float, default=0.05)
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="additional llama-bench arguments after '--'",
    )
    args = parser.parse_args(argv)
    if not 0 <= args.tolerance < 1:
        parser.error("--tolerance must be between 0 and 1")
    extra = list(args.extra_args)
    if extra[:1] == ["--"]:
        extra = extra[1:]
    report = run_benchmark(
        model=args.model,
        llama_bench=args.llama_bench,
        gpu_layers=args.gpu_layers,
        repetitions=args.repetitions,
        extra_args=extra,
        baseline=args.baseline,
        tolerance=args.tolerance,
        require_gpu=args.require_gpu,
        dry_run=args.dry_run,
        rocm_path=args.rocm,
    )
    json.dump(report, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0 if report["status"] != "fail" else 1


if __name__ == "__main__":
    raise SystemExit(main())
