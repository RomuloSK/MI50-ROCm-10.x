#!/usr/bin/env python3
"""Audit dynamic ELF dependencies in a flattened ROCm distribution.

The normal artifact gates check that required files exist, but an installed
binary can still be unusable when its shared-library RUNPATH points outside the
package.  This host-only audit runs ``ldd`` with the distribution's library
directories and reports unresolved ``not found`` entries.  It never executes a
GPU kernel and therefore cannot certify MI50 runtime support.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any


ELF_MAGIC = b"\x7fELF"
_BENIGN_LDD_MESSAGES = ("not a dynamic executable", "statically linked")


def _elf_files(root: Path) -> list[Path]:
    result: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        try:
            with path.open("rb") as stream:
                if stream.read(4) == ELF_MAGIC:
                    result.append(path)
        except OSError:
            continue
    return result


def _library_path(root: Path) -> str:
    directories = (
        root / "lib",
        root / "lib" / "rocm_sysdeps" / "lib",
        root / "lib" / "llvm" / "lib",
    )
    entries = [str(path) for path in directories if path.is_dir()]
    return os.pathsep.join(entries)


def validate(root: Path, *, timeout: int = 30) -> dict[str, Any]:
    root = root.resolve()
    elf_files = _elf_files(root) if root.is_dir() else []
    ldd_path = shutil.which("ldd")
    unresolved: list[dict[str, Any]] = []
    command_errors: list[dict[str, Any]] = []

    if elf_files and ldd_path:
        environment = dict(os.environ)
        environment["LD_LIBRARY_PATH"] = _library_path(root)
        for path in elf_files:
            try:
                result = subprocess.run(
                    [ldd_path, str(path)],
                    capture_output=True,
                    text=True,
                    env=environment,
                    timeout=timeout,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                command_errors.append(
                    {
                        "file": path.relative_to(root).as_posix(),
                        "error": str(exc),
                    }
                )
                continue
            output = "\n".join((result.stdout, result.stderr))
            missing = [line.strip() for line in output.splitlines() if "not found" in line]
            if missing:
                unresolved.append(
                    {
                        "file": path.relative_to(root).as_posix(),
                        "missing": missing,
                    }
                )
            elif result.returncode != 0 and not any(
                message in output.lower() for message in _BENIGN_LDD_MESSAGES
            ):
                command_errors.append(
                    {
                        "file": path.relative_to(root).as_posix(),
                        "error": f"ldd exited with status {result.returncode}",
                        "output": output[-4000:],
                    }
                )

    checks = {
        "distribution_present": root.is_dir(),
        "ldd_available": bool(ldd_path) or not elf_files,
        "unresolved_dependencies": not unresolved,
        "dependency_commands": not command_errors,
    }
    missing = [name for name, passed in checks.items() if not passed]
    return {
        "schema_version": 1,
        "root": str(root),
        "elf_checked": len(elf_files),
        "checks": checks,
        "unresolved": unresolved,
        "command_errors": command_errors,
        "missing": missing,
        "status": "pass" if not missing else "fail",
        "runtime_claim": "host dynamic-link audit; GPU execution remains pending-hardware",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error("--timeout must be positive")

    report = validate(args.root, timeout=args.timeout)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    for item in report["unresolved"]:
        for missing in item["missing"]:
            print(f"ELF dependency check failed: {item['file']}: {missing}")
    return 1 if args.strict and report["status"] != "pass" else 0


if __name__ == "__main__":
    raise SystemExit(main())
