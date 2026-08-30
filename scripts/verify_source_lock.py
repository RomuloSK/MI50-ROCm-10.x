#!/usr/bin/env python3
"""Verify that a TheRock source tree matches the repository lock file."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


REPOSITORY_PATHS = {
    "TheRock": Path("."),
    "rocm-libraries": Path("rocm-libraries"),
    "rocm-systems": Path("rocm-systems"),
    "llvm-project": Path("compiler/amd-llvm"),
}


def git_revision(path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    value = result.stdout.strip()
    return value if result.returncode == 0 and value else None


def verify(source_root: Path, lock_path: Path) -> dict[str, Any]:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    expected = {
        item["name"]: item["commit"] for item in lock.get("repositories", [])
    }
    repositories: list[dict[str, Any]] = []
    missing: list[str] = []
    mismatched: list[dict[str, str | None]] = []
    for name, commit in expected.items():
        relative = REPOSITORY_PATHS.get(name)
        if relative is None:
            missing.append(f"no path mapping for locked repository {name}")
            continue
        path = source_root / relative
        observed = git_revision(path) if path.is_dir() else None
        entry = {
            "name": name,
            "path": str(path),
            "expected_commit": commit,
            "observed_commit": observed,
        }
        repositories.append(entry)
        if observed is None:
            missing.append(f"{name}: no git checkout at {path}")
        elif observed.lower() != commit.lower():
            mismatched.append(entry)
    return {
        "schema_version": 1,
        "source_root": str(source_root.resolve()),
        "lock_file": str(lock_path.resolve()),
        "repositories": repositories,
        "missing": missing,
        "mismatched": mismatched,
        "status": "pass" if not missing and not mismatched else "fail",
        "policy": "all TheRock component repositories must match sources.lock.json",
    }


def verify_repository(repository_path: Path, repository_name: str, lock_path: Path) -> dict[str, Any]:
    """Verify one checkout when a standalone subproject is being built."""

    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    expected = {
        item["name"]: item["commit"] for item in lock.get("repositories", [])
    }
    expected_commit = expected.get(repository_name)
    observed = git_revision(repository_path) if repository_path.is_dir() else None
    record = {
        "name": repository_name,
        "path": str(repository_path),
        "expected_commit": expected_commit,
        "observed_commit": observed,
    }
    missing: list[str] = []
    mismatched: list[dict[str, str | None]] = []
    if expected_commit is None:
        missing.append(f"{repository_name}: not present in lock file")
    elif observed is None:
        missing.append(f"{repository_name}: no git checkout at {repository_path}")
    elif observed.lower() != expected_commit.lower():
        mismatched.append(record)
    return {
        "schema_version": 1,
        "source_root": str(repository_path.resolve()),
        "lock_file": str(lock_path.resolve()),
        "repositories": [record],
        "missing": missing,
        "mismatched": mismatched,
        "status": "pass" if not missing and not mismatched else "fail",
        "policy": "standalone repository must match its entry in sources.lock.json",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--repository-name", help="Verify one locked repository instead of a full TheRock tree")
    parser.add_argument("--repository-path", type=Path, help="Path to the standalone repository checkout")
    parser.add_argument(
        "--lock-file", type=Path, default=Path(__file__).resolve().parents[1] / "sources.lock.json"
    )
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)
    source_root = args.source_root.resolve()
    lock_path = args.lock_file.resolve()
    if not source_root.is_dir():
        parser.error(f"source root does not exist: {source_root}")
    if not lock_path.is_file():
        parser.error(f"lock file does not exist: {lock_path}")
    if bool(args.repository_name) != bool(args.repository_path):
        parser.error("--repository-name and --repository-path must be supplied together")
    if args.repository_name:
        report = verify_repository(args.repository_path.resolve(), args.repository_name, lock_path)
    else:
        report = verify(source_root, lock_path)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(payload, encoding="utf-8")
    else:
        sys.stdout.write(payload)
    return 1 if args.strict and report["status"] != "pass" else 0


if __name__ == "__main__":
    raise SystemExit(main())
