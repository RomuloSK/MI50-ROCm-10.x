#!/usr/bin/env python3
"""Verify patch files against the SHA-256 digests in sources.lock.json."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(repository_root: Path, lock_path: Path) -> dict[str, Any]:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    observed: list[dict[str, Any]] = []
    missing: list[str] = []
    mismatched: list[dict[str, str | None]] = []
    for entry in lock.get("patches", []):
        relative = str(entry.get("file", ""))
        path = repository_root / relative
        actual = sha256(path) if path.is_file() else None
        record = {
            "file": relative,
            "expected_sha256": entry.get("sha256"),
            "observed_sha256": actual,
        }
        observed.append(record)
        if actual is None:
            missing.append(relative)
        elif actual.lower() != str(entry.get("sha256", "")).lower():
            mismatched.append(record)
    return {
        "schema_version": 1,
        "repository_root": str(repository_root.resolve()),
        "lock_file": str(lock_path.resolve()),
        "patches": observed,
        "missing": missing,
        "mismatched": mismatched,
        "status": "pass" if not missing and not mismatched else "fail",
        "policy": "every patch applied by the builder must match sources.lock.json",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--lock-file", type=Path)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)
    repository_root = args.repository_root.resolve()
    lock_path = (args.lock_file or repository_root / "sources.lock.json").resolve()
    if not repository_root.is_dir():
        parser.error(f"repository root does not exist: {repository_root}")
    if not lock_path.is_file():
        parser.error(f"lock file does not exist: {lock_path}")
    report = verify(repository_root, lock_path)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(payload, encoding="utf-8")
    else:
        sys.stdout.write(payload)
    return 1 if args.strict and report["status"] != "pass" else 0


if __name__ == "__main__":
    raise SystemExit(main())
