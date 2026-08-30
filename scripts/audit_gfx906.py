#!/usr/bin/env python3
"""Audit a ROCm/TheRock source tree for gfx906 enablement hazards.

This tool is intentionally static: it never loads a GPU library and never
claims that a target works at runtime. It emits deterministic JSON suitable for
CI artifacts and returns non-zero only for explicit policy violations.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

try:  # Support both module and direct-script execution.
    from .mi50_policy import isa_override_lines
except ImportError:  # pragma: no cover
    from mi50_policy import isa_override_lines


SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "node_modules",
    "build",
    "dist",
    "out",
    "artifacts",
}

PATTERNS = {
    "gfx906": re.compile(r"gfx906", re.IGNORECASE),
    "gfx_override": re.compile(r"HSA_OVERRIDE_GFX_VERSION", re.IGNORECASE),
    "bf16": re.compile(r"\bBF16\b|bfloat16", re.IGNORECASE),
    "fp8": re.compile(r"\bFP8\b|float8", re.IGNORECASE),
    "aotriton": re.compile(r"AOTRITON|aotriton", re.IGNORECASE),
    "flash_attention": re.compile(r"flash.?attention|FLASH_ATTN", re.IGNORECASE),
    "rocwmma": re.compile(r"rocWMMA|ROCWMMA", re.IGNORECASE),
    "hipblaslt": re.compile(r"hipBLASLt|HIPBLASLT", re.IGNORECASE),
    "hipsparselt": re.compile(r"hipSPARSELt|HIPSPARSELT", re.IGNORECASE),
    "composable_kernel": re.compile(r"composable.?kernel|COMPOSABLE_KERNEL", re.IGNORECASE),
    "hiptensor": re.compile(r"hipTensor|HIPTENSOR", re.IGNORECASE),
    "rocprofiler_compute": re.compile(r"rocprofiler.?compute|ROCPROFILER_COMPUTE", re.IGNORECASE),
}


def iter_files(root: Path, excludes: set[str]):
    for current, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and d not in excludes)
        for name in sorted(files):
            path = Path(current) / name
            if path.is_symlink():
                continue
            yield path


def read_text(path: Path) -> str | None:
    try:
        data = path.read_bytes()
    except (OSError, PermissionError):
        return None
    # Avoid attempting to interpret object files and firmware as text.
    if b"\x00" in data:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("utf-8", errors="replace")


def audit(root: Path, excludes: set[str]) -> dict:
    counts: Counter[str] = Counter()
    matches: list[dict] = []
    scanned = 0
    unreadable = 0
    override_assignments: list[dict] = []

    for path in iter_files(root, excludes):
        text = read_text(path)
        if text is None:
            unreadable += 1
            continue
        scanned += 1
        relative = path.relative_to(root).as_posix()
        for line_number, line in isa_override_lines(text):
            override_assignments.append({"file": relative, "line": line_number, "text": line})
        for line_number, line in enumerate(text.splitlines(), 1):
            for name, pattern in PATTERNS.items():
                if pattern.search(line):
                    counts[name] += 1
                    if len(matches) < 5000:
                        matches.append(
                            {
                                "file": relative,
                                "line": line_number,
                                "kind": name,
                                "text": line.strip()[:500],
                            }
                        )

    return {
        "schema_version": 1,
        "root": str(root.resolve()),
        "files_scanned": scanned,
        "files_skipped_or_unreadable": unreadable,
        "match_counts": dict(sorted(counts.items())),
        "matches_truncated": len(matches) >= 5000,
        "matches": matches,
        "policy": {
            "gfx906_seen": counts["gfx906"] > 0,
            # A source tree may legitimately *read* the override variable to
            # warn users (upstream does this in rocm_agent_enumerator).  Only an
            # assignment is a policy violation, because only then can a build
            # silently masquerade as a newer ISA.
            "gfx_override_seen": bool(override_assignments),
            "gfx_override_mentions": counts["gfx_override"],
            "gfx_override_assignments": override_assignments[:50],
            "runtime_claim": "static-only; GPU execution remains pending-hardware",
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="source tree to inspect")
    parser.add_argument("--json-out", type=Path, help="write the report to this path")
    parser.add_argument("--exclude", action="append", default=[], help="directory name to skip")
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "fail if gfx906 is absent or a build installs an ISA override; naming "
            "the variable to read it is allowed"
        ),
    )
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if not root.is_dir():
        parser.error(f"root does not exist or is not a directory: {root}")

    report = audit(root, set(args.exclude))
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(payload, encoding="utf-8")
    else:
        sys.stdout.write(payload)

    if args.strict and not report["policy"]["gfx906_seen"]:
        print("audit failure: no gfx906 references found", file=sys.stderr)
        return 2
    if args.strict and report["policy"]["gfx_override_seen"]:
        print(
            "audit failure: an ISA override value is assigned in the source tree",
            file=sys.stderr,
        )
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
