#!/usr/bin/env python3
"""Make TheRock artifact slices flatten-complete by repairing their manifests.

``fileset_tool.py artifact-flatten`` is manifest driven: it copies only the
prefixes listed in each slice's ``artifact_manifest.txt``.  In the gfx906 build
some slices end up with a payload on disk but an empty (0-byte) manifest, and
thousands of installed files -- the LLVM/AMDGPU compiler, device bitcode,
``libamd_comgr.so``, ``hipcc``, ``hipconfig`` and ``hipify`` -- are then
silently dropped from the distribution and from the Python wheels.  A ROCm
install that cannot compile HIP is not a deliverable, so repair the manifest
from the bytes that actually exist before flattening, and report anything that
is still uncovered so the build fails instead of shipping a hollow tree.

The repair is purely data driven: a prefix is the ancestor directory up to the
first ``stage``/``dist`` component (falling back to three path components),
which is the layout TheRock writes.  Existing manifest entries are preserved so
a correct slice is never narrowed.

An interrupted ``Populate artifact`` step can also leave a slice whose payload
files are themselves 0 bytes while the upstream ``stage`` trees are intact.
Repairing the manifest in that case would flatten empty files into the
distribution, so ``--heal`` deletes such slices and lets the build graph
regenerate them before the manifest is trusted again.

Truncation is confirmed by comparing each 0-byte payload file against the same
relative path in the build tree (``--build-root``): a legitimately empty upstream
file -- gdb self-tests, documentation fixtures, some hipify shell helpers -- also
has an empty counterpart, so it is never mistaken for corruption.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from pathlib import Path


SLICE_NAME = re.compile(r"^(.+)_(run|lib|dev|dbg|doc|test)_(.+)$")
TERMINALS = ("stage", "dist")


def payload_entries(slice_dir: Path) -> list[Path]:
    """Return files and symlinks that belong to the slice payload."""

    return [
        path
        for path in slice_dir.rglob("*")
        if path.name != "artifact_manifest.txt" and (path.is_file() or path.is_symlink())
    ]


def zero_byte_payload_files(slice_dir: Path) -> list[Path]:
    """Return regular payload files that were written with no content.

    Symlinks legitimately report a tiny size, and some documentation or test
    fixtures can be empty, so this is only used as a corruption *signal* for a
    slice whose manifest is also unusable.
    """

    return [
        path
        for path in payload_entries(slice_dir)
        if not path.is_symlink() and path.is_file() and path.stat().st_size == 0
    ]


def truncated_payload_files(
    slice_dir: Path, build_root: Path | None
) -> list[Path]:
    """Return 0-byte payload files whose build-tree counterpart has content.

    Only these are definite truncation: an empty file that is empty upstream too
    is a real installed artifact, not damage from an interrupted copy.
    """

    if build_root is None:
        return []
    build_root = build_root.resolve()
    truncated: list[Path] = []
    for path in zero_byte_payload_files(slice_dir):
        counterpart = build_root / path.relative_to(slice_dir)
        if counterpart.is_file() and counterpart.stat().st_size > 0:
            truncated.append(path)
    return truncated


def prefix_for(relpath: str) -> str:
    parts = relpath.split("/")
    for index, part in enumerate(parts):
        if part in TERMINALS and index >= 2:
            return "/".join(parts[: index + 1])
    return "/".join(parts[:3])


def derive_prefixes(slice_dir: Path, entries: list[Path]) -> list[str]:
    prefixes: set[str] = set()
    for path in entries:
        relpath = path.relative_to(slice_dir).as_posix()
        candidate = prefix_for(relpath)
        # Only real directories can serve as pattern-matcher base directories.
        if candidate and (slice_dir / candidate).is_dir():
            prefixes.add(candidate)
    return sorted(prefixes)


def read_manifest(manifest: Path) -> list[str]:
    if not manifest.is_file():
        return []
    return [
        line.strip()
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_manifest_atomic(manifest: Path, prefixes: list[str]) -> None:
    fd, temporary_name = tempfile.mkstemp(
        prefix=".artifact_manifest.", suffix=".tmp", dir=str(manifest.parent)
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            "".join(f"{prefix}\n" for prefix in prefixes), encoding="utf-8"
        )
        os.replace(temporary, manifest)
    finally:
        temporary.unlink(missing_ok=True)


def uncovered_payload(relpaths: set[str], prefixes: list[str]) -> list[str]:
    return sorted(
        relpath
        for relpath in relpaths
        if not any(relpath.startswith(f"{prefix}/") for prefix in prefixes)
    )


def repair(
    artifacts_dir: Path,
    *,
    apply: bool = True,
    active_projects: set[str] | None = None,
    heal: bool = False,
    build_root: Path | None = None,
) -> dict:
    """Repair every slice manifest under *artifacts_dir*."""

    artifacts_dir = artifacts_dir.resolve()
    if not artifacts_dir.is_dir():
        raise ValueError(f"artifact directory does not exist: {artifacts_dir}")

    repaired: list[dict] = []
    complete = 0
    uncovered_after = 0
    payload_total = 0
    healed: list[dict] = []

    for slice_dir in sorted(p for p in artifacts_dir.iterdir() if p.is_dir()):
        match = SLICE_NAME.match(slice_dir.name)
        if not match:
            continue
        if active_projects is not None and match.group(1) not in active_projects:
            continue
        manifest = slice_dir / "artifact_manifest.txt"
        entries = payload_entries(slice_dir)
        payload_total += len(entries)
        existing = read_manifest(manifest)

        if heal and entries:
            empties = truncated_payload_files(slice_dir, build_root)
            if not empties and build_root is None and not existing:
                # No build root to compare against: fall back to the weaker
                # signal of an unusable manifest plus bulk zero-byte payload.
                empties = zero_byte_payload_files(slice_dir)
            if empties:
                healed.append(
                    {
                        "slice": slice_dir.name,
                        "truncated_files": len(empties),
                        "payload_files": len(entries),
                    }
                )
                if apply:
                    import shutil

                    shutil.rmtree(slice_dir)
                continue

        relpaths = {path.relative_to(slice_dir).as_posix() for path in entries}
        wanted = sorted(set(existing) | set(derive_prefixes(slice_dir, entries)))
        missing = uncovered_payload(relpaths, wanted)
        uncovered_after += len(missing)

        if not missing and (existing or not entries):
            complete += 1
            continue

        repaired.append(
            {
                "slice": slice_dir.name,
                "previous_prefixes": existing,
                "repaired_prefixes": wanted,
                "payload_files": len(entries),
                "uncovered_files": len(missing),
            }
        )
        if apply and wanted:
            write_manifest_atomic(manifest, wanted)

    return {
        "schema_version": 1,
        "artifacts_dir": str(artifacts_dir),
        "status": "pass" if uncovered_after == 0 else "fail",
        "payload_files_seen": payload_total,
        "slices_complete": complete,
        "slices_repaired": repaired,
        "slices_healed": healed,
        "uncovered_payload_files": uncovered_after,
        "note": (
            "artifact-flatten only copies prefixes named in "
            "artifact_manifest.txt; an empty manifest silently drops the payload"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    parser.add_argument(
        "--subprojects-json",
        type=Path,
        help="artifact_subprojects.json used to ignore inactive slices",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="report uncovered payloads without rewriting manifests",
    )
    parser.add_argument(
        "--heal",
        action="store_true",
        help="delete slices whose payload is zero-byte truncation so the build "
        "graph regenerates them from the intact stage trees",
    )
    parser.add_argument(
        "--build-root",
        type=Path,
        help="build tree used to tell a truncated payload file apart from one "
        "that is legitimately empty upstream (enables precise --heal)",
    )
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)

    active_projects: set[str] | None = None
    if args.subprojects_json and args.subprojects_json.is_file():
        active_projects = set(
            json.loads(args.subprojects_json.read_text(encoding="utf-8"))
        )

    report = repair(
        args.artifacts_dir,
        apply=not args.check_only,
        active_projects=active_projects,
        heal=args.heal,
        build_root=args.build_root,
    )
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    for record in report["slices_repaired"]:
        print(
            f"repaired {record['slice']}: {record['payload_files']} payload files, "
            f"prefixes {record['repaired_prefixes']}",
            flush=True,
        )
    for record in report["slices_healed"]:
        print(
            f"healed {record['slice']}: removed {record['truncated_files']} truncated "
            f"payload files so the slice is regenerated",
            flush=True,
        )
    return 1 if args.strict and report["status"] != "pass" else 0


if __name__ == "__main__":
    raise SystemExit(main())
