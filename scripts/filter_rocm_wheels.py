#!/usr/bin/env python3
"""Remove unsupported MI50 payloads from generated ROCm wheels.

TheRock's host ``rocm-sdk-libraries`` wheel can contain a generic
hipSPARSELt shared library even when no gfx906 device implementation is
enabled.  The stable MI50 distribution uses rocSPARSE instead.  This helper
rewrites only wheel payload members, updates ``RECORD`` hashes, and leaves the
wheel metadata/dependencies intact.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import os
from pathlib import Path
import tempfile
import zipfile


EXCLUDED_COMPONENTS = (
    "rocprofiler-compute",
    "hipsparselt",
    "rocwmma",
    "hiptensor",
    "composable-kernel",
)


def _is_excluded(name: str) -> bool:
    normalized = name.lower().replace("_", "-")
    for component in EXCLUDED_COMPONENTS:
        component = component.lower()
        if (
            f"/{component}/" in f"/{normalized}/"
            or f"/{component}." in f"/{normalized}"
            or f"/lib{component}." in f"/{normalized}"
            or normalized.endswith(f"/{component}")
        ):
            return True
    return False


def _record_line(name: str, data: bytes, record_name: str) -> str:
    if name == record_name:
        return f"{name},,"
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
    return f"{name},sha256={digest},{len(data)}"


def filter_wheel(path: Path) -> dict:
    """Rewrite *path* and return a report describing removed members."""

    path = path.resolve()
    if path.suffix != ".whl":
        raise ValueError(f"not a wheel: {path}")
    with zipfile.ZipFile(path, "r") as source:
        infos = source.infolist()
        record_candidates = [info.filename for info in infos if info.filename.endswith(".dist-info/RECORD")]
        if len(record_candidates) != 1:
            raise ValueError(f"wheel must contain exactly one RECORD: {path}")
        record_name = record_candidates[0]
        kept: list[tuple[zipfile.ZipInfo, bytes]] = []
        removed: list[str] = []
        for info in infos:
            if info.filename != record_name and _is_excluded(info.filename):
                removed.append(info.filename)
                continue
            kept.append((info, source.read(info)))

    if not removed:
        return {"wheel": path.name, "removed": [], "status": "unchanged"}

    record_rows = [
        _record_line(info.filename, data, record_name)
        for info, data in kept
    ]
    record_data = ("\n".join(sorted(record_rows)) + "\n").encode("utf-8")
    kept = [
        (info, record_data if info.filename == record_name else data)
        for info, data in kept
    ]

    with tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        with zipfile.ZipFile(temporary_path, "w") as destination:
            for info, data in kept:
                destination.writestr(info, data)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return {"wheel": path.name, "removed": sorted(removed), "status": "filtered"}


def filter_directory(directory: Path) -> list[dict]:
    return [filter_wheel(path) for path in sorted(directory.glob("*.whl"))]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    directory = args.package_dir.resolve()
    if not directory.is_dir():
        parser.error(f"package directory does not exist: {directory}")
    import json

    print(json.dumps({"schema_version": 1, "wheels": filter_directory(directory)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
