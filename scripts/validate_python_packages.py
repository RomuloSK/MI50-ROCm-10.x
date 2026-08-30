#!/usr/bin/env python3
"""Validate ROCm Python wheels produced for the MI50 gfx906 build."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
from email.parser import Parser
from pathlib import Path, PurePosixPath
import sys
import tarfile
import zipfile


EXPECTED_PACKAGES = {
    "rocm-sdk-core",
    "rocm-sdk-libraries",
    "rocm-sdk-device-gfx906",
    "rocm-sdk-devel",
}
EXCLUDED_COMPONENTS = (
    "rocprofiler-compute",
    "hipsparselt",
    "rocwmma",
    "hiptensor",
    "composable-kernel",
)


def _safe_member(name: str) -> bool:
    # Zip member names are POSIX paths even on Windows. Treat backslashes as
    # separators too so a malicious wheel cannot hide traversal behind a
    # platform-specific spelling.
    path = PurePosixPath(name.replace("\\", "/"))
    return not path.is_absolute() and ".." not in path.parts


def _digest(data: bytes) -> str:
    return base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()


def _wheel_report(path: Path, target: str, version: str) -> dict:
    checks: dict[str, bool] = {}
    errors: list[str] = []
    try:
        archive = zipfile.ZipFile(path, "r")
        infos = archive.infolist()
    except (OSError, zipfile.BadZipFile) as error:
        return {"file": path.name, "status": "fail", "errors": [str(error)]}

    names = [info.filename for info in infos]
    checks["safe_member_paths"] = all(_safe_member(name) for name in names)
    if not checks["safe_member_paths"]:
        errors.append("wheel contains an absolute or parent-traversal member")
    # Empty __init__.py files and executable link stubs are legitimate wheel
    # members.  Binary code objects, however, must never be empty because that
    # is the signature of a truncated artifact copy.
    binary_suffixes = (".so", ".a", ".bc", ".co", ".hsaco", ".o")
    checks["nonempty_binary_payloads"] = all(
        info.is_dir()
        or info.file_size > 0
        or not info.filename.lower().endswith(binary_suffixes)
        for info in infos
    )
    if not checks["nonempty_binary_payloads"]:
        errors.append("wheel contains an empty binary payload")

    metadata_names = [name for name in names if name.endswith(".dist-info/METADATA")]
    record_names = [name for name in names if name.endswith(".dist-info/RECORD")]
    checks["metadata_and_record"] = len(metadata_names) == 1 and len(record_names) == 1
    metadata = ""
    if checks["metadata_and_record"]:
        metadata = archive.read(metadata_names[0]).decode("utf-8", errors="replace")
        parsed = Parser().parsestr(metadata)
        package_name = parsed.get("Name", "").lower()
        package_version = parsed.get("Version", "")
        expected_prefix = path.name.split("-", 1)[0].replace("_", "-").lower()
        checks["metadata_name"] = package_name == expected_prefix
        checks["metadata_version"] = package_version == version
        if not checks["metadata_name"]:
            errors.append(f"metadata name {package_name!r} does not match wheel filename")
        if not checks["metadata_version"]:
            errors.append(f"metadata version {package_version!r} does not match {version!r}")

        record_rows = list(csv.reader(io.StringIO(archive.read(record_names[0]).decode("utf-8"))))
        record_map = {row[0]: row for row in record_rows if row}
        record_ok = True
        for info in infos:
            if info.is_dir() or info.filename == record_names[0]:
                continue
            row = record_map.get(info.filename)
            if row is None or len(row) < 3 or row[1] != f"sha256={_digest(archive.read(info))}" or row[2] != str(info.file_size):
                record_ok = False
                break
        checks["record_hashes"] = record_ok
        if not record_ok:
            errors.append("wheel RECORD does not match a payload member")
    else:
        errors.append("wheel must contain exactly one METADATA and RECORD")

    checks["no_isa_override_text"] = "HSA_OVERRIDE_GFX_VERSION" not in metadata
    if not checks["no_isa_override_text"]:
        errors.append("wheel metadata mentions HSA_OVERRIDE_GFX_VERSION")
    lower_names = [name.lower().replace("_", "-") for name in names]
    unsupported = [
        name
        for name in lower_names
        if any(component in name for component in EXCLUDED_COMPONENTS)
    ]
    checks["no_excluded_payloads"] = not unsupported
    if unsupported:
        errors.append(f"excluded payloads present: {unsupported[:5]}")
    if f"device-{target}" in path.name.replace("_", "-").lower():
        checks["device_target_marker"] = any(target in name.lower() for name in names)
        if not checks["device_target_marker"]:
            errors.append(f"device wheel has no {target} member")
    return {
        "file": path.name,
        "members": len(names),
        "checks": checks,
        "errors": errors,
        "status": "pass" if not errors and all(checks.values()) else "fail",
    }


def validate(package_dir: Path, target: str = "gfx906", version: str = "10.0.0+mi50.5") -> dict:
    package_dir = package_dir.resolve()
    wheel_reports = [_wheel_report(path, target, version) for path in sorted(package_dir.glob("*.whl"))]
    names = set()
    for report in wheel_reports:
        if report["file"]:
            names.add(report["file"].split("-", 1)[0].replace("_", "-").lower())
    package_checks = {
        "expected_wheels_present": EXPECTED_PACKAGES.issubset(names),
        "rocm_sdist_present": any(path.name.startswith("rocm-") and path.name.endswith(".tar.gz") for path in package_dir.glob("*.tar.gz")),
    }
    errors = []
    if not package_checks["expected_wheels_present"]:
        errors.append(f"missing expected wheels: {sorted(EXPECTED_PACKAGES - names)}")
    if not package_checks["rocm_sdist_present"]:
        errors.append("rocm sdist is missing")
    for path in sorted(package_dir.glob("rocm-*.tar.gz")):
        try:
            with tarfile.open(path, "r:gz") as archive:
                if not all(_safe_member(member.name) for member in archive.getmembers()):
                    errors.append(f"unsafe sdist member in {path.name}")
        except (OSError, tarfile.TarError) as error:
            errors.append(f"invalid sdist {path.name}: {error}")
    errors.extend(error for report in wheel_reports for error in report.get("errors", []))
    checks = {**package_checks, "all_wheels_valid": all(report["status"] == "pass" for report in wheel_reports)}
    return {
        "schema_version": 1,
        "package_dir": str(package_dir),
        "target": target,
        "version": version,
        "checks": checks,
        "wheels": wheel_reports,
        "errors": errors,
        "status": "pass" if not errors and all(checks.values()) else "fail",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--target", default="gfx906")
    parser.add_argument("--version", default="10.0.0+mi50.5")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)
    report = validate(args.package_dir, args.target, args.version)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(payload, encoding="utf-8")
    else:
        sys.stdout.write(payload)
    return 1 if args.strict and report["status"] != "pass" else 0


if __name__ == "__main__":
    raise SystemExit(main())
