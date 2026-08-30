#!/usr/bin/env python3
"""Validate a gfx906 ROCm artifact directory without requiring a GPU."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

try:  # Support both module and direct-script execution.
    from .mi50_policy import isa_override_findings
except ImportError:  # pragma: no cover
    from mi50_policy import isa_override_findings


TEXT_SUFFIXES = {
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".txt",
    ".cmake",
    ".cfg",
    ".ini",
    ".md",
    ".sh",
    ".py",
}


def files(root: Path):
    for path in sorted(root.rglob("*")):
        if path.is_file() and not path.is_symlink():
            yield path


def read_bytes(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError:
        return b""


def validate(root: Path) -> dict:
    all_files = list(files(root))
    names = [p.name.lower() for p in all_files]
    text_blobs: list[tuple[Path, str]] = []
    for path in all_files:
        data = read_bytes(path)
        if path.suffix.lower() in TEXT_SUFFIXES or b"\x00" not in data:
            text_blobs.append((path, data.decode("utf-8", errors="replace")))
    joined = "\n".join(text for _, text in text_blobs)

    checks = {
        "gfx906_marker": bool(re.search(r"gfx906", joined, re.IGNORECASE))
        or any("gfx906" in name for name in names),
        "rocblas_device_data": any(
            ("tensilelibrary" in name and "gfx906" in name) or "vega20" in name
            for name in names
        )
        or bool(re.search(r"TensileLibrary.*gfx906|vega20", joined, re.IGNORECASE)),
        "miopen_device_data": any(
            "gfx906" in name and ("kdb" in name or "fdb" in name or "miopen" in name)
            for name in names
        )
        or bool(re.search(r"gfx906.*(?:kdb|fdb)|(?:kdb|fdb).*gfx906", joined, re.IGNORECASE)),
        # Upstream ROCm helpers such as `rocm_agent_enumerator` read the
        # override variable to warn users; naming it is not the same as the
        # artifact installing an override.  Only a real assignment makes this
        # build dishonest about its ISA.
        "no_isa_override_instruction": not any(
            isa_override_findings(text) for _, text in text_blobs
        ),
    }
    # Build metadata is optional for hand-created test fixtures, but when the
    # builder emits it, validate the claims rather than treating JSON as an
    # opaque marker. This prevents a mismatched or unverified source tree from
    # being packaged under a gfx906 label.
    for metadata_name, check_prefix in (
        ("build-provenance.json", "provenance"),
        ("source-lock-verification.json", "source_lock"),
        ("patch-lock-verification.json", "patch_lock"),
    ):
        metadata_path = root / metadata_name
        if not metadata_path.is_file():
            continue
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            checks[f"{check_prefix}_valid_json"] = False
            continue
        if metadata_name == "build-provenance.json":
            checks["provenance_target"] = metadata.get("target") == "gfx906"
            checks["provenance_runtime_claim"] = str(
                metadata.get("runtime_claim", "")
            ).startswith("artifact-only")
        else:
            checks[f"{check_prefix}_status"] = metadata.get("status") == "pass"
    policy_path = root / "mi50_features.json"
    if policy_path.is_file():
        try:
            policy = json.loads(policy_path.read_text(encoding="utf-8"))
            hardware = policy.get("hardware_features") or {}
            checks["feature_policy_target"] = policy.get("llvm_target") == "gfx906"
            checks["feature_policy_no_native_bf16"] = not hardware.get("native_bf16", True)
            checks["feature_policy_no_native_fp8"] = not hardware.get("native_fp8", True)
            checks["feature_policy_no_matrix_cores"] = not hardware.get("matrix_cores", True)
        except (OSError, json.JSONDecodeError):
            checks["feature_policy_valid_json"] = False
    missing = [name for name, passed in checks.items() if not passed]
    return {
        "schema_version": 1,
        "artifact_root": str(root.resolve()),
        "files_checked": len(all_files),
        "checks": checks,
        "missing": missing,
        "status": "pass" if not missing else "fail",
        "runtime_claim": "artifact-only; GPU execution remains pending-hardware",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--strict", action="store_true", help="return non-zero when a check fails")
    args = parser.parse_args(argv)
    root = args.artifact_root.resolve()
    if not root.is_dir():
        parser.error(f"artifact root does not exist or is not a directory: {root}")

    report = validate(root)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(payload, encoding="utf-8")
    else:
        sys.stdout.write(payload)
    return 1 if args.strict and report["status"] != "pass" else 0


if __name__ == "__main__":
    raise SystemExit(main())
