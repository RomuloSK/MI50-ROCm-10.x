#!/usr/bin/env python3
"""Install a validated MI50 ROCm archive without modifying the kernel driver.

The generated archive contains the user-space ROCm stack under a top-level
``rocm/`` directory.  This installer validates its target-specific payload,
extracts into a same-filesystem staging directory, and atomically publishes it
under the requested prefix.  It deliberately leaves the inbox Linux
``amdgpu``/KFD driver and any existing ROCm installation untouched.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path, PurePosixPath
import posixpath
import shutil
import sys
import tarfile
import tempfile
from datetime import datetime, timezone

try:  # Works both as ``python -m scripts...`` and as a file entry point.
    from .validate_elf_dependencies import validate as validate_elf_dependencies
except ImportError:  # pragma: no cover - exercised by the shell entry point.
    from validate_elf_dependencies import validate as validate_elf_dependencies


TARGET = "gfx906"
SCHEMA_VERSION = 1
REQUIRED_MARKERS = {
    "hipcc": lambda name: name == "rocm/bin/hipcc",
    "llvm_llc": lambda name: name == "rocm/lib/llvm/bin/llc",
    "hip_headers": lambda name: name.startswith("rocm/include/hip/"),
    "rocr_runtime": lambda name: name.startswith("rocm/lib/libhsa-runtime64.so"),
    "llvm_ocml": lambda name: name == "rocm/lib/llvm/amdgcn/bitcode/ocml.bc",
    "rocblas_gfx906": lambda name: (
        name.startswith("rocm/lib/rocblas/library/") and TARGET in name.lower()
    ),
    "miopen_gfx906": lambda name: (
        name.startswith("rocm/share/miopen/db/") and TARGET in name.lower()
    ),
}


class ArchiveError(ValueError):
    """Raised when an archive is not a safe, target-complete ROCm package."""


def _safe_member_name(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ArchiveError(f"unsafe archive member path: {name!r}")
    if not (name == "rocm" or name.startswith("rocm/")):
        raise ArchiveError(f"archive member is outside the rocm/ root: {name!r}")
    return path


def _validate_link(member: tarfile.TarInfo) -> None:
    if not (member.issym() or member.islnk()):
        return
    target = PurePosixPath(member.linkname)
    if target.is_absolute():
        raise ArchiveError(f"absolute link target in archive: {member.name!r} -> {member.linkname!r}")
    # Relative links such as ``../libfoo.so`` are normal in a ROCm tree. They
    # are safe only when lexical normalization keeps the result below rocm/.
    parent = PurePosixPath(member.name).parent.as_posix()
    resolved = posixpath.normpath(posixpath.join(parent, member.linkname))
    if resolved != "rocm" and not resolved.startswith("rocm/"):
        raise ArchiveError(f"link escapes rocm/ in archive: {member.name!r} -> {member.linkname!r}")
    if member.islnk():
        # Tar hard-link names are commonly archive-root relative, but accept
        # a member-relative spelling when it resolves inside the same root.
        direct = posixpath.normpath(member.linkname)
        if (direct != "rocm" and not direct.startswith("rocm/")) and (
            resolved != "rocm" and not resolved.startswith("rocm/")
        ):
            raise ArchiveError(f"hard-link target is not under rocm/: {member.linkname!r}")


def inspect_archive(archive: Path) -> dict[str, object]:
    """Validate archive layout and return target-specific inventory metadata."""

    archive = archive.expanduser().resolve()
    if not archive.is_file():
        raise ArchiveError(f"archive does not exist: {archive}")

    try:
        with tarfile.open(archive, mode="r:gz") as handle:
            members = handle.getmembers()
    except (OSError, tarfile.TarError) as error:
        raise ArchiveError(f"cannot read gzip archive {archive}: {error}") from error

    if not members:
        raise ArchiveError("archive is empty")
    for member in members:
        _safe_member_name(member.name.rstrip("/") or "rocm")
        _validate_link(member)

    names = {member.name.rstrip("/") for member in members}
    checks: dict[str, object] = {}
    missing: list[str] = []
    for label, predicate in REQUIRED_MARKERS.items():
        matches = sorted(name for name in names if predicate(name))
        checks[label] = {"count": len(matches), "examples": matches[:3]}
        if not matches:
            missing.append(label)
    if missing:
        raise ArchiveError("archive is missing required gfx906 payload: " + ", ".join(missing))

    return {
        "schema_version": SCHEMA_VERSION,
        "archive": str(archive),
        "target": TARGET,
        "member_count": len(members),
        "checks": checks,
        "runtime_claim": "artifact-only; GPU execution remains pending-hardware",
    }


def _assert_prefix(prefix: Path) -> Path:
    if not prefix.is_absolute():
        raise ValueError(f"installation prefix must be absolute: {prefix}")
    raw_prefix = Path(os.path.expanduser(str(prefix)))
    if raw_prefix.is_symlink():
        raise ValueError(f"refusing to replace a symlink prefix: {raw_prefix}")
    prefix = raw_prefix.resolve()
    if prefix in {Path("/"), Path("/usr"), Path("/usr/local"), Path("/opt")}:
        raise ValueError(f"refusing to use a broad system prefix: {prefix}")
    return prefix


def _write_environment(prefix: Path) -> None:
    content = """#!/usr/bin/env bash
# MI50 ROCm 10.x user-space environment. Source this file; do not execute it.
if [[ -n \"${HSA_OVERRIDE_GFX_VERSION:-}\" || -n \"${ROCR_OVERRIDE_GFX_VERSION:-}\" ]]; then
  echo 'ISA override variables are forbidden for native gfx906 support' >&2
  return 6 2>/dev/null || exit 6
fi
MI50_ROCM_ROOT=\"$(CDPATH= cd -- \"$(dirname -- \"${BASH_SOURCE[0]}\")/rocm\" && pwd)\"
export ROCM_PATH=\"${MI50_ROCM_ROOT}\"
export ROCM_HOME=\"${MI50_ROCM_ROOT}\"
export HIP_PATH=\"${MI50_ROCM_ROOT}\"
MI50_ROCM_PATH=\"${MI50_ROCM_ROOT}/bin:${MI50_ROCM_ROOT}/lib/llvm/bin\"
if [[ -n \"${PATH:-}\" ]]; then
  IFS=: read -r -a MI50_PATH_ENTRIES <<< \"${PATH}\"
  for MI50_PATH_ENTRY in \"${MI50_PATH_ENTRIES[@]}\"; do
    if [[ -n \"${MI50_PATH_ENTRY}\" ]]; then
      MI50_ROCM_PATH=\"${MI50_ROCM_PATH}:${MI50_PATH_ENTRY}\"
    fi
  done
fi
export PATH=\"${MI50_ROCM_PATH}\"
MI50_ROCM_LIB_PATH=\"${MI50_ROCM_ROOT}/lib\"
for MI50_EXTRA_LIB_PATH in \\
  \"${MI50_ROCM_ROOT}/lib/rocm_sysdeps/lib\" \\
  \"${MI50_ROCM_ROOT}/lib/llvm/lib\"; do
  if [[ -d \"${MI50_EXTRA_LIB_PATH}\" ]]; then
    MI50_ROCM_LIB_PATH=\"${MI50_ROCM_LIB_PATH}:${MI50_EXTRA_LIB_PATH}\"
  fi
done
MI50_ROCM_LD_LIBRARY_PATH=\"${MI50_ROCM_LIB_PATH}\"
if [[ -n \"${LD_LIBRARY_PATH:-}\" ]]; then
  IFS=: read -r -a MI50_LD_PATH_ENTRIES <<< \"${LD_LIBRARY_PATH}\"
  for MI50_LD_PATH_ENTRY in \"${MI50_LD_PATH_ENTRIES[@]}\"; do
    if [[ -n \"${MI50_LD_PATH_ENTRY}\" ]]; then
      MI50_ROCM_LD_LIBRARY_PATH=\"${MI50_ROCM_LD_LIBRARY_PATH}:${MI50_LD_PATH_ENTRY}\"
    fi
  done
fi
export LD_LIBRARY_PATH=\"${MI50_ROCM_LD_LIBRARY_PATH}\"
unset MI50_EXTRA_LIB_PATH MI50_ROCM_LIB_PATH MI50_ROCM_PATH MI50_PATH_ENTRIES MI50_PATH_ENTRY
unset MI50_ROCM_LD_LIBRARY_PATH MI50_LD_PATH_ENTRIES MI50_LD_PATH_ENTRY
unset MI50_ROCM_ROOT
"""
    environment = prefix / "mi50-env.sh"
    temporary = environment.with_suffix(environment.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    temporary.chmod(0o755)
    os.replace(temporary, environment)


def install_archive(archive: Path, prefix: Path, *, force: bool = False) -> dict[str, object]:
    """Install *archive* under *prefix* and return an install manifest."""

    inventory = inspect_archive(archive)
    prefix = _assert_prefix(prefix)
    if prefix.exists() and not force:
        raise FileExistsError(f"installation prefix already exists (use --force to preserve it): {prefix}")

    prefix.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{prefix.name}.staging-", dir=prefix.parent))
    backup: Path | None = None
    published = False
    try:
        with tarfile.open(Path(inventory["archive"]), mode="r:gz") as handle:
            # Python 3.12's data filter rejects device nodes and link escapes in
            # addition to the explicit member checks above.
            handle.extractall(staging, filter="data")
        installed_root = staging / "rocm"
        if not installed_root.is_dir():
            raise ArchiveError("archive did not materialize a rocm/ directory")
        hipcc = installed_root / "bin" / "hipcc"
        if not hipcc.is_file() or hipcc.stat().st_size == 0:
            raise ArchiveError("extracted hipcc is missing or empty")

        # Marker checks prove that the expected files are present, but they do
        # not catch an unusable RUNPATH in a bundled executable.  Audit the
        # fully extracted tree before publishing it so an installation cannot
        # succeed with hidden ``not found`` ELF dependencies.
        elf_audit = validate_elf_dependencies(installed_root)
        if elf_audit["status"] != "pass":
            details = list(elf_audit.get("missing", []))
            details.extend(
                str(item.get("file", "unknown"))
                for item in elf_audit.get("unresolved", [])
            )
            suffix = ": " + ", ".join(details) if details else ""
            raise ArchiveError("extracted archive has unresolved ELF dependencies" + suffix)

        if prefix.exists():
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            backup = prefix.with_name(f"{prefix.name}.previous-{stamp}")
            if backup.exists():
                raise FileExistsError(f"refusing to overwrite prior backup: {backup}")
            os.replace(prefix, backup)
        os.replace(staging, prefix)
        published = True
        _write_environment(prefix)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "package": "rocm-mi50-gfx906-linux",
            "target": TARGET,
            "prefix": str(prefix),
            "environment": str(prefix / "mi50-env.sh"),
            "archive": inventory["archive"],
            "archive_checks": inventory["checks"],
            "elf_dependency_audit": {
                "status": elf_audit["status"],
                "elf_checked": elf_audit["elf_checked"],
                "checks": elf_audit["checks"],
                "missing": elf_audit["missing"],
                "unresolved": elf_audit["unresolved"],
                "command_errors": elf_audit["command_errors"],
            },
            "runtime_claim": inventory["runtime_claim"],
            "status": "pass",
        }
        (prefix / "mi50-install.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return manifest
    except Exception:
        if published:
            shutil.rmtree(prefix, ignore_errors=True)
        elif staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        if backup is not None and backup.exists() and not prefix.exists():
            os.replace(backup, prefix)
        raise
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--force", action="store_true", help="preserve an existing prefix as a timestamped backup")
    parser.add_argument("--dry-run", action="store_true", help="validate and report without extracting")
    args = parser.parse_args(argv)
    try:
        inventory = inspect_archive(args.archive)
        prefix = _assert_prefix(args.prefix)
        if args.dry_run:
            result = {**inventory, "prefix": str(prefix), "action": "validate-only"}
        else:
            result = install_archive(args.archive, prefix, force=args.force)
    except (ArchiveError, FileExistsError, OSError, ValueError) as error:
        print(f"install failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
