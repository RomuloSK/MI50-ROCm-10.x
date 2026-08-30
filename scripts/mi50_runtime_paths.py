#!/usr/bin/env python3
"""Shared ROCm-prefix and child-environment handling.

The compatibility package is installable either as a direct ``rocm/`` tree
or beneath an installer prefix.  Keeping normalization here prevents one
diagnostic from accidentally selecting a different ROCm installation than
another, and avoids empty ``PATH``/``LD_LIBRARY_PATH`` entries (an empty Linux
loader entry means the current working directory).
"""

from __future__ import annotations

import os
from pathlib import Path
from collections.abc import Mapping


def normalize_rocm_root(selected: str | os.PathLike[str]) -> Path:
    """Return the direct ROCm tree for an install prefix or tree."""

    root = Path(selected).expanduser().resolve()
    nested = root / "rocm"
    # A direct tree may contain an unrelated directory called ``rocm``.  Only
    # unwrap the installer layout when the direct tree has no bin directory.
    if nested.is_dir() and not (root / "bin").is_dir():
        root = nested
    return root


def _join_paths(prefixes: list[Path], inherited: str | None) -> str:
    """Prepend existing prefixes and drop empty/duplicate entries."""

    entries: list[str] = []
    for value in [*(str(path) for path in prefixes), *((inherited or "").split(os.pathsep))]:
        if value and value not in entries:
            entries.append(value)
    return os.pathsep.join(entries)


def runtime_environment(
    rocm_path: str | os.PathLike[str] | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return an environment scoped to one ROCm installation."""

    environment = dict(os.environ if environ is None else environ)
    selected = rocm_path or environment.get("ROCM_PATH")
    if not selected:
        return environment
    root = normalize_rocm_root(selected)
    environment["ROCM_PATH"] = str(root)
    environment["ROCM_HOME"] = str(root)
    environment["HIP_PATH"] = str(root)
    environment["PATH"] = _join_paths(
        [root / "bin", root / "lib" / "llvm" / "bin"], environment.get("PATH")
    )
    environment["LD_LIBRARY_PATH"] = _join_paths(
        [
            root / "lib",
            root / "lib" / "rocm_sysdeps" / "lib",
            root / "lib" / "llvm" / "lib",
        ],
        environment.get("LD_LIBRARY_PATH"),
    )
    return environment
