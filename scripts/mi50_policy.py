"""Canonical MI50/gfx906 feature policy.

This module is deliberately independent of ROCm or a GPU.  It turns the
repository support matrix into a small, machine-readable contract that build
and packaging entry points can enforce before a card is present.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MATRIX_PATH = ROOT / "support-matrix.json"


# Upstream ROCm tools legitimately *read* the ISA-override variable (for
# example `rocm_agent_enumerator` prints a warning when a user has set it), so
# a plain substring search cannot tell "the artifact honours an override" from
# "the artifact installs an override".  The build gate must only fail on the
# latter: a statement that puts a concrete value into the environment.
ISA_OVERRIDE_VARIABLES = ("HSA_OVERRIDE_GFX_VERSION", "ROCR_OVERRIDE_GFX_VERSION")

_ISA_NAMES = "|".join(ISA_OVERRIDE_VARIABLES)

# `export HSA_OVERRIDE_GFX_VERSION=9.0.8`, `-DHSA_OVERRIDE_GFX_VERSION=10.3.0`,
# `HSA_OVERRIDE_GFX_VERSION="9.0.8"` in a wrapper, or a CMake `set(... )`.
# Assignment form, including a closing subscript/quote before the equals sign:
#   export HSA_OVERRIDE_GFX_VERSION=9.0.8
#   env["HSA_OVERRIDE_GFX_VERSION"] = "9.0.8"
#   -DROCR_OVERRIDE_GFX_VERSION=10.3.0
_ISA_ASSIGNMENT = re.compile(
    rf"(?:^|(?<=[^A-Za-z0-9_])|(?<=-D))(?:{_ISA_NAMES})"
    rf"\s*[\"']?\s*\]?\s*=(?!=)\s*[\"']?[A-Za-z0-9.$]",
    re.IGNORECASE,
)
# Function/set form with the value following the name:
#   putenv("HSA_OVERRIDE_GFX_VERSION", "9.0.8")
#   set(ENV{HSA_OVERRIDE_GFX_VERSION} "9.0.8")
_ISA_SET_CALL = re.compile(
    rf"(?:{_ISA_NAMES})[\"']?\s*[,}}]?\s*[\"']\s*[0-9]",
)
# JSON/dict environment entries: `"HSA_OVERRIDE_GFX_VERSION": "9.0.8"`.
_ISA_MAPPING = re.compile(
    rf"[\"'](?:{_ISA_NAMES})[\"']\s*:\s*[\"'](?!\s*(?:unset|none|off|false|disabled?)[\"'])[^\"']*[0-9][^\"']*[\"']",
    re.IGNORECASE,
)


def isa_override_lines(text: str) -> list[tuple[int, str]]:
    """Return ``(line_number, text)`` for lines that enable an ISA override.

    Mentions, comparisons, help strings, ``unset`` and empty assignments are
    ignored: they cannot change the runtime ISA.  Only statements that put a
    concrete value into the variable are reported.
    """

    findings: list[tuple[int, str]] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not any(variable in line.upper() for variable in ISA_OVERRIDE_VARIABLES):
            continue
        if (
            _ISA_ASSIGNMENT.search(line)
            or _ISA_SET_CALL.search(line)
            or _ISA_MAPPING.search(line)
        ):
            stripped = line.strip()
            if stripped:
                findings.append((line_number, stripped[:300]))
    return findings


def isa_override_findings(text: str) -> list[str]:
    """Return the enabling lines in *text* without their line numbers."""

    return [line for _, line in isa_override_lines(text)]


def load_matrix(path: Path = MATRIX_PATH) -> dict[str, Any]:
    """Load and minimally validate the checked-in support matrix."""

    matrix = json.loads(path.read_text(encoding="utf-8"))
    target = matrix.get("target", {})
    if target.get("llvm_target") != "gfx906":
        raise ValueError("support matrix target must remain gfx906")
    if target.get("architecture") != "Vega20 / GCN5.1":
        raise ValueError("support matrix architecture must remain Vega20 / GCN5.1")
    return matrix


def component_map(matrix: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    matrix = matrix or load_matrix()
    return {component["name"]: component for component in matrix["components"]}


def feature_contract(matrix: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return the normalized contract used by build and diagnostic tools."""

    matrix = matrix or load_matrix()
    target = matrix["target"]
    precision = target["precision_policy"]
    components = component_map(matrix)
    return {
        "schema_version": 1,
        "project_version": matrix["project_version"],
        "llvm_target": target["llvm_target"],
        "architecture": target["architecture"],
        "default_features": list(target["default_features"]),
        "optimization_profile": target.get("optimization_profile", {}),
        "precision": dict(precision),
        "hardware_features": {
            "wavefront_size": 64,
            "matrix_cores": False,
            "native_bf16": False,
            "native_fp8": False,
            "sramecc": True,
            "xnack": False,
        },
        "components": {
            name: {
                "status": value["status"],
                "test": value["test"],
            }
            for name, value in sorted(components.items())
        },
        "runtime_claim": "artifact-only; GPU execution remains pending-hardware",
    }


def validate_environment(environ: dict[str, str] | None = None) -> list[str]:
    """Return policy violations in an environment used for a build/run."""

    environ = dict(os.environ if environ is None else environ)
    violations: list[str] = []
    for key in ("HSA_OVERRIDE_GFX_VERSION", "ROCR_OVERRIDE_GFX_VERSION"):
        if environ.get(key):
            violations.append(f"{key} is forbidden for a native gfx906 build")
    # Every target selector must be a single native gfx906 value.  A mixed
    # build can appear to work on the host while silently omitting the MI50
    # objects from a package, or dispatching a newer ISA at runtime.  Keep the
    # check here rather than duplicating subtly different checks in each
    # downstream build wrapper.
    target_keys = (
        "THEROCK_AMDGPU_FAMILIES",
        "THEROCK_DIST_AMDGPU_FAMILIES",
        "THEROCK_TEST_AMDGPU_FAMILIES",
        "PYTORCH_ROCM_ARCH",
        "GPU_TARGETS",
        "AMDGPU_TARGETS",
        "CMAKE_HIP_ARCHITECTURES",
    )
    for key in target_keys:
        value = environ.get(key)
        if not value:
            continue
        # CMake and a few ROCm wrappers use ';' while PyTorch commonly uses
        # ','; accepting either delimiter is safe only when the resulting set
        # still contains exactly one target.
        targets = [item.strip() for item in re.split(r"[;,]", value) if item.strip()]
        if targets != ["gfx906"]:
            violations.append(f"{key}={value!r}; expected exactly 'gfx906'")
    return violations


def require_component(name: str, *, allow_pending: bool = True) -> dict[str, Any]:
    """Return a component or raise a useful error for unsupported paths."""

    component = component_map().get(name)
    if component is None:
        raise KeyError(f"unknown MI50 component: {name}")
    if component["status"] == "unsupported-on-gfx906":
        raise RuntimeError(f"{name} is unsupported on gfx906: {component['notes']}")
    if not allow_pending and component["test"] == "GPU-test-pending":
        raise RuntimeError(f"{name} still requires MI50 hardware validation")
    return component
