#!/usr/bin/env python3
"""Parse the small, stable subset of ``rocminfo`` needed by the MI50 gate.

The parser is deliberately text-only and tolerant of extra agents, localization
noise, and formatting changes.  It never infers a GPU from an HSA override;
callers still reject override environment variables before invoking it.
"""

from __future__ import annotations

import re
from typing import Any


_NAME_RE = re.compile(r"^\s*Name:\s*(?P<name>gfx[0-9a-z-]+)\s*$", re.IGNORECASE)
_WAVEFRONT_RE = re.compile(r"^\s*Wavefront Size:\s*(?P<size>\d+)\s*$", re.IGNORECASE)
_ISA_RE = re.compile(r"amdgcn-[^\s]*gfx906", re.IGNORECASE)


def parse_rocminfo(text: str) -> dict[str, Any]:
    """Return native GPU names, wavefront sizes, and gfx906 ISA evidence."""

    names: list[str] = []
    wavefront_sizes: list[int] = []
    isa_lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        name_match = _NAME_RE.match(raw_line)
        if name_match:
            names.append(name_match.group("name").lower())
        wave_match = _WAVEFRONT_RE.match(raw_line)
        if wave_match:
            wavefront_sizes.append(int(wave_match.group("size")))
        if _ISA_RE.search(line):
            isa_lines.append(line)

    native_gfx906_agents = sorted({name for name in names if name == "gfx906"})
    return {
        "native_agent_names": sorted(set(names)),
        "native_gfx906_agent_count": len(native_gfx906_agents),
        "wavefront_sizes": sorted(set(wavefront_sizes)),
        "gfx906_isa_lines": isa_lines,
        "has_native_gfx906": bool(native_gfx906_agents),
        # A non-empty rocminfo normally reports this field for every GPU.  The
        # value is advisory when a vendor build omits it, so callers only fail
        # when a reported value contradicts MI50's wave64 contract.
        "wavefront64_observed": 64 in wavefront_sizes,
    }

