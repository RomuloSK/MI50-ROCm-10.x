#!/usr/bin/env python3
"""Load and minimally exercise a ROCr shared library without a GPU.

This test intentionally does not use an ISA override. On a host without
``/dev/kfd`` a non-success ``hsa_init`` result is expected; the important
pre-hardware gates are that the ELF loads and exports the HSA entry points.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
from pathlib import Path


def smoke(library: Path, expect_device: bool | None = None) -> dict:
    errors: list[str] = []
    loaded = False
    hsa_init_status: int | None = None
    try:
        lib = ctypes.CDLL(str(library), mode=os.RTLD_NOW | os.RTLD_LOCAL)
        loaded = True
        hsa_init = lib.hsa_init
        hsa_init.restype = ctypes.c_int
        hsa_init_status = int(hsa_init())
        if hsa_init_status == 0:
            hsa_shut_down = lib.hsa_shut_down
            hsa_shut_down.restype = ctypes.c_int
            shutdown_status = int(hsa_shut_down())
            if shutdown_status != 0:
                errors.append(f"hsa_shut_down returned {shutdown_status}")
    except OSError as exc:
        errors.append(f"shared-library load failed: {exc}")
    except AttributeError as exc:
        errors.append(f"required HSA symbol is missing: {exc}")

    device_present = Path("/dev/kfd").exists()
    if expect_device is None:
        expect_device = device_present
    if loaded and expect_device and hsa_init_status != 0:
        errors.append(f"hsa_init returned {hsa_init_status} with /dev/kfd present")

    return {
        "schema_version": 1,
        "library": str(library.resolve()),
        "loaded": loaded,
        "device_present": device_present,
        "hsa_init_status": hsa_init_status,
        "errors": errors,
        "status": "pass" if loaded and not errors else "fail",
        "runtime_claim": "host-load-only; GPU execution remains pending-hardware",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)

    library = args.library.resolve()
    if not library.is_file():
        parser.error(f"library does not exist: {library}")
    report = smoke(library)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(payload, encoding="utf-8")
    else:
        sys.stdout.write(payload)
    return 1 if args.strict and report["status"] != "pass" else 0


if __name__ == "__main__":
    raise SystemExit(main())
