#!/usr/bin/env python3
"""Inspect Linux amdgpu/KFD and Vega20 firmware readiness without changing state."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import sys


TARGET = "gfx906"
VEGA20_FIRMWARE = (
    "vega20_asd.bin",
    "vega20_ce.bin",
    "vega20_me.bin",
    "vega20_mec.bin",
    "vega20_mec2.bin",
    "vega20_pfp.bin",
    "vega20_rlc.bin",
    "vega20_sdma.bin",
    "vega20_sdma1.bin",
    "vega20_smc.bin",
    "vega20_uvd.bin",
    "vega20_vcn.bin",
)


def _read(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except (OSError, UnicodeError):
        return None


def _uevent(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    content = _read(path / "uevent")
    if content:
        for line in content.splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                values[key] = value
    return values


def _drm_devices(sysfs_root: Path) -> list[dict[str, object]]:
    drm_root = sysfs_root / "class" / "drm"
    devices: list[dict[str, object]] = []
    if not drm_root.is_dir():
        return devices
    for card in sorted(drm_root.glob("card[0-9]*")):
        device = card / "device"
        if not device.exists():
            continue
        values = _uevent(device)
        driver = values.get("DRIVER")
        if not driver and device.joinpath("driver").exists():
            try:
                driver = device.joinpath("driver").resolve().name
            except OSError:
                pass
        devices.append(
            {
                "card": card.name,
                "driver": driver,
                "pci_id": values.get("PCI_ID"),
                "pci_class": values.get("PCI_CLASS"),
                "pci_subsys_id": values.get("PCI_SUBSYS_ID"),
                "uevent": values,
            }
        )
    return devices


def _firmware_inventory(firmware_root: Path, *, amdgpu_present: bool) -> dict[str, object]:
    locations = (firmware_root / "amdgpu", firmware_root)
    found: dict[str, str] = {}
    missing: list[str] = []
    for name in VEGA20_FIRMWARE:
        match = next((location / name for location in locations if (location / name).is_file()), None)
        if match is None:
            missing.append(name)
        else:
            found[name] = str(match)
    return {
        "root": str(firmware_root),
        "expected_family": "Vega20/gfx906",
        "status": "pass" if amdgpu_present and not missing else (
            "missing" if amdgpu_present else "not-evaluable-without-amdgpu"
        ),
        "found": found,
        "missing": missing,
    }


def collect_readiness(
    *,
    sysfs_root: Path = Path("/sys"),
    firmware_root: Path = Path("/lib/firmware"),
    dev_root: Path = Path("/dev"),
) -> dict[str, object]:
    """Return a host-only readiness report; no command or device is modified."""

    sysfs_root = sysfs_root.resolve()
    firmware_root = firmware_root.resolve()
    dev_root = dev_root.resolve()
    drm_devices = _drm_devices(sysfs_root)
    amdgpu_devices = [device for device in drm_devices if device.get("driver") == "amdgpu"]
    amdgpu_module = sysfs_root / "module" / "amdgpu"
    kfd_module = sysfs_root / "module" / "kfd"
    kfd_device = (dev_root / "kfd").exists()
    dri_device = (dev_root / "dri").is_dir()
    errors: list[str] = []
    overrides = [
        key
        for key in ("HSA_OVERRIDE_GFX_VERSION", "ROCR_OVERRIDE_GFX_VERSION")
        if os.environ.get(key)
    ]
    if overrides:
        errors.append("ISA override is set: " + ", ".join(overrides))

    firmware = _firmware_inventory(firmware_root, amdgpu_present=bool(amdgpu_devices))
    if amdgpu_devices and not amdgpu_module.is_dir():
        errors.append("DRM reports amdgpu-bound devices but /sys/module/amdgpu is absent")
    if amdgpu_devices and not kfd_module.is_dir():
        errors.append("amdgpu devices are present but /sys/module/kfd is absent")
    if amdgpu_devices and not kfd_device:
        errors.append("amdgpu devices are present but /dev/kfd is unavailable")
    if amdgpu_devices and firmware["missing"]:
        errors.append("missing expected Vega20 firmware: " + ", ".join(firmware["missing"]))

    if errors:
        status = "fail"
    elif not amdgpu_devices or not kfd_device:
        status = "GPU-test-pending"
    else:
        status = "ready-for-rocr"
    return {
        "schema_version": 1,
        "target": TARGET,
        "platform": {"system": platform.system(), "release": platform.release()},
        "status": status,
        "errors": errors,
        "modules": {
            "amdgpu": {"loaded": amdgpu_module.is_dir(), "version": _read(amdgpu_module / "version")},
            "kfd": {"loaded": kfd_module.is_dir()},
        },
        "devices": {"/dev/kfd": kfd_device, "/dev/dri": dri_device},
        "drm_devices": drm_devices,
        "amdgpu_devices": amdgpu_devices,
        "firmware": firmware,
        "policy": {
            "required_driver": "Linux amdgpu + KFD",
            "required_architecture": TARGET,
            "isa_override_allowed": False,
            "runtime_claim": "readiness only; native MI50 execution still requires the full hardware test suite",
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sysfs-root", type=Path, default=Path("/sys"))
    parser.add_argument("--firmware-root", type=Path, default=Path("/lib/firmware"))
    parser.add_argument("--dev-root", type=Path, default=Path("/dev"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = collect_readiness(
        sysfs_root=args.sysfs_root,
        firmware_root=args.firmware_root,
        dev_root=args.dev_root,
    )
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    else:
        sys.stdout.write(payload)
    return 1 if report["status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
