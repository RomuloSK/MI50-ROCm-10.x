import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.mi50_kernel_readiness import VEGA20_FIRMWARE, collect_readiness


class KernelReadinessTests(unittest.TestCase):
    def _roots(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        return root / "sys", root / "firmware", root / "dev"

    def test_no_gpu_is_pending_without_false_failure(self):
        sysfs, firmware, dev = self._roots()
        report = collect_readiness(sysfs_root=sysfs, firmware_root=firmware, dev_root=dev)
        self.assertEqual(report["status"], "GPU-test-pending")
        self.assertEqual(report["errors"], [])
        self.assertEqual(report["firmware"]["status"], "not-evaluable-without-amdgpu")

    def test_bound_amdgpu_with_kfd_and_firmware_is_ready_for_rocr(self):
        sysfs, firmware, dev = self._roots()
        (sysfs / "module" / "amdgpu").mkdir(parents=True)
        (sysfs / "module" / "amdgpu" / "version").write_text("6.8\n", encoding="utf-8")
        (sysfs / "module" / "kfd").mkdir(parents=True)
        device = sysfs / "class" / "drm" / "card0" / "device"
        device.mkdir(parents=True)
        (device / "uevent").write_text("DRIVER=amdgpu\nPCI_ID=1002:66a1\n", encoding="utf-8")
        (dev / "kfd").parent.mkdir(parents=True)
        (dev / "kfd").write_bytes(b"")
        for name in VEGA20_FIRMWARE:
            (firmware / "amdgpu").mkdir(parents=True, exist_ok=True)
            (firmware / "amdgpu" / name).write_bytes(b"firmware")

        report = collect_readiness(sysfs_root=sysfs, firmware_root=firmware, dev_root=dev)
        self.assertEqual(report["status"], "ready-for-rocr")
        self.assertEqual(report["errors"], [])
        self.assertEqual(report["amdgpu_devices"][0]["driver"], "amdgpu")
        self.assertEqual(report["firmware"]["missing"], [])

    def test_drm_connector_entries_are_not_counted_as_gpu_devices(self):
        sysfs, firmware, dev = self._roots()
        (sysfs / "module" / "amdgpu").mkdir(parents=True)
        (sysfs / "module" / "kfd").mkdir(parents=True)
        card = sysfs / "class" / "drm" / "card0" / "device"
        card.mkdir(parents=True)
        (card / "uevent").write_text("DRIVER=amdgpu\nPCI_ID=1002:66a1\n", encoding="utf-8")
        connector = sysfs / "class" / "drm" / "card0-HDMI-A-1" / "device"
        connector.mkdir(parents=True)
        (connector / "uevent").write_text("DRIVER=amdgpu\nPCI_ID=1002:66a1\n", encoding="utf-8")
        (dev / "kfd").parent.mkdir(parents=True)
        (dev / "kfd").write_bytes(b"")
        for name in VEGA20_FIRMWARE:
            (firmware / "amdgpu").mkdir(parents=True, exist_ok=True)
            (firmware / "amdgpu" / name).write_bytes(b"firmware")

        report = collect_readiness(sysfs_root=sysfs, firmware_root=firmware, dev_root=dev)
        self.assertEqual(report["status"], "ready-for-rocr")
        self.assertEqual([entry["card"] for entry in report["drm_devices"]], ["card0"])

    def test_bound_amdgpu_missing_firmware_fails_explicitly(self):
        sysfs, firmware, dev = self._roots()
        (sysfs / "module" / "amdgpu").mkdir(parents=True)
        (sysfs / "module" / "kfd").mkdir(parents=True)
        device = sysfs / "class" / "drm" / "card0" / "device"
        device.mkdir(parents=True)
        (device / "uevent").write_text("DRIVER=amdgpu\n", encoding="utf-8")
        (dev / "kfd").parent.mkdir(parents=True)
        (dev / "kfd").write_bytes(b"")
        report = collect_readiness(sysfs_root=sysfs, firmware_root=firmware, dev_root=dev)
        self.assertEqual(report["status"], "fail")
        self.assertIn("missing expected Vega20 firmware", report["errors"][0])

    def test_isa_override_is_always_rejected(self):
        sysfs, firmware, dev = self._roots()
        with patch.dict(os.environ, {"HSA_OVERRIDE_GFX_VERSION": "10.3.0"}):
            report = collect_readiness(sysfs_root=sysfs, firmware_root=firmware, dev_root=dev)
        self.assertEqual(report["status"], "fail")
        self.assertIn("ISA override", report["errors"][0])


if __name__ == "__main__":
    unittest.main()
