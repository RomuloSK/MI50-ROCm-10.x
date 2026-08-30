"""Tests for the two gates that keep the HIP compiler inside the distribution.

The gfx906 build twice produced a ``dist/rocm`` tree without a usable ``hipcc``
because TheRock's manifest-driven ``artifact-flatten`` either saw a 0-byte
``artifact_manifest.txt`` or copied 0-byte truncations of intact stage files.
Both failure modes are host-visible, so both are covered here without a GPU.
"""

import tempfile
import unittest
from pathlib import Path

from scripts.repair_artifact_manifests import repair
from scripts.validate_dist_contents import REQUIRED_PATHS, validate


def _write(path: Path, content: bytes = b"payload") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


class ManifestRepairTests(unittest.TestCase):
    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp.cleanup)
        self.root = Path(self._temp.name)
        self.artifacts = self.root / "artifacts"
        self.build_root = self.root / "build"

    def _slice(self, name: str) -> Path:
        path = self.artifacts / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def test_empty_manifest_is_rebuilt_from_payload(self):
        slice_dir = self._slice("amd-llvm_run_generic")
        _write(slice_dir / "compiler" / "amd-llvm" / "stage" / "bin" / "hipcc")
        _write(slice_dir / "compiler" / "amd-llvm" / "stage" / "bin" / "hipconfig")
        manifest = _write(slice_dir / "artifact_manifest.txt", b"")

        report = repair(self.artifacts, apply=True, build_root=self.build_root)

        self.assertEqual("pass", report["status"])
        self.assertEqual(0, report["uncovered_payload_files"])
        self.assertEqual(
            ["compiler/amd-llvm/stage"],
            report["slices_repaired"][0]["repaired_prefixes"],
        )
        self.assertEqual(
            "compiler/amd-llvm/stage\n", manifest.read_text(encoding="utf-8")
        )

    def test_truncated_payload_is_healed_not_flattened(self):
        slice_dir = self._slice("amd-llvm_lib_generic")
        relative = Path("compiler") / "amd-comgr" / "stage" / "lib" / "libamd_comgr.so"
        # The slice copy is empty while the build tree it came from is intact.
        _write(slice_dir / relative, b"")
        _write(self.build_root / relative, b"a real shared library")
        _write(slice_dir / "artifact_manifest.txt", b"")

        report = repair(
            self.artifacts, apply=True, heal=True, build_root=self.build_root
        )

        self.assertEqual(
            ["amd-llvm_lib_generic"], [h["slice"] for h in report["slices_healed"]]
        )
        self.assertEqual(1, report["slices_healed"][0]["truncated_files"])
        self.assertFalse(slice_dir.exists(), "corrupt slice must be removed")

    def test_legitimately_empty_file_is_not_truncation(self):
        slice_dir = self._slice("rocgdb_test_generic")
        relative = Path("compiler") / "rocgdb" / "stage" / "share" / "empty.fixture"
        _write(slice_dir / relative, b"")
        _write(self.build_root / relative, b"")
        _write(slice_dir / "artifact_manifest.txt", b"compiler/rocgdb/stage\n")

        report = repair(
            self.artifacts, apply=True, heal=True, build_root=self.build_root
        )

        self.assertEqual([], report["slices_healed"])
        self.assertTrue(slice_dir.exists(), "valid files must survive the heal")

    def test_complete_manifest_is_left_alone(self):
        slice_dir = self._slice("core-hip_run_generic")
        _write(slice_dir / "lib" / "hip" / "stage" / "bin" / "hipcc")
        _write(slice_dir / "artifact_manifest.txt", b"lib/hip/stage\n")

        report = repair(self.artifacts, apply=True, build_root=self.build_root)

        self.assertEqual([], report["slices_repaired"])
        self.assertEqual(1, report["slices_complete"])


class DistContentTests(unittest.TestCase):
    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp.cleanup)
        self.dist = Path(self._temp.name) / "rocm"
        for relative in REQUIRED_PATHS:
            _write(self.dist / relative)
        _write(
            self.dist / "lib" / "rocblas" / "library" / "TensileLibrary_gfx906.co"
        )
        _write(self.dist / "share" / "miopen" / "db" / "gfx906_60.HIP.fdb.txt")

    def test_complete_distribution_passes(self):
        report = validate(self.dist, "gfx906")
        self.assertEqual("pass", report["status"], report["missing"])
        self.assertEqual(
            "host-only inspection; GPU execution remains pending-hardware",
            report["runtime_claim"],
        )

    def test_zero_byte_hip_compiler_fails_the_gate(self):
        _write(self.dist / "bin" / "hipcc", b"")
        report = validate(self.dist, "gfx906")
        self.assertEqual("fail", report["status"])
        self.assertIn("required:bin/hipcc", report["missing"])

    def test_missing_bitcode_fails_the_gate(self):
        (self.dist / "lib" / "llvm" / "amdgcn" / "bitcode" / "ocml.bc").unlink()
        report = validate(self.dist, "gfx906")
        self.assertIn("required:lib/llvm/amdgcn/bitcode/ocml.bc", report["missing"])

    def test_foreign_device_code_fails_the_gate(self):
        _write(self.dist / "lib" / "rocblas" / "library" / "TensileLibrary_gfx942.co")
        report = validate(self.dist, "gfx906")
        self.assertIn("single_target_device_code:gfx906", report["missing"])

    def test_architecture_named_headers_are_not_device_code(self):
        _write(self.dist / "include" / "rocwmma" / "gfx908_projection.hpp")
        report = validate(self.dist, "gfx906")
        self.assertEqual("pass", report["status"], report["missing"])
        self.assertIn("gfx908", report["details"]["foreign_targets_in_file_names"])

    def test_absent_distribution_fails_cleanly(self):
        report = validate(self.dist.parent / "nowhere", "gfx906")
        self.assertEqual("fail", report["status"])
        self.assertEqual(["distribution_present"], report["missing"])


if __name__ == "__main__":
    unittest.main()
