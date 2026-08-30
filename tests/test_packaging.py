import hashlib
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path
import tarfile
from unittest.mock import patch

from scripts.package_rocm_gfx906 import package_tree, validate_source
from scripts.install_rocm_mi50 import ArchiveError, inspect_archive, install_archive


class PackagingTests(unittest.TestCase):
    def setUp(self):
        if shutil.which("tar") is None or shutil.which("gzip") is None:
            self.skipTest("GNU tar and gzip are required for packaging tests")

    def _tree(self, root: Path) -> Path:
        source = root / "rocm"
        (source / "include" / "hip").mkdir(parents=True)
        (source / "lib" / "rocblas" / "library" / "gfx906").mkdir(parents=True)
        (source / "share" / "miopen" / "db").mkdir(parents=True)
        (source / "bin").mkdir(parents=True)
        (source / "lib" / "llvm" / "bin").mkdir(parents=True)
        (source / "lib" / "llvm" / "amdgcn" / "bitcode").mkdir(parents=True)
        (source / "lib").mkdir(exist_ok=True)
        (source / "include" / "hip" / "hip_runtime.h").write_bytes(b"// HIP\n")
        (source / "bin" / "hipcc").write_bytes(b"#!/bin/sh\n")
        (source / "lib" / "llvm" / "bin" / "llc").write_bytes(b"#!/bin/sh\n")
        (source / "lib" / "libhsa-runtime64.so").write_bytes(b"runtime")
        (source / "lib" / "llvm" / "amdgcn" / "bitcode" / "ocml.bc").write_bytes(b"bitcode")
        (source / "lib" / "rocblas" / "library" / "gfx906" / "TensileLibrary_gfx906.hsaco").write_bytes(b"code")
        (source / "share" / "miopen" / "db" / "gfx906_60.HIP.fdb.txt").write_text("db\n")
        (source / "lib" / "hipsparselt").mkdir(parents=True)
        (source / "lib" / "hipsparselt" / "unsupported.so").write_bytes(b"unsupported")
        return source

    def test_package_is_deterministic_and_contains_target_assets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._tree(root)
            first = root / "one.tar.gz"
            second = root / "two.tar.gz"
            metadata = root / "dist_info.json"
            package_tree(source, first, metadata)
            package_tree(source, second)
            self.assertEqual(
                hashlib.sha256(first.read_bytes()).digest(),
                hashlib.sha256(second.read_bytes()).digest(),
            )
            payload = json.loads(metadata.read_text(encoding="utf-8"))
            self.assertEqual(payload["target"], "gfx906")
            self.assertEqual(payload["runtime_claim"], "artifact-only; GPU execution remains pending-hardware")
            with tarfile.open(first, "r:gz") as archive:
                names = archive.getnames()
            self.assertIn("rocm/lib/rocblas/library/gfx906/TensileLibrary_gfx906.hsaco", names)
            self.assertIn("rocm/share/miopen/db/gfx906_60.HIP.fdb.txt", names)
            self.assertNotIn("rocm/lib/hipsparselt/unsupported.so", names)
            self.assertFalse(any("hipsparselt" in name for name in names))

    def test_missing_target_data_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "rocm"
            (source / "include" / "hip").mkdir(parents=True)
            (source / "lib").mkdir()
            (source / "lib" / "libhsa-runtime64.so").write_bytes(b"runtime")
            with self.assertRaises(ValueError):
                validate_source(source)

    def test_installer_validates_and_atomically_publishes_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._tree(root)
            archive = root / "rocm.tar.gz"
            package_tree(source, archive)

            inventory = inspect_archive(archive)
            self.assertEqual(inventory["target"], "gfx906")
            self.assertEqual(inventory["checks"]["hipcc"]["count"], 1)
            prefix = root / "installed-mi50"
            manifest = install_archive(archive, prefix)
            self.assertEqual(manifest["status"], "pass")
            self.assertEqual(manifest["elf_dependency_audit"]["status"], "pass")
            self.assertEqual(manifest["elf_dependency_audit"]["elf_checked"], 0)
            self.assertTrue((prefix / "rocm" / "bin" / "hipcc").is_file())
            self.assertTrue((prefix / "mi50-env.sh").is_file())
            environment = (prefix / "mi50-env.sh").read_text(encoding="utf-8")
            self.assertNotIn("set -euo pipefail", environment)
            self.assertIn("lib/llvm/bin", environment)
            self.assertIn("lib/rocm_sysdeps/lib", environment)
            self.assertIn("lib/llvm/lib", environment)
            install_manifest = (prefix / "mi50-install.json").read_text(encoding="utf-8")
            self.assertIn("\"target\": \"gfx906\"", install_manifest)

            (prefix / "sentinel").write_text("preserve\n", encoding="utf-8")
            install_archive(archive, prefix, force=True)
            backups = sorted(root.glob("installed-mi50.previous-*"))
            self.assertEqual(len(backups), 1)
            self.assertEqual((backups[0] / "sentinel").read_text(encoding="utf-8"), "preserve\n")

    def test_installer_rejects_path_traversal_before_extraction(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "unsafe.tar.gz"
            with tarfile.open(archive, "w:gz") as handle:
                member = tarfile.TarInfo("rocm/../escape")
                member.size = 1
                handle.addfile(member, fileobj=io.BytesIO(b"x"))
            with self.assertRaises(ArchiveError):
                inspect_archive(archive)

    def test_installer_rejects_unresolved_elf_dependencies_before_publish(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "rocm.tar.gz"
            package_tree(self._tree(root), archive)
            prefix = root / "installed-mi50"
            failed_audit = {
                "status": "fail",
                "elf_checked": 1,
                "checks": {},
                "missing": ["unresolved_dependencies"],
                "unresolved": [{"file": "bin/rocminfo", "missing": ["libmissing.so => not found"]}],
                "command_errors": [],
            }
            with patch("scripts.install_rocm_mi50.validate_elf_dependencies", return_value=failed_audit):
                with self.assertRaises(ArchiveError):
                    install_archive(archive, prefix)
            self.assertFalse(prefix.exists())


if __name__ == "__main__":
    unittest.main()
