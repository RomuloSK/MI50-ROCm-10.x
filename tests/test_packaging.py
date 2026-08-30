import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path
import tarfile

from scripts.package_rocm_gfx906 import package_tree, validate_source


class PackagingTests(unittest.TestCase):
    def setUp(self):
        if shutil.which("tar") is None or shutil.which("gzip") is None:
            self.skipTest("GNU tar and gzip are required for packaging tests")

    def _tree(self, root: Path) -> Path:
        source = root / "rocm"
        (source / "include" / "hip").mkdir(parents=True)
        (source / "lib" / "rocblas" / "library" / "gfx906").mkdir(parents=True)
        (source / "share" / "miopen" / "db").mkdir(parents=True)
        (source / "lib").mkdir(exist_ok=True)
        (source / "lib" / "libhsa-runtime64.so").write_bytes(b"runtime")
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


if __name__ == "__main__":
    unittest.main()
