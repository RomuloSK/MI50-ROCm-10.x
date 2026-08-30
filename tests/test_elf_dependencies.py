import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.validate_elf_dependencies import validate


class ElfDependencyTests(unittest.TestCase):
    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp.cleanup)
        self.root = Path(self._temp.name) / "rocm"
        self.root.mkdir()

    def test_non_elf_fixture_is_a_clean_host_audit(self):
        (self.root / "README.txt").write_text("host-only", encoding="utf-8")
        report = validate(self.root)
        self.assertEqual("pass", report["status"])
        self.assertEqual(0, report["elf_checked"])

    def test_unresolved_shared_library_is_reported(self):
        binary = self.root / "bin" / "sample"
        binary.parent.mkdir()
        binary.write_bytes(b"\x7fELFsynthetic")

        class Result:
            stdout = "libmissing.so => not found\n"
            stderr = ""

        with patch("scripts.validate_elf_dependencies.shutil.which", return_value="ldd"), patch(
            "scripts.validate_elf_dependencies.subprocess.run", return_value=Result()
        ):
            report = validate(self.root)

        self.assertEqual("fail", report["status"])
        self.assertFalse(report["checks"]["unresolved_dependencies"])
        self.assertEqual("bin/sample", report["unresolved"][0]["file"])

    def test_resolved_dependencies_pass(self):
        binary = self.root / "bin" / "sample"
        binary.parent.mkdir()
        binary.write_bytes(b"\x7fELFsynthetic")

        class Result:
            stdout = "libc.so.6 => /lib/libc.so.6 (0x0)\n"
            stderr = ""

        with patch("scripts.validate_elf_dependencies.shutil.which", return_value="ldd"), patch(
            "scripts.validate_elf_dependencies.subprocess.run", return_value=Result()
        ):
            report = validate(self.root)

        self.assertEqual("pass", report["status"])
        self.assertEqual([], report["unresolved"])


if __name__ == "__main__":
    unittest.main()
