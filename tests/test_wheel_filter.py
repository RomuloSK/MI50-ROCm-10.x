import base64
import csv
import hashlib
import io
import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.filter_rocm_wheels import filter_wheel


class WheelFilterTests(unittest.TestCase):
    def test_filters_unsupported_library_and_rewrites_record(self):
        with tempfile.TemporaryDirectory() as directory:
            wheel = Path(directory) / "rocm_sdk_libraries-1.0-py3-none-linux_x86_64.whl"
            members = {
                "rocm_sdk_libraries/lib/libhipsparselt.so.0": b"unsupported",
                "rocm_sdk_libraries/lib/librocblas.so.5": b"supported",
                "rocm_sdk_libraries-1.0.dist-info/METADATA": b"Name: rocm-sdk-libraries\n",
            }
            record_name = "rocm_sdk_libraries-1.0.dist-info/RECORD"
            with zipfile.ZipFile(wheel, "w") as archive:
                for name, data in members.items():
                    archive.writestr(name, data)
                archive.writestr(record_name, b"\n")

            report = filter_wheel(wheel)
            self.assertEqual(report["status"], "filtered")
            self.assertIn("rocm_sdk_libraries/lib/libhipsparselt.so.0", report["removed"])
            with zipfile.ZipFile(wheel, "r") as archive:
                self.assertNotIn("rocm_sdk_libraries/lib/libhipsparselt.so.0", archive.namelist())
                record = archive.read(record_name).decode("utf-8")
                rows = list(csv.reader(io.StringIO(record)))
            kept = members["rocm_sdk_libraries/lib/librocblas.so.5"]
            expected = base64.urlsafe_b64encode(hashlib.sha256(kept).digest()).rstrip(b"=").decode()
            self.assertIn(["rocm_sdk_libraries/lib/librocblas.so.5", f"sha256={expected}", str(len(kept))], rows)


if __name__ == "__main__":
    unittest.main()
