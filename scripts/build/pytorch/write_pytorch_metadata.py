#!/usr/bin/env python3
"""Write provenance for a completed MI50 PyTorch wheel build."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path


def git_commit(source: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(source), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() or None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel-dir", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--rocm", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--jobs", type=int, required=True)
    parser.add_argument("--hipblaslt-host", type=Path)
    parser.add_argument("--patch-dir", type=Path, required=True)
    args = parser.parse_args()

    out = args.wheel_dir.resolve()
    source = args.source.resolve()
    rocm = args.rocm.resolve()
    build_dir = args.build_dir.resolve()
    patch_dir = args.patch_dir.resolve()
    patch_root = patch_dir.parent.parent.parent
    patches = [
        {
            "file": str(patch.relative_to(patch_root)),
            "sha256": hashlib.sha256(patch.read_bytes()).hexdigest(),
        }
        for patch in sorted(patch_dir.glob("*.patch"))
    ]
    metadata = {
        "schema_version": 1,
        "project": "pytorch",
        "target": "gfx906",
        "source": str(source),
        "source_commit": git_commit(source),
        "rocm_path": str(rocm),
        "build_dir": str(build_dir),
        "wheel_dir": str(out),
        "build_options": {
            "PYTORCH_ROCM_ARCH": "gfx906",
            "USE_ROCM": "1",
            "USE_CUDA": "0",
            "USE_NCCL": "1",
            "USE_SYSTEM_NCCL": "1",
            "USE_AOTRITON": "0",
            "USE_FLASH_ATTENTION": "0",
            "USE_MEM_EFF_ATTENTION": "0",
            "USE_TRITON": "0",
            "USE_ROCM_CK_GEMM": "0",
            "USE_ROCM_CK_SDPA": "0",
            "ROCBLAS_USE_HIPBLASLT": "0",
            "PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED": "0",
        },
        "jobs": args.jobs,
        "platform": {"system": platform.system(), "release": platform.release()},
        "hipify": "tools/amd_build/build_amd.py",
        "downstream_patches": patches,
        "hipblaslt_host": str(args.hipblaslt_host.resolve()) if args.hipblaslt_host else None,
        "runtime_status": (
            "GPU-test-pending" if not Path("/dev/kfd").exists() else "hardware-validation-required"
        ),
        "hsa_override_used": bool(os.environ.get("HSA_OVERRIDE_GFX_VERSION")),
        "wheels": sorted(path.name for path in out.glob("*.whl")),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "mi50-build-metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(out / "mi50-build-metadata.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
