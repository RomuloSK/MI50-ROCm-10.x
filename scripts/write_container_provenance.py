#!/usr/bin/env python3
"""Record the resolved identity of a locally built MI50 build image."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


def write_provenance(
    output: Path,
    *,
    image: str,
    base_image: str,
    iid_file: Path,
) -> dict[str, Any]:
    """Write image inputs plus daemon-resolved identity when available."""

    try:
        image_id = iid_file.read_text(encoding="utf-8").strip()
    except OSError:
        image_id = None

    try:
        inspect = subprocess.run(
            ["docker", "image", "inspect", image, "--format", "{{json .}}"],
            capture_output=True,
            text=True,
            check=False,
        )
        inspect_status = inspect.returncode
        inspect_stdout = inspect.stdout
    except OSError as exc:
        # The normal build entry point checks for Docker first.  Keep this
        # helper deterministic when called standalone, however, so metadata
        # still records the requested inputs if the daemon is unavailable.
        inspect_status = 127
        inspect_stdout = ""
        inspect_error = str(exc)
    else:
        inspect_error = None
    payload: dict[str, Any] = {
        "schema_version": 1,
        "image": image,
        "base_image": base_image,
        "image_id": image_id,
        "docker_inspect_status": inspect_status,
    }
    if inspect_error:
        payload["docker_inspect_error"] = inspect_error
    if inspect_status == 0 and inspect_stdout.strip():
        try:
            resolved = json.loads(inspect_stdout)
        except json.JSONDecodeError:
            resolved = {"raw": inspect_stdout.strip()}
        payload["resolved"] = {
            "id": resolved.get("Id"),
            "repo_digests": resolved.get("RepoDigests", []),
            "created": resolved.get("Created"),
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--base-image", required=True)
    parser.add_argument("--iid-file", type=Path, required=True)
    args = parser.parse_args(argv)
    write_provenance(
        args.output.resolve(),
        image=args.image,
        base_image=args.base_image,
        iid_file=args.iid_file.resolve(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
