#!/usr/bin/env python3
"""Print or validate the MI50/gfx906 feature contract without a GPU."""

from __future__ import annotations

import argparse
import json
import sys

try:  # Support both module and direct-script execution.
    from .mi50_policy import feature_contract, require_component, validate_environment
except ImportError:  # pragma: no cover - exercised by the shell entry point.
    from mi50_policy import feature_contract, require_component, validate_environment


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component", help="require this component to be portable")
    parser.add_argument("--require-hardware", action="store_true")
    parser.add_argument("--check-environment", action="store_true")
    parser.add_argument("--json", action="store_true", help="emit the normalized contract")
    args = parser.parse_args(argv)

    errors: list[str] = []
    if args.component:
        try:
            require_component(args.component, allow_pending=not args.require_hardware)
        except (KeyError, RuntimeError) as exc:
            errors.append(str(exc))
    if args.check_environment:
        errors.extend(validate_environment())

    if args.json:
        payload = feature_contract()
        if errors:
            payload["errors"] = errors
        json.dump(payload, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    else:
        contract = feature_contract()
        print(f"{contract['project_version']} {contract['llvm_target']} ({contract['architecture']})")
        print("precision:", ", ".join(f"{key}={value}" for key, value in contract["precision"].items()))
        print("runtime:", contract["runtime_claim"])
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
