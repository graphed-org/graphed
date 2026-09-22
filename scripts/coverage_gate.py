#!/usr/bin/env python3
"""Per-file coverage gate: every source file >= 90%.

Python: `coverage json`'s `summary.percent_covered` (branch=true in pyproject.toml, so this
already blends line+branch). Rust: `cargo llvm-cov --json`'s `summary.lines.percent`
(--rust; --exclude drops a substring-matched filename, e.g. lib.rs).
Always prints the full per-file table and the checked count -- a gate whose confirming
outcome is silent output is not acceptable.
"""

import argparse
import json
import sys
from typing import Any

THRESHOLD = 90.0


def python_files(data: dict[str, Any]) -> dict[str, float]:
    return {path: info["summary"]["percent_covered"] for path, info in data["files"].items()}


def rust_files(data: dict[str, Any], exclude: str | None) -> dict[str, float]:
    files = data["data"][0]["files"]
    return {
        f["filename"]: f["summary"]["lines"]["percent"]
        for f in files
        if not (exclude and exclude in f["filename"])
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("report", help="coverage.json (Python) or llvm-cov --json output (Rust)")
    ap.add_argument("--rust", action="store_true", help="parse as cargo-llvm-cov --json")
    ap.add_argument("--exclude", default=None, help="Rust only: substring to drop, e.g. lib.rs")
    args = ap.parse_args()

    with open(args.report) as fh:
        data = json.load(fh)
    files = rust_files(data, args.exclude) if args.rust else python_files(data)

    for path in sorted(files):
        print(f"{files[path]:6.2f}%  {path}")
    under = {p: pct for p, pct in files.items() if pct < THRESHOLD}
    print(f"{len(files)} files checked, {len(under)} under {THRESHOLD:.0f}%")
    if not files:
        print("FAIL: report contains no files -- the gate is mis-scoped or the data is missing")
        return 1
    if under:
        print(f"FAIL: {len(under)} file(s) below the {THRESHOLD:.0f}% per-file gate:")
        for path in sorted(under):
            print(f"  {under[path]:6.2f}%  {path}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
