"""Shared helpers for the m62 URL-store suites (plain functions: unit C re-collects the tests)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import analyses


def record_objects(store: Any, prefix: str) -> list[bytes]:
    """The bytes of every object under ``prefix``, in name order."""
    return [store.fs.cat_file(p) for p in sorted(store.fs.find(prefix))]


def local_lines(path: Path) -> list[bytes]:
    return path.read_bytes().splitlines(keepends=True) if path.exists() else []


def hist_plan(n: int = 6) -> Any:
    return analyses.build_plan("analyses:histogram_chunk", n)


_CHILD = (
    "import json, sys\n"
    "from graphed.core import DurablePlan\n"
    "from graphed.checkpoint import FsspecStore, run_resumable\n"
    "plan = DurablePlan.from_bytes(open(sys.argv[1], 'rb').read())\n"
    "res = run_resumable(plan, FsspecStore(sys.argv[2]))\n"
    "print(json.dumps([res.report.executed, res.report.skipped, res.value.tolist()]))\n"
)


def resume_in_child(plan: Any, url: str, workdir: Path) -> tuple[int, int, list[int]]:
    """Resume ``plan`` in a fresh interpreter given only the plan file's path and the URL."""
    plan_file = workdir / "plan.bin"
    plan_file.write_bytes(plan.to_bytes())
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD, str(plan_file), url],
        cwd=workdir,
        env={**os.environ},
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stderr
    executed, skipped, value = json.loads(proc.stdout.strip().splitlines()[-1])
    return executed, skipped, value
