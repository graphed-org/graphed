"""m62 unit B — the store choice changes no plan byte, task id, record byte or result byte."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

import m62_url_helpers as h
import pytest

from graphed.checkpoint import FsspecStore, Store, run_resumable


def test_plan_bytes_and_task_ids_do_not_depend_on_the_store(store_url, tmp_path) -> None:
    plan = h.hist_plan(6)
    before = plan.to_bytes()
    assert before == h.hist_plan(6).to_bytes()
    local = Store(tmp_path / "local")
    remote = FsspecStore(store_url)
    local_value = run_resumable(plan, local).value
    remote_value = run_resumable(plan, remote).value
    assert plan.to_bytes() == before

    task_ids = {plan.task_id(p) for p in plan.partitions}
    local_records = h.local_lines(Path(local.journal_path))
    remote_records = h.record_objects(remote, remote.journal_path)
    assert {json.loads(r)["task_id"] for r in local_records} == task_ids
    assert {json.loads(r)["task_id"] for r in remote_records} == task_ids
    assert sorted(remote_records) == sorted(local_records)
    assert remote_value.tobytes() == local_value.tobytes()


def test_import_does_not_load_fsspec(tmp_path) -> None:
    child = (
        "import sys\n"
        "import graphed.checkpoint\n"
        "before = 'fsspec' in sys.modules\n"
        f"graphed.checkpoint.FsspecStore('memory://m62-{uuid.uuid4().hex}')\n"
        "print(before, 'fsspec' in sys.modules)\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", child],
        cwd=tmp_path,
        env={**os.environ},
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.split() == ["False", "True"]


def test_missing_fsspec_names_the_extra(monkeypatch) -> None:
    for name in [k for k in sys.modules if k == "fsspec" or k.startswith("fsspec.")]:
        monkeypatch.setitem(sys.modules, name, None)
    monkeypatch.setitem(sys.modules, "fsspec", None)
    with pytest.raises(ImportError, match=r"graphed\[checkpoint\]"):
        FsspecStore(f"memory://m62-{uuid.uuid4().hex}")
