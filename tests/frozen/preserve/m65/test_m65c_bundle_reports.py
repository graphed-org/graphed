"""M65 C frozen suite (graphed-preserve slice): a run report kept in a bundle's own Store, outside its
fingerprint, and rendered by ``inspect()`` without executing (plan-C C-5, C-6)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "m9"))

import agc
import awkward as ak

import graphed.core as gc
import graphed.debug as gd
import graphed.preserve as gp
from graphed import Session
from graphed.awkward import AwkwardBackend, from_awkward
from graphed.core import GraphStore, Partition, Plan, Task
from graphed.debug import SourceFrame, StageError

ERR = StageError(
    op="map",
    frames=(SourceFrame("analysis.py", 12, "build", "ev.x.map(f)"),),
    input_forms=("var * float64",),
    partition="u::1/4",
    cause_type="IndexError",
    cause_message="boom",
    opt_level=0,
)


def _add(a: float, b: float) -> float:
    return a + b


def _zero() -> float:
    return 0.0


def _ok(p: Partition, _r: object) -> float:
    return 1.0


def _raise_on_1(p: Partition, _r: object) -> float:
    if p.blind_step == 1:
        raise ERR
    return 1.0


def _plan(fn: Any) -> Plan[float]:
    return Plan(
        process=fn,
        combine=_add,
        empty=_zero,
        tasks=[Task(k, Partition.blind("u", "t", k, 4)) for k in range(4)],
    )


def _failed(container_digest: str | None = None) -> Any:
    rec = gd.RunRecorder()
    with pytest.raises(StageError) as info:
        gc.SequentialRunner(monitor=rec).run(_plan(_raise_on_1))
    return rec.report(error=info.value, container_digest=container_digest)


def _finished(container_digest: str | None = None) -> Any:
    rec = gd.RunRecorder()
    res = gc.SequentialRunner(monitor=rec).run(_plan(_ok))
    return rec.report(result=res, container_digest=container_digest)


def _header(text: str, digest: str) -> str:
    (line,) = [ln for ln in text.splitlines() if ln.startswith(f"    {digest[:12]} ")]
    return line


def test_attach_leaves_the_fingerprint_alone(tmp_path: Path) -> None:
    bundle, _ = agc.build_agc(tmp_path)
    fp = bundle.fingerprint()
    manifest = (bundle.root / "manifest.json").read_bytes()
    r = _failed()
    digest = gp.attach_run_report(bundle, r.to_json())
    assert isinstance(digest, str) and digest
    reopened = gp.Bundle.open(bundle.root)
    assert reopened.fingerprint() == fp
    assert (bundle.root / "manifest.json").read_bytes() == manifest
    assert json.loads(reopened.store.get(digest)) == r.to_json()
    assert f"run-report:{digest}" in reopened.store.completed()


def test_run_reports_round_trip_and_dedupe(tmp_path: Path) -> None:
    bundle, _ = agc.build_agc(tmp_path)
    failed, finished = _failed(), _finished()
    first = gp.attach_run_report(bundle, failed.to_json())
    second = gp.attach_run_report(bundle, finished.to_json())
    assert first != second
    assert gp.Bundle.open(bundle.root).run_reports() == [failed.to_json(), finished.to_json()]
    assert gp.attach_run_report(bundle, failed.to_json()) == first
    assert gp.Bundle.open(bundle.root).run_reports() == [failed.to_json(), finished.to_json()]


def test_inspect_renders_reports_only_when_present(tmp_path: Path) -> None:
    bundle, _ = agc.build_agc(tmp_path / "agc")
    assert "run reports:" not in gp.inspect(bundle)

    r = _failed()
    digest = gp.attach_run_report(bundle, r.to_json())
    other = _finished(container_digest="sha256:other")
    other_digest = gp.attach_run_report(bundle, other.to_json())
    multiline = gd.RunRecorder().report(error=RuntimeError("x\ny"))
    gp.attach_run_report(bundle, multiline.to_json())

    text = gp.inspect(gp.Bundle.open(bundle.root))
    lines = text.splitlines()
    assert "  run reports:" in lines
    head = _header(text, digest)
    assert "outcome=failed" in head and "tasks=4 finished=1 errored=1" in head and "env=same" in head
    assert f"      environment_digest={r.environment_digest}" in lines
    failed_lines = [ln for ln in lines if ln.startswith("      failed: ")]
    assert any(str(r.failure.user_frame) in ln for ln in failed_lines)
    assert "      failed: RuntimeError: x y" in lines
    done = next(t for t in r.tasks if t.state == "finished")
    assert any(
        done.partition in ln and f"duration={done.duration_s:.6f}s" in ln
        for ln in lines
        if ln.startswith("      task ")
    )
    assert any("duration=-" in ln for ln in lines if ln.startswith("      task "))
    assert "env=differs" in _header(text, other_digest)

    data = ak.Array({"x": [1.0, 2.0, 3.0]})
    s = Session(AwkwardBackend())
    ev = from_awkward(s, "events", data)
    tagged = gp.build_bundle(
        tmp_path / "tagged",
        session=s,
        value=ev.x * 2,
        datasets={"events": data},
        container_digest="sha256:abc",
    )
    same = _finished(container_digest="sha256:abc")
    same_digest = gp.attach_run_report(tagged, same.to_json())
    assert "env=same" in _header(gp.inspect(tagged), same_digest)


def test_inspect_reads_reports_without_executing(tmp_path: Path) -> None:
    bundle, _ = agc.build_agc(tmp_path)
    digest = gp.attach_run_report(bundle, _failed().to_json())
    objects = bundle.root / "store" / "objects"
    for blob in bundle.manifest["sources"].values():
        (objects / blob).unlink()
    for e in bundle.manifest["externals"]:
        (objects / e["store"]).unlink()
    text = gp.inspect(gp.Bundle.open(bundle.root))
    assert "run reports:" in text
    assert "outcome=failed" in _header(text, digest)


def test_graph_block_stays_contiguous(tmp_path: Path) -> None:
    bundle, _ = agc.build_agc(tmp_path)
    gp.attach_run_report(bundle, _failed().to_json())
    gp.attach_run_report(bundle, gd.RunRecorder().report(error=RuntimeError("n1\n    n2")).to_json())
    ir = bundle.store.get(bundle.manifest["analysis"]["ir"])
    n_nodes = len(GraphStore.deserialize(ir).nodes())
    lines = gp.inspect(gp.Bundle.open(bundle.root)).splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.strip().startswith("graph (IR"))
    end = next(i for i in range(start + 1, len(lines)) if not lines[i].startswith("    n"))
    assert end - (start + 1) == n_nodes
    reports = lines.index("  run reports:")
    assert reports > end
    assert not any(ln.startswith("    n") for ln in lines[reports:])
