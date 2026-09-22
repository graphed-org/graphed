"""m61 — a partition-wise driver replays the compiled IR once per chunk, so it must refuse what
per-chunk replay cannot compute, and ship what per-chunk replay needs.

Three shapes, one operation (`refuse_chunk_partials` / `external_evaluators` / `file_bases`):
a reduction node inside a partitioned plan is a per-chunk PARTIAL — sound only as an output the
driver's `combine` folds, never as another node's input and never as a writer's output; a writer
evaluates the same IR a plan does, so it wires the session's External evaluators; and a part index
derived from the partition alone cannot tell two listings of one input apart.
"""

from __future__ import annotations

from pathlib import Path

import awkward as ak
import numpy as np
import pytest

import graphed
import graphed.numpy as gn
from graphed import GraphedError, compile_ir, refuse_chunk_partials
from graphed.awkward import AwkwardBackend, from_parquet, gak, to_parquet
from graphed.core.execution import SequentialRunner
from graphed.numpy import io as npio

pytest.importorskip("pyarrow")

X = np.arange(10.0)  # dyadic: every sum below is exact


@pytest.fixture
def parquet_path(tmp_path: Path) -> str:
    path = str(tmp_path / "in.parquet")
    ak.to_parquet(ak.Array({"x": X}), path)
    return path


def _ak_source(path: str | list[str]) -> graphed.Array:
    return from_parquet(graphed.Session(AwkwardBackend()), "ev", path)


def _np_source(path: str | list[str]) -> graphed.Array:
    return npio.from_parquet(graphed.Session(gn.NumpyBackend()), "ev", path)


def _total(plan: graphed.core.execution.Plan[float]) -> float:
    return float(SequentialRunner().run(plan).value)


def _sum_plan(out: graphed.Array, steps: int) -> graphed.core.execution.Plan[float]:
    return graphed.aggregate_plan(
        out, reduce=lambda v: float(v[0]), combine=lambda a, b: a + b, empty=lambda: 0.0, steps_per_file=steps
    )


# ---- reductions inside a partitioned plan ----------------------------------------------------
def test_refusal_reads_the_compiled_ir(parquet_path: str) -> None:
    ev = _ak_source(parquet_path)
    partial_output = compile_ir(ev.session, ev.x[2:8])
    refuse_chunk_partials(partial_output, as_outputs=False)  # a plan may fold an output partial
    with pytest.raises(GraphedError, match="'slice' reduces the partitioned axis"):
        refuse_chunk_partials(partial_output, as_outputs=True)
    with pytest.raises(GraphedError, match="'slice' .* feeds another node"):
        refuse_chunk_partials(compile_ir(ev.session, gak.sum(ev.x[2:8])), as_outputs=False)
    refuse_chunk_partials(compile_ir(ev.session, ev.x * 2), as_outputs=True)  # row-local: fine


def test_aggregate_plan_refuses_an_interior_reduction(parquet_path: str) -> None:
    ev = _ak_source(parquet_path)
    with pytest.raises(GraphedError, match="'slice' reduces the partitioned axis and feeds another node"):
        _sum_plan(gak.sum(ev.x[2:8]), steps=2)


def test_aggregate_plan_still_folds_an_output_reduction(parquet_path: str) -> None:
    ev = _ak_source(parquet_path)
    assert _total(_sum_plan(gak.sum(ev.x), steps=2)) == float(X.sum())
    assert _total(_sum_plan(gak.sum(ev.x[ev.x > 4]), steps=3)) == float(X[X > 4].sum())  # a mask is row-local


@pytest.mark.parametrize("backend", ["awkward", "numpy"])
def test_writers_refuse_a_reduction_output(parquet_path: str, tmp_path: Path, backend: str) -> None:
    ev = _ak_source(parquet_path) if backend == "awkward" else _np_source(parquet_path)
    write = to_parquet if backend == "awkward" else npio.to_parquet
    with pytest.raises(GraphedError, match="'slice' reduces the partitioned axis: a partitioned write"):
        write(ev.x[2:8], str(tmp_path / "out"), steps_per_file=2)
    assert not (tmp_path / "out").exists()


# ---- the session's External evaluators reach the write workers -------------------------------
def _double(x: object) -> object:
    return x * 2  # type: ignore[operator]


def test_awkward_writer_wires_externals(parquet_path: str, tmp_path: Path) -> None:
    ev = _ak_source(parquet_path)
    paths = to_parquet(gak.zip({"a": graphed.apply(_double, ev.x)}), str(tmp_path / "out"), steps_per_file=2)
    back = ak.concatenate([ak.from_parquet(p) for p in paths])
    assert back.a.tolist() == (X * 2).tolist()


def test_numpy_writer_wires_externals(parquet_path: str, tmp_path: Path) -> None:
    ev = _np_source(parquet_path)
    paths = npio.to_parquet(graphed.apply(_double, ev.x), str(tmp_path / "out"), steps_per_file=2)
    back = ak.concatenate([ak.from_parquet(p) for p in paths])
    assert back.data.tolist() == (X * 2).tolist()


# ---- a duplicated input is refused, not silently overwritten ---------------------------------
@pytest.mark.parametrize("backend", ["awkward", "numpy"])
def test_writers_refuse_a_duplicated_input(parquet_path: str, tmp_path: Path, backend: str) -> None:
    ev = _ak_source([parquet_path, parquet_path]) if backend == "awkward" else _np_source([parquet_path, parquet_path])
    write = to_parquet if backend == "awkward" else npio.to_parquet
    with pytest.raises(ValueError, match="duplicate input"):
        write(ev, str(tmp_path / "out"), steps_per_file=2)
