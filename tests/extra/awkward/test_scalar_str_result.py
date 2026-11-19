"""A per-chunk function returning a Python ``str`` or ``bytes`` is typed as a scalar (#260)."""

from __future__ import annotations

import json
import operator
import warnings
from pathlib import Path
from typing import Any

import awkward as ak
import numpy as np
import pytest

import graphed
from graphed import GraphedWarning
from graphed.awkward import AwkwardBackend, from_awkward, from_parquet, gak
from graphed.core.execution import SequentialRunner
from graphed.errors import StageError
from graphed.write import PartWrite

CHUNKS = [ak.Array({"w": [1.5, -2.0, 3.0]}), ak.Array({"w": [1.0, 0.5, 2.0]})]


def _source(tmp_path: Path) -> tuple[graphed.Session, Any]:
    session = graphed.Session(AwkwardBackend())
    paths = [str(tmp_path / f"in{i}.parquet") for i in range(len(CHUNKS))]
    for chunk, path in zip(CHUNKS, paths, strict=True):
        ak.to_parquet(chunk, path)
    return session, from_parquet("ev", paths, session=session).w


def summary(a: ak.Array) -> str:
    return json.dumps({"s": float(ak.sum(a)), "n": len(a)})


def _part_metadata(tmp_path: Path, x: Any, value: Any) -> list[Any]:
    got: list[Any] = []

    def codec(_arr: Any, _path: str, meta: Any) -> None:
        got.append(meta["m"])

    write = PartWrite(
        x,
        str(tmp_path / "out"),
        name=lambda p: Path(p.uri).stem,
        codec=codec,
        metadata={"m": value},
    )
    plan = graphed.aggregate_plan(
        gak.sum(x), writes=[write], reduce=list, combine=operator.add, empty=list
    )
    SequentialRunner().run(plan)
    return got


def test_a_str_result_is_inferred_as_a_scalar_string_without_a_warning(tmp_path: Path) -> None:
    session, x = _source(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("error", GraphedWarning)
        value = graphed.apply(summary, x)
    assert session.form(value).describe() == "str"
    assert _part_metadata(tmp_path, x, value) == [summary(chunk.w) for chunk in CHUNKS]


def test_the_whole_dataset_value_is_the_function_over_every_row(tmp_path: Path) -> None:
    session, x = _source(tmp_path)
    value = graphed.apply(summary, x)
    assert session.materialize(value) == summary(ak.concatenate([c.w for c in CHUNKS]))


def test_a_long_str_round_trips_into_part_metadata_unchanged(tmp_path: Path) -> None:
    _session, x = _source(tmp_path)

    def long(a: ak.Array) -> str:
        return json.dumps({"n": len(a), "pad": "x" * 10_000})

    value = graphed.apply(long, x)
    got = _part_metadata(tmp_path, x, value)
    assert got == [long(chunk.w) for chunk in CHUNKS]
    assert all(len(m) > 10_000 for m in got)


def test_a_bytes_result_is_a_scalar_bytes(tmp_path: Path) -> None:
    session, x = _source(tmp_path)
    value = graphed.apply(lambda a: summary(a).encode(), x, name="as_bytes")
    assert session.materialize(value) == summary(ak.concatenate([c.w for c in CHUNKS])).encode()


def test_a_declared_scalar_string_needs_no_inference(tmp_path: Path) -> None:
    session, x = _source(tmp_path)

    def untraceable(a: ak.Array) -> str:
        if not len(a):
            raise RuntimeError("needs rows")
        return summary(a)

    with warnings.catch_warnings():
        warnings.simplefilter("error", GraphedWarning)
        value = graphed.apply(untraceable, x, output_type="scalar string")
    assert _part_metadata(tmp_path, x, value) == [summary(chunk.w) for chunk in CHUNKS]
    assert session.materialize(value) == summary(ak.concatenate([c.w for c in CHUNKS]))


@pytest.mark.parametrize(
    ("declared", "returns"),
    [("scalar string", lambda a: len(a)), ("scalar int64", summary), ("string", summary)],
    ids=["int-for-string", "string-for-int", "rows-for-scalar"],
)
def test_the_wrong_result_type_raises_output_type_error(
    tmp_path: Path, declared: str, returns: Any
) -> None:
    session, x = _source(tmp_path)
    value = graphed.apply(returns, x, output_type=declared, name=f"wrong_{declared}")
    with pytest.raises(StageError) as err:
        session.materialize(value)
    assert err.value.cause_type == "OutputTypeError"


def test_an_inferred_str_result_refuses_another_type_on_data(tmp_path: Path) -> None:
    session, x = _source(tmp_path)

    def changes(a: ak.Array) -> Any:
        return summary(a) if not len(a) else len(a)

    value = graphed.apply(changes, x)
    with pytest.raises(StageError) as err:
        session.materialize(value)
    assert err.value.cause_type == "OutputTypeError"


def test_a_scalar_declaration_names_what_it_can_hold() -> None:
    session = graphed.Session(AwkwardBackend())
    x = from_awkward("e", ak.Array([1.0]), session=session)
    with pytest.raises(graphed.GraphedTypeError, match="primitive dtype, string or bytes"):
        graphed.apply(summary, x, output_type="scalar var * float64", name="bad_scalar")


def test_an_input_of_unknown_regular_size_says_why_it_has_no_length_zero_call() -> None:
    session = graphed.Session(AwkwardBackend())
    x = ak.to_regular(from_awkward("e", ak.Array([[1.0, 2.0], [3.0, 4.0]]), session=session))
    with pytest.warns(GraphedWarning, match="regular size unknown at record time"):
        graphed.apply(lambda a: ak.Array(np.asarray(ak.to_numpy(a)) * 2), x, name="regular")
