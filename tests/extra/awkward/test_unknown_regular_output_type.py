"""An ``output_type`` with a regular size unknown at record time fits any size (#270)."""

from __future__ import annotations

import operator
from collections.abc import Callable
from pathlib import Path
from typing import Any

import awkward as ak
import pytest

import graphed
from graphed.awkward import AwkwardBackend, from_awkward, from_parquet, gak
from graphed.awkward.backend import canonical_output_type
from graphed.core.execution import SequentialRunner
from graphed.errors import StageError

WIDE = {"two": [[1.0, 2.0], [3.0, 4.0]], "three": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]}


def _source(tmp_path: Path, *widths: str) -> Any:
    session = graphed.Session(AwkwardBackend())
    paths = []
    for i, width in enumerate(widths):
        paths.append(str(tmp_path / f"in{i}.parquet"))
        ak.to_parquet(ak.Array({"w": WIDE[width]}), paths[-1])
    return from_parquet("ev", paths, session=session).w


def _regular(a: ak.Array) -> ak.Array:
    return ak.to_regular(a, axis=1)


def _per_chunk(out: Any) -> list[list[Any]]:
    plan = graphed.aggregate_plan(out, reduce=list, combine=operator.add, empty=list)
    return [ak.to_list(chunk) for chunk in SequentialRunner().run(plan).value]


def test_the_type_to_regular_records_declares_an_apply() -> None:
    session = graphed.Session(AwkwardBackend())
    x = gak.to_regular(from_awkward("x", ak.Array({"w": WIDE["three"]}), session=session).w)
    out = graphed.apply(lambda a: a, x, output_type=x.type.content)
    assert out.typestr == x.typestr
    assert ak.to_list(session.materialize(out)) == WIDE["three"]


def test_an_unknown_size_spelling_is_a_fixpoint() -> None:
    assert canonical_output_type("## * float64") == "## * float64"
    assert canonical_output_type("{a: var * ## * ?int64}") == "{a: var * ## * ?int64}"


def test_an_unknown_size_fits_each_chunks_own_width(tmp_path: Path) -> None:
    x = _source(tmp_path, "two", "three")
    out = graphed.apply(_regular, x, output_type="## * float64", name="regular")
    assert _per_chunk(out) == [WIDE["two"], WIDE["three"]]


DECLARED: dict[str, tuple[str, Callable[[ak.Array], Any]]] = {
    "a known size stays strict": ("3 * float64", _regular),
    "the leaf is still compared": ("## * int64", _regular),
    "regular is still not var": ("## * float64", lambda a: a),
    "an option is still compared": ("## * ?float64", _regular),
}


@pytest.mark.parametrize(("declared", "fn"), DECLARED.values(), ids=DECLARED.keys())
def test_everything_but_an_unknown_size_is_still_compared(
    tmp_path: Path, declared: str, fn: Callable[[ak.Array], Any]
) -> None:
    out = graphed.apply(fn, _source(tmp_path, "two"), output_type=declared, name="checked")
    with pytest.raises(StageError) as err:
        _per_chunk(out)
    assert err.value.cause_type == "OutputTypeError"


RECORDED: dict[str, Callable[[Any], Any]] = {
    "to_regular": gak.to_regular,
    "a record of a regular": lambda w: ak.zip({"a": gak.to_regular(w)}, depth_limit=1),
    "a regular under a list": lambda w: gak.unflatten(gak.to_regular(w), 1),
    "local_index of a regular": lambda w: ak.local_index(gak.to_regular(w)),
}


@pytest.mark.parametrize("build", RECORDED.values(), ids=RECORDED.keys())
def test_a_recorded_unknown_size_type_round_trips_through_output_type(
    build: Callable[[Any], Any],
) -> None:
    session = graphed.Session(AwkwardBackend())
    x = build(from_awkward("x", ak.Array({"w": WIDE["two"]}), session=session).w)
    out = graphed.apply(lambda a: a, x, output_type=x.type.content, name="same")
    assert out.typestr == x.typestr
    assert ak.to_list(session.materialize(out)) == ak.to_list(session.materialize(x))
