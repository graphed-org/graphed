"""An ``output_type`` with a regular size unknown at record time fits any size (#270)."""

from __future__ import annotations

import operator
import tracemalloc
from collections.abc import Callable
from pathlib import Path
from typing import Any

import awkward as ak
import pytest

import graphed
from graphed.awkward import AwkwardBackend, from_awkward, from_parquet, gak
from graphed.awkward.backend import (
    _matched,
    canonical_output_type,
    check_output_type,
)
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


def test_a_quoted_hash_hash_is_text_not_a_size() -> None:
    spelling = 'struct[{a: int64}, parameters={"__doc__": "## * x"}]'
    assert canonical_output_type(spelling) == spelling


def test_union_members_fit_in_any_order_beside_an_unknown_size() -> None:
    value = ak.concatenate([ak.to_regular(ak.Array([[1.0, 2.0]]), axis=1), ak.Array([1])])
    inputs = [ak.Array([1, 2])]
    assert check_output_type(value, inputs, "output_type", "union[int64, ## * float64]") is None
    assert check_output_type(value, inputs, "output_type", "union[int64, ## * int64]") is not None


def test_a_union_member_an_unknown_size_took_first_is_given_back() -> None:
    f = ak.types.NumpyType("float64")
    sizes = [ak.types.RegularType(f, n) for n in (2, 3)]
    declared = [ak.types.RegularType(f, ak.typetracer.unknown_length), sizes[0]]
    assert _matched(sizes, declared)
    assert not _matched([sizes[0], sizes[0]], [declared[0], sizes[1]])


def test_the_stand_in_size_does_not_multiply_the_known_ones() -> None:
    tracemalloc.start()
    try:
        canonical_output_type("## * 4000 * float64")
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < 16_000_000


def test_a_size_zero_regular_from_the_length_zero_call_is_not_inferred() -> None:
    session = graphed.Session(AwkwardBackend())
    x = from_awkward("x", ak.Array(WIDE["two"]), session=session)
    with pytest.warns(UserWarning, match=r"'0 \* 0 \* float64' \(an unknown regular size"):
        out = graphed.apply(lambda a: ak.from_numpy(ak.to_numpy(_regular(a))), x)
    assert "0 *" not in out.typestr
