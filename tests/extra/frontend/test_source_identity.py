"""A re-read of one source reuses its binding; another source under a taken name is refused."""

from __future__ import annotations

import sys
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import graphed.numpy as gn
from graphed import Session
from graphed.numpy import NumpyBackend, NumpyForm
from graphed.write import steps_for

REFUSAL = r"already bound to other data in this Session; pass name=.*graphed\.new_session\(\)"


@pytest.fixture
def parquet(tmp_path: Path) -> tuple[Any, list[str]]:
    ak = pytest.importorskip("awkward")
    pytest.importorskip("pyarrow")
    paths = [str(tmp_path / f"{i}.parquet") for i in range(2)]
    for path in paths:
        ak.to_parquet(ak.Array({"x": [1.0, 2.0], "y": [3, 4]}), path)
    from graphed.awkward import AwkwardBackend, from_parquet  # noqa: PLC0415

    return (Session(AwkwardBackend()), from_parquet), paths


def test_a_parquet_reread_reuses_its_node(parquet: tuple[Any, list[str]]) -> None:
    (s, from_parquet), paths = parquet
    first = from_parquet("ev", paths, session=s)
    count = s.node_count()
    assert from_parquet("ev", paths, session=s).node_id == first.node_id
    assert s.node_count() == count


def test_the_partitioning_is_part_of_a_parquet_identity(parquet: tuple[Any, list[str]]) -> None:
    (s, from_parquet), paths = parquet
    first = from_parquet("ev", paths, steps_per_file=3, session=s)
    count = s.node_count()
    assert from_parquet("ev", paths, steps_per_file=3, session=s).node_id == first.node_id
    assert s.node_count() == count
    variants: list[dict[str, Any]] = [{}, {"steps_per_file": 3, "open_files": True}]
    for variant in variants:
        with pytest.raises(ValueError, match=REFUSAL):
            from_parquet("ev", paths, session=s, **variant)
    # each refused read would have partitioned its plans differently from the held one
    plans = {
        _default_partitions(from_parquet, paths, s, **variant)
        for variant in [{"steps_per_file": 3}, *variants]
    }
    assert len(plans) == 3


def _default_partitions(
    from_parquet: Any, paths: list[str], like: Session, **variant: Any
) -> tuple[Any, ...]:
    fresh = from_parquet("ev", paths, session=Session(type(like.backend)()), **variant)
    (data,) = fresh.session.sources().values()
    return tuple(data.partitions(steps_for(data, None)))


@pytest.mark.parametrize(
    ("first", "second"),
    [((None, None), (1, None)), ((None, None), (None, ["x", "y"]))],
    ids=["other files", "other columns, one form"],
)
def test_other_files_or_columns_under_a_taken_name_are_refused(
    parquet: tuple[Any, list[str]], first: tuple[Any, Any], second: tuple[Any, Any]
) -> None:
    (s, from_parquet), paths = parquet
    from_parquet("ev", paths[: first[0]], columns=first[1], session=s)
    with pytest.raises(ValueError, match=REFUSAL):
        from_parquet("ev", paths[: second[0]], columns=second[1], session=s)


def test_a_reused_binding_still_follows_its_params() -> None:
    s = Session(NumpyBackend())
    a = np.arange(4.0)
    nodes = [gn.from_array("x", a, chunks=c, session=s).node_id for c in (2, 4, 2)]
    assert nodes == [0, 1, 0]


def test_equal_ndarrays_are_still_other_data() -> None:
    s = Session(NumpyBackend())
    gn.from_array("x", np.arange(3.0), session=s)
    with pytest.raises(ValueError, match=REFUSAL):
        gn.from_array("x", np.arange(3.0), session=s)


class _Same:
    def source_identity(self) -> object:
        return "same"


def test_an_equal_identity_under_another_form_is_refused_by_the_form() -> None:
    s = Session(NumpyBackend())
    s.source("n", form=NumpyForm(np.dtype("float64"), shape=(None,)), data=_Same())
    with pytest.raises(ValueError, match="already a slot of another form in this Session; pass"):
        s.source("n", form=NumpyForm(np.dtype("int64"), shape=(None,)), data=_Same())
    assert (
        s.source("n", form=NumpyForm(np.dtype("float64"), shape=(None,)), data=_Same()).node_id
        == 0
    )


@pytest.fixture
def _fast_switching() -> Iterator[None]:
    interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    yield
    sys.setswitchinterval(interval)


@pytest.mark.usefixtures("_fast_switching")
@pytest.mark.parametrize(
    "create",
    [
        lambda s: gn.zeros((3,), session=s),
        lambda s: gn.default_rng(7, session=s).normal(3),
    ],
    ids=["zeros", "default_rng"],
)
def test_two_threads_creating_one_content_bind_it_once(create: Callable[[Session], Any]) -> None:
    refused = 0
    for _ in range(2000):
        s = Session(NumpyBackend())
        barrier = threading.Barrier(2)
        out: list[object] = []

        def go(
            s: Session = s, barrier: threading.Barrier = barrier, out: list[object] = out
        ) -> None:
            barrier.wait()
            try:
                out.append(create(s).node_id)
            except ValueError as exc:
                out.append(exc)

        threads = [threading.Thread(target=go) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        refused += any(isinstance(o, ValueError) for o in out)
        assert s.node_count() == 1
    assert refused == 0


def test_a_named_creation_over_a_nearby_fill_is_other_data() -> None:
    s = Session(NumpyBackend())
    gn.full((1,), np.array([0.1]), name="w", session=s)
    with pytest.raises(ValueError, match=REFUSAL):
        gn.full((1,), np.array([0.1000000001]), name="w", session=s)


_ELIDED = np.arange(2000.0)
_ELIDED_OTHER = _ELIDED.copy()
_ELIDED_OTHER[1000] = -1.0  # inside the span ndarray repr elides


@pytest.mark.parametrize(
    ("first", "second"),
    [(np.array([0.1]), np.array([0.1000000001])), (_ELIDED, _ELIDED_OTHER)],
    ids=["past 8 digits", "past 1000 elements"],
)
def test_creations_over_nearby_fills_are_distinct_sources(
    first: np.ndarray, second: np.ndarray
) -> None:
    s = Session(NumpyBackend())
    a = gn.full(first.shape, first, session=s)
    b = gn.full(second.shape, second, session=s)
    assert a.node_id != b.node_id
    np.testing.assert_array_equal(s.materialize(b), second)


def test_draws_with_nearby_array_parameters_are_distinct_sources() -> None:
    s = Session(NumpyBackend())
    a = gn.default_rng(7, session=s).normal(1, loc=np.array([0.1]))
    b = gn.default_rng(7, session=s).normal(1, loc=np.array([0.1000000001]))
    assert a.node_id != b.node_id
