"""A plugin External with no stated type records what one length-zero call gives (#253)."""

from __future__ import annotations

import warnings
from typing import Any

import awkward as ak
import numpy as np
import pytest

import graphed
from graphed.awkward import AwkwardBackend, from_awkward
from graphed.errors import GraphedWarning, StageError
from graphed.preserve.externals import ExternalPlugin, record_external, sha256_bytes

CALLS: list[int] = []


def _events() -> Any:
    session = graphed.Session(AwkwardBackend())
    return from_awkward("ev", ak.Array({"n": [1, 2, 3]}), session=session)


def _plugin(kind: str, evaluate: Any, **kw: Any) -> ExternalPlugin:
    def counted(resource: Any, params: Any, inputs: list[Any]) -> Any:
        CALLS.append(len(inputs[0]))
        return evaluate(inputs[0])

    return ExternalPlugin(
        kind=kind, content_hash=sha256_bytes, evaluate=counted, samples=lambda: [b"a", b"b"], **kw
    )


HALF = _plugin("half", lambda n: n * 0.5)


def _record(plugin: ExternalPlugin, **kw: Any) -> tuple[Any, list[str]]:
    ev = _events()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = record_external(ev.session, plugin, b"a", [ev.n], **kw)
    return out, [str(w.message) for w in caught if issubclass(w.category, GraphedWarning)]


def test_the_type_is_the_length_zero_calls_and_is_checked() -> None:
    CALLS.clear()
    out, caught = _record(HALF)
    assert (out.typestr, caught) == ("## * float64", [])
    assert 0 in CALLS
    np.testing.assert_array_equal(graphed.materialize(out), [0.5, 1.0, 1.5])


def test_a_value_unlike_the_length_zero_call_is_refused() -> None:
    liar = _plugin("liar", lambda n: n * 0.5 if len(n) == 0 else n)
    out, _ = _record(liar)
    with pytest.raises(StageError, match=r"output_type .float64.; its value is .int64."):
        graphed.materialize(out)


def test_a_plugin_that_cannot_run_on_no_rows_warns_and_keeps_the_input_form() -> None:
    def picky(n: Any) -> Any:
        if len(n) == 0:
            raise ValueError("needs rows")
        return n * 0.5

    out, caught = _record(_plugin("picky", picky))
    assert out.typestr == "## * int64"
    assert len(caught) == 1
    assert "could not infer the type of the 'picky' plugin" in caught[0]
    assert "ValueError: needs rows" in caught[0] and caught[0].endswith("pass output_type=")


@pytest.mark.parametrize("stated", ["output_type", "output_dtype"])
def test_a_stated_type_runs_no_length_zero_call(stated: str) -> None:
    CALLS.clear()
    if stated == "output_type":
        out, caught = _record(HALF, output_type="float64")
    else:
        out, caught = _record(_plugin("typed", lambda n: n * 0.5, output_dtype="float64"))
    assert (out.typestr, caught, CALLS) == ("## * float64", [], [])


def test_a_plugin_whose_value_is_no_array_is_not_inferred() -> None:
    CALLS.clear()
    out, caught = _record(_plugin("filler", lambda n: object(), array_value=False))
    assert (out.typestr, caught, CALLS) == ("## * int64", [], [])


def test_a_length_zero_numpy_result_is_no_type() -> None:
    # numpy types an array built from no values float64, whatever the values would be
    out, caught = _record(_plugin("listed", lambda n: np.array([v * 2 for v in n])))
    assert out.typestr == "## * int64"
    assert len(caught) == 1 and "returns ndarray, not an array" in caught[0]
    np.testing.assert_array_equal(graphed.materialize(out), [2, 4, 6])
