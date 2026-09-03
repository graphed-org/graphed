"""m51 anchor L — the numpy idiom REFUSES a varied write (§6.4f, plan §10).

`to_parquet` is per idiom: the awkward idiom gains variation-aware write-out, the numpy idiom does
NOT (it hard-caps output at one 1-D column). §6.4f binds the numpy refusal precisely:

  * the entry point is the MODULE path `graphed.numpy.io.to_parquet` — `graphed.numpy.to_parquet` is
    deliberately NOT a package attribute (absent from the `graphed.numpy` namespace and `__all__`);
  * the trigger is a `Varied` FIRST POSITIONAL, raised as a `graphed` error naming the AWKWARD
    backend (not §2.2's reserved-name `AttributeError` that a bare `varied.session` read produces);
  * the numpy idiom gains NO `select=` keyword — a `select=` call stays an ordinary `TypeError`, and
    §6.4f/§10 forbid freezing a graphed error for that arm.

THIS FILE IS AWKWARD-FREE BY CONSTRUCTION (plan §10's frozen-layout rule): the required 3.14t
`test-freethreaded` CI job collects the whole `tests/frozen/numpy` subtree in one process under
`pytest hypothesis numpy` alone — no awkward wheel. The `Varied` is built through the numpy idiom
(`graphed.numpy.from_array` + `graphed.vary`), importing neither `awkward` nor `gak`.
"""

from __future__ import annotations

import numpy as np
import pytest

import graphed
import graphed.numpy as gn
import graphed.numpy.io as gio
from graphed import Session
from graphed.errors import GraphedError
from graphed.numpy import NumpyBackend


def _varied_numpy_array() -> object:
    """A `Varied` whose members are numpy-idiom source arrays — no awkward anywhere."""
    session = Session(NumpyBackend())
    g = gn.from_array(session, "x", np.arange(12.0))
    varied = graphed.vary(g, "jes", up=g * 1.1, down=g * 0.9)
    assert isinstance(varied, graphed.Varied)  # instrument live: the fixture really built a container
    assert graphed.labels(varied) == ("nominal", "jes_up", "jes_down")
    return varied


def test_numpy_to_parquet_refuses_a_varied_first_positional(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """§6.4f: a `Varied` first positional is refused with a graphed error naming the awkward backend.

    Non-vacuous: with no guard, `to_parquet` reads `varied.session` first, and §2.2's reserved-name
    rule makes that an `AttributeError` — a *different* exception type — so `pytest.raises(GraphedError)`
    is red until the awkward-naming guard is added, and stays red for any guard that does not name the
    backend the writer points the analyst at.
    """
    varied = _varied_numpy_array()
    with pytest.raises(GraphedError, match=r"awkward"):
        gio.to_parquet(varied, str(tmp_path / "out"))


def test_numpy_to_parquet_module_path_is_the_entry_point_not_a_package_attribute() -> None:
    """§6.4f: the refusal lives on the MODULE function; `graphed.numpy.to_parquet` stays unexported."""
    assert not hasattr(gn, "to_parquet")
    assert "to_parquet" not in gn.__all__
    assert callable(gio.to_parquet)  # positive control: the module path IS the entry point


def test_numpy_to_parquet_gains_no_select_keyword(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """§6.4f/§10: the numpy idiom adds NO `select=`; a `select=` call is an ordinary `TypeError`.

    This guards the reverse mistake — an implementer wiring `select=` into the numpy idiom to mirror
    the awkward one. It must stay a plain `TypeError` (unexpected keyword), never a graphed error.
    """
    session = Session(NumpyBackend())
    g = gn.from_array(session, "x", np.arange(6.0))
    with pytest.raises(TypeError) as excinfo:
        gio.to_parquet(g, str(tmp_path / "nope"), select=g > 2.0)  # type: ignore[call-arg]
    assert not isinstance(excinfo.value, GraphedError)
    assert "select" in str(excinfo.value)
