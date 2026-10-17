"""An op recorded or refused inside installed third-party code names the analysis line that called
it, and the first frame outside graphed when every other frame is installed (graphed#73)."""

from __future__ import annotations

import os
import sysconfig
import types
from collections.abc import Callable
from typing import Any

import awkward as ak
import numpy as np
import pytest

import graphed
import graphed.provenance as provenance
from graphed.awkward import AwkwardBackend, from_awkward
from graphed.errors import GraphedTypeError

bh = pytest.importorskip("boost_histogram")
hist = pytest.importorskip("hist")
correctionlib = pytest.importorskip("correctionlib")
cs = pytest.importorskip("correctionlib.schemav2")

FAKE_LIB = os.path.join(sysconfig.get_path("purelib"), "gw73_fakelib", "mod.py")


def _x() -> Any:
    session = graphed.Session(AwkwardBackend())
    return from_awkward(session, "ev", ak.Array({"x": [1.0, 2.0, 3.0]})).x


def _correction() -> Any:
    corr = cs.Correction(
        name="c",
        version=1,
        inputs=[cs.Variable(name="x", type="real")],
        output=cs.Variable(name="w", type="real"),
        data=1.0,
    )
    return correctionlib.CorrectionSet(cs.CorrectionSet(schema_version=2, corrections=[corr]))["c"]


def _fake_lib() -> types.ModuleType:
    """A module whose code object says it lives in site-packages, as an installed library's does."""
    module = types.ModuleType("gw73_fakelib.mod")
    exec(compile("def truth(x):\n    return bool(x)\n", FAKE_LIB, "exec"), module.__dict__)
    return module


CALLS: dict[str, Callable[[Any], Any]] = {
    "np.histogram": lambda x: np.histogram(x),
    "ak.from_iter": lambda x: ak.from_iter([x]),
    "boost_histogram fill": lambda x: bh.Histogram(bh.axis.Regular(3, 0, 3)).fill(x),
    "hist fill": lambda x: hist.Hist.new.Reg(3, 0, 3, name="x").Double().fill(x=x),
    "correctionlib evaluate": lambda x: _correction().evaluate(x),
    "installed library": lambda x: _fake_lib().truth(x),
}


@pytest.mark.parametrize("call", CALLS)
def test_a_refusal_inside_installed_code_names_the_calling_line(call: str) -> None:
    with pytest.raises(GraphedTypeError) as info:
        CALLS[call](_x())
    where = info.value.provenance
    assert (where.filename, where.function) == (__file__, "<lambda>"), where


def test_with_every_frame_installed_the_first_frame_outside_graphed_is_named(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(provenance, "_INSTALLED", ("",))
    with pytest.raises(GraphedTypeError) as info:
        _fake_lib().truth(_x())
    where = info.value.provenance
    assert (where.filename, where.lineno, where.function) == (FAKE_LIB, 2, "truth"), where
