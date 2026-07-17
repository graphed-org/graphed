"""Real provenance (M3): sub-expression text, toggle, and survival through realistic call patterns."""

from __future__ import annotations

import sys

from toy import ToyBackend, source

from graphed import Session, is_enabled, set_enabled
from graphed import provenance as prov


def test_captures_filename_lineno_and_source_text() -> None:
    s = Session(ToyBackend())
    a = source(s, "a")
    b = source(s, "b")
    line = sys._getframe().f_lineno + 1
    c = a + b
    p = s.provenance(c)
    assert p.filename == __file__
    assert p.lineno == line
    assert "a + b" in p.source  # sub-expression text via executing


def test_provenance_survives_helper_functions() -> None:
    s = Session(ToyBackend())
    a = source(s, "a")

    def build_mass(x: object) -> object:
        return x.pt * 2  # the user's helper line

    out = build_mass(a)
    p = s.provenance(out)
    assert p.filename == __file__
    assert p.function == "build_mass"


def test_provenance_survives_list_comprehension() -> None:
    s = Session(ToyBackend())
    a = source(s, "a")
    outs = [a + i for i in range(2)]
    p = s.provenance(outs[0])
    assert p.filename == __file__


def test_toggle_disables_capture() -> None:
    s = Session(ToyBackend())
    a = source(s, "a")
    assert is_enabled()
    set_enabled(False)
    try:
        c = a + a
        assert s.provenance(c).filename == "<provenance-disabled>"
    finally:
        set_enabled(True)
    assert is_enabled()


def test_capture_outside_graphed_returns_user_frame() -> None:
    p = prov.capture()
    assert p.filename == __file__
