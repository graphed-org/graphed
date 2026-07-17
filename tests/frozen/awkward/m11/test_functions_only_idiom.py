"""M11 factorization: the awkward idiom is FUNCTIONS over arrays, never member functions.

Design review 2026-06-10: `graphed.Array` is backend-idiom-neutral and backends complete their own
idiom. graphed-numpy supplies a method/property proxy (NumpyArray); graphed-awkward deliberately
does NOT — awkward's core design applies every operation as a function taking arrays (`gak.*`),
so the awkward backend keeps the base proxy and its user surface stays in the `gak` namespace.
"""

from __future__ import annotations

import awkward as ak
from graphed import Array, Session

from graphed_awkward import AwkwardBackend, from_awkward, gak


def test_awkward_sessions_return_the_base_neutral_proxy() -> None:
    s = Session(AwkwardBackend())
    a = from_awkward(s, "a", ak.Array([[1.0], [2.0, 3.0]]))
    assert type(a) is Array  # no array_type override: the neutral proxy IS the awkward idiom
    assert type(a + a) is Array


def test_no_method_idiom_on_the_proxy_the_surface_is_gak() -> None:
    # the numpy method idiom must not exist here; the same operations are gak functions
    for name in ("sum", "mean", "std", "num", "argmin", "argmax"):
        assert name not in vars(Array), f"method idiom {name!r} leaked onto the shared proxy"
    for fn in ("sum", "num", "argmin", "argmax", "flatten", "zip"):
        assert callable(getattr(gak, fn)), f"gak.{fn} missing: the functions-only surface is the idiom"


def test_gak_functions_record_over_the_base_proxy() -> None:
    s = Session(AwkwardBackend())
    a = from_awkward(s, "a", ak.Array([[1.0], [2.0, 3.0]]))
    out = gak.sum(gak.num(a, axis=1))
    assert type(out) is Array
