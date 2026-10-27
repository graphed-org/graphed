"""M11 factorization: the awkward idiom is FUNCTIONS over arrays, never member functions.

Design review 2026-06-10: `graphed.Array` is backend-idiom-neutral and backends complete their own
idiom. graphed-numpy supplies a method/property proxy (NumpyArray); graphed-awkward deliberately
does NOT — awkward's core design applies every operation as a function taking arrays (`gak.*`),
so the awkward backend's proxy (AwkwardArray) adds no member functions, only awkward's
`__awkward_function__` hook routing `ak.*` to `gak`, and its user surface stays in the `gak`
namespace.
"""

from __future__ import annotations

import awkward as ak

from graphed import Array, Session
from graphed.awkward import AwkwardArray, AwkwardBackend, from_awkward, gak


def test_awkward_sessions_return_the_base_neutral_proxy() -> None:
    s = Session(AwkwardBackend())
    a = from_awkward(s, "a", ak.Array([[1.0], [2.0, 3.0]]))
    assert isinstance(a, Array) and type(a) is AwkwardArray  # the neutral proxy plus ak.* dispatch
    assert type(a + a) is AwkwardArray


def test_no_method_idiom_on_the_proxy_the_surface_is_gak() -> None:
    # the numpy method idiom must not exist here; the same operations are gak functions
    for name in ("sum", "mean", "std", "num", "argmin", "argmax"):
        assert name not in vars(Array), f"method idiom {name!r} leaked onto the shared proxy"
    for fn in ("sum", "num", "argmin", "argmax", "flatten", "zip"):
        assert callable(getattr(gak, fn)), (
            f"gak.{fn} missing: the functions-only surface is the idiom"
        )


def test_gak_functions_record_over_the_base_proxy() -> None:
    s = Session(AwkwardBackend())
    a = from_awkward(s, "a", ak.Array([[1.0], [2.0, 3.0]]))
    out = gak.sum(gak.num(a, axis=1))
    assert isinstance(out, Array) and type(out) is AwkwardArray
