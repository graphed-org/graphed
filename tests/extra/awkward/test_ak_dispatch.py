"""``ak.<fn>(deferred)`` records exactly the node ``gak.<fn>`` records, through awkward's
``__awkward_function__`` hook on ``AwkwardArray``/``AwkwardVaried``."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

import awkward as ak
import pytest

import graphed
from graphed import Session
from graphed.awkward import AwkwardArray, AwkwardBackend, AwkwardVaried, from_awkward, gak
from graphed.errors import GraphedError


@pytest.fixture
def arrays() -> dict[str, Any]:
    s = Session(AwkwardBackend())
    return {
        "s": s,
        "a": from_awkward(s, "a", ak.Array([[1.0, 2.0], [], [3.0]])),
        "w": from_awkward(s, "w", ak.Array([[0.5, 2.0], [], [1.0]])),
        "r": from_awkward(s, "r", ak.Array([{"pt": 1.0, "eta": 2.0}, {"pt": 3.0, "eta": 4.0}])),
        "n3": from_awkward(s, "n3", ak.Array([[[1.0, 2.0], [3.0]], [[4.0, 5.0, 6.0]]])),
    }


Call = Callable[[dict[str, Any]], Any]
# (ak spelling, the gak call it must equal); the keyword/positional cases are where the signatures differ
PAIRS: dict[str, tuple[Call, Call]] = {
    "sum": (lambda x: ak.sum(x["a"], axis=1), lambda x: gak.sum(x["a"], axis=1)),
    "sum-array-keyword": (
        lambda x: ak.sum(array=x["a"], axis=1),
        lambda x: gak.sum(x["a"], axis=1),
    ),
    "mean-positional-weight": (
        lambda x: ak.mean(x["a"], x["w"], axis=1),
        lambda x: gak.mean(x["a"], axis=1, weight=x["w"]),
    ),
    "std-positional-ddof": (
        lambda x: ak.std(x["a"], x["w"], 1, axis=1),
        lambda x: gak.std(x["a"], axis=1, ddof=1, weight=x["w"]),
    ),
    "mask-keyword": (
        lambda x: ak.mask(x["a"], mask=x["a"] > 1),
        lambda x: gak.mask(x["a"], x["a"] > 1),
    ),
    "full_like-keyword": (
        lambda x: ak.full_like(x["a"], fill_value=3),
        lambda x: gak.full_like(x["a"], 3),
    ),
    "values_astype-keyword": (
        lambda x: ak.values_astype(x["a"], to="float32"),
        lambda x: gak.values_astype(x["a"], "float32"),
    ),
    "with_field-keywords": (
        lambda x: ak.with_field(x["r"], what=x["r"].pt, where="x"),
        lambda x: gak.with_field(x["r"], x["r"].pt, "x"),
    ),
    "without_field-keyword": (
        lambda x: ak.without_field(x["r"], where="pt"),
        lambda x: gak.without_field(x["r"], "pt"),
    ),
    "with_parameter-keyword": (
        lambda x: ak.with_parameter(x["r"], parameter="k", value="v"),
        lambda x: gak.with_parameter(x["r"], "k", "v"),
    ),
    "isclose-keyword": (
        lambda x: ak.isclose(x["a"], b=x["w"]),
        lambda x: gak.isclose(x["a"], x["w"]),
    ),
    "where-varargs": (
        lambda x: ak.where(x["a"] > 1, x["a"], -1.0),
        lambda x: gak.where(x["a"] > 1, x["a"], -1.0),
    ),
    "zip": (
        lambda x: ak.zip({"p": x["a"], "q": x["w"]}, depth_limit=1),
        lambda x: gak.zip({"p": x["a"], "q": x["w"]}, depth_limit=1),
    ),
    "softmax-default": (lambda x: ak.softmax(x["n3"]), lambda x: gak.softmax(x["n3"], axis=-1)),
    "combinations": (
        lambda x: ak.combinations(x["a"], 2, fields=["p", "q"]),
        lambda x: gak.combinations(x["a"], 2, fields=["p", "q"]),
    ),
}


@pytest.mark.parametrize("name", PAIRS)
def test_ak_call_records_the_gak_node(arrays: dict[str, Any], name: str) -> None:
    via_ak, via_gak = PAIRS[name]
    out = via_ak(arrays)
    count = arrays["s"].node_count()
    expected = via_gak(arrays)
    assert type(out) is AwkwardArray
    assert out.node_id == expected.node_id
    assert arrays["s"].node_count() == count  # gak found the node ak recorded: nothing new


def test_tuple_returning_and_eager_metadata_match_gak(arrays: dict[str, Any]) -> None:
    pair = ak.broadcast_arrays(arrays["a"], arrays["w"])
    assert [p.node_id for p in pair] == [
        p.node_id for p in gak.broadcast_arrays(arrays["a"], arrays["w"])
    ]
    assert ak.fields(arrays["r"]) == gak.fields(arrays["r"]) == ["pt", "eta"]


def test_provenance_is_the_user_line(arrays: dict[str, Any]) -> None:
    out = ak.prod(arrays["a"], axis=1)
    prov = arrays["s"].provenance(out)
    assert (prov.filename, prov.source[:8]) == (__file__, "ak.prod(")


def test_options_gak_lacks_are_refused_like_gak(arrays: dict[str, Any]) -> None:
    with pytest.raises(TypeError, match="highlevel"):
        ak.sum(arrays["a"], axis=1, highlevel=False)
    with pytest.raises(TypeError, match="where"):  # gak requires the name ak defaults to None
        ak.with_field(arrays["r"], arrays["r"].pt)


@pytest.mark.parametrize(
    ("spelled", "call"),
    [
        ("ak.to_numpy", lambda a: ak.to_numpy(a)),
        ("ak.to_arrow_table", lambda a: ak.to_arrow_table(a)),
        (
            "ak.str.join",
            lambda a: ak.str.join(a, "-"),
        ),  # named "join" like gak.join, a different function
    ],
)
def test_functions_without_a_gak_equivalent_refuse(
    arrays: dict[str, Any], spelled: str, call: Callable[[Any], Any]
) -> None:
    count = arrays["s"].node_count()
    with pytest.raises(
        GraphedError,
        match=rf"^graphed records no deferred {re.escape(spelled)}: .*gak has no equivalent",
    ):
        call(arrays["a"])
    assert arrays["s"].node_count() == count


def test_varied_dispatches_like_gak(arrays: dict[str, Any]) -> None:
    a = arrays["a"]
    v: Any = graphed.vary(a, "jes", up=a * 1.1, down=a * 0.9)
    out, expected = ak.sum(v, axis=1), gak.sum(v, axis=1)
    assert type(v) is AwkwardVaried
    assert type(out) is AwkwardVaried
    labels = ("nominal", "jes_up", "jes_down")
    assert [graphed.member_of(out, k).node_id for k in labels] == [
        graphed.member_of(expected, k).node_id for k in labels
    ]


@pytest.mark.parametrize("cls", [AwkwardArray, AwkwardVaried])
def test_the_awkward_proxy_adds_no_member_functions(cls: type) -> None:
    assert "__awkward_function__" in vars(cls)
    assert [name for name in vars(cls) if not name.startswith("_")] == []
