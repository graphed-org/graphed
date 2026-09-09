"""The correctionlib plugin evaluates identically-laid-out arguments on their flat buffers.

`correctionlib.highlevel._wrap_awkward` broadcasts and `ak.transform`s on every call; on option-typed
jagged inputs that wrapper costs more than the correction formula. When every array argument already
shares one layout the plugin projects once, evaluates the flat buffers and rebuilds the offsets/index
stack. The guard is structural (identical layer classes and identical offsets/index buffers), and the
contract is bit-identity with the wrapper: values, None mask and type.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import awkward as ak
import correctionlib
import numpy as np
import pytest

import graphed.preserve.externals.correctionlib_external as clx

_CSET_JSON = json.dumps(
    {
        "schema_version": 2,
        "corrections": [
            {
                "name": "sf",
                "version": 1,
                "inputs": [
                    {"name": "systematic", "type": "string"},
                    {"name": "x", "type": "real"},
                    {"name": "y", "type": "real"},
                ],
                "output": {"name": "sf", "type": "real"},
                "data": {
                    "nodetype": "category",
                    "input": "systematic",
                    "content": [
                        {
                            "key": "nominal",
                            "value": {
                                "nodetype": "formula",
                                "expression": "2.5*x + 0.125*y",
                                "parser": "TFormula",
                                "variables": ["x", "y"],
                            },
                        }
                    ],
                },
            }
        ],
    }
)
PARAMS = {"name": "sf", "args": json.dumps(["nominal", "$0", "$1"])}


@pytest.fixture(scope="module")
def cset() -> correctionlib.CorrectionSet:
    return correctionlib.CorrectionSet.from_string(_CSET_JSON)


def _evaluate(monkeypatch: pytest.MonkeyPatch, cset: object, inputs: list[ak.Array]) -> tuple[ak.Array, bool]:
    """`eval_correctionlib`'s answer, plus whether the fast path is the one that produced it."""
    engaged: list[bool] = []
    real = clx._flat_buffer_fast_path

    def spy(evaluate: Callable[..., Any], call: list[object]) -> object:
        out = real(evaluate, call)
        engaged.append(out is not None)
        return out

    monkeypatch.setattr(clx, "_flat_buffer_fast_path", spy)
    got = clx.eval_correctionlib(cset, PARAMS, inputs)
    assert len(engaged) == 1, "the plugin no longer consults the fast path exactly once"
    return got, engaged[0]


def _assert_identical(got: ak.Array, ref: ak.Array) -> None:
    assert str(ak.type(got)) == str(ak.type(ref))
    assert ak.to_list(got) == ak.to_list(ref)  # values AND the None pattern


def _wrapper_result(cset: object, inputs: list[ak.Array]) -> ak.Array:
    """What correctionlib's own awkward wrapper returns — the bit-identity reference."""
    return cset["sf"].evaluate("nominal", *inputs)  # type: ignore[index]


OPTION_JAGGED = [
    ak.Array([[1.0, None, 3.0], [], [4.0, 5.0], [None]]),
    ak.Array([[8.0, None, 6.0], [], [4.0, 2.0], [None]]),
]


def test_matched_option_layout_takes_the_fast_path(monkeypatch: pytest.MonkeyPatch, cset: object) -> None:
    got, engaged = _evaluate(monkeypatch, cset, OPTION_JAGGED)
    assert engaged, "option-typed jagged inputs sharing one layout must not reach the wrapper"
    _assert_identical(got, _wrapper_result(cset, OPTION_JAGGED))
    assert ak.to_list(got) == [[3.5, None, 8.25], [], [10.5, 12.75], [None]]


def test_differing_none_masks_fall_back(monkeypatch: pytest.MonkeyPatch, cset: object) -> None:
    """The discriminating case: same offsets, different index — projecting once would misalign x
    against y, so the guard must refuse it and the wrapper's broadcast must own the answer."""
    inputs = [
        ak.Array([[1.0, None, 3.0], [4.0]]),
        ak.Array([[1.0, 2.0, None], [4.0]]),
    ]
    got, engaged = _evaluate(monkeypatch, cset, inputs)
    assert not engaged, "inputs with different None masks must not share one projection"
    _assert_identical(got, _wrapper_result(cset, inputs))
    assert ak.to_list(got) == [[2.625, None, None], [10.5]]


def test_non_ascending_shared_index_rebuilds_in_position_order(
    monkeypatch: pytest.MonkeyPatch, cset: object
) -> None:
    """A member the guard admits and the naive rebuild would get wrong: an IndexedOptionArray whose
    valid entries do not appear in content order. Projection reorders, so the rebuilt index is
    renumbered by position rather than reusing the original."""
    index = np.array([2, -1, 0, 1], dtype=np.int64)
    offsets = np.array([0, 2, 4], dtype=np.int64)

    def shuffled(values: list[float]) -> ak.Array:
        opt = ak.contents.IndexedOptionArray(
            ak.index.Index64(index), ak.contents.NumpyArray(np.array(values))
        )
        return ak.Array(ak.contents.ListOffsetArray(ak.index.Index64(offsets), opt))

    inputs = [shuffled([1.0, 2.0, 3.0]), shuffled([8.0, 16.0, 24.0])]
    got, engaged = _evaluate(monkeypatch, cset, inputs)
    assert engaged
    _assert_identical(got, _wrapper_result(cset, inputs))
    assert ak.to_list(got) == [[10.5, None], [3.5, 7.0]]


@pytest.mark.parametrize(
    ("label", "inputs"),
    [
        ("plain jagged", [ak.Array([[1.0, 2.0], [3.0]]), ak.Array([[8.0, 16.0], [24.0]])]),
        ("flat", [ak.Array([1.0, 2.0, 3.0]), ak.Array([8.0, 16.0, 24.0])]),
        ("flat option", [ak.Array([1.0, None, 3.0]), ak.Array([8.0, None, 24.0])]),
    ],
)
def test_non_option_and_flat_inputs_are_unchanged(
    monkeypatch: pytest.MonkeyPatch, cset: object, label: str, inputs: list[ak.Array]
) -> None:
    got, _ = _evaluate(monkeypatch, cset, inputs)
    _assert_identical(got, _wrapper_result(cset, inputs))


def test_a_layout_outside_the_guard_falls_back(monkeypatch: pytest.MonkeyPatch, cset: object) -> None:
    """A member the class must refuse: a multidimensional NumpyArray has no flat buffer to evaluate,
    so the wrapper keeps the call."""
    inputs = [ak.Array(np.arange(6.0).reshape(3, 2)), ak.Array(np.arange(6.0).reshape(3, 2) * 8.0)]
    got, engaged = _evaluate(monkeypatch, cset, inputs)
    assert not engaged
    _assert_identical(got, _wrapper_result(cset, inputs))


def test_a_sliced_jagged_array_falls_back(monkeypatch: pytest.MonkeyPatch, cset: object) -> None:
    """Offsets that do not start at 0 leave content outside the listed range; evaluating the whole
    buffer would feed the correction entries no event asked for, so the guard refuses them."""
    inputs = [
        ak.Array([[1.0, 2.0], [3.0], [4.0, 5.0]])[1:],
        ak.Array([[8.0, 16.0], [24.0], [32.0, 40.0]])[1:],
    ]
    assert np.asarray(inputs[0].layout.offsets.data)[0] != 0
    got, engaged = _evaluate(monkeypatch, cset, inputs)
    assert not engaged
    _assert_identical(got, _wrapper_result(cset, inputs))


@pytest.mark.parametrize(
    ("label", "column"),
    [
        ("string", ak.Array([["a", "b"], ["c"]])),
        ("record", ak.Array([{"p": 1.0}, {"p": 2.0}])),
    ],
)
def test_non_numeric_columns_are_refused(cset: object, label: str, column: ak.Array) -> None:
    """The other end of the class: layouts whose innermost buffer is not a numeric column. A string
    peels to its `char` codes and a record has no single buffer at all — neither is evaluable."""
    assert clx._flat_buffer_fast_path(cset["sf"].evaluate, ["nominal", column, column]) is None  # type: ignore[index]


@pytest.mark.parametrize("order", [("short first", 0), ("long first", 1)])
def test_length_one_broadcasting_matches_the_wrapper(
    monkeypatch: pytest.MonkeyPatch, cset: object, order: tuple[str, int]
) -> None:
    """No layer buffer separates two flat arrays of different length; correctionlib broadcasts the
    length-1 one either way, so whichever path runs must land on the wrapper's answer."""
    short, long_ = ak.Array([8.0]), ak.Array([1.0, 2.0, 3.0])
    inputs = [short, long_] if order[1] == 0 else [long_, short]
    got, _ = _evaluate(monkeypatch, cset, inputs)
    _assert_identical(got, _wrapper_result(cset, inputs))
