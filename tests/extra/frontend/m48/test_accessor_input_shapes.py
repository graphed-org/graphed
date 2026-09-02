"""§2.2's remaining `labels`/`universe`/`nominal` input shapes, and §9.1's error branches.

The frozen m48 tree reaches the `Varied` and event-context shapes; the `{label: hist}` RESULT
MAPPING and the bare histogram are `graphed-histogram`'s anchors, in a different distribution, and
this tree must stay importable under `pytest hypothesis numpy` alone — so the histogram here is a
duck-typed stand-in on the one attribute §2.2 keys on. Awkward-free, like the frozen tree beside it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import graphed
from graphed import GraphedError, Session
from graphed.numpy import NumpyBackend, from_array


class _Hist:
    """A bare unvaried histogram: duck-typed on `axes`, which is what §2.2 reads."""

    axes = ()


def _varied() -> tuple[Session, Any, Any]:
    session = Session(NumpyBackend())
    x = from_array(session, "x", np.arange(1.0, 5.0))
    return session, x, graphed.vary(x, "jes", up=x * 1.5)


def test_a_result_mapping_reads_as_its_keys_with_nominal_first() -> None:
    result = {"jes_up": _Hist(), "nominal": _Hist()}
    assert graphed.labels(result) == ("nominal", "jes_up")
    assert graphed.universe(result, "jes_up") is result["jes_up"]
    assert graphed.nominal(result) is result["nominal"]


def test_a_mapping_without_nominal_is_not_given_one() -> None:
    """§2.2 binds both verbs to the same input shapes, so they must not disagree about which
    labels exist: seating `"nominal"` unconditionally claims a label `universe` refuses."""
    result = {"jes_up": _Hist(), "jes_down": _Hist()}
    assert graphed.labels(result) == ("jes_up", "jes_down")
    with pytest.raises(KeyError, match="nominal"):
        graphed.nominal(result)


def test_an_unknown_label_on_a_result_mapping_lists_the_valid_ones() -> None:
    result = {"nominal": _Hist()}
    with pytest.raises(KeyError, match="jer_up"):
        graphed.universe(result, "jer_up")


def test_a_bare_histogram_reads_as_the_single_label_nominal() -> None:
    hist = _Hist()
    assert graphed.labels(hist) == ("nominal",)
    assert graphed.universe(hist, "nominal") is hist
    assert graphed.nominal(hist) is hist
    with pytest.raises(KeyError, match="jes_up"):
        graphed.universe(hist, "jes_up")


def test_a_plain_Array_is_refused_by_both_verbs_rather_than_read_as_one_universe() -> None:
    """Answering `("nominal",)` here would let a DROPPED container read as a legal single
    universe — the §2.5 silent drop this refusal exists to prevent."""
    _s, x, _v = _varied()
    with pytest.raises(GraphedError, match="carries no variations"):
        graphed.labels(x)
    with pytest.raises(GraphedError, match="carries no variations"):
        graphed.universe(x, "nominal")


def test_an_unreadable_shape_is_named_in_the_error() -> None:
    with pytest.raises(GraphedError, match="int"):
        graphed.labels(3)
    with pytest.raises(GraphedError, match="int"):
        graphed.universe(3, "nominal")


def test_weight_refuses_anything_that_is_not_an_event_context() -> None:
    _s, x, varied = _varied()
    for value in (x, varied, None):
        with pytest.raises(GraphedError, match="ambient weight registry"):
            graphed.weight(value)


def test_reindex_to_a_context_free_target_refuses_naming_the_handle() -> None:
    """A context-free target has no row space to re-index INTO, so the identity would silently
    keep the value in the source context's rows."""
    session = Session(NumpyBackend())
    x = from_array(session, "x", np.arange(1.0, 5.0))
    assert graphed.reindex_to(x, None) is x  # context-free value: identity, not an error
    stamped = graphed.accessors.with_context(x, object())
    with pytest.raises(GraphedError, match="context-free target"):
        graphed.reindex_to(stamped, None)


def test_broadcast_like_is_the_bound_no_op_for_a_backend_without_the_seam() -> None:
    """numpy is rectilinear, so its backend supplies no `broadcast_like` and a genuine shape
    mismatch surfaces at execution rather than being papered over at record time."""
    session, x, varied = _varied()
    factor = x * 2.0
    assert not hasattr(session.backend, "broadcast_like")  # the control for "bound no-op"
    assert graphed.broadcast_like(x, factor) is factor

    spread = graphed.broadcast_like(varied, factor)
    assert isinstance(spread, graphed.Varied)
    assert list(graphed.labels(spread)) == ["nominal", "jes_up"]
    assert graphed.nominal(spread).node_id == factor.node_id
