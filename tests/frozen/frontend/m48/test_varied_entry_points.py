"""§2.3b: the plain-`Array` entry points learn `Varied`.

The corpus requires it — unvaried photons/muons sliced by a JES-varied selection. Today
`Array.__getitem__` reaches its final `TypeError` on a `Varied` mask and `Array.filter` has no
runtime check at all, letting the container fall through into `record_op` and die on `.node_id`;
both wrong shapes are what these assertions reject.
"""

from __future__ import annotations

import pytest
from vary_fixtures import loose_varied, vector_source

import graphed


def _varied_mask() -> tuple[graphed.Session, graphed.Array, graphed.Varied]:
    session, x = vector_source()
    return session, x, loose_varied(x) > 3.0


def test_getitem_with_a_varied_mask_returns_a_varied_carrying_the_masks_labels() -> None:
    _s, x, mask = _varied_mask()
    sliced = x[mask]
    assert isinstance(sliced, graphed.Varied)
    assert list(graphed.labels(sliced)) == list(graphed.labels(mask))
    for label in graphed.labels(mask):
        assert graphed.universe(sliced, label).node_id == x[graphed.universe(mask, label)].node_id


def test_filter_with_a_varied_mask_returns_a_varied_carrying_the_masks_labels() -> None:
    _s, x, mask = _varied_mask()
    filtered = x.filter(mask)
    assert isinstance(filtered, graphed.Varied)
    assert list(graphed.labels(filtered)) == list(graphed.labels(mask))
    for label in graphed.labels(mask):
        assert graphed.universe(filtered, label).node_id == x.filter(graphed.universe(mask, label)).node_id


def test_the_varied_branch_is_not_a_blanket_except() -> None:
    """One `__getitem__` must both accept the container and keep refusing a genuinely unsupported
    index — a `try: delegate except: raise` rewrite satisfies only half of this."""
    _s, x, mask = _varied_mask()
    assert isinstance(x[mask], graphed.Varied)
    with pytest.raises(TypeError, match="unsupported index"):
        _ = x[1.5]
    with pytest.raises(TypeError, match="unsupported index"):
        _ = x[object()]
