"""§2.2 `Varied.apply` — a record-time `Array -> Array` applied per universe.

Named `apply`, not `map`: `Array.map` is an EXECUTION-time data callable and the two contracts must
not share a name.
"""

from __future__ import annotations

import pytest
from vary_fixtures import loose_varied, vector_source

import graphed
from graphed import Array, GraphedError


def test_apply_maps_the_function_over_every_universe() -> None:
    _s, x = vector_source()
    varied = loose_varied(x)
    applied = varied.apply(lambda member: member * 2.0)
    assert list(graphed.labels(applied)) == list(graphed.labels(varied))
    for label in graphed.labels(varied):
        member = graphed.universe(varied, label)
        # node-id equality, sound by interning: an implementation that applies `fn` to nominal
        # alone (or returns its argument) leaves the non-nominal members at the wrong id.
        assert graphed.universe(applied, label).node_id == (member * 2.0).node_id
    assert graphed.universe(applied, "jes_up").node_id != graphed.universe(applied, "nominal").node_id


def test_apply_returns_the_nominal_members_idiom() -> None:
    _s, x = vector_source()
    applied = loose_varied(x).apply(lambda member: member + 1.0)
    assert type(graphed.nominal(applied)) is type(x)
    assert isinstance(graphed.nominal(applied), Array)


def test_a_function_returning_a_varied_is_refused_with_guidance() -> None:
    """The error contract: containers are combined by ORDINARY ops (§2.4's label-aligned union),
    never by returning one out of `apply`, which has no label alignment to apply."""
    _s, x = vector_source()
    other = loose_varied(x, "jer")  # the container `fn` closed over
    with pytest.raises(GraphedError, match="ordinary ops"):
        loose_varied(x).apply(lambda member: other)
