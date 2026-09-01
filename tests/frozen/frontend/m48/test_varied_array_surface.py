"""§2.3a: `Varied` carries the full `Array` PUBLIC surface, plus §2.2's property dispositions.

The inventory is enumerated DYNAMICALLY from `type(graphed.nominal(v))`, so an idiom subclass is
covered without a literal list, and each name is resolved ON THE CLASS: an instance-level
`hasattr` is unconditionally satisfied by `Varied`'s label-mapping field access. Carries §2.3c's
non-vacuity floor and §2.3e(4)'s own floor in the same test.
"""

from __future__ import annotations

import inspect

import pytest
from vary_fixtures import loose_varied, record_source, vector_source

import graphed
import graphed.numpy as gnp
from graphed import GraphedError

#: the freeze-time size of the discovered surface on the numpy idiom's `Array` class
SURFACE_FLOOR = 67
#: the freeze-time count of base-`Array` elementwise/structural dunders (a containment floor: the
#: broadcast class can only grow, and m51's added verbs must not shrink it)
BROADCAST_FLOOR = 36


def _discover(cls: type) -> set[str]:
    """§2.3a's bound discovery: public methods PLUS the dunder set, off the resolving class."""
    return {
        name
        for name, _ in inspect.getmembers(cls, inspect.isfunction)
        if not name.startswith("_") or (name.startswith("__") and name.endswith("__"))
    }


def _properties(cls: type) -> set[str]:
    return {
        name
        for name, _ in inspect.getmembers(cls, lambda m: isinstance(m, property))
        if not name.startswith("_")
    }


def _dispositions() -> dict[str, str]:
    """The `Array`-surface classification, read as the UNION over the packages that own it — the
    same per-package shape §2.3d's floor takes over its three enumerations."""
    return {**graphed.SURFACE_DISPOSITIONS, **gnp.SURFACE_DISPOSITIONS}


def test_every_discovered_surface_name_resolves_on_the_varied_class() -> None:
    _s, x = vector_source()
    varied = loose_varied(x)
    discovered = _discover(type(graphed.nominal(varied)))
    assert len(discovered) >= SURFACE_FLOOR
    assert {"__array_ufunc__", "__getitem__", "__and__", "filter"} <= discovered  # §2.3c's floor
    missing = sorted(name for name in discovered if getattr(type(varied), name, None) is None)
    assert not missing, f"{missing} are answered only by the field-access fallback, not implemented"


def test_class_resolution_is_what_makes_the_parity_gate_discriminating() -> None:
    """The instance never refuses: `Array.__getattr__` records a `field` op for any non-underscore
    name, so an instance-level inventory would report EVERY name as present."""
    _s, record = record_source()
    varied = graphed.vary(record, "sf", up=record, down=record)
    assert getattr(type(varied), "pt", None) is None
    assert list(graphed.labels(varied["pt"])) == list(graphed.labels(varied))  # field access answers


def test_surface_floor_names_repartition_as_the_only_refusing_member() -> None:
    dispositions = _dispositions()
    refusing = {name for name, kind in dispositions.items() if kind == "refusing"}
    broadcast = {name for name, kind in dispositions.items() if kind == "broadcast"}
    assert refusing == {"repartition"}  # §2.3e(4); a re-classification would hide a stub here
    assert len(broadcast) >= BROADCAST_FLOOR


def test_one_behavioural_probe_per_disposition_class() -> None:
    _s, x = vector_source()
    varied = loose_varied(x)
    broadcast = varied > 3.0
    assert isinstance(broadcast, graphed.Varied)
    assert list(graphed.labels(broadcast)) == list(graphed.labels(varied))
    assert graphed.universe(broadcast, "jes_up").node_id == (graphed.universe(varied, "jes_up") > 3.0).node_id
    with pytest.raises(GraphedError):
        varied.repartition(n=2)  # refusing (§5.4); NOT `TypeError: not callable`


def test_properties_are_classified_by_measurement_not_by_a_literal_list() -> None:
    """§2.2's three-class rule. The discriminator between the eager and the recording class is the
    `Session.node_count()` delta of the access on the PLAIN nominal `Array`, so a blanket "delta 0
    for every property" reds a correct implementation on `T`."""
    session, x = vector_source()
    varied = loose_varied(x)
    discovered = _properties(type(graphed.nominal(varied)))
    assert {"node_id", "session", "dtype", "T"} <= discovered
    eager, recording = [], []
    for name in sorted(discovered - {"node_id", "session"}):
        before = session.node_count()
        getattr(x, name)
        (recording if session.node_count() > before else eager).append(name)
    assert eager and recording, "both property classes must be represented, or the split is untested"

    for name in eager:
        before = session.node_count()
        answer = getattr(varied, name)
        assert session.node_count() == before  # answered on the nominal member, nothing recorded
        assert answer == getattr(graphed.nominal(varied), name)
    for name in recording:
        answer = getattr(varied, name)
        assert isinstance(answer, graphed.Varied)  # takes its underlying METHOD's disposition
        assert list(graphed.labels(answer)) == list(graphed.labels(varied))


def test_node_id_and_session_raise_while_the_field_of_that_name_still_reads() -> None:
    """The reserved-name rule is a PROPERTY rule: `compile_ir` reads `arr.node_id` per output, so a
    recorded `field` op named "node_id" would silently compile. A blanket `__getattr__` refusal
    would also eat a tree branch of that name, which the record fixture carries literally."""
    _s, x = vector_source()
    varied = loose_varied(x)
    for name in ("node_id", "session"):
        with pytest.raises(AttributeError):
            getattr(varied, name)

    _rs, record = record_source()
    varied_record = graphed.vary(record, "sf", up=record, down=record)
    field = varied_record["node_id"]
    assert isinstance(field, graphed.Varied)
    nominal_field = graphed.nominal(field)
    assert nominal_field.node_id == record["node_id"].node_id
