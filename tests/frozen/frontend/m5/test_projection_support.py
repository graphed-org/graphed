"""Frontend M5 support: the generic graph walk, source helpers, Projection type, on-fail policy."""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence

import pytest
from graphed_core import PayloadDescriptor

from graphed import (
    CONSERVATIVE,
    Array,
    Projection,
    ProjectionError,
    Session,
    handle_opaque,
)


class _Backend:
    def op_form(self, op: str, inputs: Sequence[object], params: Mapping[str, object]) -> str:
        return op

    def eval_stage(self, op: str, inputs: Sequence[object], params: Mapping[str, object]) -> object:
        if op == "add":
            return inputs[0] + inputs[1]  # type: ignore[operator]
        if op == "inc":
            return inputs[0] + 1  # type: ignore[operator]
        return inputs[0]

    def boundary_ops(self) -> frozenset[str]:
        return frozenset({"source"})

    def project(self, op: str, used: object, params: Mapping[str, object]) -> object:
        return used

    def external_payload(self, op: str, params: Mapping[str, object]) -> PayloadDescriptor | None:
        return PayloadDescriptor(
            kind="x", content_hash="h", framework="f", version="v", io_schema="s", preprocessing_ref=None
        )


def _src(s: Session, name: str, value: object) -> Array:
    return s.source(name, form="f", data=value)


def test_walk_evaluates_like_materialize() -> None:
    s = Session(_Backend())
    a = _src(s, "a", 3)
    b = _src(s, "b", 5)
    out = (a + b).reduce("inc")
    # materialize uses walk under the hood
    assert s.materialize(out) == 9  # (3+5)+1


def test_walk_with_custom_handlers_counts_sources() -> None:
    s = Session(_Backend())
    a = _src(s, "a", 1)
    b = _src(s, "b", 2)
    out = a + b
    seen: set[int] = set()
    s.walk(
        out,
        source=lambda nid: seen.add(nid) or nid,
        op=lambda nid, name, ins, params: nid,
        external=lambda nid, fn, ins: nid,
    )
    assert seen == set(s.source_ids())


def test_source_helpers() -> None:
    s = Session(_Backend())
    a = _src(s, "events", 1)
    assert s.source_ids() == [a.node_id]
    assert s.source_name(a.node_id) == "events"
    assert s.form_of(a.node_id) == "f"


def test_source_value_resolves_eager_and_lazy() -> None:
    s = Session(_Backend())
    eager = _src(s, "a", 5)
    lazy = s.source("b", form="f", data=lambda: 7)  # a lazy loader (resolved on access)
    assert s.source_value(eager.node_id) == 5
    assert s.source_value(lazy.node_id) == 7


def test_projection_type() -> None:
    p = Projection({"events": frozenset({"pt", "eta"}), "other": frozenset()})
    assert p.columns_for("events") == frozenset({"pt", "eta"})
    assert p.columns_for("missing") == frozenset()
    assert p.total_columns() == 2


def test_on_fail_policy_raise() -> None:
    with pytest.raises(ProjectionError):
        handle_opaque("map", "raise")


def test_on_fail_policy_warn() -> None:
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = handle_opaque("map", "warn")
    assert result is CONSERVATIVE
    assert len(w) == 1


def test_on_fail_policy_pass() -> None:
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = handle_opaque("map", "pass")
    assert result is None
    assert len(w) == 0


def test_invalid_on_fail_policy_rejected() -> None:
    with pytest.raises(ValueError):
        handle_opaque("map", "nonsense")
