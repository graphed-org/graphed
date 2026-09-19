"""m58 integ-m58-E1/E2/E3 — an External's stand-in is its DECLARED form.

Projection replays the graph on typetracers. Where an External node is, replay continues with a
stand-in: a typetracer of the node's RECORDED form (the one `record_external(descriptor=, form=)`
declared), carrying the session backend's behavior — not the first input, whose type the node's
declaration may deliberately differ from. Its inputs stay reported as fully read.
"""

from __future__ import annotations

import pytest
from m58_declaration_fixtures import (
    BEHAVIOR,
    NESTED,
    PAIRS,
    declared_external,
    external_session,
    form_of,
)

import graphed.awkward as ga
from graphed import BufferNeed, GraphedTypeError
from graphed.awkward import gak


def test_an_external_declared_one_level_deeper_projects_on_its_own_form() -> None:
    session, root = external_session()
    deeper = declared_external(session, [root.x], form_of(NESTED), "deeper")
    assert session.form(deeper).describe() == "## * var * var * float64"

    with pytest.raises(GraphedTypeError):  # the op is well-typed on the DECLARATION alone
        gak.num(root.x, axis=2)
    counts = gak.num(deeper, axis=2)
    assert session.form(counts).describe() == "## * var * int64"

    assert ga.project(counts, on_fail="pass").read_columns == {"events": frozenset({"x"})}
    buffers = ga.project_buffers(counts, on_fail="pass").read_buffers
    assert buffers == {"events": {"x": BufferNeed.DATA}}  # the External's input, fully read


def test_a_shape_preserving_external_reports_exactly_what_it_reported_before() -> None:
    session, root = external_session()
    same = declared_external(session, [root.x], session.form(root.x), "same")
    assert session.form(same).describe() == "## * var * float64"
    output = gak.num(same, axis=1) + gak.num(root.y, axis=1)

    assert ga.project(output, on_fail="pass").read_columns == {"events": frozenset({"x"})}
    buffers = ga.project_buffers(output, on_fail="pass").read_buffers
    assert buffers == {"events": {"x": BufferNeed.DATA, "y": BufferNeed.OFFSETS}}


def test_the_stand_in_carries_the_session_backends_behavior() -> None:
    session, root = external_session(behavior=BEHAVIOR)
    pair = declared_external(session, [root.z], form_of(PAIRS), "pair")
    assert session.form(pair).describe() == "## * m58pair[a: float64, b: float64]"

    total = pair.total  # a behavior PROPERTY of the declared form, not a field of the input
    assert session.form(total).describe() == "## * float64"

    assert ga.project(total, on_fail="pass").read_columns == {"events": frozenset({"z"})}
    buffers = ga.project_buffers(total, on_fail="pass").read_buffers
    assert buffers == {"events": {"z": BufferNeed.DATA}}
