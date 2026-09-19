"""m59 integ-m59-I3/I5 — what a tuple key refuses at record time, and what it defers to run time.

A key that would consume or restructure the partitioned axis is a record-time `TypeError` naming
the spelling that works (`a[1:3][:, 0]`); a member that is not a `slice`/`int`/`None`/`Ellipsis` is
refused whatever axis it sits on. A key that is legal but hits an empty row at run time is NOT a
record-time error: it reaches the executor and comes back as a `StageError` at the user's line.
"""

from __future__ import annotations

import inspect

import pytest
from m59_idiom_fixtures import (
    CONSUMES_AXIS0,
    MALFORMED,
    RAGGED,
    eager_form,
    session_over,
    tracer,
)

import graphed.debug as gd


@pytest.mark.parametrize("label", list(CONSUMES_AXIS0) + list(MALFORMED))
def test_a_refused_tuple_key_records_nothing(label: str) -> None:
    key = (CONSUMES_AXIS0 | MALFORMED)[label]
    session, events = session_over()
    jets = events.x  # recorded here, so `before` counts only what the refused key would add
    before = session.node_count()

    with pytest.raises(TypeError):
        jets[key]

    assert session.node_count() == before


def test_an_array_member_inside_a_tuple_key_is_refused() -> None:
    session, events = session_over()
    jets = events.x
    mask = jets > 1.0
    before = session.node_count()

    with pytest.raises(TypeError):
        jets[:, mask]

    assert session.node_count() == before


def test_consuming_the_partitioned_axis_is_refused_by_naming_the_chained_spelling() -> None:
    _session, events = session_over()

    with pytest.raises(TypeError) as info:
        events.x[1:3, 0]

    # the message hands the user the two-subscript spelling that does work: `a[1:3][:, 0]`
    assert "][" in str(info.value)


def test_an_index_past_an_empty_row_surfaces_as_the_executors_stage_error() -> None:
    session, ragged = session_over(RAGGED)

    first = ragged[:, 0]  # records: a typetracer cannot see that one row is empty

    assert session.form(first).describe() == eager_form(tracer(RAGGED)[:, 0])
    with pytest.raises(gd.StageError) as info:
        gd.run(session, first, opt_level=1)
    error = info.value
    frame = inspect.currentframe()
    assert error.cause_type == "IndexError"
    assert "[:, 0]" in error.user_frame.source
    assert frame is not None and error.user_frame.function == frame.f_code.co_name
