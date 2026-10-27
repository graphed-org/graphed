"""``StageError`` compares by its fields, and leaves a comparison with anything else to Python."""

from __future__ import annotations

from graphed.debug import SourceFrame, StageError


def _error(op: str) -> StageError:
    return StageError(
        op=op,
        frames=(SourceFrame("analysis.py", 3),),
        input_forms=("var * float64",),
        partition="p0",
        cause_type="ValueError",
        cause_message="boom",
        opt_level=1,
    )


def test_equal_fields_compare_equal_and_other_types_defer() -> None:
    assert _error("add") == _error("add")
    assert _error("add") != _error("mul")
    assert _error("add") != "add"
