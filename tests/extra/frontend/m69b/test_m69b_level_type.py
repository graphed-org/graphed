"""`aggregate_plan(opt_level=)` takes the ints 0 and 1 only; equal non-ints are refused."""

from __future__ import annotations

from typing import Any

import m69b_opt_fixtures as fx
import pytest

from graphed import aggregate_plan


@pytest.mark.parametrize("level", [True, False, 0.0, 1.0])
def test_a_level_equal_to_0_or_1_but_not_an_int_is_refused(level: Any) -> None:
    _s, events = fx.record()
    with pytest.raises(ValueError, match=r"is 0 .* or 1 "):
        aggregate_plan(
            *fx.merged_pair(events),
            reduce=fx.sums,
            combine=fx.add,
            empty=fx.zero,
            opt_level=level,
        )
