"""``RetryElsewhere`` re-raises the last failure once every fresh context has failed too."""

from __future__ import annotations

from typing import Any

import pytest

from graphed.checkpoint import RetryElsewhere
from graphed.core import Partition


def test_the_last_failure_is_raised_after_every_attempt() -> None:
    seen: list[Any] = []

    def process(partition: Partition, resources: Any) -> Any:
        seen.append(resources)
        raise ValueError(f"attempt {len(seen)}")

    with pytest.raises(ValueError, match="attempt 2"):
        RetryElsewhere(attempts=2, new_resources=dict).recover(
            Partition("u", "t", 0, 1),
            RuntimeError("first"),
            process=process,
            combine=lambda a, b: a,
            resources={"bad": True},
        )
    assert seen == [{}, {}]
