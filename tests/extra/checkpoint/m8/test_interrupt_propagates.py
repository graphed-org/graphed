"""An interrupt or exit inside a task stops ``run_resumable``; a task error is dead-lettered."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import analyses
import pytest

from graphed.checkpoint import Store, run_resumable
from graphed.checkpoint import runner as runner_module


def _raising(exc: BaseException) -> Any:
    def run_one(*args: Any, **kwargs: Any) -> Any:
        raise exc

    return run_one


@pytest.mark.parametrize("exc", [KeyboardInterrupt(), SystemExit(3)], ids=type)
def test_an_interrupt_in_a_task_stops_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exc: BaseException
) -> None:
    store = Store(tmp_path)
    monkeypatch.setattr(runner_module, "_run_one", _raising(exc))
    with pytest.raises(type(exc)):
        run_resumable(analyses.build_plan("analyses:histogram_chunk", 2), store)
    assert store.dead_letters() == []


def test_a_task_error_is_still_dead_lettered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = Store(tmp_path)
    monkeypatch.setattr(runner_module, "_run_one", _raising(ValueError("bad chunk")))
    result = run_resumable(analyses.build_plan("analyses:histogram_chunk", 2), store)
    assert result.report.dead == 2
    assert len(store.dead_letters()) == 2
