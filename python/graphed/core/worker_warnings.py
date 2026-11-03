"""A task's warnings, carried to the driver in its partial when it ran in another process.

A worker process has no channel to the driver that every runner shares, so the partial is it. Only
a task that caught something carries one, and ``finalize`` reports what the run caught on the
driver, once per message.
"""

from __future__ import annotations

import contextlib
import os
import pickle
import threading
import uuid
import warnings
from collections.abc import Callable, Iterator, Mapping
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, TextIO

_Caught = dict[tuple[type[Warning], str], tuple[str, int]]

_sink: ContextVar[_Caught | None] = ContextVar("graphed_task_warnings", default=None)
_install_lock = threading.Lock()
_shown: Callable[..., None] = warnings.showwarning
# the pid tells a forked child from its parent; the uuid, another host's process with the same pid
_interpreter = uuid.uuid4().hex
_worker_id: str | None = None


def process_id() -> str:
    """Return this process's identity, which a runner hands its workers as the driver's."""
    return f"{_interpreter}:{os.getpid()}"


def enter_worker(driver: str) -> None:
    """Mark this process a worker for the driver ``driver``, unless it is that driver.

    Only a marked worker keeps its tasks' warnings; a runner that never calls this leaves them
    where they are issued.
    """
    global _worker_id
    if driver != process_id():
        _worker_id = process_id()


def _carried(category: type[Warning]) -> type[Warning]:
    """``category``, or its builtin base when it would not survive the trip.

    A warning must not fail its task at the pickle, nor the run where it is issued from a message.
    """
    try:
        pickle.dumps(category)
        category("")
    except Exception:
        return next(k for k in category.__mro__ if k.__module__ == "builtins")
    return category


def _keep_or_show(
    message: Warning | str,
    category: type[Warning],
    filename: str,
    lineno: int,
    file: TextIO | None = None,
    line: str | None = None,
) -> None:
    sink = _sink.get()
    if sink is None:
        _shown(message, category, filename, lineno, file, line)
    else:
        sink.setdefault((category, str(message)), (filename, lineno))


def _install() -> None:
    global _shown
    with _install_lock:
        if warnings.showwarning is not _keep_or_show:
            _shown, warnings.showwarning = warnings.showwarning, _keep_or_show
            # the default action warns once per location per process, and workers outlive a run
            warnings.simplefilter("always", RuntimeWarning)


@contextlib.contextmanager
def keeping_warnings() -> Iterator[_Caught]:
    """Collect this task's warnings, when it runs in a worker of another process.

    ``warnings.showwarning`` is process-wide, so the task is told from the others by a context
    variable: ``catch_warnings`` is not thread-safe.
    """
    caught: _Caught = {}
    if _worker_id != process_id():
        yield caught
        return
    _install()
    token = _sink.set(caught)
    try:
        yield caught
    except BaseException as exc:
        # a failed task has no partial, but its failure reaches the driver
        for (category, message), (filename, lineno) in caught.items():
            exc.add_note(f"{category.__name__} before this: {message} ({filename}:{lineno})")
        raise
    finally:
        _sink.reset(token)


@dataclass(frozen=True)
class Warned:
    """A partial and the warnings it carries, each with the first chunk that raised it."""

    value: Any
    caught: Mapping[tuple[type[Warning], str], tuple[str, int, str]]


def carrying(value: Any, caught: _Caught, chunk: str) -> Any:
    """Return ``value``, carrying ``caught`` if there is any."""
    if not caught:
        return value
    return Warned(
        value,
        {(_carried(kind), message): (*where, chunk) for (kind, message), where in caught.items()},
    )


def _split(value: Any) -> tuple[Any, Mapping[Any, Any]]:
    return (value.value, value.caught) if isinstance(value, Warned) else (value, {})


def without_warnings(value: Any) -> Any:
    """Return ``value`` without the warnings it carries, which are issued here."""
    plain, caught = _split(value)
    for (category, message), (filename, lineno, chunk) in caught.items():
        warnings.warn_explicit(f"{message} [first in {chunk}]", category, filename, lineno)
    return plain


@dataclass(frozen=True)
class WarnedCombine:
    """A plan's ``combine`` over partials that may carry warnings; the left chunk's are kept."""

    fn: Callable[[Any, Any], Any]

    def __call__(self, a: Any, b: Any) -> Any:
        """Combine ``a`` and ``b``, and what they carry."""
        (left, left_caught), (right, right_caught) = _split(a), _split(b)
        caught, value = {**right_caught, **left_caught}, self.fn(left, right)
        return Warned(value, caught) if caught else value


def on_value(fn: Callable[[Any], Any], value: Any) -> Any:
    """``fn`` over ``value``'s partial, its warnings kept on the outside."""
    plain, caught = _split(value)
    return Warned(fn(plain), caught) if caught else fn(plain)


@dataclass(frozen=True)
class WarnedFinalize:
    """A plan's ``finalize``, issuing what the run's partial carries first."""

    fn: Callable[[Any], Any] | None

    def __call__(self, value: Any) -> Any:
        """Issue what ``value`` carries, then finalize the plain value."""
        value = without_warnings(value)
        return value if self.fn is None else self.fn(value)
