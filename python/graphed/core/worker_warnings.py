"""A task's warnings, carried to the driver in its partial when it ran in another process.

A worker process has no channel to the driver that every runner shares, so the partial is it. Only
a task that caught something carries one, and ``finalize`` reports what the run caught on the
driver, once per message.
"""

from __future__ import annotations

import contextlib
import os
import threading
import warnings
from collections.abc import Callable, Iterator, Mapping
from contextvars import ContextVar
from dataclasses import dataclass, replace
from typing import Any, TextIO

_Caught = dict[tuple[type[Warning], str], tuple[str, int]]

_sink: ContextVar[_Caught | None] = ContextVar("graphed_task_warnings", default=None)
_install_lock = threading.Lock()
_shown: Callable[..., None] = warnings.showwarning
_driver_pid: int | None = None


def mark_driver() -> None:
    """Record this process as a driver: one that builds plans, where a warning is already home."""
    global _driver_pid
    _driver_pid = os.getpid()


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
    """Collect this task's warnings, unless it runs in a driver.

    ``warnings.showwarning`` is process-wide, so the task is told from the others by a context
    variable: ``catch_warnings`` is not thread-safe.
    """
    caught: _Caught = {}
    if os.getpid() == _driver_pid:
        yield caught
        return
    _install()
    token = _sink.set(caught)
    try:
        yield caught
    finally:
        _sink.reset(token)


@dataclass(frozen=True)
class Warned:
    """A partial and the warnings it carries, each with the first chunk that raised it."""

    value: Any
    caught: Mapping[tuple[type[Warning], str], tuple[str, int, str]]

    def emit(self) -> Any:
        """Issue the warnings here and return the plain partial."""
        for (category, message), (filename, lineno, chunk) in self.caught.items():
            warnings.warn_explicit(f"{message} [first in {chunk}]", category, filename, lineno)
        return self.value


def carrying(value: Any, caught: _Caught, chunk: str) -> Any:
    """Return ``value``, carrying ``caught`` if there is any."""
    if not caught:
        return value
    return Warned(value, {key: (*where, chunk) for key, where in caught.items()})


def without_warnings(value: Any) -> Any:
    """Return ``value`` without the warnings it carries, which are issued here."""
    return value.emit() if isinstance(value, Warned) else value


@dataclass(frozen=True)
class WarnedCombine:
    """A plan's ``combine`` over partials that may carry warnings; the left chunk's are kept."""

    fn: Callable[[Any, Any], Any]

    def __call__(self, a: Any, b: Any) -> Any:
        """Combine ``a`` and ``b``, and what they carry."""
        if not (isinstance(a, Warned) or isinstance(b, Warned)):
            return self.fn(a, b)
        left, right = (x.caught if isinstance(x, Warned) else {} for x in (a, b))
        return Warned(
            self.fn(*(x.value if isinstance(x, Warned) else x for x in (a, b))),
            {**right, **left},
        )


def resolving(resolve: Callable[[Any], Any], value: Any) -> Any:
    """``resolve`` over ``value``'s partial, its warnings kept."""
    if isinstance(value, Warned):
        return replace(value, value=resolve(value.value))
    return resolve(value)


@dataclass(frozen=True)
class WarnedFinalize:
    """A plan's ``finalize``, issuing what the run's partial carries first."""

    fn: Callable[[Any], Any] | None = None

    def __call__(self, value: Any) -> Any:
        """Issue what ``value`` carries, then finalize the plain value."""
        value = without_warnings(value)
        return value if self.fn is None else self.fn(value)
