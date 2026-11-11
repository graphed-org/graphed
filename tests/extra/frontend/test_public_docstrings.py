"""#156: public docstrings speak to users: no internal ids, no private names, runnable examples."""

from __future__ import annotations

import doctest
import importlib
import inspect
import re
import types
from collections.abc import Iterator
from typing import Any

import pytest

import graphed
import graphed.core
import graphed.debug as gd
from graphed.awkward import from_awkward, from_parquet, gnano, to_parquet

INTERNAL_ID = re.compile(
    r"§\s?[\dA-Z]|\bplan [A-F]\.\d|\(plan M\d|\bM\d{1,2}\b|\bm\d{2}\b|\bR\d{1,2}\.\d"
)
PRIVATE_NAME = re.compile(r"(?<![\w./])_[a-z]\w*")
PACKAGES = (
    "graphed",
    "graphed.core",
    "graphed.awkward",
    "graphed.numpy",
    "graphed.debug",
    "graphed.checkpoint",
    "graphed.preserve",
)
ENTRY_POINTS: dict[str, Any] = {
    "Session": graphed.Session,
    "materialize": graphed.materialize,
    "aggregate_plan": graphed.aggregate_plan,
    "vary": graphed.vary,
    "collate": graphed.collate,
    "apply": graphed.apply,
    "join": graphed.join,
    "repartition": graphed.repartition,
    "current_session": graphed.current_session,
    "new_session": graphed.new_session,
    "from_parquet": from_parquet,
    "from_awkward": from_awkward,
    "to_parquet": to_parquet,
    "gnano.events": gnano.events,
    "format_traceback": gd.format_traceback,
    "gd.run": gd.run,
    "gd.replay": gd.replay,
    "SequentialRunner": graphed.core.SequentialRunner,
}


def _public() -> Iterator[tuple[str, object]]:
    """Each package's members and moved names, with each class's own public methods."""
    for name, member in _members():
        yield name, member
        if inspect.isclass(member) and member.__module__.startswith("graphed"):
            for key, value in vars(member).items():
                value = getattr(value, "__func__", value)
                value = value.fget if isinstance(value, property) else value
                if not key.startswith("_") and inspect.isfunction(value):
                    yield f"{name}.{key}", value


def _members() -> Iterator[tuple[str, object]]:
    """Each package, its `__all__` members, the members of a module there, and the moved names."""
    for package in PACKAGES:
        module = importlib.import_module(package)
        yield package, module
        for name in module.__all__:
            member = getattr(module, name)
            yield f"{package}.{name}", member
            if isinstance(member, types.ModuleType):
                for key, value in vars(member).items():
                    own = getattr(value, "__module__", None) == member.__name__
                    if not key.startswith("_") and own and callable(value):
                        yield f"{package}.{name}.{key}", value
    for name, home in graphed._MOVED.items():
        yield f"{home}.{name}", getattr(importlib.import_module(home), name)


def _offending(pattern: re.Pattern[str]) -> list[str]:
    return sorted(
        {
            name
            for name, member in _public()
            if (inspect.isclass(member) or callable(member) or inspect.ismodule(member))
            and pattern.search(member.__doc__ or "")
        }
    )


def test_the_population_is_the_public_surface() -> None:
    names = {name for name, _ in _public()}
    assert {
        "graphed.vary",
        "graphed.awkward.gak.max",
        "graphed.systematics.labels",
        "graphed.Session.materialize",
    } <= names


def test_no_public_docstring_cites_an_internal_id() -> None:
    assert _offending(INTERNAL_ID) == []


def test_no_public_docstring_names_a_private_helper() -> None:
    assert _offending(PRIVATE_NAME) == []


@pytest.mark.parametrize("name", sorted(ENTRY_POINTS))
def test_an_entry_point_carries_an_example_that_runs(name: str) -> None:
    obj = inspect.unwrap(ENTRY_POINTS[name])
    tests = doctest.DocTestFinder(recurse=False).find(obj, name, globs={})
    assert sum(len(test.examples) for test in tests) > 0
    runner = doctest.DocTestRunner(optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE)
    for test in tests:
        runner.run(test)
    assert runner.failures == 0
