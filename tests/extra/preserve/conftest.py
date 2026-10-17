"""torch and xgboost each vendor an OpenMP runtime; allow both to load in one process (as m26 does).

Every test here starts and ends with the built-in plugin registry: a frozen preserve test that ran
earlier in the process may have replaced a built-in kind (m9 registers its own ``triton_model``)."""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

import graphed.preserve  # noqa: F401  (registers the built-ins before the snapshot)
from graphed.preserve.externals import _base

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# taken at collection, before any test runs, so it holds the built-ins only
_BUILTINS = dict(_base._REGISTRY)


def _restore_builtins() -> None:
    _base._REGISTRY.clear()
    _base._REGISTRY.update(_BUILTINS)


@pytest.fixture(autouse=True)
def _builtin_plugins() -> Iterator[None]:
    _restore_builtins()
    yield
    _restore_builtins()
