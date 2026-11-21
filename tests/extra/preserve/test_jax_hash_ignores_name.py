"""A JAX export's content hash is its computation, not the exported function's name (#225)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from graphed.preserve.externals import JAX_PLUGIN

jax = pytest.importorskip("jax")
from jax import export  # noqa: E402


def _export(fn: Callable[[Any], Any]) -> bytes:
    spec = jax.ShapeDtypeStruct(export.symbolic_shape("b, 1"), jax.numpy.float32)
    return bytes(export.export(jax.jit(fn))(spec).serialize())


def f(x: Any) -> Any:
    return jax.nn.sigmoid(0.5 * x[:, 0] + 0.1)


def renamed(x: Any) -> Any:
    return jax.nn.sigmoid(0.5 * x[:, 0] + 0.1)


def other(x: Any) -> Any:
    return jax.nn.sigmoid(0.25 * x[:, 0] + 0.1)


def test_a_renamed_or_lambda_export_hashes_as_the_same_model() -> None:
    lam = lambda x: jax.nn.sigmoid(0.5 * x[:, 0] + 0.1)  # noqa: E731
    hashes = {JAX_PLUGIN.content_hash(_export(fn)) for fn in (f, renamed, lam)}
    assert len(hashes) == 1
    assert JAX_PLUGIN.content_hash(_export(other)) not in hashes


def _helper_a(v: Any) -> Any:
    return jax.nn.sigmoid(v)


def _helper_b(v: Any) -> Any:
    return jax.nn.sigmoid(v)


def _scale(v: Any) -> Any:
    return v * 3.0


def _scale_more(v: Any) -> Any:
    return v * 4.0


def test_a_renamed_nested_function_hashes_as_the_same_model() -> None:
    helper_a, helper_b, scale, scale_more = map(
        jax.jit, (_helper_a, _helper_b, _scale, _scale_more)
    )

    def with_a(x: Any) -> Any:
        return scale(helper_a(0.5 * x[:, 0]))

    def with_b(x: Any) -> Any:
        return scale(helper_b(0.5 * x[:, 0]))

    def with_more(x: Any) -> Any:
        return scale_more(helper_a(0.5 * x[:, 0]))

    a, b, more = (JAX_PLUGIN.content_hash(_export(fn)) for fn in (with_a, with_b, with_more))
    assert a == b != more
