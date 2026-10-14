"""``AwkwardArray``: the awkward-idiom proxy that lets ``ak.<fn>(deferred)`` record ``gak.<fn>``.

awkward's high-level functions offer every array-like argument an ``__awkward_function__`` hook
(``awkward._dispatch.named_high_level_function``). This class answers it by calling the ``gak``
function of the same name, so ``ak.sum(a, axis=1)`` records exactly what ``gak.sum(a, axis=1)``
records, refusals included. ``AwkwardVaried`` carries the same hook for containers of universes.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

import awkward as ak

from graphed import Array
from graphed.errors import GraphedError
from graphed.provenance import register_internal
from graphed.varied import Varied, register_varied

from . import functions as gak

# a node recorded through ak.<fn> points at the user's line, not at awkward's dispatcher
register_internal("awkward._dispatch")

#: ak keyword names whose gak parameter is spelled differently (a gak name of its own wins)
_RENAMED = {
    "b": "y",
    "fill_value": "value",
    "mask": "condition",
    "parameter": "key",
    "to": "dtype",
    "what": "value",
    "where": "field",
}


def _awkward_function(
    self: object, dispatch: Callable[..., Any], array_likes: object, args: Any, kwargs: Any
) -> Any:
    name = dispatch.__name__
    # identity, not the name alone: ak.str.join is also named "join"
    if getattr(ak, name, None) is not dispatch or not hasattr(gak, name):
        raise GraphedError(
            f"graphed records no deferred {dispatch.__module__}.{name}: graphed.awkward.gak has no "
            "equivalent; materialize the array first (Session.materialize) to use it eagerly"
        )
    fn = getattr(gak, name)
    signature = inspect.signature(dispatch)
    bound = signature.bind(*args, **kwargs).arguments
    gak_params = inspect.signature(fn).parameters
    positional: list[object] = []
    keywords: dict[str, object] = {}
    for i, (param, value) in enumerate(bound.items()):
        if signature.parameters[param].kind is inspect.Parameter.VAR_POSITIONAL:
            positional.extend(value)
        elif i == 0:
            positional.append(value)
        else:
            keywords[param if param in gak_params else _RENAMED.get(param, param)] = value
    return fn(*positional, **keywords)


class AwkwardArray(Array):
    """The awkward idiom's proxy: the neutral ``Array`` plus awkward's function-dispatch hook."""

    __slots__ = ()

    __awkward_function__ = _awkward_function


class AwkwardVaried(Varied):
    """A container of awkward-idiom universes; ``ak.<fn>`` on it expands as ``gak.<fn>`` does."""

    __slots__ = ()

    __awkward_function__ = _awkward_function


register_varied(AwkwardArray, AwkwardVaried)
