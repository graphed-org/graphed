"""``AwkwardArray``: the awkward-idiom proxy that lets ``ak.<fn>(deferred)`` record ``gak.<fn>``.

awkward's high-level functions offer every array-like argument an ``__awkward_function__`` hook
(``awkward._dispatch.named_high_level_function``). This class answers it by calling the ``gak``
function of the same name, so ``ak.sum(a, axis=1)`` records exactly what ``gak.sum(a, axis=1)``
records, refusals included. ``AwkwardVaried`` carries the same hook for containers of universes.
Both also take ak's field assignment, ``events["Photon"] = photons``, as a recorded ``with_field``.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

import awkward as ak

from graphed import Array
from graphed.errors import GraphedError, GraphedTypeError
from graphed.provenance import capture, register_internal
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
        spelled = next(
            (
                f"{prefix}.{name}"
                for prefix, ns in (("ak", ak), ("ak.str", ak.str))
                if getattr(ns, name, None) is dispatch
            ),
            f"{dispatch.__module__}.{name}",
        )
        raise GraphedError(
            f"graphed records no deferred {spelled}: graphed.awkward.gak has no "
            "equivalent; materialize the array first (Session.materialize) to use it eagerly"
        )
    fn = getattr(gak, name)
    signature = inspect.signature(dispatch)
    bound = signature.bind(*args, **kwargs).arguments
    gak_params = inspect.signature(fn).parameters
    positional: list[object] = []
    keywords: dict[str, object] = {}
    for i, param in enumerate(signature.parameters.values()):
        target = param.name if param.name in gak_params else _RENAMED.get(param.name, param.name)
        if param.name in bound:
            value = bound[param.name]
        elif target in gak_params and gak_params[target].default not in (param.default, param.empty):
            value = param.default  # an omitted ak argument means ak's default (ak.softmax: axis=-1)
        else:
            continue
        if param.kind is inspect.Parameter.VAR_POSITIONAL:
            positional.extend(value)
        elif i == 0:
            positional.append(value)
        else:
            keywords[target] = value
    return fn(*positional, **keywords)


def _with_path(target: Any, path: tuple[str, ...], value: Any) -> Any:
    if len(path) > 1:
        value = _with_path(target[path[0]], path[1:], value)
    return gak.with_field(target, value, path[0])


def _setitem(self: Any, where: object, what: object) -> None:
    """ak's ``Array.__setitem__``: a field name, or a tuple path into nested records, set by recording
    ``with_field`` and rebinding this proxy to it; other proxies of the old node are untouched."""
    path = (where,) if isinstance(where, str) else where
    if not (isinstance(path, tuple) and path and all(isinstance(part, str) for part in path)):
        raise GraphedTypeError(
            "__setitem__", capture(), "only fields may be assigned in-place (by field name)"
        )
    new = _with_path(self, path, what)
    if not isinstance(new, type(self)):
        raise GraphedTypeError(
            "__setitem__",
            capture(),
            "assigning a varied value turns this array into a container of universes, which an "
            "in-place assignment cannot do; rebind instead: x = gak.with_field(x, value, field)",
        )
    for cls in type(new).__mro__:
        for slot in getattr(cls, "__slots__", ()):
            if slot != "__weakref__":
                setattr(self, slot, getattr(new, slot))


class AwkwardArray(Array):
    """The awkward idiom's proxy: the neutral ``Array`` plus awkward's function-dispatch hook."""

    __slots__ = ()

    __awkward_function__ = _awkward_function
    __setitem__ = _setitem


class AwkwardVaried(Varied):
    """A container of awkward-idiom universes; ``ak.<fn>`` on it expands as ``gak.<fn>`` does."""

    __slots__ = ()

    __awkward_function__ = _awkward_function
    __setitem__ = _setitem


register_varied(AwkwardArray, AwkwardVaried)
