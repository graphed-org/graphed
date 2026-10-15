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
import numpy as np
from awkward._connect.numpy import implemented as _ak_numpy

from graphed import Array
from graphed.array import no_eager_value
from graphed.errors import GraphedError
from graphed.provenance import register_internal
from graphed.varied import Varied, register_varied

from . import functions as gak

# a node recorded through ak.<fn> points at the user's line, not at awkward's dispatcher
register_internal("awkward._dispatch")
register_internal("awkward._connect.numpy")

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


def _clip(a: object, a_min: object = None, a_max: object = None) -> object:
    """``np.clip`` as its two ufuncs: awkward has no ``np.clip`` of its own."""
    low = a if a_min is None else np.maximum(a, a_min)
    return low if a_max is None else np.minimum(low, a_max)


def _array_function(self: object, func: Any, types: object, args: Any, kwargs: Any) -> Any:
    """A numpy function that is not a ufunc, answered as ``ak.Array`` answers it: by awkward's own
    implementation, whose ``ak.<fn>`` calls dispatch back through ``__awkward_function__``."""
    impl = _clip if func is np.clip else _ak_numpy.get(func)
    if impl is None:  # awkward would convert eagerly; a deferred array has nothing to convert
        raise no_eager_value(f"np.{func.__name__}(...)", "numpy value")
    return impl(*args, **kwargs)


class AwkwardArray(Array):
    """The awkward idiom's proxy: the neutral ``Array`` plus awkward's function-dispatch hook."""

    __slots__ = ()

    __awkward_function__ = _awkward_function
    __array_function__ = _array_function


class AwkwardVaried(Varied):
    """A container of awkward-idiom universes; ``ak.<fn>`` on it expands as ``gak.<fn>`` does."""

    __slots__ = ()

    __awkward_function__ = _awkward_function
    __array_function__ = _array_function


register_varied(AwkwardArray, AwkwardVaried)
