"""``NumpyVaried``: the numpy-idiom `Varied` subclass (§2.2's per-idiom container).

The neutral `graphed.Varied` carries `Array`'s surface; this subclass completes the numpy idiom
the way `NumpyArray` completes `Array` — the metadata properties and the method set, so §2.3a's
parity gate (enumerated from `type(graphed.nominal(v))`) resolves every discovered name on the
container's own class. It re-exposes only what `Varied` must answer per label; the recording logic
behind those names stays on the members.
"""

from __future__ import annotations

from typing import Any

from graphed.accessors import nominal
from graphed.varied import Varied, expand, install_surface, register_varied

from .array import NumpyArray

_BROADCAST_SURFACE = (
    "__array_function__", "all", "any", "argmax", "argmin", "astype", "clip", "cumprod",
    "cumsum", "max", "mean", "min", "prod", "ravel", "reshape", "round", "squeeze", "std",
    "sum", "swapaxes", "take", "transpose", "var",
)  # fmt: skip
#: §2.3a: the numpy idiom's own additions to the `Array` surface, all *broadcast*
SURFACE_DISPOSITIONS: dict[str, str] = dict.fromkeys(_BROADCAST_SURFACE, "broadcast")


class NumpyVaried(Varied):
    """A container of numpy-idiom universes."""

    __slots__ = ()

    # §2.2's three-class property rule: `shape`/`dtype`/`ndim` are form-answered — they record
    # nothing on a plain `Array` — so they answer EAGERLY on the nominal member (sound by §2.1's
    # form compatibility), while `T` is a plain alias for the recorded `transpose` op and so takes
    # that method's *broadcast* disposition.
    @property
    def shape(self) -> Any:
        return nominal(self).shape

    @property
    def dtype(self) -> Any:
        return nominal(self).dtype

    @property
    def ndim(self) -> Any:
        return nominal(self).ndim

    @property
    def T(self) -> Any:
        return expand(lambda member: member.T, (self,), {})


install_surface(NumpyVaried, SURFACE_DISPOSITIONS)
register_varied(NumpyArray, NumpyVaried)
