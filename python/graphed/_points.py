"""The §4.2 point value type: a universe's coordinates in nuisance space.

A **coordinate** is a canonical tag. Three input spellings share one value space — an identifier
token kept verbatim, a numeric string, and a `int`/`float`/`Fraction`, which is decomposed to an
exact (sign, digits, power-of-ten) triple and rendered by `_tags._render`, the same path a numeric
string takes. A number is never `str()`-ed into a tag (`str(Fraction(1, 2))` is `"1/2"`, which is
no tag at all) and coordinates are never compared as floats.

A **point** is `{nuisance -> coordinate}` stored as a tuple of pairs sorted by nuisance name, so a
point is hashable, comparable, and orders deterministically (§5.3). Numeric coordinates are stored
already reduced to their value's e-form, which is what makes `"0p5"`, `"0.5"`, `0.5` and
`Fraction(1, 2)` one coordinate and makes `render` a plain read (§4.10: rendering is BY VALUE).

Sibling to `_tags`, and importing nothing else from the package: no `Varied` contact, no `vary`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from fractions import Fraction

from ._tags import _normalize, _render, canonical_tag, numeric_value
from .errors import GraphedError


class Point(tuple[tuple[str, str], ...]):
    """A name-sorted tuple of `(nuisance, canonical coordinate)` pairs.

    Constructing from a mapping is the EXPLICIT `points=` mode, which zero-drops: a coordinate
    whose value is 0 says "this axis sits at its central value", which is what absence already
    says, so `{jes: 1, btag: 0}` and `{jes: 1}` are one point and `Point({})` is the origin.
    """

    __slots__ = ()

    def __new__(cls, mapping: Mapping[str, object]) -> Point:
        pairs = ((_nuisance(name), coordinate(value)) for name, value in mapping.items())
        return _build(pair for pair in pairs if numeric_value(pair[1]) != 0)


def default(name: str, tag: str) -> Point:
    """§4.4's default point `{name: tag}`, which is NEVER zero-dropped.

    Its coordinate is the tag the analyst registered — a *name* for that universe rather than a
    displacement — so the legal tag `0` carries the ordinary universe `f"{name}_0"`, distinct from
    `nominal`, exactly as it is today.
    """
    return _build([(_nuisance(name), coordinate(tag))])


def restrict(point: Point, axes: frozenset[str]) -> Point:
    """§4.6's whole resolution machinery: drop every coordinate off `axes`, zero-dropping nothing."""
    return _build(pair for pair in point if pair[0] in axes)


def render(point: Point) -> dict[str, str]:
    """§4.10's coordinate view: nuisance-sorted, each coordinate by VALUE, so two equal points
    render alike and the rendering predicts resolution rather than round-tripping the spelling."""
    return dict(point)


def _build(pairs: Iterable[tuple[str, str]]) -> Point:
    return tuple.__new__(Point, sorted(pairs))


def _nuisance(name: object) -> str:
    """A nuisance name is a Python identifier — the same rule `vary` applies to a family name."""
    if not isinstance(name, str) or not name.isidentifier():
        raise GraphedError(f"a nuisance name must be a Python identifier, got {name!r}")
    return name


def coordinate(value: object) -> str:
    """The canonical coordinate a spelling names, reduced to its VALUE when it has one."""
    if isinstance(value, str):
        tag = canonical_tag(value)
        exact = numeric_value(tag)
        return tag if exact is None else _decimal(exact, value)
    if isinstance(value, bool) or not isinstance(value, int | float | Fraction):
        raise GraphedError(
            f"a coordinate is a variation tag or a number, got {value!r} — spell an identifier "
            "coordinate as a string"
        )
    # a float goes through its shortest round-tripping decimal (never its IEEE expansion); an int
    # and a Fraction are already exact
    return _decimal(Fraction(repr(value)) if isinstance(value, float) else Fraction(value), value)


def _decimal(value: Fraction, source: object) -> str:
    """`value` as an exact decimal tag, or a refusal naming the value it cannot spell."""
    denominator, twos, fives = value.denominator, 0, 0
    while denominator % 2 == 0:
        denominator, twos = denominator // 2, twos + 1
    while denominator % 5 == 0:
        denominator, fives = denominator // 5, fives + 1
    if denominator != 1:
        raise GraphedError(
            f"coordinate {source!r} has no finite decimal expansion ({value}); a coordinate names "
            "an exact tag, and no decimal spelling names this value"
        )
    scale = max(twos, fives)
    digits = value.numerator * 10**scale // value.denominator
    return _render(*_normalize(digits < 0, str(abs(digits)), -scale), f"{value}")
