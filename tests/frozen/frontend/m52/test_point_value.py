"""C1 / design §4.2, §4.10, §6-C1: the point value type.

`graphed._points` is resolved inside each body through `m52_point_fixtures.point_api`, never at
module scope, so this file COLLECTS against a tree with no m52 implementation and fails at RUN time
(TEST_SANITY §5-1).

The frozen surface is four names on that module: `Point(mapping)` (explicit construction, which
zero-drops), `default(name, tag)` (the never-zero-dropped default point), `restrict(point, axes)`
and `render(point)`. The origin is spelled `Point({})`.
"""

from __future__ import annotations

from fractions import Fraction
from itertools import combinations

import pytest
from m52_point_fixtures import point_api

from graphed.errors import GraphedError

#: five spellings of one coordinate - design §4.2's "three input spellings, one value space"
HALF_SPELLINGS = ("0p5", "0.5", "5em1", 0.5, Fraction(1, 2))

#: 21 significant digits, inside `_tags.MAX_TAG_CHARS`; no IEEE double separates it from 1
NEAR_ONE = "1.00000000000000000001"


def test_numeric_spellings_of_one_coordinate_are_one_point() -> None:
    """A `dict[str, str]` implementation yields two points here: `canonical_tag("0p5")` is `"0p5"`
    while `canonical_tag("0.5")` is `"5em1"`."""
    api = point_api()

    points = [api.Point({"jes": spelling}) for spelling in HALF_SPELLINGS]
    for left, right in combinations(points, 2):
        assert left == right
        assert hash(left) == hash(right)


def test_a_float_decomposes_exactly_and_not_through_ieee() -> None:
    api = point_api()

    # the admitted members: one coordinate under four spellings of unity
    one = api.Point({"jes": 1})
    assert one == api.Point({"jes": 1.0}) == api.Point({"jes": "1"}) == api.Point({"jes": "1.0"})
    # and the discriminator a float-keyed implementation cannot see
    assert api.Point({"jes": NEAR_ONE}) != one


def test_a_fraction_with_no_finite_decimal_is_refused_naming_the_value() -> None:
    api = point_api()

    # the positive control: two fractions that DO have finite decimal expansions
    assert api.Point({"jes": Fraction(1, 2)}) == api.Point({"jes": "0.5"})
    assert api.Point({"jes": Fraction(3, 8)}) == api.Point({"jes": "0.375"})

    with pytest.raises(GraphedError) as caught:
        api.Point({"jes": Fraction(1, 3)})
    assert "1/3" in str(caught.value)


def test_identifier_coordinates_are_not_promoted_to_numbers() -> None:
    """design §4.2's stated refusal: the HistFactory reading of `up` as +1 sigma is a stats-export
    convention, not a frontend change."""
    api = point_api()

    assert api.Point({"jes": "up"}) != api.Point({"jes": "1"})
    assert api.Point({"jes": "up"}) != api.Point({"jes": 1})


def test_a_point_is_name_sorted_and_order_independent() -> None:
    api = point_api()

    one = api.Point({"jes": 1, "btag": -1})
    other = api.Point({"btag": -1, "jes": 1})
    assert one == other
    assert list(api.render(one)) == list(api.render(other)) == ["btag", "jes"]


def test_an_explicit_map_zero_drops_and_an_all_zero_map_is_the_origin() -> None:
    """design §4.2: in an explicit map a zero coordinate says "this axis sits at its central
    value", which is what absence already says."""
    api = point_api()

    assert api.Point({"jes": 1, "btag": 0}) == api.Point({"jes": 1})
    assert api.Point({"jes": 0}) == api.Point({})
    assert api.Point({"jes": "0"}) == api.Point({})


def test_a_default_point_is_never_zero_dropped() -> None:
    """The zero asymmetry. A default point's coordinate is the tag the analyst REGISTERED - a name
    for that universe - so the legal tag `0` mints an ordinary universe, not `nominal`."""
    api = point_api()

    assert api.default("shift", "0") != api.Point({})
    assert api.default("shift", "0") != api.default("other", "0")
    assert api.default("shift", "0") == api.default("shift", "0")


def test_restrict_drops_only_the_coordinates_off_the_axes() -> None:
    api = point_api()

    point = api.Point({"jes": 1, "btag": -1})
    assert api.restrict(point, frozenset({"jes"})) == api.Point({"jes": 1})
    assert api.restrict(point, frozenset()) == api.Point({})
    assert api.restrict(point, frozenset({"jes", "btag", "jer"})) == point  # superset is identity

    # restriction must not zero-drop either: the default point survives its own axis
    zero = api.default("shift", "0")
    assert api.restrict(zero, frozenset({"shift"})) == zero
    assert api.restrict(zero, frozenset({"shift"})) != api.Point({})


def test_rendering_is_by_value_so_two_equal_points_render_alike() -> None:
    """design §4.10: a numeric coordinate renders THROUGH its value, so the rendered view predicts
    resolution and deliberately does not round-trip the tag spelling."""
    api = point_api()

    assert api.render(api.Point({"jes": "0p5"})) == {"jes": "5em1"}
    assert api.render(api.Point({"jes": "0.5"})) == {"jes": "5em1"}
    assert api.render(api.Point({"jes": "up"})) == {"jes": "up"}  # non-numeric stays verbatim
    assert api.render(api.Point({})) == {}


@pytest.mark.parametrize("name", ["1jes", "jes.x", ""])
def test_a_nuisance_name_must_be_a_python_identifier(name: str) -> None:
    """The same rule `vary` applies to a family name."""
    api = point_api()

    with pytest.raises(GraphedError) as caught:
        api.Point({name: 1})
    assert repr(name) in str(caught.value)


def test_an_identifier_nuisance_name_is_accepted() -> None:
    """The positive control for the parametrized refusal above."""
    api = point_api()

    assert api.Point({"jes": 1}) != api.Point({})
    assert api.Point({"jes_2": 1}) != api.Point({"jes": 1})
