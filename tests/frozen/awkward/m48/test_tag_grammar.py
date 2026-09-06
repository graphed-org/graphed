"""§1.1: the tag grammar, its e-canonicalization, and every call-time rejection.

Validation and canonicalization are CHANNEL-INDEPENDENT across all three tag channels — kwarg
names, `points={tag: ...}`, and the shift form's inner collection-mapping keys — because
literal kwarg syntax cannot spell a dotted or digit-leading tag while `**`-unpacking admits any
string key. Every rejection below is a call-time `GraphedError`; the two rejections around the
32-character cap are asserted by CAUSE, with mutually exclusive messages, because a diagnostic
that blames the wrong property sends the user to the wrong fix.
"""

from __future__ import annotations

import pytest
from vary_ctx_fixtures import (
    events_context,
    jes_kwargs,
    pu_weight,
    reserved_names_context,
    shifted_jets,
)

import graphed
from graphed import GraphedError

#: the two cap refusals, split by CAUSE (§1.1); each assertion below also asserts the ABSENCE of
#: the other, so no single message can discharge both
MAGNITUDE = "magnitude"
TAG_LENGTH = "canonical tag length"


def _weight_context(ctx: object, name: str, **kwargs: object) -> object:
    """The §2.1(b) weight form: a per-event central factor plus per-tag factors."""
    return graphed.vary(ctx, name, pu_weight(ctx, 1.0), is_weight=True, **kwargs)


def _only_tag_label(varied: object) -> str:
    """The single user label of a one-tag container."""
    labels = list(graphed.labels(varied))
    assert len(labels) == 2 and labels[0] == "nominal"
    return labels[1]


def test_kwarg_tags_canonicalize_by_exact_decimal_arithmetic() -> None:
    """`"2"`, `"2.0"`, `"2e0"` and `"20e-1"` are ONE value, so they are ONE label — the labels are
    never an IEEE round-trip, so no float artifact reaches a name."""
    _s, ctx = events_context()
    for spelling in ("2", "2.0", "2e0", "20e-1"):
        varied = _weight_context(ctx, "murf", points={spelling: pu_weight(ctx, 2.0)})
        assert list(graphed.labels(varied)) == ["nominal", "murf_2"]


def test_the_variations_channel_carries_tags_kwarg_syntax_cannot_spell() -> None:
    _s, ctx = events_context()
    eps = _weight_context(ctx, "eps", points={"1e-8": pu_weight(ctx, 1.0)})
    assert list(graphed.labels(eps)) == ["nominal", "eps_1em8"]
    pdf = _weight_context(ctx, "pdf", points={f"{i}": pu_weight(ctx, float(i)) for i in (1, 2, 102)})
    assert list(graphed.labels(pdf)) == ["nominal", "pdf_1", "pdf_2", "pdf_102"]  # indices untouched


def test_the_collection_mapping_inner_keys_are_the_third_channel() -> None:
    """The shift form's tags arrive as the INNER keys of a collection mapping and take the same
    grammar — a channel-specific canonicalizer would leave `murf_0.5` on disk."""
    _s, ctx = events_context()
    varied = graphed.vary(ctx, "murf", Jet={"0.5": shifted_jets(ctx, 0.5)})
    assert list(graphed.labels(varied)) == ["nominal", "murf_5em1"]


def test_non_minimal_canonical_grammar_tags_are_re_rendered() -> None:
    """A hand-typed non-minimal e-form would otherwise ride through as an ordinary identifier tag:
    two labels, one value."""
    _s, ctx = events_context()
    assert _only_tag_label(_weight_context(ctx, "murf", points={"50em2": pu_weight(ctx, 0.5)})) == (
        "murf_5em1"
    )
    assert _only_tag_label(_weight_context(ctx, "murf", points={"05": pu_weight(ctx, 5.0)})) == "murf_5"


def test_two_spellings_of_one_value_unify_ACROSS_calls() -> None:
    _s, ctx = events_context()
    dotted = _weight_context(ctx, "murf", points={"0.5": pu_weight(ctx, 0.5)})
    e_form = _weight_context(ctx, "murf", points={"5em1": pu_weight(ctx, 0.5)})
    assert list(graphed.labels(dotted)) == list(graphed.labels(e_form)) == ["nominal", "murf_5em1"]


def test_two_spellings_of_one_value_are_a_duplicate_rejection_WITHIN_one_call() -> None:
    """The other reading of the same sentence: unification is across calls, never inside one."""
    _s, ctx = events_context()
    with pytest.raises(GraphedError):
        _weight_context(
            ctx, "murf", points={"0.5": pu_weight(ctx, 0.5), "5em1": pu_weight(ctx, 0.5)}
        )


def test_cross_notation_numeric_equal_pairs_are_rejected_within_one_call() -> None:
    """The p-form deliberately does NOT canonicalize, so it is the one residual duplicate class:
    two labels for one value would mean two StrCategory bins and two content hashes."""
    _s, ctx = events_context()
    for pair in (("0.5", "0p5"), ("2", "2p0")):
        with pytest.raises(GraphedError):
            _weight_context(ctx, "murf", points={t: pu_weight(ctx, 1.0) for t in pair})


def test_cross_notation_pairs_are_rejected_across_two_STACKING_calls() -> None:
    """A family is the tags one `name` carries on one container INCLUDING inherited labels, so a
    per-call check would accept `"0.5"` now and `"0p5"` next call, uncatchable afterwards."""
    _s, ctx = events_context()
    stacked = _weight_context(ctx, "murf", points={"0.5": pu_weight(ctx, 0.5)})
    with pytest.raises(GraphedError):
        _weight_context(stacked, "murf", points={"0p5": pu_weight(stacked, 0.5)})


def test_duplicate_after_canonicalization_inside_one_collection_mapping() -> None:
    _s, ctx = events_context()
    with pytest.raises(GraphedError):
        graphed.vary(
            ctx, "murf", Jet={"0.5": shifted_jets(ctx, 0.5), "5em1": shifted_jets(ctx, 0.5)}
        )


def test_malformed_and_non_string_tags_are_rejected() -> None:
    _s, ctx = events_context()
    for tag in ("inf", "nan", "+2", "1_000", " 2", "2 ", "1.5e31 "):
        with pytest.raises(GraphedError):
            _weight_context(ctx, "sig", points={tag: pu_weight(ctx, 1.0)})
    with pytest.raises(GraphedError):
        _weight_context(ctx, "sig", points={0.5: pu_weight(ctx, 1.0)})  # a Python float, not a string


def test_negative_zero_canonicalizes_to_zero() -> None:
    """One value, one label: `m0` would be a second name for the same universe."""
    _s, ctx = events_context()
    varied = _weight_context(ctx, "shift", points={"-0": pu_weight(ctx, 1.0)})
    assert list(graphed.labels(varied)) == ["nominal", "shift_0"]


def test_the_cap_boundary_pair_refuses_by_cause() -> None:
    """`"1.5e31"` renders `15` followed by 30 zeros — exactly 32 characters, and LEGAL. The naive
    "mantissa digits + exponent" sum computes 33 for it and rejects it; `"1e40"` alone cannot catch
    that off-by-one, which is why the accepted half of the pair is here."""
    _s, ctx = events_context()
    accepted = _weight_context(ctx, "scan", points={"1.5e31": pu_weight(ctx, 1.0)})
    tag = _only_tag_label(accepted).removeprefix("scan_")
    assert tag == "15" + "0" * 30 and len(tag) == 32

    with pytest.raises(GraphedError) as over_cap:
        _weight_context(ctx, "scan", points={"1.5e32": pu_weight(ctx, 1.0)})
    assert MAGNITUDE in str(over_cap.value) and TAG_LENGTH not in str(over_cap.value)

    with pytest.raises(GraphedError) as negative_twin:
        _weight_context(ctx, "scan", points={"-1.5e31": pu_weight(ctx, 1.0)})
    # the magnitude is the one just certified LEGAL; the `m` sign marker is what takes the rendered
    # tag to 33 characters, so blaming the magnitude here would blame the wrong property
    assert TAG_LENGTH in str(negative_twin.value) and MAGNITUDE not in str(negative_twin.value)


def test_an_integer_magnitude_over_the_cap_names_the_magnitude() -> None:
    _s, ctx = events_context()
    with pytest.raises(GraphedError) as excinfo:
        _weight_context(ctx, "scan", points={"1e40": pu_weight(ctx, 1.0)})
    assert MAGNITUDE in str(excinfo.value) and TAG_LENGTH not in str(excinfo.value)
    # the positive half: a small exponent is ordinary sugar, not a magnitude problem
    small = _weight_context(ctx, "eps", points={"1e-8": pu_weight(ctx, 1.0)})
    assert _only_tag_label(small) == "eps_1em8"


def test_signature_shadowed_names_reach_tags_through_the_mapping_channels() -> None:
    """`nominal`, `is_weight`, `points` and `collections` are legal tags AND legal collection
    names, so they can only arrive through a mapping — `collections` by self-reference."""
    _s, ctx = events_context()
    shadowed = {name: pu_weight(ctx, 1.0) for name in ("nominal", "is_weight", "points", "collections")}
    varied = _weight_context(ctx, "pu", points=shadowed)
    assert list(graphed.labels(varied)) == [
        "nominal",
        "pu_nominal",
        "pu_is_weight",
        "pu_points",
        "pu_collections",
    ]
    _rs, tree = reserved_names_context()  # a tree branch literally named `collections`
    self_referenced = graphed.vary(tree, "jes", collections={"collections": {"up": tree["collections"]}})
    assert list(graphed.labels(self_referenced)) == ["nominal", "jes_up"]


def test_the_tag_nominal_is_legal_and_yields_an_ordinary_label() -> None:
    """`"nominal"` is unreachable as a user label BY CONSTRUCTION — every user label carries a `_`
    — so there is no "label equals nominal" rejection to freeze."""
    _s, ctx = events_context()
    varied = _weight_context(ctx, "pu", points={"nominal": pu_weight(ctx, 1.1)})
    assert list(graphed.labels(varied)) == ["nominal", "pu_nominal"]
    assert graphed.universe(varied, "pu_nominal") is not graphed.universe(varied, "nominal")


def test_variations_is_refused_in_the_shift_form() -> None:
    _s, ctx = events_context()
    with pytest.raises(GraphedError):
        graphed.vary(ctx, "jes", points={"up": shifted_jets(ctx, 1.05)})


def test_nominal_is_refused_in_the_shift_form_with_an_error_naming_collections() -> None:
    """The collections' central members come from the target context, so `nominal=` has nothing to
    mean there — and as a shadowed name it cannot be read as a tag either."""
    _s, ctx = events_context()
    with pytest.raises(GraphedError, match="collections"):
        graphed.vary(ctx, "jes", shifted_jets(ctx, 1.0), **jes_kwargs(ctx))


def test_no_label_ever_contains_a_dot_or_a_dash() -> None:
    """The measured ground: `ak.from_parquet(columns=["murf_0.5"])` is silently empty, RNTuple
    splits a field path on `.`, and the TTree writer uses `.` as its nesting separator."""
    _s, ctx = events_context()
    varied = _weight_context(ctx, "murf", points={"0.5": pu_weight(ctx, 0.5), "-2": pu_weight(ctx, 2.0)})
    assert list(graphed.labels(varied)) == ["nominal", "murf_5em1", "murf_m2"]
    assert not any("." in label or "-" in label for label in graphed.labels(varied))


def test_a_tag_given_twice_and_an_empty_tag_set_are_rejected() -> None:
    _s, ctx = events_context()
    with pytest.raises(GraphedError):
        _weight_context(ctx, "pu", up=pu_weight(ctx, 1.1), points={"up": pu_weight(ctx, 1.2)})
    with pytest.raises(GraphedError):
        _weight_context(ctx, "pu")
