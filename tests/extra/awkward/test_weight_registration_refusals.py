"""What a weight registration must refuse, and what a refused one must leave behind.

`vary.check_members` compares a factor only against its OWN nominal, so two factors whose nominals
multiply happily can still collide at one label — a shift label both of them are non-nominal at.
A lazily composed ambient would raise that clash at the first read, long after `vary`'s
transactional mint has released the labels, so the composition's own walk is run over FORMS at
registration instead — the same grouping, the same operands, every form identity a `describe()`
and never a repr. The walk's grouping is load-bearing: `mul` inference is not associative for
raise/no-raise, and the walk forms products (the complement of each factor) that no single label
needs.

Awkward-idiom because the admitted member needs a backend whose `add` and `mul` disagree: awkward
broadcasts `## * float64` against `## * 3 * float64` (so `check_members` admits the container) and
refuses `## * 3 * float64` against `## * 5 * float64` (so the product at that label does not exist).
Numpy has no such member — `NumpyBackend.op_form` routes both through the same broadcasting, so any
pair `mul` refuses at a label, `add` has already refused inside the container.
"""

from __future__ import annotations

from typing import Any

import awkward as ak
import numpy as np
import pytest

import graphed
import graphed.awkward as ga
from graphed import Session
from graphed.awkward import AwkwardBackend, from_awkward
from graphed.context import _two_level
from graphed.errors import GraphedError, GraphedTypeError
from graphed.varied import Varied, member_of, rebuild

SHAPES = ak.Array(
    {
        "var": ak.unflatten(np.ones(18), np.full(6, 3)),
        "r3": np.ones((6, 3)),
        "r5": np.ones((6, 5)),
    }
)

EVENTS = ak.Array(
    {
        "w": np.ones(6),
        "r3": np.arange(18.0).reshape(6, 3) * 0.1 + 1.0,
        "r5": np.arange(30.0).reshape(6, 5) * 0.1 + 1.0,
        "Jet": ak.zip({"pt": ak.unflatten(np.arange(12.0), np.full(6, 2))}),
    }
)


def _shifted_context() -> tuple[Session, Any]:
    """A context carrying a `jes` shift, so a factor read through it varies at `jes_up`."""
    session = Session(AwkwardBackend())
    ctx = ga.gnano.events(from_awkward(session, "ev", EVENTS))
    jets = ctx.Jet
    shifted = {
        tag: ga.gak.with_field(jets, jets.pt * scale, "pt") for tag, scale in (("up", 1.05), ("down", 0.95))
    }
    return session, graphed.vary(ctx, "jes", collections={"Jet": shifted})


def _factor(ctx: Any, wide: Any, scale: float, *, nominal: Any = None) -> Any:
    """A weight factor computed on shifted objects: `nominal`-shaped (flat unless named) at
    nominal, `wide`-shaped at `jes_up`."""
    flat = ctx["w"] * scale
    return rebuild(
        {"nominal": flat if nominal is None else nominal * scale, "jes_up": wide * scale, "jes_down": flat},
        context=ctx,
    )


def test_two_factors_that_collide_at_one_shift_label_are_refused_at_registration() -> None:
    session, shifted = _shifted_context()
    first = _factor(shifted, shifted["r3"], 1.0)
    registered = graphed.vary(shifted, "A", first, is_weight=True, up=first * 1.1)

    second = _factor(shifted, shifted["r5"], 2.0)  # nominals multiply; the `jes_up` members do not
    nominals = [session.form(graphed.nominal(first)), session.form(graphed.nominal(second))]
    session.backend.op_form("mul", nominals, {})  # must not raise, or this is not the per-label case

    with pytest.raises(GraphedTypeError, match="jes_up"):
        graphed.vary(registered, "B", second, is_weight=True, up=second * 1.1)
    assert "B_up" not in session._points, "a refused registration left its label minted"

    # the admitted member: a second factor that agrees at every label still registers
    admitted = _factor(shifted, shifted["r3"], 3.0)
    ok = graphed.vary(registered, "B", admitted, is_weight=True, up=admitted * 1.1)
    assert "B_up" in graphed.labels(graphed.weight(ok))


def test_the_refusal_survives_a_derivation_between_the_two_registrations() -> None:
    """The running form must follow the factor list at every site that sets it. A derivation
    MATERIALISES — the child's form map is rebuilt from the composed container's members, not
    copied — so the clash has to be refused through that rebuilt state too."""
    session, shifted = _shifted_context()
    first = _factor(shifted, shifted["r3"], 1.0)
    registered = graphed.vary(shifted, "A", first, is_weight=True, up=first * 1.1)

    derived = registered[registered["w"] > -1.0]
    second = _factor(derived, derived["r5"], 2.0)
    with pytest.raises(GraphedTypeError, match="jes_up"):
        graphed.vary(derived, "B", second, is_weight=True, up=second * 1.1)
    assert "B_up" not in session._points


def test_a_refused_weight_leaves_no_residue_for_the_shift_after_weight_report() -> None:
    """§2.5's diagnostic reads `Session._weight_factors`, so the form check has to run BEFORE that
    append: a weight whose registration was refused must not be reported against a shift it never
    got to precede. The refused factor here READS the collection the shift varies, which is what
    makes the report's cone walk answer yes for it."""
    session, shifted = _shifted_context()
    jets = shifted.Jet
    factor = ga.gak.prod(1.0 + jets.pt * 0.01, axis=1)  # a per-event weight that reads Jet
    registered = graphed.vary(shifted, "btag", factor, is_weight=True, up=factor * 1.1)

    with pytest.raises(GraphedTypeError):  # a record-typed factor: no product with the ambient
        graphed.vary(registered, "bad", jets, is_weight=True, up=jets)

    later = graphed.vary(
        registered, "jer", collections={"Jet": {"up": ga.gak.with_field(jets, jets.pt * 1.02, "pt")}}
    )
    ambient = graphed.weight(later)
    universes = [graphed.universe(ambient, label) for label in graphed.labels(ambient)]
    assert graphed.compile_ir(session, *universes).shift_after_weight == (("btag", "Jet"),)


def _colliding_shapes() -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Two regular shapes whose forms render identically under `str` and differently under
    `describe`. Awkward elides the middle of a deep type; the depth at which it starts is a repr
    detail that moves between versions, so it is searched rather than pinned."""
    session = Session(AwkwardBackend())
    for depth in range(6, 15):
        deep, wide = (2,) * depth, (2,) * (depth - 1) + (3,)
        record = from_awkward(
            session, f"probe{depth}", ak.Array({"d": np.ones((2, *deep)), "e": np.ones((2, *wide))})
        )
        left, right = session.form(record["d"]), session.form(record["e"])
        if str(left) == str(right) and left.describe() != right.describe():
            return deep, wide
    pytest.fail("awkward's type repr no longer abbreviates; the memo-key collision is unreachable")


def test_a_factor_whose_form_only_LOOKS_like_a_memoised_one_is_still_refused() -> None:
    """The running-form memo keys on the `Form` protocol's `describe()`, never on `str(form)`.

    An abbreviating repr makes two forms that multiply DIFFERENTLY render alike, so a `str` key
    would hand the clashing pair the memoised product form of a pair that multiplies fine, admit
    the registration, and defer the raise to the first read — past `vary`'s transactional mint,
    with the labels bound for the Session's life.
    """
    deep, wide = _colliding_shapes()
    rows = len(EVENTS)
    session = Session(AwkwardBackend())
    events = ak.Array(
        {
            "w": np.ones(rows),
            "deep": np.ones((rows, *deep)),
            "wide": np.ones((rows, *wide)),
            "Jet": EVENTS.Jet,
        }
    )
    ctx = ga.gnano.events(from_awkward(session, "ev", events))
    jets = ctx.Jet
    shifted = graphed.vary(
        ctx,
        "jes",
        collections={
            "Jet": {
                tag: ga.gak.with_field(jets, jets.pt * scale, "pt")
                for tag, scale in (("up", 1.05), ("down", 0.95))
            }
        },
    )
    assert str(session.form(shifted["deep"])) == str(session.form(shifted["wide"])), (
        "the two member forms must be indistinguishable by `str`, or the memo is not under test"
    )

    # seed the memo with the deep pair, so a `str` key would answer for the wide one
    registered = shifted
    for name in ("A", "B"):
        factor = _factor(registered, registered["deep"], 1.0)
        registered = graphed.vary(registered, name, factor, is_weight=True, up=factor * 1.1)
    seeded = len(session._mul_forms)
    assert seeded, "no product form was memoised; the collision cannot be exercised"

    clashing = _factor(registered, registered["wide"], 2.0)
    with pytest.raises(GraphedTypeError, match="jes_up"):
        graphed.vary(registered, "C", clashing, is_weight=True, up=clashing * 1.1)
    assert "C_up" not in session._points, "a refused registration left its label minted"

    # the admitted end: a genuinely identical pair still hits the memo rather than re-inferring
    admitted = _factor(registered, registered["deep"], 3.0)
    ok = graphed.vary(registered, "C", admitted, is_weight=True, up=admitted * 1.1)
    assert "C_up" in graphed.labels(graphed.weight(ok))


def test_a_variation_member_whose_form_only_LOOKS_like_the_central_one_is_refused() -> None:
    """`check_members`' compatibility fast path is the same class as the memo key, one registration
    earlier: a member whose form only RENDERS like the central universe's must still meet the
    backend's own inference, or an abbreviated repr waves an incompatible member straight past the
    check that exists to catch it. Same two shapes."""
    deep, wide = _colliding_shapes()
    session = Session(AwkwardBackend())
    events = ak.Array({"deep": np.ones((6, *deep)), "wide": np.ones((6, *wide))})
    ctx = ga.gnano.events(from_awkward(session, "ev", events))
    central, incompatible = ctx["deep"] * 1.0, ctx["wide"] * 1.0

    with pytest.raises(GraphedError):
        graphed.vary(ctx, "A", central, is_weight=True, up=incompatible)
    assert "A_up" not in session._points

    ok = graphed.vary(ctx, "A", central, is_weight=True, up=central * 1.1)  # the admitted end
    assert "A_up" in graphed.labels(graphed.weight(ok))


def _joint_pair(
    shifted: Any, b_up_wide: Any, *, a_nominal: Any = None, a_up_wide: Any = None
) -> tuple[Any, Any, Any]:
    """A registered `A` plus the `(central, up)` pair for a `B` whose only disagreement with `A`
    lives at the joint `B_up__jes_up` that this very registration mints — `B`'s central agrees with
    `A` at `jes_up`, so nothing already in the running form map can refuse or admit the pair."""
    a_up_wide = shifted["r3"] if a_up_wide is None else a_up_wide
    first = _factor(shifted, a_up_wide, 1.0, nominal=a_nominal)
    registered = graphed.vary(shifted, "A", first, is_weight=True, up=first * 1.1)
    central = _factor(shifted, a_up_wide, 2.0)
    return registered, central, _factor(shifted, b_up_wide, 3.0)


def test_a_clash_at_a_joint_this_registration_mints_is_refused() -> None:
    """The record-time check's operands must be the composition's, at a label brand new to the
    running form map. The earlier factors resolve a minted joint through their own point
    restriction — here onto `jes_up`, whose member is wide — not through their nominal, so folding
    the running nominal would admit a pair whose product does not exist and leave the minted labels
    bound for the Session's life."""
    session, shifted = _shifted_context()
    registered, central, clashing = _joint_pair(shifted, shifted["r5"])
    assert "B_up__jes_up" not in registered._recorded, "the joint is not new to the union; wrong case"
    at_jes_up = [
        session.form(_two_level(container, "jes_up")) for container in (registered._factors[0], central)
    ]
    session.backend.op_form("mul", at_jes_up, {})  # jes_up itself agrees, or the joint is not the cause

    with pytest.raises(GraphedTypeError, match="B_up__jes_up"):
        graphed.vary(registered, "B", central, is_weight=True, up=clashing)
    for label in ("B_up", "B_up__jes_up", "B_up__jes_down"):
        assert label not in session._points, "a refused registration left its label minted"

    admitted = _factor(shifted, shifted["r3"], 3.0)  # the name is free again
    ok = graphed.vary(registered, "B", central, is_weight=True, up=admitted)
    assert "B_up__jes_up" in graphed.labels(graphed.weight(ok))


def test_a_rolled_back_mint_leaves_no_resolution_the_restored_registry_no_longer_gives() -> None:
    """A rollback is the one place a resolution moves BACKWARDS, and the walk has already answered
    at the minted joint by the time the clash at it refuses: the earlier factor resolved there
    through its own point restriction, onto `jes_up`. Once the label is unminted that answer is the
    central universe again, so the resolution memo — which stores on the premise that the registry
    only grows — must not be allowed to keep the restricted one.
    """
    session, shifted = _shifted_context()
    registered, central, clashing = _joint_pair(shifted, shifted["r5"])
    factor = registered._factors[0]

    with pytest.raises(GraphedTypeError, match="B_up__jes_up"):
        graphed.vary(registered, "B", central, is_weight=True, up=clashing)
    assert "B_up__jes_up" not in session._points

    restricted = _two_level(factor, "jes_up")
    assert _two_level(factor, "B_up__jes_up").node_id == _two_level(factor, "nominal").node_id
    assert restricted.node_id != _two_level(factor, "nominal").node_id, "the two answers coincide"


def test_a_valid_pair_that_only_LOOKS_clashing_through_the_running_nominal_still_registers() -> None:
    """The other end of the same class. `A` is wide at nominal and flat at `jes_up`, so at the
    minted joint the composition multiplies flat by `B`'s wide member and succeeds — a check that
    folded the running NOMINAL there would refuse a program the composition composes."""
    session, shifted = _shifted_context()
    registered, central, wider = _joint_pair(
        shifted, shifted["r5"], a_nominal=shifted["r3"], a_up_wide=shifted["w"]
    )
    with pytest.raises(ValueError, match="broadcast"):  # the pair a nominal fold would have formed
        running_nominal = session.form(_two_level(registered._factors[0], "nominal"))
        session.backend.op_form("mul", [running_nominal, session.form(_two_level(wider, "jes_up"))], {})

    ok = graphed.vary(registered, "B", central, is_weight=True, up=wider)
    ambient = graphed.weight(ok)
    assert "B_up__jes_up" in graphed.labels(ambient)
    assert session.form(graphed.universe(ambient, "B_up__jes_up")).describe()


def _plain_context() -> tuple[Session, Any]:
    """No shift: the association cases below are about the NOMINAL products the walk forms."""
    session = Session(AwkwardBackend())
    return session, ga.gnano.events(from_awkward(session, "assoc", SHAPES))


def _mul(session: Session, *forms: Any) -> Any:
    """Left-to-right, the association a check that MODELLED the walk would use."""
    product = forms[0]
    for form in forms[1:]:
        product = session.backend.op_form("mul", [product, form], {})
    return product


def _register_all(ctx: Any, fields: tuple[str, ...]) -> Any:
    for name, field in zip("ABCD", fields, strict=False):  # the names are just labels
        central = ctx[field] * 1.0
        ctx = graphed.vary(ctx, name, central, is_weight=True, up=central * 1.1)
    return ctx


def test_a_product_only_the_composition_s_own_grouping_forms_is_refused_at_registration() -> None:
    """`mul` inference is not associative for raise/no-raise, so the check has to RUN the
    composition's walk rather than fold the same operands in registration order: the balanced tree
    over three nominals groups the last two, and `3 * float64` against `5 * float64` has no
    product, while folding those same three left to right does."""
    session, ctx = _plain_context()
    forms = {field: session.form(ctx[field] * 1.0) for field in ("var", "r3", "r5")}
    _mul(session, forms["var"], forms["r3"], forms["r5"])  # the fold types, or nothing is refused
    with pytest.raises(ValueError, match="broadcast"):
        _mul(session, forms["var"], _mul(session, forms["r3"], forms["r5"]))

    registered = _register_all(ctx, ("var", "r3"))
    third = registered["r5"] * 1.0
    with pytest.raises(GraphedTypeError, match="mul"):
        graphed.vary(registered, "C", third, is_weight=True, up=third * 1.1)
    assert sorted(session._points) == ["A_up", "B_up"], "the refused registration moved the registry"

    fine = registered["var"] * 2.0  # the name is free again
    ok = graphed.vary(registered, "C", fine, is_weight=True, up=fine * 1.1)
    assert "C_up" in graphed.labels(graphed.weight(ok))


def test_a_clashing_pair_the_walk_never_forms_as_a_pair_still_registers() -> None:
    """The other end: the walk refuses what it FORMS, not what it merely contains. These four
    nominals include the same `3 *` / `5 *` pair, but every product the tree and the complement
    pushdown build has a ragged factor in it first, so all four register and the ambient composes
    at every label."""
    session, ctx = _plain_context()
    with pytest.raises(ValueError, match="broadcast"):  # the pair itself has no product
        _mul(session, session.form(ctx["r3"] * 1.0), session.form(ctx["r5"] * 1.0))

    registered = _register_all(ctx, ("r3", "var", "var", "r5"))
    ambient = graphed.weight(registered)
    assert graphed.labels(ambient) == ("nominal", "A_up", "B_up", "C_up", "D_up")
    for label in graphed.labels(ambient):
        assert session.form(graphed.universe(ambient, label)).describe()


def test_a_factor_that_nests_past_the_one_level_the_composition_flattens_is_refused() -> None:
    """§2.2 lets a factor's member be a container — a factor computed on shifted objects — and the
    composition reads two levels deep to flatten it. A member that varies past that level would
    compose into an ambient universe that is itself a container, which no consumer can read, so it
    is refused inside `vary` instead of at the read that trips over it."""
    session, shifted = _shifted_context()
    flat = shifted["w"] * 1.0
    inner = rebuild({"nominal": flat, "q_up": flat * 3.0}, context=shifted)
    central = rebuild({"nominal": flat, "jes_up": inner, "jes_down": flat}, context=shifted)

    assert isinstance(member_of(central, "jes_up"), Varied), "the member is not nested; wrong case"
    with pytest.raises(GraphedTypeError, match="jes_up"):
        graphed.vary(shifted, "A", central, is_weight=True, up=flat * 1.1)
    assert "A_up" not in session._points


def _clash_at_an_unwalked_label() -> tuple[Session, Any]:
    """Two weight factors whose members clash at `q_up` — a label no collection carries, so it is
    not in the recorded union and the record-time walk never asks about it. The JOINT over it is in
    the union, put there by the hand-built escape hatch, waiting for a mint to reach it."""
    session = Session(AwkwardBackend())
    ctx = ga.gnano.events(from_awkward(session, "ev", EVENTS))
    ctx._collections = {**ctx._collections, "hand": rebuild({"nominal": ctx["w"], JOINT: ctx["w"] * 2.0})}
    flat = ctx["w"] * 1.0
    for name, field in (("A", "r3"), ("B", "r5")):
        factor = rebuild({"nominal": flat, "q_up": ctx[field] * 1.0}, context=ctx)
        ctx = graphed.vary(ctx, name, factor, is_weight=True, up=factor * 1.1)
    return session, ctx


JOINT = "b_up__q_up"


def test_a_clash_a_later_mint_creates_raises_at_the_read_and_leaves_that_varys_labels_bound() -> None:
    """The one raise this design moves out of `vary`. A clash at a label NO registration could see
    — neither factor's registration walked it, because nothing carried it — becomes reachable when
    a later, unrelated `vary` mints the joint over it. Resolving as of the read is what finds it;
    the read is where it is reported, naming the label, and the loose `vary` that minted keeps its
    own labels, because that `vary` is not the one at fault and refusing it would blame it.

    The ambient composes for all its labels at once, so the clash takes the whole container down:
    that context reads at NO label until it is resolved. What stays readable is any other context,
    which is what says the raise belongs to the clashing composition and not to the Session.
    """
    session, ctx = _clash_at_an_unwalked_label()
    assert JOINT not in session._points
    labels = graphed.labels(graphed.weight(ctx))
    assert labels, "the ambient does not compose before the mint"
    unclashed = ga.gnano.events(from_awkward(session, "clean", EVENTS))
    factor = unclashed["w"] * 1.0
    unclashed = graphed.vary(unclashed, "C", factor, is_weight=True, up=factor * 1.1)

    minting = graphed.vary(ctx["w"] * 3.0, "q", up=ctx["w"] * 4.5)
    graphed.vary(minting * 2.0, "b", up=minting * 2.2)
    assert {"q_up", JOINT} <= set(session._points)

    for label in labels:
        with pytest.raises(GraphedTypeError, match=JOINT):
            graphed.universe(graphed.weight(ctx), label)
    assert {"q_up", "b_up", JOINT} <= set(session._points), "the loose vary's labels were rolled back"
    assert graphed.universe(graphed.weight(unclashed), "C_up").node_id
