"""Guards for the O(N) ambient weight composition (§4/§5 of the weight-composition design).

The frozen suites verify the ambient's VALUES, and a composition quadratic in the registered
weight families satisfies every one of them. The cost laws are therefore pinned here, in
multiply/getitem counts rather than wall time — the counts are deterministic, so a law is
assertable exactly.

Numpy-idiom only, so the guards stay cheap and portable across the CI matrix. Two registration
cases cannot be expressed here and live in `tests/extra/awkward`: the per-LABEL form clash needs
a backend whose `add` and `mul` disagree, and the §2.5 residue case needs a refused factor that
reads the shifted collection, which no numpy op builds out of one.
"""

from __future__ import annotations

import gc
from typing import Any

import numpy as np
import pytest

import graphed
import graphed.accessors
from graphed import Array, Kind, Session, compile_ir, context
from graphed.context import EventContext
from graphed.errors import GraphedTypeError
from graphed.numpy import NumpyBackend, from_record
from graphed.varied import Varied, member_of, rebuild

VEC = np.arange(1.0, 13.0)


def _ops(session: Session, op: str) -> int:
    return sum(1 for recorded, _params, _inputs in session._ops.values() if recorded == op)


def _context() -> tuple[Session, Any, Any]:
    """A numpy-idiom event context over one flat record, plus that record."""
    session = Session(NumpyBackend())
    record = from_record(session, "ev", pt=VEC, w=np.ones(12))
    collections = {"pt": record["pt"], "w": record["w"]}
    return session, EventContext(session, record["pt"], collections=collections), record


def _ambient(ctx: Any) -> Array | Varied:
    """`graphed.weight` answers `... | None`; every context read below has registered factors."""
    ambient = graphed.weight(ctx)
    assert ambient is not None
    return ambient


def _shift(ctx: Any, name: str, pt: Any, scale: float = 1e-3) -> Any:
    return graphed.vary(ctx, name, collections={"pt": {"up": pt * (1 + scale), "down": pt * (1 - scale)}})


def _program(count: int, kind: str, *, shift: bool, read_every: bool, register: bool) -> int:
    """Run one weight program in its own Session; answer the multiplies it recorded.

    Three shapes, which is what makes the guard's three legs distinct: `independent` factors vary at
    their own labels only (every label has one varying factor — the complement pushdown); `shared`
    factors are computed on shifted objects, so every one of them varies at the shift's labels (the
    tree walk, with no minted joint anywhere); `joint` lets the m53 fanout mint the joint labels and
    walks the tree over those; `mixed` alternates shifted and flat factors, so at the shift's labels
    some indices vary and some do not — the only shape in which the tree walk has whole nominal
    subtrees to fold in, and one an implementation handling only "every factor varies" would pass
    without.

    `register=False` builds the same factor arrays and registers none of them: the control the
    composition's own multiplies are differenced against. The count is taken over the whole program
    and not around the read, because a registration onto an already-composed ambient folds there
    rather than at the read.
    """
    session, ctx, record = _context()
    if shift:
        ctx = _shift(ctx, "jes", record["pt"])
    base = _ops(session, "mul")
    for index in range(count):
        if kind == "independent" or (kind == "mixed" and index % 2):
            central = record["w"] * (1.0 + 1e-3 * (index + 1))
            extra: dict[str, Any] = {"down": central * 0.9}
        else:
            central = ctx["pt"] * (1.0 + 1e-3 * (index + 1))
            extra = {"composes_as_union": True} if kind in ("shared", "mixed") else {}
        up = central * 1.1
        if not register:
            continue
        ctx = graphed.vary(ctx, f"f{index}", central, is_weight=True, up=up, **extra)
        if read_every:
            graphed.weight(ctx)
    if register and not read_every:
        graphed.weight(ctx)
    return _ops(session, "mul") - base


def _composition_muls(count: int, kind: str, *, shift: bool = False, read_every: bool = False) -> int:
    """The multiplies the COMPOSITION records: the program minus the identical unregistered one."""
    return _program(count, kind, shift=shift, read_every=read_every, register=True) - _program(
        count, kind, shift=shift, read_every=read_every, register=False
    )


@pytest.mark.parametrize("count", [4, 8, 16, 32])
def test_composing_independent_weight_families_is_linear_in_the_family_count(count: int) -> None:
    # The pushdown leg: N two-tag families, no factor varying at another's label, so every label has
    # exactly one varying factor. `5N - 5` is the closed form of the product tree (N-1) plus the
    # complement pushdown plus one multiply per non-nominal label. A per-registration eager fold
    # costs `N^2 + 2N - 3` on the same program — 1085 against 155 at N=32.
    assert _composition_muls(count, "independent") == 5 * count - 5


@pytest.mark.parametrize("kind", ["shared", "joint", "mixed"])
@pytest.mark.parametrize("count", [4, 8, 16])
def test_a_label_that_several_factors_vary_at_composes_below_the_left_fold(kind: str, count: int) -> None:
    """The tree-walk leg, in three forms: without any minted joint label (weight factors read
    through a shift), with them (the m53 fanout's joints), and with only half the factors varying at
    the shared label, which is where the walk folds whole nominal subtrees into the product.

    The reference is the same program reading at every intermediate context, which costs exactly the
    left fold (pinned below) — so an implementation that quietly fell back to the chain lands ON the
    reference instead of under it.
    """
    _session, ctx, record = _context()
    shifted = _shift(ctx, "jes", record["pt"])
    # the shape that makes this the tree-walk leg: every registered factor is non-nominal at `jes_up`
    families = [shifted["pt"] * (1.0 + 1e-3 * (index + 1)) for index in range(count)]
    assert all(
        graphed.universe(family, "jes_up").node_id != graphed.nominal(family).node_id for family in families
    ), "this program does not put two varying factors at one label"

    once = _composition_muls(count, kind, shift=True)
    chain = _composition_muls(count, kind, shift=True, read_every=True)
    assert once < chain, f"{kind} composition costs the left fold's {chain} multiplies"


def _left_fold(count: int) -> int:
    """The multiplies a per-registration eager fold records over `count` independent families: at
    registration `i > 0` one multiply per recorded label, and the union carries `2i + 3` of them."""
    return count * count + 2 * count - 3


@pytest.mark.parametrize("count", [4, 8, 16])
def test_reading_the_ambient_at_every_context_costs_exactly_the_left_fold(count: int) -> None:
    # The residual the laziness leaves: a registration onto an ambient that already stands composed
    # FOLDS onto it, so a program that reads at every intermediate context pays ONE eager chain —
    # remaking the composition at each of those reads instead would cost twice it.
    assert _composition_muls(count, "independent", read_every=True) == _left_fold(count)


def _derived(weights: int, shifts: int, *, with_weights: bool = True) -> tuple[int, int, int]:
    """One `ctx[mask]` on a context carrying `shifts` shift families and `weights` weight families:
    the getitems and multiplies it records, and the child ambient's label count."""
    session, ctx, record = _context()
    for index in range(shifts):
        ctx = _shift(ctx, f"s{index}", record["pt"], 1e-3 * (index + 1))
    if with_weights:
        ctx = _register_context(ctx, record, weights)
    mask = ctx["pt"] > 2.0
    getitems, muls = _ops(session, "getitem"), _ops(session, "mul")
    ambient = graphed.weight(ctx[mask])
    return (
        _ops(session, "getitem") - getitems,
        _ops(session, "mul") - muls,
        len(graphed.labels(ambient)) if ambient is not None else 0,
    )


def _register_context(ctx: Any, record: Any, count: int) -> Any:
    for index in range(count):
        central = record["w"] * (1.0 + 1e-3 * (index + 1))
        ctx = graphed.vary(ctx, f"f{index}", central, is_weight=True, up=central * 1.1, down=central * 0.9)
    return ctx


@pytest.mark.parametrize(("weights", "shifts"), [(4, 0), (4, 8), (16, 8)])
def test_one_derivation_re_indexes_the_ambient_once_per_label(weights: int, shifts: int) -> None:
    """Two axes, because per-LABEL cost is what separates this design from re-indexing every factor:
    holding the weight families and growing the shift families must keep the ambient at one re-index
    per label, while per-factor re-indexing would charge factors x labels (at (16, 8): 784, not 49).

    The ambient's own share is isolated by differencing against the identical program with no weight
    registered — the record, the collections and the mask are re-indexed either way.
    """
    getitems, _muls, labels = _derived(weights, shifts)
    without, _m, _l = _derived(weights, shifts, with_weights=False)
    assert getitems - without == labels


@pytest.mark.parametrize(("weights", "shifts"), [(4, 0), (4, 8), (16, 8)])
def test_a_derivation_forces_only_the_multiplies_a_read_would_have_forced(weights: int, shifts: int) -> None:
    # Composing at the parent before re-indexing is what bounds the derivation: the nodes it forces
    # are the ones the parent's own `graphed.weight` builds, hash-consed with them.
    _getitems, muls, _labels = _derived(weights, shifts)

    session, ctx, record = _context()
    for index in range(shifts):
        ctx = _shift(ctx, f"s{index}", record["pt"], 1e-3 * (index + 1))
    ctx = _register_context(ctx, record, weights)
    base = _ops(session, "mul")
    graphed.weight(ctx)
    assert muls == _ops(session, "mul") - base


def test_a_record_typed_weight_factor_is_refused_inside_vary_and_leaves_its_labels_free() -> None:
    """The cross-factor form clash `check_members` cannot see: it compares a factor against its own
    nominal, never against what is already registered. Refused at REGISTRATION, so `vary`'s
    transactional mint rolls the labels back and the name stays usable."""
    session, ctx, record = _context()
    weight = record["w"]
    registered = graphed.vary(ctx, "pu", weight * 1.0, is_weight=True, up=weight * 1.1)

    with pytest.raises(GraphedTypeError, match="mul"):
        graphed.vary(registered, "sf", record, is_weight=True, up=record)
    assert sorted(session._points) == ["pu_up"]

    # the admitted member: the same name, a well-typed factor, still registers
    again = graphed.vary(registered, "sf", weight * 1.02, is_weight=True, up=weight * 1.2)
    assert graphed.labels(graphed.weight(again)) == ("nominal", "pu_up", "sf_up")


def test_a_label_reached_only_through_a_factors_own_array_is_still_reported_unreached() -> None:
    """§2.5's unreached-label diagnostic. `register()` splits: the labels are recorded at vary time,
    but the `_labels` stamps go on the COMPOSED members, never on the user's own `up=` array — so an
    output built from that array does not count the label as reached, which is the silent-cost case.

    Two families and not one: with a single family the ambient's member IS the user's array, so a
    one-family program cannot tell the two placements apart.
    """
    session, ctx, record = _context()
    weight = record["w"]
    first = graphed.vary(ctx, "pu", weight, is_weight=True, up=weight * 1.1, down=weight * 0.9)
    btag_up = weight * 1.2
    graphed.vary(first, "btag", weight * 1.05, is_weight=True, up=btag_up, down=weight * 0.95)

    reported = compile_ir(session, btag_up * 2.0).unreached_labels
    assert reported == ("btag_down", "btag_up", "pu_down", "pu_up")


def test_a_factor_nests_one_level_and_no_further() -> None:
    """§2.2's one legal level — a factor whose member is itself a container, which is what a factor
    computed on shifted objects is — composes, because the walk reads two levels deep and lands on
    an array. A second level would compose into an ambient universe that is itself a container,
    which no consumer can read (`graphed.universe(...).node_id` included), so it is refused while
    `vary` can still roll the label back.
    """
    session, ctx, record = _context()
    weight = record["w"]
    inner = rebuild({"nominal": weight * 1.0, "a_up": weight * 1.1})
    registered = graphed.vary(ctx, "deep", inner, is_weight=True, up=weight * 1.3)
    assert graphed.labels(graphed.weight(registered)) == ("nominal", "deep_up")

    twice = rebuild({"nominal": inner, "b_up": weight * 1.2})
    with pytest.raises(GraphedTypeError, match="nominal"):
        graphed.vary(ctx, "deeper", twice, is_weight=True, up=weight * 1.3)
    assert "deeper_up" not in session._points


@pytest.mark.parametrize("read_the_parent_first", [False, True])
def test_the_composed_ambient_carries_the_registering_context_not_the_reading_one(
    read_the_parent_first: bool,
) -> None:
    """§2.3e's ORIGINATION handle on a DEFERRED ambient is the context that last CHANGED the factor
    list, never the one that happens to compose it. Stamping the reader would make
    `graphed.weight(parent)` observable: two sibling `vary` children would answer with two
    different handles and §2.3e would refuse to combine them — an evaluation-order effect on a
    program whose structure did not change.
    """
    _session, ctx, record = _context()
    weight, pt = record["w"], record["pt"]
    registered = graphed.vary(ctx, "pu", weight, is_weight=True, up=weight * 1.1)
    if read_the_parent_first:
        graphed.weight(registered)
    left = _shift(registered, "jes", pt)
    right = _shift(registered, "jer", pt, scale=2e-3)

    assert graphed.accessors.context_of(_ambient(left)) is registered
    assert graphed.accessors.context_of(_ambient(right)) is registered
    # what the moved handle would break: §2.3e refuses to combine a descendant-stamped ambient
    # with a sibling's read, and refuses to re-index it back to the context that registered it
    product = graphed.nominal(graphed.weight(left)) * right["w"]
    assert graphed.accessors.context_of(product) is right
    assert graphed.accessors.reindex_to(_ambient(left), registered) is graphed.weight(left)


def test_variations_answers_an_empty_registry_on_a_projected_context() -> None:
    """§2.2's projection drops the registry, and `graphed.variations` reports that drop rather than
    faulting on it: a projected context's ambient factor is one label's bare member, which carries
    no §1.1 tag map. Reading the map off the COMPOSED container instead reaches for `_tags` on a
    plain `Array`."""
    _session, ctx, record = _context()
    weight = record["w"]
    registered = graphed.vary(ctx, "pu", weight, is_weight=True, up=weight * 1.1)
    assert graphed.variations(registered) == {"pu": {"up": (Kind.WEIGHT, None)}}
    assert graphed.variations(graphed.universe(registered, "pu_up")) == {}
    assert graphed.variations(graphed.nominal(registered)) == {}


def _count_resolutions(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Count §4.6 member resolutions, the walk's own unit of work."""
    calls: list[int] = []
    original = Varied._member_for

    def counted(self: Varied, label: str) -> Any:
        calls.append(1)
        return original(self, label)

    monkeypatch.setattr(Varied, "_member_for", counted)
    return calls


def _walk_resolutions(calls: list[int], count: int) -> tuple[int, int, Any]:
    """Register `count` independent weight families without reading, and answer the resolutions the
    record-time walk performed, the (factor, label) pairs it asked about, and the Session."""
    del calls[:]
    session, ctx, record = _context()
    for index in range(count):
        central = record["w"] * (1.0 + 1e-3 * (index + 1))
        ctx = graphed.vary(ctx, f"f{index}", central, is_weight=True, up=central * 1.1, down=central * 0.9)
    return len(calls), len(ctx._factors) * len(ctx._recorded), session


@pytest.mark.parametrize("count", [4, 8, 16])
def test_the_record_time_walk_resolves_each_factor_and_label_once(
    count: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§5's check re-runs the whole composition at every registration, so what decides construction
    cost is how many §4.6 resolutions those re-runs perform, not the multiplies they record.
    Memoised per (factor container, label), the walk pays a fixed number of resolutions per pair
    however many families are registered; re-resolving puts a factor list's worth of them on every
    registration, one order above the node count this composition exists to bound.
    """
    calls = _count_resolutions(monkeypatch)
    reference, reference_pairs, _session = _walk_resolutions(calls, 2)
    resolutions, pairs, session = _walk_resolutions(calls, count)

    assert resolutions * reference_pairs == reference * pairs
    assert sum(len(page) for page in session._universes.values()) == pairs


JOINT = "b_up__jes_up"


def _hand_built_joint(ctx: Any, record: Any) -> Any:
    """A context carrying a label the point registry has never seen.

    `graphed.varied.rebuild` is the hand-built escape hatch, so a collection can carry `JOINT` into
    the recorded union long before any `vary` mints it — which is what lets a mint land on a label
    the composition has already been asked about.
    """
    return {**ctx._collections, "hand": rebuild({"nominal": record["w"], JOINT: VEC * 1.0})}


def _resolved_product(session: Session, ctx: Any, label: str) -> np.ndarray:
    """The oracle: the eager product of what each registered factor resolves to at `label` under
    the registry as it stands NOW, read two levels deep and multiplied here rather than by the
    composition under test."""
    product: Any = None
    for factor in ctx._factors:
        member = member_of(member_of(factor, label), label)
        product = member if product is None else product * member
    resolved: np.ndarray = np.asarray(session.materialize(product))
    return resolved


def _minted_by_the_second_registration(*, read_first: bool) -> tuple[Session, Any]:
    """Two weight families, where the SECOND registration's own fanout mints `JOINT` and so moves
    what the first factor resolves to there."""
    session, ctx, record = _context()
    shifted = _shift(ctx, "jes", record["pt"])
    shifted._collections = _hand_built_joint(shifted, record)
    left = graphed.vary(shifted, "a", shifted["pt"] * 0.5, is_weight=True, up=shifted["pt"] * 0.6)
    assert JOINT in left._recorded and JOINT not in session._points
    if read_first:
        graphed.weight(left)
    both = graphed.vary(left, "b", shifted["pt"] * 1.0, is_weight=True, up=shifted["pt"] * 1.2)
    assert JOINT in session._points
    return session, both


def _minted_by_a_later_loose_vary(*, read_first: bool) -> tuple[Session, Any]:
    """Two weight families, then an unrelated LOOSE `vary` elsewhere in the analysis mints `JOINT`
    after the last registration — where no record-time check runs at all."""
    session, ctx, record = _context()
    shifted = _shift(ctx, "jes", record["pt"])
    shifted._collections = _hand_built_joint(shifted, record)
    for name in ("a", "b_weight"):
        factor: Any = rebuild(
            {"nominal": record["w"] * 1.0, "jes_up": record["w"] * 1.01, "jes_down": record["w"] * 1.0},
            context=shifted,
        )
        shifted = graphed.vary(shifted, name, factor, is_weight=True, up=factor * 1.1)
    if read_first:
        graphed.weight(shifted)
    depends_on_jes = shifted["pt"] * 1.0
    graphed.vary(depends_on_jes, "b", up=depends_on_jes * 1.1)
    assert JOINT in session._points
    return session, shifted


@pytest.mark.parametrize("program", [_minted_by_the_second_registration, _minted_by_a_later_loose_vary])
def test_a_mint_after_the_walk_first_asked_about_a_label_moves_what_the_ambient_resolves_there(
    program: Any,
) -> None:
    """§3 clause 1: a factor's member is resolved against the registry AS OF THE READ. A label the
    registry could not decompose when the factor was registered therefore stops answering with the
    central universe once a mint reaches it, in both shapes that can mint one late — the next
    registration's own fanout, and a loose `vary` elsewhere in the analysis.

    Whether `graphed.weight()` was called before that mint must not enter the answer: the composed
    ambient is a cache under the Session's mint epoch, so a read before the mint is remade by the
    read after it. The oracle multiplies the members itself, so the guard cannot pass by agreeing
    with the composition's own association.
    """
    unread_session, unread = program(read_first=False)
    read_session, read = program(read_first=True)

    value = np.asarray(unread_session.materialize(graphed.universe(graphed.weight(unread), JOINT)))
    again = np.asarray(read_session.materialize(graphed.universe(graphed.weight(read), JOINT)))
    oracle = _resolved_product(unread_session, unread, JOINT)

    assert np.allclose(value, oracle, rtol=1e-12, atol=0.0)
    assert np.allclose(again, oracle, rtol=1e-12, atol=0.0)
    # the discriminator: the pre-mint answer is the central universe, and it is NOT this one
    central = _resolved_product(unread_session, unread, "nominal")
    assert not np.allclose(value, central, rtol=1e-12, atol=0.0)


def test_an_unsettled_ambient_leaves_the_record_time_check_walking_the_original_factors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One predicate decides both whether a registration folds onto the composed ambient and which
    operands the record-time check walks, so the check validates exactly what the read composes.

    An ambient carrying a label no `vary` has minted is where the two can part: it stands composed
    at the current epoch, so it looks foldable, but a mint can still move an operand of it and the
    read therefore remakes from the original factors. `mul` inference is not associative for
    raise/no-raise, so a check walking the fold instead judges a product the read never builds.
    """
    session, ctx, record = _context()
    shifted = _shift(ctx, "jes", record["pt"])
    shifted._collections = _hand_built_joint(shifted, record)
    left = graphed.vary(shifted, "a", shifted["pt"] * 0.5, is_weight=True, up=shifted["pt"] * 0.6)
    graphed.weight(left)
    memo = left._memo
    assert memo is not None and not memo[3], "this shape must leave the ambient composed and unsettled"
    del session

    walked: list[list[Any]] = []
    checked = context._check_forms

    def spy(session: Any, factors: Any, labels: Any, overlays: Any, fixed: Any) -> None:
        walked.append(list(factors))
        checked(session, factors, labels, overlays, fixed)

    monkeypatch.setattr(context, "_check_forms", spy)
    graphed.vary(left, "b", shifted["pt"] * 1.0, is_weight=True, up=shifted["pt"] * 1.2)

    assert [id(operand) for operand in walked[-1][:-1]] == [id(factor) for factor in left._factors]
    assert all(operand is not memo[2] for operand in walked[-1]), "the check walked the fold"


def test_an_unrelated_mint_after_the_last_registration_costs_the_next_read_no_new_multiplies() -> None:
    """The remake is a re-walk, not a rebuild: M1 interns every node, so recomposing from the same
    factors after a mint that moved nothing the ambient reads records no node at all. This is what
    keeps §3 clause 1's read-time resolution affordable — the cost of a mint is a dict walk.
    """
    session, ctx, record = _context()
    for index in range(8):
        central = record["w"] * (1.0 + 1e-3 * (index + 1))
        ctx = graphed.vary(ctx, f"f{index}", central, is_weight=True, up=central * 1.1)
    first = graphed.weight(ctx)

    loose = record["pt"] * 3.0
    graphed.vary(loose, "elsewhere", up=loose * 1.1)
    before = _ops(session, "mul")
    again = graphed.weight(ctx)

    assert again is not first, "the mint did not invalidate the composition, so nothing was remade"
    assert _ops(session, "mul") == before, "the remake minted a node"
    assert graphed.labels(again) == graphed.labels(first)
    assert all(
        graphed.universe(again, label).node_id == graphed.universe(first, label).node_id
        for label in graphed.labels(first)
    )


def test_a_projection_leaves_a_bare_factor_the_walk_still_multiplies() -> None:
    """§2.2's projection replaces the factor list with ONE label's member, which is an array and not
    a container, so the walk has a factor with no universes to resolve. It still multiplies, and a
    family registered on the projected context composes onto it — taking no memo entry, which would
    hold the array for the Session and buy nothing."""
    session, ctx, record = _context()
    weight = record["w"]
    projected = graphed.universe(graphed.vary(ctx, "pu", weight, is_weight=True, up=weight * 1.1), "pu_up")
    assert not isinstance(projected._factors[0], Varied)

    later = graphed.vary(projected, "sf", weight * 2.0, is_weight=True, up=weight * 2.1)
    ambient = graphed.weight(later)
    assert graphed.labels(ambient) == ("nominal", "sf_up")
    assert graphed.nominal(ambient).node_id == (projected._factors[0] * (weight * 2.0)).node_id
    assert id(projected._factors[0]) not in session._universes


def test_a_projected_weight_stays_the_member_across_a_later_mint() -> None:
    """A projected context's ambient is ONE resolved member. A mint elsewhere in the Session moves
    the epoch, and the read must still hand back that member — not a one-label container around
    it — so what a projection returns does not depend on what registered since."""
    _session, ctx, record = _context()
    weight = record["w"]
    registered = graphed.vary(ctx, "pu", weight, is_weight=True, up=weight * 1.1)
    projected = graphed.universe(registered, "pu_up")
    before = graphed.weight(projected)
    graphed.vary(registered, "sf", weight * 2.0, is_weight=True, up=weight * 2.1)  # a mint, elsewhere
    after = graphed.weight(projected)
    assert before is not None and after is not None
    assert not isinstance(before, Varied)
    assert not isinstance(after, Varied)
    assert after.node_id == before.node_id == (weight * 1.1).node_id


def _minting_shapes() -> dict[str, Any]:
    """Every way a program can change the point registry, as a callable on a fresh context."""

    def weight(ctx: Any, record: Any) -> None:
        graphed.vary(ctx, "pu", record["w"], is_weight=True, up=record["w"] * 1.1)

    def shift(ctx: Any, record: Any) -> None:
        _shift(ctx, "jes", record["pt"])

    def loose(_ctx: Any, record: Any) -> None:
        graphed.vary(record["pt"] * 1.0, "scale", up=record["pt"] * 1.1)

    def rolled_back(ctx: Any, record: Any) -> None:
        registered = graphed.vary(ctx, "pu", record["w"], is_weight=True, up=record["w"] * 1.1)
        with pytest.raises(GraphedTypeError):
            graphed.vary(registered, "sf", record, is_weight=True, up=record)

    return {"weight": weight, "shift": shift, "loose": loose, "rolled back": rolled_back}


@pytest.mark.parametrize("shape", list(_minting_shapes()))
def test_every_change_to_the_point_registry_moves_the_mint_epoch(shape: str) -> None:
    """The epoch is what tells a composed ambient it may still stand, so a registry change it does
    not see is a resolution that moves under a cache claiming to be current. `vary._bind_points` and
    `vary`'s rollback are the only two writers of the registry, and both move it — including the
    rollback, which leaves the registry as it found it but can move a resolution BACKWARDS.
    """
    session, ctx, record = _context()
    _minting_shapes()[shape](ctx, record)  # the rolled-back shape has to mint before it refuses
    epoch, registry = session._mint_epoch, dict(session._points)
    _minting_shapes()[shape](ctx, record)
    assert session._mint_epoch != epoch
    if shape == "rolled back":
        assert session._points == registry, "this shape is meant to leave the registry as it was"


@pytest.mark.parametrize("read_every", [False, True])
def test_the_resolution_memo_holds_no_container_the_analysis_has_dropped(read_every: bool) -> None:
    """The memo is keyed by container identity, so it must not be what keeps a container alive:
    a long analysis would otherwise accumulate every factor it ever registered.

    Both ends of the class, because what a page STORES decides it: a resolved member is an array
    carrying its context, and a context holds the factors and the composed containers whose pages
    those are — a cycle rooted in the live Session, which gc cannot break and no finalizer sees.
    Reading once at the end never composes an intermediate ambient, so it cannot show that; reading
    at every context — the shape a histogram fill has, one ambient read per fill — does.
    """
    session, ctx, record = _context()

    def program() -> int:
        inner = ctx
        for index in range(4):
            central = record["w"] * (1.0 + 1e-3 * (index + 1))
            inner = graphed.vary(inner, f"f{index}", central, is_weight=True, up=central * 1.1)
            if read_every:
                graphed.weight(inner)
        graphed.weight(inner)
        return len(session._universes)

    assert program() > 0, "no container was memoised, so this guard measures nothing"
    gc.collect()
    assert session._universes == {}
