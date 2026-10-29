"""§4's composition: the ambient at every label as one product walk, over arrays or over forms.

The same walk runs at the read and, over forms, at registration (`_check_forms`), so the
record-time type check cannot disagree with the composition it guards.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from ...errors import GraphedError, GraphedTypeError
from ...provenance import capture
from ..points import restrict
from ..varied import (
    Varied,
    key_for,
    labels_of,
    member_nodes,
    member_of,
    members_of,
    point_registry,
    registered_points,
    session_of,
)


def _marks(
    entries: Sequence[Any], slots: Sequence[int], overlays: frozenset[int]
) -> frozenset[int]:
    """Return the overlay slots as the OBJECT ids `_compose` tests.

    See `_materialise`.
    """
    return frozenset(
        id(entry) for entry, slot in zip(entries, slots, strict=True) if slot in overlays
    )


def two_level(container: Any, label: str) -> Any:
    """§2.1's `factor[L]`: the container's member for L, then.

    When that member is ITSELF a `Varied` (a registered factor computed on shifted objects) — its
    own member for L. The composed ambient weight is therefore always FLAT.

    Memoised on the Session, because the answer is asked again at every later registration: the
    record-time check re-runs the composition over the whole factor list, and §4.6's resolution
    (a point restriction and a scan of the container's registered points) is not a dict lookup.

    What is stored is the resolved LABEL PATH, never the member: a member is an array carrying its
    context, and a context holds its factors, so a stored member roots the whole analysis in the
    live Session and the page's finalizer never fires. On a hit the member is rebuilt from the live
    container by two dict lookups, which is what the memo was buying in the first place.
    """
    if not isinstance(container, Varied):
        return container
    session = session_of(container)
    if session is None:
        return member_of(member_of(container, label), label)
    page = session.systematics.universe_page(container)
    path = page.get(label)
    if path is not None:
        member = members_of(container)[path[0]]
        return member if path[1] is None else members_of(member)[path[1]]
    outer_key = key_for(container, label)
    member = members_of(container)[outer_key]
    inner_key = key_for(member, label) if isinstance(member, Varied) else None
    if _settled(container, label):
        page[label] = (outer_key, inner_key)
    return member if inner_key is None else members_of(member)[inner_key]


def _settled(container: Any, label: str) -> bool:
    """Whether a later mint could still move `two_level(container, label)`.

    The one thing that makes the memo above safe to keep across registrations.

    §4.6 reads the point registry at each level, and every read that finds NOTHING is an answer
    the next `vary` can change: an unregistered label falls to the central universe, and an
    unregistered own label is off the container's axes, so a label minted later can start
    resolving onto it. Those are recomputed rather than stored. A label the container carries
    outright never reads the registry at all, and `"nominal"` is registered nowhere by
    construction.
    """
    return _fixed(container, label) and _fixed(member_of(container, label), label)


def _all_settled(session: Any, operands: Sequence[Any], labels: Sequence[str]) -> bool:
    """Whether every resolution the composition just made is one no later mint can move.

    Read off `two_level`'s memo rather than recomputed: that memo stores an answer exactly when
    `_settled` holds, so a page missing a label is that label's unsettled resolution. It must be
    read straight after the composition — a mint can settle a label the composition read before it.
    A composition that is not settled must not be FOLDED onto, because a mint could move an operand
    it already multiplied; it is remade from the original factors instead.
    """
    state = session.systematics
    return all(
        all(label in page for label in labels)
        for page in (
            state.universe_page(operand) for operand in operands if isinstance(operand, Varied)
        )
    )


def _fixed(value: Any, label: str) -> bool:
    if not isinstance(value, Varied):
        return True
    members = members_of(value)
    if label in members:
        return True
    registry = point_registry(value)
    return label in registry and all(own in registry for own in members if own != "nominal")


# ---- §4's composition: linear in the registered factors ------------------------------------
def _member(member: Any, label: str) -> Any:
    """Return the composition's own operand at a label: the member itself."""
    return member


def _times(left: Any, right: Any, label: str) -> Any:
    """Return the composition's multiply, naming the LABEL it failed at.

    Almost every clash is refused at registration by the walk over forms below, which names the
    label the same way. The one that reaches here is a clash a later mint created at a label no
    registration could walk (§7), and without the label the message says only that some multiply in
    some universe is ill-typed.
    """
    try:
        return left * right
    except GraphedError as exc:
        raise GraphedTypeError(
            "mul", capture(), f"the ambient weight at variation label {label!r}: {exc}"
        ) from exc


def _compose(
    factors: Sequence[Any],
    labels: Sequence[str],
    project: Callable[[Any, str], Any] = _member,
    mul: Callable[[Any, Any, str], Any] = _times,
    overlays: frozenset[int] = frozenset(),
    fixed: frozenset[int] = frozenset(),
) -> dict[str, Any]:
    """Return the ambient at every label.

    The product of each factor's member for that label, read two levels deep so a factor computed
    on shifted objects contributes in the label's own universe.

    An OVERLAY among the entries makes the ambient an ordered sequence of operations rather than a
    product, so it takes the walk below instead of this one.

    The association is chosen so the work shared between labels is built once — a balanced product
    tree over the nominal members, a complement pushdown handing each index the product of all the
    others, and a tree walk for a label two or more factors vary at. `D`, the set of varying
    indices, is decided by MEMBER IDENTITY: a factor's `_tags` never mentioning a label says
    nothing about whether its member there is the nominal one.

    `project` and `mul` make this one walk serve both times it is needed: over arrays when the
    ambient materialises, and over FORMS at registration (`_check_forms`). The backend's multiply
    inference is not associative for raise/no-raise, so a check that MODELLED this walk instead of
    running it would admit clashes the composition then refuses at the read.
    """
    if overlays:
        return _compose_ordered(factors, labels, project, mul, overlays, fixed)
    return _product_over(factors, labels, project, mul)


def _product_over(
    factors: Sequence[Any],
    labels: Sequence[str],
    project: Callable[[Any, str], Any],
    mul: Callable[[Any, Any, str], Any],
) -> dict[str, Any]:
    """Return the balanced product of `factors` at every label.

    `_compose` without the overlays, so a run of product operations between two overlays composes
    exactly as a whole list of them does. An empty run answers with no labels at all, which is how
    a leading or adjacent overlay reads.
    """
    if not factors:
        return {}
    centrals = [two_level(factor, "nominal") for factor in factors]
    central_ids = [member_nodes(central) for central in centrals]
    count = len(centrals)
    tree = _product_tree([project(central, "nominal") for central in centrals], mul)
    complements = _complements(tree, count, mul)
    composed: dict[str, Any] = {}
    for label in labels:
        applied = [two_level(factor, label) for factor in factors]
        varying = [i for i in range(count) if member_nodes(applied[i]) != central_ids[i]]
        if not varying:
            rest = tree[(0, count)]
        elif len(varying) == 1:
            rest = complements[varying[0]]
        else:
            rest = _outside(tree, count, varying, mul)
        parts = [project(applied[i], label) for i in varying]
        composed[label] = _product(parts if rest is None else [rest, *parts], mul, label)
    return composed


def _compose_ordered(
    factors: Sequence[Any],
    labels: Sequence[str],
    project: Callable[[Any, str], Any],
    mul: Callable[[Any, Any, str], Any],
    overlays: frozenset[int],
    fixed: frozenset[int] = frozenset(),
) -> dict[str, Any]:
    """§2.1's ORDERED operations, for a factor list carrying an overlay.

    Each RUN of product operations between two overlays composes with the same balanced tree the
    whole list takes when there is no overlay, so the ambient stays one tree walk per label plus
    one multiply per overlay — an overlay changes the shape of the composition, never its cost
    class. An OVERLAY then replaces the running value at the labels its family spans, because its
    members ARE the whole ambient rescaled: the operations up to it compose exactly to the node
    they were built from, so multiplying would count it twice. Whether an overlay spans a label is
    the same MEMBER-IDENTITY test the product walk uses, EXCEPT at a universe another family
    PLACED at a point carrying the overlay's coordinate (§2.1): the member declared there is the
    user's value for that point, and the join reads the same member two-level, so both outcomes
    agree there. An overlay a projection into its OWN universe FIXED replaces at every label
    instead, because there every value is that universe (§2.3). Order is registration order, so a
    factor registered after an overlay multiplies the overlay's result.
    """
    runs: list[dict[str, Any]] = []
    marks: list[Any] = []
    run: list[Any] = []
    for factor in factors:
        if id(factor) in overlays:
            runs.append(_product_over(run, labels, project, mul))
            marks.append(factor)
            run = []
        else:
            run.append(factor)
    runs.append(_product_over(run, labels, project, mul))
    nominal_ids = [member_nodes(two_level(overlay, "nominal")) for overlay in marks]
    composed: dict[str, Any] = {}
    for label in labels:
        running: Any = runs[0].get(label)
        for index, overlay in enumerate(marks):
            applied = two_level(overlay, label)
            spans = id(overlay) in fixed or (
                member_nodes(applied) != nominal_ids[index]
                and not _placed_elsewhere(overlay, label)
            )
            if running is None or spans:
                running = project(applied, label)
            part = runs[index + 1].get(label)
            if part is not None:
                running = part if running is None else mul(running, part, label)
        composed[label] = running
    return composed


def _resolvable(operands: Sequence[Any], labels: Sequence[str]) -> bool:
    """Whether the fold base can answer the labels this registration mints.

    §4.6 resolves a label by restricting its point to the container's OWN axes and matching an own
    label exactly. A COMPOSED container merges every factor's axes, so a point that reaches two of
    them (a placement, `{muR: 2, muF: 2}`) matches nothing and falls to the central universe, where
    the factor list restricts each axis to its own factor and finds it. Folding onto the base would
    freeze that miss — a placed universe reading its central value, and differently depending on
    whether a `weight()` read happened to leave a memo — so the composition is remade from the
    list.
    """
    for operand in operands:
        if not isinstance(operand, Varied):
            continue
        registry = point_registry(operand)
        carried = registered_points(operand)
        axes = frozenset(nuisance for own in carried.values() for nuisance, _ in own)
        for label in labels:
            point = registry.get(label)
            if point is None or label in members_of(operand):
                continue
            if restrict(point, axes) and key_for(operand, label) == "nominal":
                return False
    return True


def _placed_elsewhere(overlay: Any, label: str) -> bool:
    """§2.1: whether `label` is a universe ANOTHER family PLACED at this coordinate.

    The point it was placed at carries this overlay's coordinate — the tour's `scale_upup` over a
    `muR` factor and a relative-delta `muF`. The member the placing family declared there is the
    user's value for that point, and the join reads the same member two-level, so the overlay must
    leave it alone for the two outcomes to agree.

    `_route` drops the placing family's OWN axis from an additive placement's point, so a placed
    universe is exactly one that no coordinate of its own point names; a default one-at-a-time
    label and a fanned-out joint both carry theirs (`{name: tag}` and `{name: tag, **foreign}`
    under `f"{name}_{tag}"` and `f"{name}_{tag}__{foreign label}"`). The overlay's own universes
    are its members, placed or not, and it still replaces there.
    """
    if label in labels_of(overlay):
        return False
    point = point_registry(overlay).get(label)
    if point is None:
        return False
    return not any(label == f"{n}_{t}" or label.startswith(f"{n}_{t}__") for n, t in point)


def _product(parts: Sequence[Any], mul: Callable[[Any, Any, str], Any], label: str) -> Any:
    product = parts[0]
    for part in parts[1:]:
        product = mul(product, part, label)
    return product


def _product_tree(
    centrals: Sequence[Any], mul: Callable[[Any, Any, str], Any]
) -> dict[tuple[int, int], Any]:
    """`(lo, hi)` -> the product of `centrals[lo:hi]`, split at the midpoint.

    `N-1` multiplies, shared by every label, and a pure function of the registration order.
    """
    tree: dict[tuple[int, int], Any] = {}

    def build(lo: int, hi: int) -> Any:
        if hi - lo > 1:
            mid = (lo + hi) // 2
            tree[(lo, hi)] = mul(build(lo, mid), build(mid, hi), "nominal")
        else:
            tree[(lo, hi)] = centrals[lo]
        return tree[(lo, hi)]

    if centrals:
        build(0, len(centrals))
    return tree


def _complements(
    tree: Mapping[tuple[int, int], Any], count: int, mul: Callable[[Any, Any, str], Any]
) -> list[Any]:
    """Return `out[i]`, the product of every nominal but the `i`-th.

    In O(N) multiplies shared down the tree — so a label exactly one factor varies at costs one
    more. `None` at `count == 1`, where the product of everything else is empty.
    """
    out: list[Any] = [None] * count

    def push(lo: int, hi: int, outside: Any) -> None:
        if hi - lo == 1:
            out[lo] = outside
            return
        mid = (lo + hi) // 2
        left, right = tree[(lo, mid)], tree[(mid, hi)]
        push(lo, mid, right if outside is None else mul(outside, right, "nominal"))
        push(mid, hi, left if outside is None else mul(outside, left, "nominal"))

    if count:
        push(0, count, None)
    return out


def _outside(
    tree: Mapping[tuple[int, int], Any],
    count: int,
    varying: Sequence[int],
    mul: Callable[[Any, Any, str], Any],
) -> Any:
    """Return the maximal subtrees holding none of `varying`, folded left to right.

    `None` when the varying indices cover every leaf.
    """
    parts: list[Any] = []

    def walk(lo: int, hi: int) -> None:
        if not any(lo <= index < hi for index in varying):
            parts.append(tree[(lo, hi)])
        elif hi - lo > 1:
            mid = (lo + hi) // 2
            walk(lo, mid)
            walk(mid, hi)

    walk(0, count)
    return _product(parts, mul, "nominal") if parts else None


# ---- the same walk, over forms: the record-time refusal -------------------------------------
def _check_forms(
    session: Any,
    factors: Sequence[Any],
    labels: Sequence[str],
    overlays: frozenset[int],
    fixed: frozenset[int] = frozenset(),
) -> None:
    """Type-check the composition by RUNNING it over forms, node-free (§5).

    A cross-factor clash is the one thing the composition itself can raise, and `check_members`
    cannot see it — it compares a factor only against its own nominal. Running the walk here keeps
    that refusal inside `vary`'s §4.5 transactional scope, where the minted labels still roll back;
    deferred to the first read it would leave them bound for the Session's life.
    """
    _compose(
        factors,
        labels,
        lambda member, label: _member_form(session, member, label),
        lambda left, right, label: _mul_form(session, left, right, label),
        overlays,
        fixed,
    )


def _member_form(session: Any, member: Any, label: str) -> Any:
    """Return the form of a member the composition multiplies.

    `two_level` resolves §2.2's one legal level of nesting, so an ordinary factor arrives here as
    an array. One that varies past that level would compose into an ambient universe that is
    itself a container — which no consumer can read, `graphed.universe(...).node_id` included — so
    it is refused here, inside `vary`, rather than at the read that trips over it.
    """
    if isinstance(member, Varied):
        raise GraphedTypeError(
            "mul",
            capture(),
            f"the ambient weight at variation label {label!r}: this factor's member is itself a "
            f"container over {list(members_of(member))}, one universe deeper than a weight factor "
            "nests",
        )
    return session.form(member)


def _mul_form(session: Any, left: Any, right: Any, label: str) -> Any:
    """Return the product form of two member forms, memoised per Session (`mul_forms`).

    The walk multiplies one factor's handful of member forms at a time, so the distinct pairs stay
    few whatever the union's size; without the memo it would pay a backend inference per multiply.
    The key is the `Form` protocol's own `describe()` and NEVER `str(form)`: a repr may abbreviate
    — awkward's elides the middle of a deep type — so two forms that multiply differently would
    share a key and the memo would admit the very clash the walk exists to refuse.
    """
    key = (left.describe(), right.describe())
    memo = session.systematics.mul_forms
    hit = memo.get(key)
    if hit is not None:
        return hit
    try:
        form = session.backend.op_form("mul", [left, right], {})
    except Exception as exc:  # the clash the composition would raise, named at its own label
        raise GraphedTypeError(
            "mul", capture(), f"the ambient weight at variation label {label!r}: {exc}"
        ) from exc
    memo[key] = form
    return form


def _union(*groups: Sequence[str]) -> tuple[str, ...]:
    out: dict[str, None] = {"nominal": None}
    for group in groups:
        for label in group:
            out.setdefault(label, None)
    return tuple(out)
