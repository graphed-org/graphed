"""§2.3d: every public Array-consuming module verb carries a `Varied` disposition.

An UNDISPOSED verb does not fail loudly — `Varied`'s label-mapping `getattr` turns an unhandled
duck-typed read into a recorded op and silently compiles nonsense — so the enumeration is
EXHAUSTIVE and DYNAMIC: a new verb is fixed in `src`, never by editing this file. The refusal table
is SPLIT BY CONTRACT because §2.3d binds two of them; one frozen contract for both would red a
conforming implementation. Also carries §2.2's property half, classified by MEASUREMENT, and the
`graphed.context_of`-on-a-container discriminator.
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest
from vary_ctx_fixtures import (
    awkward_session,
    events_context,
    loose_varied,
    partitioned_vector_source,
    pu_weight,
    record_source,
    vector_source,
)

import graphed
import graphed.awkward as ga
import graphed.numpy as gnp
from graphed import BufferProjection, GraphedError, Projection, read_columns
from graphed.awkward import gak

#: §2.3d's bound class set
CLASSES = {"refusing", "expanding", "broadcasting", "eager-metadata", "accepting"}
#: the classes `graphed`'s own table can host at m48 — `accepting` is `Histogram.fill`'s, in the
#: other repo, and `to_parquet`'s only from m51, so this stays a CONTAINMENT floor
M48_CLASS_FLOOR = {"refusing", "expanding", "broadcasting", "eager-metadata"}
#: named because the annotation filter cannot reach them: `compile_ir`'s parameters annotate
#: `Session`/`Any`, and `context_of`/`broadcast_like` are the sole representatives of
#: eager-metadata and broadcasting
FLOOR = {("graphed", "compile_ir"), ("graphed", "context_of"), ("graphed", "broadcast_like")}
#: freeze-time size of the union of the three enumerations (8 discovered in `graphed` + this floor
#: + 6 numpy + 2 awkward); a containment floor, so later verbs cannot red it
UNION_FLOOR = 19


def _mentions_array(fn: object) -> bool:
    """§2.3d's bound filter: ANY parameter annotation mentioning `Array` — not the first parameter,
    which misses `compile_ir` and `apply` (Array operand not first) and `read_columns`
    (`Sequence[Array]`)."""
    try:
        signature = inspect.signature(fn)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    return any("Array" in str(p.annotation) for p in signature.parameters.values())


def _enumerate(module: Any, package: str) -> set[tuple[str, str]]:
    discovered = set()
    for name in getattr(module, "__all__", ()):
        member = getattr(module, name, None)
        if inspect.isfunction(member) and _mentions_array(member):
            discovered.add((package, name))
    return discovered


def _discovered() -> set[tuple[str, str]]:
    """`graphed.vary` is excluded BY NAME: it PRODUCES containers rather than consuming them, so no
    disposition class exists for it, yet its annotations mention `Array`."""
    core = _enumerate(graphed, "graphed") - {("graphed", "vary")}
    return core | FLOOR | _enumerate(gnp, "graphed.numpy") | _enumerate(ga, "graphed.awkward")


def _dispositions() -> dict[tuple[str, str], str]:
    tables = {
        "graphed": graphed.VERB_DISPOSITIONS,
        "graphed.numpy": gnp.VERB_DISPOSITIONS,
        "graphed.awkward": ga.VERB_DISPOSITIONS,
    }
    return {(pkg, name): kind for pkg, table in tables.items() for name, kind in table.items()}


def test_every_discovered_verb_carries_a_disposition() -> None:
    discovered = _discovered()
    dispositions = _dispositions()
    undisposed = sorted(discovered - set(dispositions))
    assert not undisposed, f"{undisposed} would silently compile a recorded field op on a Varied"
    assert {dispositions[key] for key in discovered} <= CLASSES


def test_the_floor_is_asserted_over_the_union_of_the_three_enumerations() -> None:
    """Never per enumeration: neither idiom package hosts a floor member, so a per-enumeration
    floor is unsatisfiable against a correct implementation."""
    discovered = _discovered()
    dispositions = _dispositions()
    assert discovered >= FLOOR
    assert len(discovered) >= UNION_FLOOR
    assert {dispositions[key] for key in discovered} >= M48_CLASS_FLOOR
    # the dynamic half is what makes this self-repairing; an empty one passes tautologically
    assert _enumerate(graphed, "graphed") - {("graphed", "vary")}
    assert _enumerate(gnp, "graphed.numpy") and _enumerate(ga, "graphed.awkward")


def test_evaluate_ir_is_outside_the_array_consuming_surface() -> None:
    """It never receives an `Array`, so there is no `Varied` to refuse and no positive control."""
    assert ("graphed", "evaluate_ir") not in _discovered()
    assert ("graphed", "evaluate_ir") not in _dispositions()


def test_to_parquet_carries_no_disposition_until_m51() -> None:
    """Freezing an m48 entry whose VALUE m51 must change is a freeze-order trap: it is outside
    `graphed.numpy.__all__` and `graphed.awkward.to_parquet` annotates its first parameter `Any`."""
    dispositions = _dispositions()
    for package in ("graphed.numpy", "graphed.awkward"):
        assert (package, "to_parquet") not in _discovered()
        assert (package, "to_parquet") not in dispositions


def test_the_idiom_packages_classifications_are_bound_per_verb() -> None:
    dispositions = _dispositions()
    for verb in ("apply_gufunc", "empty_like", "full_like", "ones_like", "zeros_like"):
        assert dispositions[("graphed.numpy", verb)] == "broadcasting"
    # each expanding verb returns its OWN type per label, never `read_columns`' union treatment
    assert dispositions[("graphed.numpy", "project")] == "expanding"
    assert dispositions[("graphed.awkward", "project")] == "expanding"
    assert dispositions[("graphed.awkward", "project_buffers")] == "expanding"


def test_the_boundary_and_plan_verbs_refuse_without_silently_compiling() -> None:
    """Contract one. m48 asserts only that they raise; §5.4's exact message shape is m49's. The
    positive control is what makes it a REFUSAL rather than a broken signature."""
    _s, root = awkward_session()
    plain = root.MET.pt
    varied = graphed.vary(plain, "sig", up=plain * 1.1)
    calls = {
        "repartition": lambda operand: graphed.repartition(operand, n=2),
        "pack_key": lambda operand: graphed.pack_key(operand, on=["MET"]),
        "join": lambda operand: graphed.join(operand, plain, on=["MET"]),
        "shuffle_plan": lambda operand: graphed.shuffle_plan(
            operand, reduce=list, combine=lambda a, b: a, empty=list
        ),
        "join_plan": lambda operand: graphed.join_plan(operand),
    }
    for verb, call in calls.items():
        raised: Exception | None = None
        try:
            call(varied)
        except Exception as exc:
            raised = exc
        assert raised is not None, f"graphed.{verb} silently accepted a Varied"
    assert graphed.repartition(plain, n=2).node_id != plain.node_id  # the control: plain still works


def test_the_compile_and_aggregate_verbs_refuse_naming_graphed_universe() -> None:
    """Contract two. They consume `arr.node_id`/`arr.session` directly, and §2.2's reserved-name
    rule makes that a clean seam — the varied route to a plan is §6.1c's group API."""
    session, x = partitioned_vector_source()
    varied = loose_varied(x)
    reducer = {"reduce": lambda values: values, "combine": lambda a, b: a, "empty": list}
    with pytest.raises(GraphedError, match=r"graphed\.universe"):
        graphed.compile_ir(session, varied)
    with pytest.raises(GraphedError, match=r"graphed\.universe"):
        graphed.aggregate_plan(varied, **reducer)
    assert graphed.compile_ir(session, x * 2.0).ir  # plain-Array positive controls
    assert graphed.aggregate_plan(x * 2.0, **reducer).process is not None


def test_the_expanding_verbs_are_asserted_PER_VERB() -> None:
    """A blanket per-label-shape wording is false for `read_columns`, which answers with ONE
    union read set, not a mapping."""
    _s, root = awkward_session()
    varied = graphed.vary(root.MET.pt, "sig", up=root.MET.pt * 1.1)
    expanded = graphed.apply(lambda member: member, varied, name="identity")
    assert isinstance(expanded, graphed.Varied)
    assert list(graphed.labels(expanded)) == list(graphed.labels(varied))

    union = graphed.vary(gak.num(root.Jet) * 1.0, "sig", up=root.MET.pt)
    assert read_columns([union], root.node_id) == ("Jet", "MET")
    # `None` DOMINATES: a plain set union would silently narrow the conservative label's read list
    opaque = root.map(lambda events: events).MET.pt
    conservative = graphed.vary(root.MET.pt, "sig", up=opaque)
    assert read_columns([conservative], root.node_id) is None
    assert read_columns([opaque], root.node_id) is None  # the member really is the conservative one


def test_the_idiom_expanding_verbs_return_their_own_type_per_label() -> None:
    _s, root = awkward_session()
    varied = graphed.vary(root.MET.pt, "sig", up=root.MET.pt * 1.1)
    projections = ga.project(varied)
    buffers = ga.project_buffers(varied)
    assert set(projections) == set(graphed.labels(varied))
    assert all(isinstance(value, Projection) for value in projections.values())
    assert all(isinstance(value, BufferProjection) for value in buffers.values())


def test_reindex_to_broadcasts_and_unify_contexts_carries_no_disposition() -> None:
    """Both m48 verbs are disposed here so the gate does not discover `reindex_to` unclassified;
    `unify_contexts` takes context HANDLES rather than `Array`s, like `evaluate_ir`."""
    assert _dispositions()[("graphed", "reindex_to")] == "broadcasting"
    assert ("graphed", "unify_contexts") not in _dispositions()
    assert ("graphed", "unify_contexts") not in _discovered()


def test_node_id_and_session_raise_while_the_field_of_that_name_still_reads() -> None:
    _s, x = vector_source()
    varied = loose_varied(x)
    for name in ("node_id", "session"):
        with pytest.raises(AttributeError):
            getattr(varied, name)
    # the rule is a PROPERTY rule, so it cannot be a blanket `__getattr__` refusal: a record
    # carrying a literal `node_id` FIELD must still read it through string getitem
    _rs, record = record_source()
    varied_record = graphed.vary(record, "sf", up=record, down=record)
    assert graphed.nominal(varied_record["node_id"]).node_id == record["node_id"].node_id


def test_the_property_half_is_classified_by_measurement() -> None:
    """`dtype`/`ndim`/`shape` are `_form_meta`-backed and record NOTHING; `T` is a plain alias for
    the recorded `transpose` op. A blanket "delta 0 for every property" reds a correct
    implementation on `T`. The fixture is a 1-D VECTOR source by necessity: `NumpyArray.T` raises
    on a >=2-D partitioned form and on any record form, so the measurement itself would raise."""
    session, x = vector_source()
    varied = loose_varied(x)

    before = session.node_count()
    _eager = x.dtype
    assert session.node_count() == before  # the eager representative, measured not assumed
    before = session.node_count()
    _recorded = x.T
    assert session.node_count() == before + 1  # the recording representative

    before = session.node_count()
    assert varied.dtype == graphed.nominal(varied).dtype
    assert session.node_count() == before
    transposed = varied.T
    assert isinstance(transposed, graphed.Varied)
    assert list(graphed.labels(transposed)) == list(graphed.labels(varied))


def test_context_of_on_a_container_answers_with_the_MOST_DERIVED_handle() -> None:
    """§2.3e: `vary` accepts members whose handles differ along one ancestry chain, so the
    most-derived handle may belong to a NON-nominal member — which is what §6.4a(2a)'s
    `context_of(mask) is context_of(record)` predicate reads. The fixture is a `vary` IDENTITY
    link, which keeps the row space fixed."""
    _s, events = events_context()
    events2 = graphed.vary(events, "pu", pu_weight(events, 1.0), is_weight=True, up=pu_weight(events, 1.1))
    container = graphed.vary(events.Jet, "jes", up=events2.Jet)
    assert graphed.context_of(container) is events2
    assert graphed.context_of(graphed.nominal(container)) is events
    _s2, root = awkward_session()
    assert graphed.context_of(root) is None  # context-free reads answer `None`, not a stub handle
