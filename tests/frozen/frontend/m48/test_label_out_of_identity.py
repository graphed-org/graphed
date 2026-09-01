"""§1.2: variation labels are frontend metadata, never structural identity.

Scoped to the DEFAULT SIBLING lowering — §1.2's §6.2 carve-out deliberately makes both clauses
false in m50's axis mode, where the labels become output content. Also carries §5.2a's dedup
witness and, for §7.2's merge guard, the UNVARIED SCOPE positive control that lives in `graphed`.
"""

from __future__ import annotations

import numpy as np
from vary_fixtures import VECTOR, loose_varied, sibling_outputs, vector_source

import graphed
from graphed import compile_ir, evaluate_ir
from graphed.core import GraphStore
from graphed.numpy import NumpyBackend, from_array


def _recorded_text(session: graphed.Session) -> list[str]:
    """Every `name` and every params key/value the record-time store carries, as text.

    `s._store.nodes()` is the house route to the record-time store
    (`graphed-histogram tests/frozen/m29/test_multi_weight_fills.py`).
    """
    text: list[str] = []
    for node in session._store.nodes():
        text.append(str(node["name"]))
        for key, value in dict(node["params"]).items():
            text.append(f"{key}={value}")
    return text


def test_no_node_name_or_params_carries_a_label() -> None:
    s, x = vector_source()
    varied = loose_varied(x)
    labels = graphed.labels(varied)
    assert labels[0] == "nominal" and len(labels) == 3  # the program under test really is varied
    haystack = _recorded_text(s)
    assert haystack  # the store was read, not an empty walk
    for label in labels:
        assert not any(label in entry for entry in haystack), (
            f"label {label!r} entered a NodeKey name/param; §1.2 forbids it, and a label in params "
            "forks interning so two structurally identical universes stop deduplicating"
        )


def test_renaming_every_label_leaves_the_compiled_ir_byte_identical() -> None:
    """The clause that subsumes any token wording: it reaches the fused `stage` nodes' members,
    which the record-time store never carries."""
    ir = []
    for name, tags in (("jes", ("up", "down")), ("btag", ("hi", "lo"))):
        s, x = vector_source()
        compiled = compile_ir(s, *sibling_outputs(loose_varied(x, name, tags=tags)))
        ir.append(compiled.ir)
    assert ir[0] == ir[1], "renaming a systematic recompiled it; §1.2's AddressTable precedent"
    # non-vacuity: the compared artifact really does carry the fused members, so an implementation
    # leaking a label into a stage member (invisible to the record-time store) is caught here.
    members = [
        m for n in GraphStore.deserialize(ir[0]).nodes() if n["kind"] == "stage" for m in n["members"]
    ]
    assert members


def test_structurally_identical_members_intern_to_one_node() -> None:
    """§5.2a's dedup witness: arena delta 0 and ONE node id for two structurally equal universes."""
    s, x = vector_source()
    up = x * 1.5
    before = s.node_count()
    down = x * 1.5  # structurally identical to `up`
    varied = graphed.vary(x, "jes", up=up, down=down)
    assert s.node_count() - before == 0
    assert graphed.universe(varied, "jes_up").node_id == graphed.universe(varied, "jes_down").node_id
    # positive control on the same counter: a genuinely different member DOES grow the arena, so
    # the delta-0 assertion above is not reading a dead instrument.
    before = s.node_count()
    other = graphed.vary(x, "jer", up=x * 1.25, down=x * 0.75)
    assert s.node_count() - before == 2
    assert graphed.universe(other, "jer_up").node_id != graphed.universe(other, "jer_down").node_id


def test_unvaried_optimizer_merged_program_still_compiles_and_runs() -> None:
    """§7.2 SCOPE control: the merge REFUSAL is fill-shaped and varied-only, so an UNVARIED
    multi-output program whose outputs the optimizer merges must keep working exactly as today
    (§6.3's "no-variation paths are unchanged"). It needs no fill, so it stays in `graphed`."""
    s = graphed.Session(NumpyBackend())
    b = from_array(s, "x", VECTOR) * 2.0
    compiled = compile_ir(s, b, b * 1.0)
    assert len(GraphStore.deserialize(compiled.ir).outputs()) == 1  # the optimizer merged them
    values = evaluate_ir(compiled, NumpyBackend(), {"x": VECTOR})
    assert len(values) == 1
    assert np.array_equal(values[0], VECTOR * 2.0)
