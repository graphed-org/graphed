"""M40 — GIR1 codec extension for the ``Join`` boundary (plan §2.3, T_JOIN=6).

The tag is APPENDED (``T_JOIN=6``, after M39's ``T_EXCHANGE=5``) and the magic stays ``b"GIR1"``:
existing graphs (tags 0..5) serialize byte-identically, so the M8 determinism gate stays untouched;
a ``Join``-bearing blob round-trips byte-identically; and a node tag this build does not know is
rejected LOUDLY (a ``ValueError`` from ``BadTag``), never silently mis-parsed into a wrong node —
the difference between a reader that fails on a foreign/newer tag and one that corrupts silently.

Pinned Python surface (mirrors M39 ``add_exchange``; contract E1/E2): a ``Join`` has NO ``name`` —
its identity is its ``scheme`` ParamMap + its **two** logical inputs (build side, probe side):
``GraphStore.add_join(inputs: list[int], params) -> int`` and ``nodes()[i]["kind"] == "join"``.
"""

from __future__ import annotations

import pytest

from graphed.core import GraphStore

_SCHEME = {"how": "inner", "on": "event"}


def _join_graph() -> tuple[GraphStore, list[int]]:
    """src_l -> ptl ; src_r -> ptr ; JOIN(ptl, ptr) -> sum(output)."""
    g = GraphStore()
    lsrc = g.add_source("left", {"uri": "l.root"})
    rsrc = g.add_source("right", {"uri": "r.root"})
    lpt = g.add_op("pt", [lsrc])
    rpt = g.add_op("pt", [rsrc])
    j = g.add_join([lpt, rpt], _SCHEME)
    out = g.add_reduction("sum", [j])
    return g, [out]


def test_magic_is_unchanged_even_with_a_join() -> None:
    # the tag is APPENDED, not a format break: a blob CONTAINING a Join still starts with GIR1.
    g, outs = _join_graph()
    blob = g.serialize(outputs=outs)
    assert blob[:4] == b"GIR1", "the magic must stay GIR1 (append a tag, never bump the magic)"


def test_join_blob_roundtrips_byte_identically() -> None:
    g, outs = _join_graph()
    blob = g.serialize(outputs=outs)
    back = GraphStore.deserialize(blob)
    assert back.node_count() == g.node_count()
    # M8/M22 pattern (see M39 freeze-M39-1): compare deserialize-vs-deserialize, since a
    # blob-marked `back` cannot equal the unmarked builder `g` under the read-only-serialize pin.
    assert back.to_dot() == GraphStore.deserialize(blob).to_dot()
    assert back.serialize() == blob, "re-serialize of a decoded Join graph must be byte-identical"


def test_join_scheme_and_both_inputs_survive_the_roundtrip() -> None:
    g, outs = _join_graph()
    back = GraphStore.deserialize(g.serialize(outputs=outs))
    j = next(n for n in back.nodes() if n["kind"] == "join")
    assert j["params"] == _SCHEME, "the join's scheme ParamMap must round-trip through the codec"
    # a Join carries TWO logical inputs (build, probe) — not one like an Exchange; both survive.
    assert len(j["inputs"]) == 2, "a Join is a two-input boundary; both edges must round-trip"


def test_an_unknown_node_tag_is_rejected_not_silently_misparsed() -> None:
    # THE loud-failure witness: a node tag this build cannot interpret (e.g. a Join blob read by an
    # older reader, or any corruption) must raise, NOT be swallowed by a catch-all default that
    # mis-decodes it as some other node kind. Poison the first node tag with an out-of-range value.
    g, outs = _join_graph()
    blob = bytearray(g.serialize(outputs=outs))
    blob[8] = 200  # byte 8 = first node's tag (4 magic + 4 node-count); 200 is no valid tag
    with pytest.raises(ValueError):
        GraphStore.deserialize(bytes(blob))


def test_tag_one_past_join_is_rejected() -> None:
    # discriminates a decoder whose match accidentally admits tag 7 (or treats >=6 as valid): the
    # first tag AFTER T_JOIN=6 must still be rejected loudly, pinning the exhaustive-match contract.
    g, outs = _join_graph()
    blob = bytearray(g.serialize(outputs=outs))
    blob[8] = 7
    with pytest.raises(ValueError):
        GraphStore.deserialize(bytes(blob))


def test_low_tags_and_magic_are_byte_frozen() -> None:
    # M8-determinism CONTROL (mirrors M39's `test_legacy_graphs_are_byte_stable`): a golden byte
    # oracle for a graph of only pre-M40 node kinds. It PASSES today by design and is the guard that
    # adding T_JOIN=6 did not renumber tags 0..2 (source=0/op=1/reduction=2) or bump the magic — a
    # renumber/bump is self-consistent on a fresh round-trip, so only an external literal catches it.
    g = GraphStore()
    src = g.add_source("events", {"uri": "f.root"})
    pt = g.add_op("pt", [src], {"thr": 30})
    out = g.add_reduction("sum", [pt])
    golden = (
        b"GIR1\x03\x00\x00\x00\x00\x06\x00\x00\x00events\x01\x00\x00\x00\x03\x00\x00\x00uri\x03"
        b"\x06\x00\x00\x00f.root\x01\x02\x00\x00\x00pt\x01\x00\x00\x00\x03\x00\x00\x00thr\x00"
        b"\x1e\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x02"
        b"\x03\x00\x00\x00sum\x00\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00"
        b"\x01\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00"
    )
    assert g.serialize(outputs=[out]) == golden, "M8 golden: tags 0..2 + GIR1 magic must not move"
