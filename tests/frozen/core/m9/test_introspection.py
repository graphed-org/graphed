"""M9 support — `GraphStore.nodes()` exposes the interned IR as structured, read-only data.

The M9 preservation bundle both *renders* the IR (``inspect``) and *interprets* it (``reproduce``);
both need to walk the graph as plain data — node kind, op name, params, inputs, output flag, and the
``External`` payload descriptor. These tests pin that introspection (which must agree with the rest
of the IR surface — ``to_dot`` / ``serialize`` / a deserialized round trip)."""

from __future__ import annotations

from graphed_core import GraphStore, PayloadDescriptor


def _graph() -> tuple[GraphStore, int]:
    g = GraphStore()
    src = g.add_source("events", {"uri": "ds"})
    pt = g.add_op("pt", [src], {"thr": 30.0})
    ext = g.add_external(
        PayloadDescriptor(
            kind="correctionlib",
            content_hash="sha256:beef",
            framework="correctionlib",
            version="2",
            io_schema="btagSF",
            preprocessing_ref=None,
        ),
        [pt],
        {"path": "sf.json", "name": "btagSF"},
    )
    out = g.add_reduction("hist", [ext])
    return g, out


def test_nodes_report_kind_name_params_inputs_in_id_order() -> None:
    nodes = _graph()[0].nodes()
    assert [n["id"] for n in nodes] == [0, 1, 2, 3]
    assert [n["kind"] for n in nodes] == ["source", "op", "external", "reduction"]
    assert nodes[0]["name"] == "events" and nodes[0]["params"] == {"uri": "ds"}
    assert nodes[1]["params"] == {"thr": 30.0} and nodes[1]["inputs"] == [0]
    assert nodes[3]["inputs"] == [2]


def test_output_flag_marks_only_outputs() -> None:
    # [freeze-M22-1 respin: outputs are given per request — the flags appear in the artifact
    # serialized FOR that output, and nowhere else]
    g, out = _graph()
    assert [n["output"] for n in g.nodes()] == [False, False, False, False]  # no setter exists
    back = GraphStore.deserialize(g.serialize(outputs=[out]))
    assert [n["output"] for n in back.nodes()] == [False, False, False, True]


def test_external_descriptor_is_exposed() -> None:
    ext = next(n for n in _graph()[0].nodes() if n["kind"] == "external")
    desc = ext["descriptor"]
    assert desc["kind"] == "correctionlib"
    assert desc["content_hash"] == "sha256:beef"
    assert desc["io_schema"] == "btagSF"
    assert desc["preprocessing_ref"] is None
    assert ext["params"] == {"path": "sf.json", "name": "btagSF"}


def test_nodes_survive_a_serialize_roundtrip() -> None:
    g, out = _graph()
    back = GraphStore.deserialize(g.serialize(outputs=[out]))
    flagged = GraphStore.deserialize(back.serialize())  # the marks carried in the bytes persist
    assert back.nodes() == flagged.nodes()  # introspection is stable through the durable form


def test_param_value_types_are_preserved() -> None:
    g = GraphStore()
    g.add_source("s", {"i": 7, "f": 1.5, "b": True, "s": "x"})
    params = g.nodes()[0]["params"]
    assert params == {"i": 7, "f": 1.5, "b": True, "s": "x"}
    assert isinstance(params["i"], int) and isinstance(params["f"], float)
    assert isinstance(params["b"], bool) and isinstance(params["s"], str)
