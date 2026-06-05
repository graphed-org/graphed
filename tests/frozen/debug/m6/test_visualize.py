"""`visualize` renders a small, legible stage-graph in Mermaid/Graphviz, annotated with provenance
and (optionally) projected columns (plan M6)."""

from __future__ import annotations

import graphed_numpy as gn
import numpy as np
import pytest
from analyses import numpy_oob
from graphed import Session

import graphed_debug as gd


def test_mermaid_renders_stage_nodes_and_edges() -> None:
    s, bad = numpy_oob()
    g = gd.lower(s, bad, opt_level=1)
    out = gd.visualize(g, fmt="mermaid")
    assert out.startswith("flowchart TD")
    assert out.count("-->") == len(g.stages) - 1  # a linear pipeline: one edge per non-root stage
    assert "@ " in out  # provenance annotation present


def test_graphviz_renders() -> None:
    s, bad = numpy_oob()
    g = gd.lower(s, bad, opt_level=1)
    out = gd.visualize(g, fmt="graphviz")
    assert out.startswith("digraph graphed_debug")
    assert "->" in out


def test_canonical_analysis_is_a_small_legible_stage_graph() -> None:
    # a realistic small numpy analysis reduces to a handful of stages (snapshot the structure)
    s = Session(gn.NumpyBackend())
    ev = gn.from_record(s, "events", pt=np.arange(5.0), eta=np.arange(5.0), phi=np.arange(5.0))
    out = ((ev["pt"] * 2.0 + ev["eta"]) - ev["phi"]).reduce("sum")
    g = gd.lower(s, out, opt_level=1)
    assert len(g.stages) <= 6  # small + legible, not one node per op
    mermaid = gd.visualize(g, fmt="mermaid")
    assert mermaid.count("[") <= 6


def test_visualize_is_deterministic() -> None:
    s, bad = numpy_oob()
    g = gd.lower(s, bad, opt_level=1)
    assert gd.visualize(g, fmt="mermaid") == gd.visualize(g, fmt="mermaid")


def test_unknown_format_rejected() -> None:
    s, bad = numpy_oob()
    g = gd.lower(s, bad, opt_level=1)
    with pytest.raises(ValueError):
        gd.visualize(g, fmt="svg")
