"""``visualize`` — render a ``LoweredGraph`` as Graphviz or Mermaid (plan M6).

Nodes are stages, annotated with their member ops + the user provenance of the stage head, and
(optionally) the projected columns per source from an M5 ``Projection``. Output is deterministic
source text (Graphviz ``digraph`` / Mermaid ``flowchart``); rendering to an image is left to the
caller.
"""

from __future__ import annotations

from collections.abc import Mapping

from .lowering import LoweredGraph, LoweredStage


def _stage_label(stage: LoweredStage, projection: Mapping[str, frozenset[str]] | None) -> str:
    head = stage.head
    if head.kind == "source":
        cols = projection.get(head.op) if projection else None  # head.op is the source name
        extra = f"\\ncols: {', '.join(sorted(cols))}" if cols else ""
        return f"source {head.op}{extra}\\n@ {head.provenance}"
    ops = " → ".join(m.op for m in stage.members)
    kind = "boundary" if stage.boundary else "stage"
    return f"{kind}[{ops}]\\n@ {head.provenance}"


def _stage_inputs(graph: LoweredGraph, stage: LoweredStage) -> set[int]:
    """The stage ids that feed this stage (external inputs of its members)."""
    member_ids = {m.node_id for m in stage.members}
    head_of: dict[int, int] = {}
    for s in graph.stages:
        for m in s.members:
            head_of[m.node_id] = s.node_id
    feeders: set[int] = set()
    for m in stage.members:
        for i in m.input_ids:
            if i not in member_ids:
                feeders.add(head_of[i])
    return feeders


def visualize(
    graph: LoweredGraph,
    *,
    fmt: str = "mermaid",
    projection: Mapping[str, frozenset[str]] | None = None,
) -> str:
    """Render ``graph`` as ``"mermaid"`` or ``"graphviz"`` source text."""
    if fmt not in ("mermaid", "graphviz"):
        raise ValueError(f"unknown fmt {fmt!r}; use 'mermaid' or 'graphviz'")
    edges = [(src, stage.node_id) for stage in graph.stages for src in sorted(_stage_inputs(graph, stage))]
    if fmt == "mermaid":
        lines = [f"flowchart TD  %% opt_level={graph.opt_level}"]
        for stage in graph.stages:
            lines.append(f'  n{stage.node_id}["{_stage_label(stage, projection)}"]')
        for a, b in edges:
            lines.append(f"  n{a} --> n{b}")
        return "\n".join(lines)
    lines = [f"digraph graphed_debug {{  // opt_level={graph.opt_level}", "  rankdir=TB;"]
    for stage in graph.stages:
        label = _stage_label(stage, projection).replace("\\n", "\\l") + "\\l"
        lines.append(f'  n{stage.node_id} [shape=box, label="{label}"];')
    for a, b in edges:
        lines.append(f"  n{a} -> n{b};")
    lines.append("}")
    return "\n".join(lines)
