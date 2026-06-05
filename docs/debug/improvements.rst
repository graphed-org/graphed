Future improvements
===================

Catalogued, not silently dropped (plan A.7 / Part F).

- **Interactive debugger / time-travel replay** is explicitly Phase 2 (M6 guardrail): this milestone
  ships static lowering + traceback surfacing only.
- **Richer inter-op assertions** at ``opt_level=0`` (dtype/shape contracts per op) beyond the current
  consistency checks.
- **Fused-kernel single-call execution.** The debug runner currently executes a fused stage's members
  sequentially to localize an error; a real fused kernel (M7 executor) would execute once and, on
  failure, re-run op-by-op to localize. The provenance mapping is identical either way.
- **Graphviz rendering to image** (the ``visualize`` text output is Graphviz/Mermaid source; rendering
  to PNG/SVG is left to the caller's toolchain).
