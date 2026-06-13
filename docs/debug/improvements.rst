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
- **Dashboard run-control** (M37 / root prompt R20.6): browser→run commands (pause/cancel) need a
  websocket and a control seam; SSE is one-directional, so the MVP dashboard is observe-only.
- **Persisted run-report** — folding a run's event log + flamegraph into the M9 preservation bundle
  as a reproducible artifact (the live dashboard is in-memory only today).
- **Distributed dashboard transport** — the M37 seam already forwards worker events over a queue; a
  network transport would let it observe a future non-local executor unchanged.
