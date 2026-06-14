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
- **Dashboard run-control** (M37 / root prompt R20.6): the websocket transport is bidirectional, so
  browser→run commands (pause/cancel) are a natural extension — they need a control seam back into
  the executor, which the MVP deliberately omits (the dashboard is observe-only).
- **Persisted run-report** — folding a run's event log + profile into the M9 preservation bundle as a
  reproducible artifact (the live Perspective tables are in-memory only today).
- **Profile as a flamegraph** — the profile is currently a Perspective table (group-by-function self
  µs, plus a treemap view); a true flamegraph plugin would render the call tree's shape.
- **Per-worker push** — workers currently forward through the driver's collector; a future
  distributed executor could have each remote worker open its own ``NetworkMonitor`` to the server
  (the transport already supports it).
