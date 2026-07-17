Improvements
============

Tracked design improvements and limitations for ``graphed`` (plan M0 requires this file).

Current limitations
-------------------

- **Provenance is a stub.** M2 captures only (filename, lineno) of the first non-graphed frame.
  M3 adds sub-expression text (stack_data/executing), thread-safe capture, toggling, benchmarking.
- **Reference evaluation only.** ``Session.materialize`` walks the graph node-by-node; the real
  morsel-driven executor is M7. No fusion (M4).
- **Opaque map results have unknown form.** ``map`` yields an object form, so it cannot feed a
  typed op (e.g. ``sum``) — by design; typed transforms should be ops, not opaque callables.
- **The awkward backend is M3.** numpy is the trivial seam-prover here.

Planned
-------

- Real provenance + sourcemaps (M3), column projection via ``Backend.project`` (M5), and the
  execution contract (M7) all build on this frontend.
