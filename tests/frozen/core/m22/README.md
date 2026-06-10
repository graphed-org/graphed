# Frozen acceptance suite — M22 (graphed-core): output-scoped reduction & serialization

Fixes the compile_ir output-accumulation footgun (recorded 2026-06-10 in mvp-shortcomings,
user-confirmed fix plan): graph outputs are a property of the COMPILE REQUEST, not store state.
Traceability:

| Test file | Verifies | Item |
|---|---|---|
| `test_output_scoped_reduction.py` | `reduce(outputs=)` / `serialize(outputs=)` / `IncrementalReducer.finalize(outputs=)` use EXACTLY the requested set (stored marks ignored; byte-identical to a fresh single-mark store); sequential compiles from one store are history-independent and write no store state; the marks path (`outputs=None`) is unchanged; deliberate multi-output requests keep all outputs; invalid ids rejected; concurrent compiles of different outputs are isolated (read-only compilation, 3.14t) | M22 |
