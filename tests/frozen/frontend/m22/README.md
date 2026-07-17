# Frozen acceptance suite — M22 (graphed): output-scoped compiles

Fixes the compile_ir output-accumulation footgun (recorded 2026-06-10 in mvp-shortcomings,
user-confirmed plan): outputs are a property of the COMPILE REQUEST, not session/store state.
Traceability (with graphed-core's m22 suite pinning the Rust surface):

| Test file | Verifies | Item |
|---|---|---|
| `test_output_scoped_compiles.py` | the footgun reproducer (compile A then B from ONE session: each artifact single-output, never the union); compiles are session-history-independent BYTE FOR BYTE (equal to fresh-session compiles, in both directions); `serialized_ir` scoped on the optimized AND `optimize=False` paths; incremental (M10) sessions identical; deliberate multi-output compiles still carry every requested output | M22 |
