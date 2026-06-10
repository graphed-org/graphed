# M22 attempts — graphed (output-scoped compiles; the compile_ir footgun fix)

## Iteration 0 — TEST_AUTHORING/TEST_SANITY/IMPLEMENTING — 2026-06-10 (freeze-M22-0)

- frozen suite tests/frozen/m22 (5 tests, reusing the m10 toy backend); NON-VACUOUS (the 4
  footgun tests fail pre-implementation; the deliberate-multi-output pin passes, as intended).
- serialized_ir/compile_ir pass outputs= explicitly to reduce/serialize/finalize (graphed-core
  M22): artifacts carry EXACTLY the requested outputs; compiles are session-history-independent
  BYTE FOR BYTE (pinned against fresh-session compiles, both directions); holds on the one-shot,
  incremental (M10), and optimize=False paths; deliberate multi-output compiles keep every
  requested output.
- DEVIATION FROM THE PLAN, recorded: the plan said "stop calling mark_output". The frozen m8
  test test_optimized_ir_keeps_the_output_and_is_the_reduced_graph pins the MARK SIDE EFFECT
  (serialized_ir then marks-path store.reduce() must agree), so the legacy mark_output call is
  RETAINED for back-compat — it is never read by the compile path, the artifacts are fully
  output-scoped, and no frozen test was altered (per the standing rule: amendments only on
  explicit user authorization).
- gates: frozen 188/188 PASS · coverage 93.85% (>=90, branch) · ruff+format clean ·
  mypy --strict clean · determinism green · sphinx -W clean. Full sweep: all 10 repos + the
  uproot fork green.
