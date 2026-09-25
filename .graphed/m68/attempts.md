# m68 graphed implementer — iteration log

Freeze: freeze-preserve-m68 (1aabf38). `git diff freeze-preserve-m68 -- tests/frozen/` must stay empty.

## Iteration 0 — baseline (source = 3c46e01)
`./scripts/run-tests.sh`: every subtree green except preserve/m68 (29 failed, pre-implementation).
The combined `pytest tests -x` cannot collect on the base (duplicate basenames:
`tests/frozen/awkward/m3/test_dispatch.py` imports checkpoint/m8's `analyses`), so the per-subtree
runner is the instrument.

## Iteration 1 — commit-1 scope (services.py, Session registry, Plan/DurablePlan.services, bind hook, Triton)
m68: 45 pass except the 4 commit-2 tests (bundle manifest, inspect, RunReport.endpoints); preserve
m26/m27/m9 green (C1 kept: `_transport_module_and_factory({})` still returns `triton_http_transport`).
Diff coverage vs 3c46e01: 100% (151 lines); per-file: only the 4 ML plugin files under 90% (frameworks
not installed locally; pre-existing). Precommit --fast ok (integrity advisory: ci.yml modified — an
added live-leg no-skip guard in the triton job).
NEW FAILURE: frontend/m48 `test_plan_schema_is_unchanged_by_a_varied_program` pins Plan's literal
field set; §3.2's `Plan.services` field breaks it. Dispute filed:
`.graphed/m68/disputes/frontend-m48-test_plan_schema_is_unchanged_by_a_varied_program.md`. STOPPED,
nothing committed; work staged on m68-services.

## Iteration 2 — owner rulings R-A / R-B applied (freeze-preserve-m68-2)
R-B: `ExternalPlugin.load_params` (Triton `("url", "transport")`), cache key `(kind, content_hash,
endpoint, {p: params[p] for p in load_params})`. `uv lock`: the 3c46e01 lock was already stale
(`uv lock --check` fails on a `git archive 3c46e01`; 54 packages missing); this change adds only grpcio.
Runner: 4 F, all commit-2 scope (bundle manifest, inspect, RunReport.endpoints); m48 R-A test green.
diff-cover vs 3c46e01: 100% (152 lines); changed files 97–100%. Precommit --fast ok.
