# m68 graphed implementer — iteration log

Freeze: freeze-preserve-m68-2. `git diff freeze-preserve-m68-2 -- tests/frozen/` must stay empty.

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

## Iteration 3 — commit 2 (bundle manifest, inspect, RunReport.endpoints)
`graphed.services.referenced_services` is the one "specs the IR names" rule, used by aggregate_plan and
build_bundle. RunReport.endpoints: read-only mapping, written to JSON only when non-empty, `__reduce__`
so a report still pickles (extra test fails without it: "cannot pickle 'mappingproxy'").
`reproduce` refuses a node naming a service (PreserveError) instead of the tritonclient KeyError on the
missing url (extra test fails without it). Runner: 0 F (rc=0); diff-cover vs 3c46e01 100% (183 lines);
changed files 97–100%. Precommit --fast ok.

## Iteration 4 — review r1 repairs
- Medium (docs): `docs/preserve/design.rst` "Not supported yet" now states that `reproduce` raises
  `PreserveError` at an operation naming a service; run it as a plan (`bind_services` or an executors runner).
- A1: the awkward `to_parquet` plans (`_WritePart`, `_VariedWritePart`) carry `Plan.services` (via
  `aggregate.plan_services`, shared with `aggregate_plan`) and bind through `services.bind_externals`.
  numpy `to_parquet` unchanged: the numpy backend refuses plugin Externals (no payload descriptor).
- A2: `Launch.__hash__` over frozensets of env/resources (the §3 defaults rule fixes the mappingproxy
  representation). A3: `RunReport.__setstate__` fills `endpoints` for a 0.0.6 pickle.
- Exit round: `referenced_services(names=None)`, `aggregate_plan(services=None)`; freeze ref above.
- New extra tests each fail their mutant (hook removed ×2, write_plan dropping services, no hash, no setstate).
- Runner rc=0 (2437 pass, 33 skip); diff-cover vs 3c46e01 100% (199 lines); changed files 97–100%;
  precommit --fast ok; sphinx -W ok. Trap: a mutate/restore within one second leaves a stale .pyc
  (same size, same mtime second) — touch the restored file.
