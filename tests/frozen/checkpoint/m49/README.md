# `tests/frozen/checkpoint/m49` — traceability

The m49 checkpoint tree: the durable-plan half of systematic variations. Three anchors, all over one
varied program built through `graphed.vary`'s loose primitive on a partitioned numpy source and
lowered to one marked output per universe.

Plan: `systematics-vary-plan.md` §7.3, §7.4, §8.2(i), §10/m49.

## Fixture bindings the anchors rest on

- **`m49_analyses.py`, not `analyses.py`.** `tests/frozen/checkpoint` and `tests/extra/checkpoint`
  collect in ONE pytest process, and `checkpoint/m8/analyses.py` is on `pythonpath`, so the bare
  name is bound repo-wide.
- **Every closure operand is module-level** (§7.3): the `PartitionedSource` and the
  reduce/combine/empty callables. A `__main__`-defined operand cloudpickles by value and digests
  differently across `PYTHONHASHSEED`, which would make the determinism anchor red against a correct
  implementation.
- **The `DurablePlan` is built BY VALUE** — `OpSpec.from_callable(plan.process)` over the `Plan`
  `aggregate_plan` returned (§7.3). `run_resumable` takes a `DurablePlan`; `aggregate_plan` returns a
  plain `Plan`, and no bridge ships. `OpSpec.from_ref` would keep the closure's fields out of the
  plan bytes, so the §8.2(i) anchor could not see them.
- **The (beta) hook is supplied by the fixture** for the determinism anchor, returning a populated
  sorted payload in §8.2(i)'s entry layout. m48's `None` default pickles seed-independently and would
  freeze that anchor green against the shape it exists to ban. Legal because §8.2(i)'s
  self-supplied-hook ban is worded over the LABEL association, which no anchor here asserts.

## Test → anchor

| Test | Anchor |
|---|---|
| `test_varied_resume.py::test_the_journal_unit_is_the_whole_variation_composite` | §7.1/§7.3 — one plan, one IR, one marked output per universe; one partition read, not one per universe |
| `test_varied_resume.py::test_kill_then_resume_is_byte_identical_and_does_less_work` | §7.3 — the N-variation composite partial is the journal unit; resumed result byte-identical, measurably less work |
| `test_varied_resume.py::test_no_double_count_at_any_kill_boundary` | §7.3 — every universe contributes each partition exactly once, wherever the crash landed |
| `test_varied_resume.py::test_resume_after_completion_recomputes_no_universe` | §7.3 — a journaled composite is reused, not re-read |
| `test_varied_dead_letter.py::test_a_poison_free_twin_dead_letters_nothing` | §7.4 — positive control for the three below |
| `test_varied_dead_letter.py::test_one_poisoned_variation_dead_letters_the_whole_composite` | §7.4 — dead-lettering is partition-atomic; the oracle is the same plan over the surviving partitions, the discriminator the nominal universe's total |
| `test_varied_dead_letter.py::test_retry_reruns_the_whole_composite_not_the_failing_universe` | §7.4 — retry is partition-atomic: `process` is the composite, so an attempt reads the whole partition |
| `test_varied_dead_letter.py::test_the_descriptor_keeps_its_fixed_key_list_and_gains_no_variation_key` | §7.4 — the descriptor's structured half gains no variation key; no dead-letter edit is an m49 target |
| `test_varied_plan_determinism.py::test_a_populated_sorted_payload_is_seed_independent` | §8.2(i)/§3.2 — byte-identical `DurablePlan.to_bytes()` and `task_id`s across `PYTHONHASHSEED` with the field populated |
| `test_varied_plan_determinism.py::test_a_frozenset_payload_is_seed_dependent` | §8.2(i) — the live control: the banned shape does move the plan bytes, so the assertion above is not vacuous |
| `test_varied_plan_determinism.py::test_the_payload_reaches_the_plan_bytes` | §7.3/§8.2(i) — the BY-VALUE clause: the field is IN the bytes and in `task_id` |

## Not anchored here

The LABEL the dead-letter surface names (§7.4, §8.2) rides the `StageError` anchor in
`graphed-executors`. §8.2(ii)'s wrap and (iii)'s attribution ride `tests/frozen/debug/m49`.
`variation_labels`' bound producer is `graphed-histogram`'s group-plan builder, anchored there.
