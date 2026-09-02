# m50 frozen suite — `graphed` / `tests/frozen/debug/m50`

Milestone m50, Brief C (m49 carryover, test-only over source shipped at `01d908b`). Authority:
`systematics-vary-plan.md` r44 §8.2(ii); the m50 DECOMPOSE artifact's Brief C.

The External ARM of §8.2(ii)'s attribution. `evaluate_ir` dispatches attribution at three sites —
the op loop, the inline stage-member loop, and the External payload's evaluator — and no frozen
anchor reached the third. A failure raised inside an External evaluator at a key the worker's label
channel has an entry for becomes a labelled `StageError`; a key with no entry re-raises untouched.

| Test | Clause | Discriminates |
|---|---|---|
| `test_external_failure_at_an_entried_key_becomes_a_labelled_stage_error` | §8.2(ii) External arm | red (raw `Boom`) when the External arm bypasses `_dispatch`/drops `on_failure` |
| `test_external_failure_with_no_entry_for_its_key_re_raises_untouched` | §8.2(ii) entry gate | red if the arm attributes a key with no channel entry (unconditional labelling) |
