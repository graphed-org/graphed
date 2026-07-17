# M39 frozen suite — graphed-checkpoint (multi-stage journal + shuffle resume)

Milestone **M39**. graphed-checkpoint owns the multi-stage journal schema + resume replay for the
shuffle. It does NOT run the shuffle executor (that is exec-local), so this suite drives an ABSTRACT
two-phase `DurablePlanV2` with deterministic, importable-by-ref stage processes — exactly as the M8
suite drives an abstract histogram plan. **Frozen — read-only after `freeze-M39-0`.**

## Files → plan clause / theme

| File | Covers | Plan clause / theme |
|---|---|---|
| `shuffle_analyses.py` | a self-contained two-phase (map_write → gather) `DurablePlanV2` + deterministic stage processes | §4.4 |
| `test_multistage_journal.py` | `JournalEntry` gains `stage`+`deps`; V1 default still writes `journal.log`; journal-per-writer (`journal.<node>.log`) with UNION replay; owner journals a stolen task's manifest | §7.3, §6.3 |
| `test_shuffle_resume.py` | **(e)** resume a half-done shuffle = bit-for-bit vs uninterrupted; `skipped>0` witnesses reused blocks; two-phase (kill mid-shuffle, gather resumes); kill at every boundary matches | §5.3, §7.3 |

## Pinned surface (test-author decisions)

- `graphed_checkpoint.Store(root, node=None)` — `node=None` keeps the V1 `journal.log` (M8 untouched);
  `node="A"` writes `journal.A.log`; `completed()` replays the UNION of all journal files.
- `Store.record_done(task_id, partition, blob, *, stage="", deps=())` and `JournalEntry(..., stage,
  deps)`.
- `run_shuffle_resumable(plan_v2, store, *, resources=None, _kill_after=None) -> ShuffleResumeResult`
  with `.value` (tuple of content-addressed gather-block hashes) and `.report` (M8-style
  `executed`/`skipped`/`did_less_work`). Stage-process convention:
  `process(task, inputs, resources) -> bytes` (`inputs` = upstream dep block payloads, empty for
  stage 0); each block journaled with its `stage` + `deps`.

## Non-vacuity

Pre-implementation: `Store.record_done` has no `stage`/`deps`, `Store.__init__` has no `node`,
`run_shuffle_resumable` and `DurablePlanV2` are absent → right-reason failures. Note this suite runs a
CONTROLLED two-phase plan (abstract processes) — the real end-to-end shuffle resume with the executor
+ Store is an exec-local integration concern; here the JOURNAL + resume-skip MACHINERY is what is
frozen.
