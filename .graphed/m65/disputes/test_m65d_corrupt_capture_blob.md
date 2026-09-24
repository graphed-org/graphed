# Dispute record: a corrupt capture blob has no frozen witness

**Artifact:** `tests/frozen/debug/m65/test_m65d_replay.py` (`freeze-m65d-fixup`). Test 6 pins the
store arm of D-2 through `Store.completed()` and `store.get(entry.blob)` on intact blobs; no frozen
test replays a task whose captured blob is present but does not hash to its name.

**Gap:** review D r1 N1 (`graphed-workdir/lanes/debug/reviews/unit-D-impl-r1.md`): `replaying._decode`
validated the store read with `assert`, which `python -O` strips, so a corrupt blob then failed inside
the codec with an unrelated error. The owner ruled (2026-09-24) that the check becomes an explicit
`FileNotFoundError` naming the blob, folded into the commit that introduced it (the integrity scan's
`assertion_removed` rule refuses the edit as a later commit). With that fold the raise line has no
frozen hit: `diff-cover --compare-branch=origin/main` over the frozen suite gives 136/139 lines,
97.8 % < 98 % (§B.3 diff gate, frozen hits); the D frozen files alone leave `replaying.py` line 52
missing beside the two `StageError` raises the extra binding test covers.

**Why not routed around:** the raise is the reviewer's finding and the owner's ruling; a one-line
form or a helper only relocates the uncovered line; an extra test does not count as a frozen hit.

**Proposed correction (owner ruling "Resolve by refreeze", 2026-09-24;
`--allow-refreeze tests/frozen/debug/m65/`):**
- NEW frozen file `tests/frozen/debug/m65/test_m65d_replay_capture_blob.py` (test author, not the
  implementer): capture a run with `aggregate_plan(store=)`, overwrite one task's captured blob
  bytes in place (a deleted blob leaves `completed()` and takes the re-read arm), then `replay` of
  that task reports `input_source == "store"` and raises `FileNotFoundError` whose message names the
  blob. Closes when the test fails on the `assert` tree (`ac62f52`, `AssertionError`) and passes at
  HEAD, and the D-frozen diff-cover leaves only the two `StageError` raises missing in `replaying.py`.
- Tag `freeze-m65d-fixup2` (precedent: `freeze-m65a3-fixup`); a `README-m65d-fixup2.md` row.

**Disposition:** RULED — "Resolve by refreeze" (owner, 2026-09-24). Recorded by the lane lead
(team-lead), 2026-09-24.
