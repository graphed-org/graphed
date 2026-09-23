# Test Dispute — §8.2(ii) pins the "no entry → raw re-raise" arm that the owner has overruled

**Filed by:** team-lead, 2026-09-22, implementing the owner ruling "`_PartitionReduce._attribute` always attribute".
**Status:** RESOLVED — owner ruled (2026-09-22). Correction applied by team-lead under that authorization;
`systematics-vary-plan.md` §8.2(ii) now reads: every failure at a key with a §8.2(i) frame is attributed
(unvaried programs included, with an empty variation); the label channel only adds the label; a frameless key
re-raises the original untouched. The class, enumerated by one full `scripts/run-tests.sh` run over the change
(every subtree ran; three members, all `checkpoint/m49` + `debug/m49`):

| member | what it pinned | correction |
|---|---|---|
| `debug/m49/test_variation_attribution.py::test_a_worker_failure_with_no_label_channel_reraises_the_original` | no label channel → raw `PoisonError` | renamed `..._is_attributed_with_no_variation`: `StageError`, `variation == ""`, cause type/message, the user's line |
| `debug/m49/test_variation_attribution.py::test_a_failure_whose_key_has_no_entry_reraises_the_original` | entry-less key → raw `PoisonError` | renamed `..._is_attributed_with_no_variation`: same assertions; a NEW test pins the surviving raw arm at a key with no frame (`dataclasses.replace(process, frames=())`) |
| `checkpoint/m49/test_varied_dead_letter.py::test_the_descriptor_keeps_its_fixed_key_list_and_gains_no_variation_key` | dead-letter key set == `DESCRIPTOR_KEYS` and `error_type == "ValueError"` | the poison is now a `StageError`, so M8's `stage_error` sub-descriptor joins the fixed key set (`DESCRIPTOR_KEYS | {"stage_error"}`, no `variation` key), `error_type == "StageError"`, `stage_error.cause_type == "ValueError"` |

## The tests

`tests/frozen/debug/m49/test_variation_attribution.py` binds §8.2(ii)'s worker-side wrap; its docstring made
"with NO entry it re-raises the original exception untouched" law. `tests/frozen/checkpoint/m49/test_varied_dead_letter.py`
reads the dead-letter descriptor's `error_type` off the same arm.

## The clause they contradict

The owner's ruling (2026-09-22) and the amended §8.2(ii): the frame, not the label entry, is what classifies a
failure as attributable. `StageError` needs frames at construction, so the raw arm survives only where the
compiled correspondence has no frame for the failing key.

## Proposed correction (applied)

`_PartitionReduce` gains `frames: tuple[tuple[Key, Frame], ...] = ()`, filled by `aggregate_plan` from
`compiled.correspondence.frames`; `_attribute` falls back to that frame with `labels = ()` when the label channel
has no entry. The three tests above assert the new arm; every other assertion is unchanged.
