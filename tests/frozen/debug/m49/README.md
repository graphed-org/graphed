# m49 frozen suite — `graphed` / `tests/frozen/debug/m49`

Milestone m49, the error-path tree. Authority: `systematics-vary-plan.md` r41 (§8.1, §8.2(ii),
§8.2(iii)) and the m49 DECOMPOSE artifact's Brief D. This source is `graphed`'s, so its §B.3 diff
coverage must come from this repo's own frozen suite — no `graphed-executors` test can supply it.

`scripts/run-tests.sh` collects `tests/frozen/debug` WHOLE in one process, so basenames here are
unique against every debug milestone and the helper module carries the `m49_` prefix (`m6/analyses.py`
is live and would otherwise bind the bare name first).

| anchor | plan clause | test |
|---|---|---|
| `vary-m49-D1` | §8.1 `variation` in `__eq__`/`__hash__` (+ hash-tuple control) | `test_variation_attribution.py::test_variation_participates_in_equality_and_hash` |
| `vary-m49-D1` | §8.1 `""` default = nominal/unvaried; `summary()` line | `test_variation_attribution.py::test_the_empty_string_is_the_default_and_a_label_reaches_the_summary` |
| `vary-m49-D2` | §8.2(ii) UNATTRIBUTED arm — no channel at all | `test_variation_attribution.py::test_a_worker_failure_with_no_label_channel_reraises_the_original` |
| `vary-m49-D2` | §8.2(ii) UNATTRIBUTED arm — populated field, no entry for the failing KEY | `test_variation_attribution.py::test_a_failure_whose_key_has_no_entry_reraises_the_original` |
| `vary-m49-D3` | §8.2(ii)/(iii) attributed arm: the failing key's label + the user's line | `test_variation_attribution.py::test_an_attributed_failure_carries_its_own_keys_label_and_the_users_line` |
| `vary-m49-D4` | §8.2(ii) frame TIE-BREAK — lowest record id mapping to a key wins | `test_variation_attribution.py::test_two_lines_merged_onto_one_key_report_the_lowest_record_ids_frame` |
| `vary-m49-D5` | §8.2(ii)/(iii) across a spawned process (M6 contract extended) | `test_variation_process_boundary.py::test_a_labelled_stage_error_survives_a_spawned_worker_process` |
| — | poisoned toy backend, merged-key topology, §8.2(i) payload builders, spawn worker | `m49_analyses.py` |

## What the fixtures witness

`merged_key_program()` records the poisoned op and its `+ 0.0` additive identity on two DIFFERENT
lines; equality saturation quotients them onto one reduced node, keeping the earlier one. Three
recorded nodes, two reduced keys — so exactly one key is reached from two record ids at two lines,
which is what makes the tie-break assertion discriminating (a last-writer-wins implementation reports
the other line).

The suite supplies its own (β) hook. §5.2a's self-derivation ban is worded over the LABEL
association, whose only bound producer is `graphed-histogram`'s group-plan builder; what these tests
assert is the wrap's two arms, the keying and the frame tie-break. The frames are COPIED off the
`CompiledGraph` that `compile_ir` built — never fabricated — so the tie-break is read out of the
artifact and not out of the hook.

`m49_analyses.artifact_frames` locates §8.2(i)'s per-key frame association list on the artifact by
LAYOUT (`((reduced_node_id, member_index | None), frame)`) rather than by field name: the field's
spelling is pinned by the accessor anchor in `tests/frozen/frontend/m49`, and this tree deliberately
does not compete for it. It refuses ambiguity — zero or several candidate layouts fail loudly.
