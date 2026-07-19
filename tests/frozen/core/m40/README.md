# M40 frozen suite — graphed-core (Join IR + codec + plan staging)

Milestone **M40** (distributed joins). graphed-core owns the IR/codec/plan seam for the relational
`Join` boundary: the `NodeKey::Join { scheme, inputs }` variant (contract E2), its GIR1 codec tag
`T_JOIN=6` (appended after M39's `T_EXCHANGE=5`), and the two-producer staging in the existing
additive `DurablePlanV2`. **Frozen — read-only after freeze.** Mirrors `tests/frozen/core/m39/`.

## Files → plan clause / theme

| File | Covers | Plan clause |
|---|---|---|
| `test_join_ir.py` | `Join` variant: records a **two-input** boundary; interns/CSEs; scheme + input-order in structural identity; ENDS a stage (survives reduce as itself, splits 3 stages); DCE keeps a reachable / drops an unreachable join; byte-deterministic reduction | §2.1, E2; IR/optimizer theme |
| `test_join_serialize.py` | `T_JOIN=6` appended; MAGIC stays `GIR1`; Join blob (scheme + both inputs) round-trips byte-identical; unknown/one-past tag rejected loudly; low-tag golden byte oracle (M8 gate untouched) | §2.3; serialize theme |
| `test_join_plan.py` | two producer `map_write` stages + a `gather_join` depending on **both** (`inputs=(0,1)`); plan bytes deterministic; embedded Join IR survives; `task_id` folds `backend_id` | §4.4, §7.2, E1; determinism theme |

## Pinned Python surface (test-author decisions the implementer must match)

- `GraphStore.add_join(inputs: list[int], params) -> int` — a `Join` has **no** `name` (mirrors
  M39 `add_exchange`; contract E2 `Join { scheme, inputs }`); `inputs` has **exactly two** ids
  (build side, probe side, order significant); `params` is the `scheme` map. `nodes()[i]["kind"]
  == "join"`, `nodes()[i]["params"]` is the scheme.
- `Join` is a **boundary** (`is_boundary` is `!Op`, so no change): fusion never crosses it and it
  survives reduction as itself, with its two input edges remapped to the reduced nodes.
- Codec: `T_JOIN = 6`, magic stays `b"GIR1"`; tags 0..5 unchanged; an unknown node tag decodes to a
  loud `ValueError` (`DecodeError::BadTag`), never a silent mis-parse.
- Plan: a join stages as two `map_write` producers + one `gather_join` with `inputs=(0, 1)` in the
  M39 `DurablePlanV2`/`StageSpec` API (no schema change).

## Traceability — the WRONG implementation each test discriminates

| Test | Witnessed mechanism | Discriminating property (wrong impl it FAILS) |
|---|---|---|
| `test_add_join_records_a_two_input_boundary_node_with_its_scheme` | `nodes()` exposes `kind=="join"`, two inputs `[b,d]`, scheme round-trips | a one-input node (Exchange-shaped), or scheme dropped from `nodes()` |
| `test_join_interns_and_cses_like_any_node` | identical joins share one id; `node_count==3` | Join not in the intern table / not `Hash+Eq` → two nodes (CSE broken) |
| `test_join_scheme_and_input_order_participate_in_structural_identity` | `{lr, rl, left_outer}` are 3 distinct ids | impl that sorts/canonicalizes the two inputs (loses build/probe asymmetry) or drops scheme from the hash → collisions |
| `test_join_ends_stages_and_survives_reduce` | reduced graph has `join==1`, `stage==3`, 7 nodes; join inputs point at stages | **Join treated as a fusible `Op`** → fusion runs through it: 1 stage, `join==0` |
| `test_dce_keeps_a_reachable_join_and_drops_an_unreachable_one` | only the output-reachable join survives reduce | DCE that keeps unreachable nodes → 2 joins survive |
| `test_reduction_over_a_join_is_byte_deterministic` | two identical builds → identical reduced `to_dot` | any nondeterminism in join reduction/labeling |
| `test_magic_is_unchanged_even_with_a_join` | `blob[:4]==b"GIR1"` on a join blob | MAGIC bump (format break) instead of an appended tag |
| `test_join_blob_roundtrips_byte_identically` | deser→re-ser byte-identical; `to_dot` stable | asymmetric encode/decode of `Join` (drops/reorders scheme or inputs) |
| `test_join_scheme_and_both_inputs_survive_the_roundtrip` | decoded join has scheme + `len(inputs)==2` | encoding that writes one input (Exchange copy) or omits the scheme |
| `test_an_unknown_node_tag_is_rejected_not_silently_misparsed` | poisoned tag 200 → `ValueError` | catch-all/default decode arm → **silent unknown-tag corruption** |
| `test_tag_one_past_join_is_rejected` | poisoned tag 7 → `ValueError` | decoder that admits `tag >= 6` (e.g. `>=` range check) instead of exact match |
| `test_low_tags_and_magic_are_byte_frozen` | golden byte oracle for source/op/reduction + GIR1 | adding `T_JOIN=6` **renumbers tags 0..2 or bumps the magic** (a renumber is self-consistent on round-trip; only an external literal catches it) |
| `test_join_plan_is_byte_identical_across_runs` | `to_bytes()` identical twice | nondeterministic plan serialization for the join path |
| `test_join_plan_roundtrips_both_producer_edges` | `gather_join.inputs == (0, 1)` survives | join gather modelled as single-producer `(0,)` (shuffle copy) → drops a side |
| `test_join_ir_survives_the_plan_roundtrip` | plan's embedded IR keeps its `Join` node | IR not carried verbatim (cloudpickle/lossy) — the Join is lost in the plan |
| `test_join_plan_task_ids_are_deterministic_and_backend_folded` | task id stable + changes with `backend_id` | id ignores `backend_id` → cross-backend cache poisoning (B-r5.2) |

## Non-vacuity

Pre-implementation every join test fails for the right reason: `GraphStore.add_join` is absent, so
each helper (`_join_graph` / `_two_sided_join` / `_join_ir`) raises `AttributeError` on the missing
symbol. The single control `test_low_tags_and_magic_are_byte_frozen` PASSES pre-implementation by
design — it exercises only today's codec (source/op/reduction) and is the deliberate M8-determinism
regression guard that adding `T_JOIN=6` does not disturb the existing tags or magic (mirrors M39's
`test_legacy_graphs_are_byte_stable`).
