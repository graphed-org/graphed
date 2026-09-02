# m49 frozen suite — `graphed` / `tests/frozen/awkward/m49`

Milestone m49 (shift path + impact + executor end-to-end). Authority: `systematics-vary-plan.md`
r41. Every anchor here imports `graphed.awkward`, so none of them can live in
`tests/frozen/frontend/m49`, which a required free-threaded CI job collects under
`pytest hypothesis numpy` alone (§10/m49 partition rule (2)). No whole-subtree job collects
`tests/frozen/awkward`; `scripts/run-tests.sh` runs this directory in its own process, and helper
basenames carry an `m49_` prefix regardless (prepend import mode binds a bare top-level name to
whichever sibling directory imported first).

| anchor | plan clause | test |
|---|---|---|
| `vary-m49-B1` | §2.5 shift-after-weight, the violating program (pairing, not membership) | `test_shift_after_weight.py::test_a_weight_registered_before_the_shift_it_reads_is_reported_with_its_collection` |
| `vary-m49-B1` | §2.5 positive control one — the correct ORDER | `test_shift_after_weight.py::test_a_weight_registered_after_the_shift_reports_nothing` |
| `vary-m49-B1` | §2.5 positive control two — an ambient weight the walk answers no for | `test_shift_after_weight.py::test_an_ambient_weight_that_does_not_read_the_shifted_collection_reports_nothing` |
| `vary-m49-B2` | §2.6c derivation identity is the mask's PER-LABEL node ids (carried `ir-F3`) | `test_context_label_regressions.py::test_two_selections_sharing_a_nominal_universe_derive_distinct_contexts` |
| `vary-m49-B3` | §2.2/§6.1d(3) a projected context keeps the labels §2.5's walk reads (carried `ir-F5`) | `test_context_label_regressions.py::test_a_label_read_through_a_projected_context_still_reaches_the_output` |
| `vary-m49-B4` | §5.4 refusal MESSAGE — the verb and the container's labels, through `gak.join` | `test_join_refusal.py::test_a_varied_operand_is_refused_naming_the_verb_and_the_containers_labels` |
| `vary-m49-B4` | §5.4 positive control — a variation downstream of the join compiles per universe | `test_join_refusal.py::test_a_variation_downstream_of_the_join_compiles_and_answers_per_universe` |
| `vary-m49-B5` | §6.1d broadcast blame + its compatible-factor control, both structure paths | `test_broadcast_blame.py::test_a_structure_mismatch_agreeing_on_outer_length_blames_the_factor` |

## Spellings this freeze pins

* **§2.5's second diagnostic channel** is `CompiledGraph.shift_after_weight`: a sorted tuple of
  `(factor family name, collection name)` pairs, `()` when the order is sound. Detection is
  record-time (§2.5 — the registry's weak reference is dead by compile, and no Session-retained
  object carries a collection name); the field is where it surfaces.
* **§5.4's message** names the refusing verb as the user spelled it (`gak.join`) and every label
  `graphed.labels` answers for the offending container, `"nominal"` included.
* **§6.1d's blame** is a `GraphedError` naming the operand role (`factor`) and the factor's own
  awkward type below its outer length — the part a typetracer form and a materialized partition
  render identically, so one message shape serves both paths.

## Notes for the implementer

* `vary-m49-B5` brackets a CLASS with a member at each end: the regular-vs-regular mismatch is
  decidable from the typetracer forms and raises at RECORD time (already a `GraphedError`, naming
  two anonymous `RegularArray`s), while the jagged pair differs only in the data and raises at
  EXECUTION time (a raw `ValueError`). Both operands reach `graphed.awkward`'s `ak.broadcast_arrays`
  arm, which serves the form path and the eval path alike.
* `vary-m49-B4`'s refusal is one class over three message sites in the repo — the `gak` dispatch
  wrapper, `refuse_boundary`, and `Varied`'s refusing methods. This tree exercises the first.
* `vary-m49-B2` and `vary-m49-B3` are green against the code as it stands: both were repaired
  during the m48 review cycle without a frozen witness, and the m48 suite stays green under either
  regression. They are anchors against re-introduction, not new behaviour.
