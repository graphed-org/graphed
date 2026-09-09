# Test Dispute — tests/frozen/awkward/m48/test_vary_stacking.py::test_a_factor_read_at_the_parent_is_accepted_and_re_indexed_to_the_derived_row_space

## The test
Registers `sf` on the masked child `sel` (ambient: `pu` weight + `jes` shift carried in), with
members `pu_weight(events, 1.05/1.1)` — `Varied` over `jes` (they read the jes-varied MET). For
every label of the ambient it asserts the weight's row count equals the mask's rows at
`chosen = label if label in mask_labels else "nominal"`.

## The clause it contradicts
Plan `systematics-vary-plan.md` §2 (m53 dependency fan-out): a member that reads an object shifted
by a registered nuisance is a genuine dependency and mints the joint. On main the SAME registration
at the parent `events` mints `sf_up__jes_up`/`sf_up__jes_down`; at the masked child it does not,
only because the adopted composed ambient's tag map picks up the leaked shift `jes` and m53's
kind-based exclusion fires (`scratchpad/m56/m48probe.py`). The test therefore pins a lineage
inconsistency, not a rule. With the m56 node-based test the child mints the joints too, and each
joint is re-indexed to ITS coordinate's row space: `sf_up__jes_up` has 23 rows = the mask's rows at
`jes_up` (§2.1(b)), where the test's `chosen` maps a joint label to `"nominal"` (19 rows).

## Proposed correction
`chosen` resolves a label through its registered point: the mask's label for any coordinate whose
nuisance the mask carries (`jes_up` for `sf_up__jes_up`), `"nominal"` otherwise. The assertion and
every other leg stay as they are. Re-freeze requires the owner's affirmation.

## Resolution
Owner affirmed the re-freeze 2026-09-08 ("the proposed solution gives correct behavior"); applied
as the correction above, in its own commit on `m56/both-kind-fanout`.
