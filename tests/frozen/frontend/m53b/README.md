# frontend/m53b — additive / off-grid `points=` mode (traceability)

A SECOND freeze increment on the m53 dependency-fanout core (frozen at `m53-freeze`). The core's
`points=` PRUNES a dependent member's auto-grid; this increment adds the complement — an off-grid
`points=` entry over an INDEPENDENT member RE-POINTS its one-at-a-time label from the default
`{name: tag}` to a foreign-only point (own axis dropped), restoring the retired m52 mechanism in the
unified m53 LIST form. Owner-ruled Option 2: `points=` is always an iterable of coordinate maps,
routed per-entry by the named member's genuine foreign dependence (`_foreign(member)` non-empty →
prune, empty → additive).

Awkward-free numpy-idiom: every additive re-point is a record-time point-registry fact read back
through `graphed.points()`; none needs backend materialization, so this increment stays entirely on
the frontend tree (the required free-threaded CI job collects `tests/frozen/frontend` whole with
only pytest+hypothesis+numpy). The `m53b_` prefix is load-bearing under prepend import mode.

Fixture: `m53b_offgrid_fixtures.py` — two-axis carriers (`jes`+`jer`) for the three overloads and
refusal classes; `scale_grid()` (independent `muR`/`muF`, tags `2`/`0.5`) for the μR×μF grid;
`offdiag_axes()` (`jes`/`btag`, tags `1`/`-1`) for the prescribed off-diagonal; `triple_numeric()`
for the zero-drop survivor; `unregistered_context()` for carrier-reachability refusal. An independent
member is read off `graphed.nominal(...)` (or a plain record array), so a plain `vary` mints it
one-at-a-time and `points=` must re-point it.

| Work item (plan §) | Test |
|---|---|
| §3.4/§5 additive re-point on the loose overload (own axis dropped, label kept) | `test_additive_repoint.py::test_additive_repoints_on_the_loose_overload` |
| §3.4/§5 additive re-point on the weight overload | `test_additive_repoint.py::test_additive_repoints_on_the_weight_overload` |
| §3.4/§5 additive re-point on the shift overload | `test_additive_repoint.py::test_additive_repoints_on_the_shift_overload` |
| §7.2 μR×μF 7-point grid: exactly 7 universes | `test_scale_grid.py::test_the_grid_has_exactly_seven_universes` |
| §7.2 the two diagonals are 2-coordinate off-grid points (own axis dropped) | `test_scale_grid.py::test_the_diagonals_are_two_coordinate_off_grid_points` |
| §7.2 the whole grid registry matches; labels keep their spelling | `test_scale_grid.py::test_the_grid_matches_the_seven_point_map_and_keeps_the_labels` |
| §7.3 prescribed off-diagonal `{jes:1,btag:-1}` re-points; unregistered axes refused (class) | `test_offdiagonal_prescribed.py::test_a_prescribed_off_diagonal_repoints_an_independent_member` |
| §7.4 per-entry MIXING: one prune entry + one additive entry, disjoint members | `test_additive_mixing.py::test_a_prune_entry_and_an_additive_entry_both_land_in_one_call` |
| §6 own-axis rule: prune keeps own axis, additive drops it (same mixed call) | `test_additive_mixing.py::test_the_own_axis_rule_holds_across_the_mixed_call` |
| §6 mixed registry deterministic across two runs | `test_additive_mixing.py::test_the_mixed_registry_is_deterministic_across_two_runs` |
| §4/§3.1 refuse an own-tag naming no member of the call (typo) | `test_additive_refusals.py::test_an_own_tag_naming_no_member_of_this_call_is_refused` |
| §3.2 refuse an entry with only its own coordinate | `test_additive_refusals.py::test_an_entry_with_only_its_own_coordinate_is_refused` |
| §4 refuse a single-foreign point already owned by its axis label | `test_additive_refusals.py::test_a_single_foreign_coordinate_already_owned_by_its_axis_label_is_refused` |
| §4 refuse an unreachable additive coordinate, naming what is registered | `test_additive_refusals.py::test_an_unreachable_additive_coordinate_is_refused_naming_what_is_registered` |
| §5 refuse two additive entries re-pointing to one point (collision) | `test_additive_refusals.py::test_two_additive_entries_re_pointing_to_one_point_are_refused` |
| §4 all-zero point drops to empty (refused); a zero beside two live coords survives | `test_additive_refusals.py::test_an_all_zero_point_drops_to_empty_and_is_refused_but_a_zero_beside_live_survives` |
| §3 (guardrail) a dependent member's uncarried axis is refused by prune, not re-routed | `test_additive_refusals.py::test_a_dependent_members_uncarried_axis_is_refused_by_prune_not_silently_additive` |
| §7.7(a) §1b carve-out: a genuinely-dependent member auto-mints its compound, no `points=` | `test_additive_regression.py::test_a_genuinely_dependent_member_still_auto_mints_its_compound` |
| §7.7(b) a pure prune selection is byte-identical to the frozen m53 behavior | `test_additive_regression.py::test_a_pure_prune_selection_is_byte_identical_to_the_frozen_behavior` |
| §7.8 additive registry byte-identical across two `PYTHONHASHSEED`s | `test_additive_regression.py::test_the_additive_registry_is_byte_identical_across_hash_seeds` |

## Non-vacuity (fails on the current `m53-freeze` tree for the right reason)

Additive off-grid over an independent member is refused today (probe "(b)"): the fanout treats the
member as independent and derives no joint, so every additive entry raises
`GraphedError: points= entry {...} names no joint the fanout of '<name>' derives; the derived joints
are []`. 17 of the 20 tests fail at their first additive `vary` call for this feature-absence; the
witness is `graphed.points()` reporting the annotated multi-axis foreign-only point (e.g.
`scale_dndn` → `{muR: 5em1, muF: 5em1}`, `corr_off` → `{jes: 1, btag: m1}`), NOT the default
`{scale: dndn}` / `{corr: off}` a bare mint leaves, and NOT a silent nominal collapse.

The mixing entry carries a sharper signature — `... the derived joints are ['corr_dep__jes_down',
'corr_dep__jes_up']` — proving the independent `corr_ind` names none of the dependent member's
joints (it must route additive, not prune).

## Positive controls (PASS on the current tree — the harness is live, not merely failing)

- `test_a_genuinely_dependent_member_still_auto_mints_its_compound` — §1b: `jer_hi__jes_up` →
  `{jer: hi, jes: up}` minted with no `points=`.
- `test_a_pure_prune_selection_is_byte_identical_to_the_frozen_behavior` — the frozen prune registry.
- `test_a_dependent_members_uncarried_axis_is_refused_by_prune_not_silently_additive` — the prune
  guardrail (a dependent member's typo'd foreign axis stays refused).
</content>
