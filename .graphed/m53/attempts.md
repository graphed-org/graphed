# m53 implementer attempts

## Target
42 FAIL / 30 PASS (measured pre-m53). Make the 42 pass; survivors + m48 independent-union stay green.

## C1 — discriminator + joint generator + composes + mint (vary.py, loose overload)

C1 wires the discriminator (`_foreign`) + generator (`_fanout`) through all three overloads (the
tuple return couples them: `gather_members` -> `(one_at_a_time, joints)`, so context.py moves with
vary.py or mypy/runtime break). Two refinements the frozen suite forced beyond the plan's literal
`registered_points` discriminator:
 - STACKED weight: a foreign nuisance in `old._tags` composes into the union via `_two_level(old,)`,
   so it is excluded (`composed`). Distinguishes stacked (btag in _tags) from the §8-g ambient-carry
   (jes carried but _tags empty), which DOES fan out.
 - SPECTATOR: a carrier varied by some OTHER nuisance but not the member's foreign one collapses it
   (m48 test_vary_stacking, loose `vary(base_jes,"jer",up=inner_varied)`); a fresh carrier or one
   sharing the foreign nuisance fans out. Both member objects are byte-identical, so the signal is
   the CARRIER, not the member.
Result: 56 pass, 16 fail (all C3 points= prune + C4 guard). awkward/m48 green (regression fixed).

## C3 — points= prune inversion
`_prune` resolves each `{nuisance:coord}` map to an auto-grid joint by point-equality: own-family
coord checked via canonical_tag tag-membership (so '0.5'!='0p5'), foreign coords via `_check_reachable`
(names registered universes), origin (no foreign coord) refused. Kept joints pruned from the grid.
All 14 points= tests pass. 2 guard tests remain (C4).

## C4 — the loud guard
`_guard` raises when the un-selected default grid `prod(family sizes)` (this family x each foreign
family, nominal included) exceeds `max_universes` (default 64), naming the count and families. points=
and composes_as_union bypass it; an independent family (no joints) is never guarded. Target suite: 72
passed (42 made green + 30 survivors).

## Source complete (C1 e4b9703, C3 13de600, C4 154a9cd, style c832fa0)
Target suite 72 passed; frozen diff empty; ruff/mypy clean; determinism (two-seed) green; diff
coverage vary.py 97.6% / context.py 100% from the FROZEN suite. All vary-touching frozen milestones
(awkward m48-m53, frontend m48-m53, corpus, checkpoint, numpy m51) green. No disputes.

## C5 notebooks — status: BLOCKED on lead design call
ADL graphed-adl-benchmarks.ipynb: cell 32 auto-fans 7->15 cleanly (no off-grid points=). Unblocked;
awaits lead's execution-scope preference (the notebook's own parallel-speedup study is heavy + m53-
unrelated).
Tour graphed-vary-systematics-tour.ipynb: Levels 0-14 + new headline/prune/guard cells feasible.
Levels 15-18 (muR x muF grid, PDF eigenvector correlations, numeric sigma, the 2-foreign {jes:1,btag:-1})
build OFF-GRID joints via m52 additive points= over INDEPENDENT members. m53 points= only PRUNES an
auto grid (member-resolution reachability, dispute ruling #1) and refuses off-grid points — verified:
`points=[{scale:dndn,muR:0.5,muF:0.5}]` -> "names no joint the fanout of 'scale' derives; …[]".
Plan §3's "precision knob for off-grid points" was never realized in the frozen suite (its only
precision test is grid-resident). Surfaced to lead: (A) add explicit additive off-grid points= mode,
(B) declare correlated-grid-over-independent-nuisances Phase-2 + recast 15-18 as markdown [recommended],
(C) other. Tour rewrite PAUSED pending decision.

## C5 notebooks — DONE (consolidated rulings, 2026-09-05)
Two separate commits in coffea-benchmarks-graphed-mvp @ graphed-mvp (docs; only the edited notebook
staged each time — coffea-adl-benchmarks.ipynb's pre-existing edit left untouched). Executed via
nbclient on the m52 .venv python3 kernel. NOTE: an earlier combined commit (5be40c4) that markdown-
recast tour Levels 15-18 as "Phase-2" was reset before push — superseded by the ruling below.

Ruling reconciliation: I first recast off-grid Levels 15-18 (muR x muF/PDF) as Phase-2 markdown per an
option-B reading; surfaced the tension; the OWNER then chose to BUILD the additive off-grid points=
mode (its own gated cycle, in design). So 15-18 are HELD OUT of the tour, NOT markdown-recast — they
will be authored against the new mode.

ADL b1bc05f: cells 31 (markdown 7->15, joints named) + 32 (STYLE programmatic over the 15 labels,
joint colour = b-tag component + jes-leg linestyle; titles 7->15; comment reframed to auto-fanout).
vary call UNCHANGED — auto-fanout fires on the real correctionlib/gak construction (verified 15
universes, 8 joints, 2-axis points). Executed cell 1 (setup) + cell 32 only; spliced outputs
preserving execution_count=21; Q1-Q8 (cells 2-31) and the parallel-speedup study (33+) byte-identical.
Witness numbers unchanged (49625.198758 -> 49634.770145) since m53 augments the 7.

Tour 7513eba: Levels 0-9 (cells 1-20) byte-identical. Cell 0 map: mechanism 3 = auto-fanout (was
points=), companion = full-15-grid. Second half rebuilt (44->34 cells), levels END at 15: L10
auto-fanout headline (9u), L11 projection resolution, L12 factorization-error (auto-fanned joints;
ordering 2.21% > 0.62% > 1.5e-16 control holds), L13 real-data+pool (9u survives pool; composes_as_union
control = 5), L14 grid control (union 5 / points= diagonal 7 / max_universes=8 guard), L15 the 3 m53
refusals (unreachable coord / prune+collapse / over-budget). Off-grid Levels 15-18 and the tangential
numeric-canonicalization level HELD OUT. Full end-to-end nbclient run: 0 errored, exec_counts 1-16
monotonic.

## Held-out tour levels — rebuild notes (for the off-grid Option-2 cycle)
Two old tour levels are held out of 61f4f47; the off-grid rebuild decides whether to restore them:
- old L17 "numeric coordinates canonicalized by VALUE": `vary(x,"morph",variations={numeric-string tags})`
  — the label keeps your spelling, `graphed.points()` renders the value; two spellings of one value
  ("0.5"/"0p5"/0.5/Fraction(1,2)) are ONE universe and registering both is refused. Fold into the
  μR×μF level (it uses numeric coordinates), where it is naturally motivated.
- old L19 "zero asymmetry": (a) a TAG literally named "0" mints a real universe `shift_0` (point
  {shift:0}) distinct from nominal; (b) a zero COORDINATE inside a points= map is dropped to origin,
  so a point reducing to a duplicate-of-default or to {} is refused. (b) is additive off-grid points=
  semantics → rebuild with the off-grid mode; (a) now entangles with auto-fanout + the injective-label
  check under m53 (`shift_0__jes_up` collapses onto `jes_up` → refused) and is partly covered by the
  m53 frozen mint-refusal contract (dispute ruling #4 names "0"/"central"/"nominal").
