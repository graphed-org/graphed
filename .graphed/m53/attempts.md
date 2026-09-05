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
