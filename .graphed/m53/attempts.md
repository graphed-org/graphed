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
- REBUILD prose note (lead): phrase any off-grid reference in ANALYST terms — drop internal-process
  framing like "second gated cycle, in design" from user-facing tour cells. The rebuild replaces
  cell 0 anyway, so the current 61f4f47 pointer stands as-is until then.

## Off-grid / additive `points=` (m53-freeze2) — implementer iteration

Target: `tests/frozen/frontend/m53b` 17 FAIL / 3 PASS -> 20/20; CORE (corpus/m52 + awkward/m52
+ awkward/m53 + frontend/m52 + frontend/m53) stays 72/0. One concern, one file: `python/graphed/vary.py`.

Change (plan §3/§7): reworked `_prune` -> per-entry router `_route`, returning
`(kept_joint_labels, additive_overrides)`. Per entry, `t = canonical_tag(E[name])`; the discriminator
is `foreign_by_tag[t]` (already computed in `_fanout`): non-empty (member DEPENDENT) -> the frozen
prune path unchanged (member-resolution, own axis kept); empty (member INDEPENDENT) -> ADDITIVE —
validate the foreign coords by carrier-reachability (`_check_reachable(name, Point(foreign),
_reachable(...))`) and re-point the one-at-a-time label `f"{name}_{t}"` from `{name:t}` to the
foreign-only point (own axis dropped). `_fanout` now returns the additive overrides as a 3rd element;
`_mint_defaults` SKIPS any tag with an additive override then mints the overrides (the ONE coordination
change — else `_check_unique` rejects two points for one label). `context.py` untouched (gather_members
signature `(one_at_a_time, joints)` unchanged). The shared "only-name-coordinate" check (`Point(entry)`
zero-drops) refuses both the foreign-empty entry and the all-zero point in one place; the collision and
single-foreign-already-owned refusals fall out of `_check_unique`.

Gates: m53b 20/20; core 72/0; `git diff m53-freeze2 -- tests/frozen` and `git diff HEAD -- tests/frozen`
both empty (no frozen file touched — vary.py is the only working-tree change). ruff check + format clean;
mypy clean (78 files). Determinism: μRxμF grid registry byte-identical across PYTHONHASHSEED 0/12345,
plus the two frozen determinism tests. Diff line+branch coverage on the changed vary.py lines from the
FROZEN suite = 24/26 = 92.3% (>90 gate). The one frozen-uncovered line is the prune "names no joint"
refusal for a DEPENDENT member named with a reachable-but-uncrossed axis (on m53-freeze that line was
hit by the m53b additive entries themselves, which the OLD prune refused; they now correctly route
additive). Added `tests/extra/m53/test_a_dependent_members_reachable_uncrossed_axis_is_refused_by_prune`
(distinct from the m53b `nosuch` guardrail, which is refused earlier at `_check_reachable`) -> combined
diff coverage 26/26 = 100%.

---

## m53-unified — unify `points=` into one `variations=` surface + `VariationError` (implementer, freeze `m53-unified-freeze` @ e8c351a)

Front-door reshape only; the APPROVED router (`d090625`) stays byte-identical. Source-only, 3 files.

- `errors.py`: `VariationError(GraphedError)` — `__init__(situation, entry, *, valid=None, detail="")`
  storing `.situation/.entry/.valid/.detail`, message `f"{situation}: {detail}"`. Exported from
  `__init__.py` (`__all__` + import).
- `vary.py`: dropped the `points=` param; widened `variations=` to `Mapping | Iterable[entry]`. New
  `_parse_variations(variations)` splits a non-Mapping iterable into (declares dict, placements list)
  and feeds the EXISTING internal `variations`(declares)/`points`(placements) seam unchanged — a
  2-`tuple` is a declare, a `Mapping` a placement, anything else -> `GraphedTypeError(capture())`. A
  Mapping/None passes straight through as declares. Empty channels collapse to `None` so the two-param
  seam stays byte-identical.
- Re-typed the §4 raises (control flow untouched, only the exception class + stale `points=` wording):
  `_route` (no-joint->`unresolved`, own-not-a-tag->`unresolved`, own-only->`empty`), `_check_reachable`
  (unknown-nuisance->`unresolved`, unreachable-value->`unreachable`), `_check_unique` (both->`duplicate`),
  `gather_members` composes+placement guard (->`conflict`). Every asserted message substring preserved
  (enumerated across all vary-arc frozen trees: joint/nosuch/down/sideways/jer/jes_btag_up/central/
  nominal/composes_as_union/"already registered under"/typo/btag/corr/nowhere/up — none was "points").
  `_guard`'s user-facing message ("pass points=[...]") reworded to "pass variations= placements"
  (test_guard asserts only 15/jes/btag); logic unchanged, stays `GraphedError`.

Gates (all reproduced):
- Frozen vary-arc suite `pytest corpus/m52 awkward/{m52,m53} frontend/{m52,m53,m53b}`: **101 passed / 0
  failed / 0 skipped** (frontend 83, corpus+awkward 18). The 49 pre-impl failures now pass, 33 controls
  green; `test_variation_error.py` 9/9 (8 situations + positive control).
- `git diff m53-unified-freeze -- tests/frozen` EMPTY (no frozen file touched; source diff = errors.py
  +19, __init__.py +3, vary.py). 
- Diff coverage from the FROZEN suite alone: **100% (33/33 lines, 0 missing)** via `diff-cover
  --compare-branch=m53-unified-freeze`. `coverage report -m` branch partials (`vary.py 635->640` in
  `register()`, `errors.py 39->41` in pre-existing `GraphedTypeError`) are OLD lines outside the diff;
  no changed line carries an arc-partial (the `_parse_variations` if/elif/else + None-return branches
  are all exercised: Mapping/None/list-with-declares/list-with-placements/malformed).
- ruff check + ruff format --check clean; mypy --strict clean (78 files).
- Determinism: determinism-bearing frozen tests identical across two runs (`diff` empty); frozen
  `test_...byte_deterministic_across_two_runs` (awkward, fresh interpreters) + `test_the_mixed_registry
  _is_deterministic_across_two_runs` pass.
- Integrity: no `# type: ignore`/`except: pass`/`NotImplementedError`/`todo!`/`pragma: no cover`/`skip`/
  `xfail` in changed source (only pre-existing PLC0415 import-cycle noqa). No named target stubbed.

Diff coverage is fully carried by the frozen suite (no NEW `tests/extra` needed for coverage), BUT
the reshape broke two STALE `points=` call-sites in the pre-existing extra suite that I failed to
re-run at first pass (reviewer REJECT: one blocking finding + a sibling). Swept the WHOLE
`tests/extra` tree:
- `tests/extra/m53/test_discriminator.py`: dependent-member prune refusal re-authored to the list
  form `variations=[("a", dependent), {"corr": "a", "jer": "up"}]` (raises `VariationError`, is-a
  `GraphedError`; "names no joint" now in `.detail`).
- `tests/extra/frontend/m52/test_intra_call_point_uniqueness.py`: the m52 MAPPING form `points={tag:
  point}` no longer exists. Members made INDEPENDENT (`graphed.nominal(ctx["pt"]) * 0.5`) so both
  placements route ADDITIVE and collide at ONE point via `_check_unique`'s intra-call half — collision
  call re-points corr_a/corr_b to one point (refused, naming both); admit call re-points to distinct
  points (both mint, own axis dropped, `points["corr_a"]==JOINT`). Docstring premise (JOINT absent
  from the registry at call start) preserved.
- `uv run pytest tests/extra -p no:cacheprovider`: 70 passed / 5 skipped (pre-existing optional-dep
  imports: graphed_histogram/exec_local/executors) / 0 failed. `git diff m53-unified-freeze --
  tests/frozen` stays EMPTY (only `tests/extra/**` touched this iteration).
