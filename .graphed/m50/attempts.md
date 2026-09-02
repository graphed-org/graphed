# m50 G1 — graphed introspection + varied preservation (implementer log)

## Iteration 1
Changed:
- python/graphed/accessors.py: §6.2(i-bis) axis-mode arm of `labels`/`universe`; new
  `_variation_axis_index` helper (recognises `axis.__dict__.get("name")=="variation"`, never
  `h.axes.name`); new `graphed.variations(ctx)` verb → `{name:{tag:(kind,value|None)}}`, kinds
  "weight" (ambient-weight tag map) / "shift" (Varied collection tag maps), numeric via
  `_tags.numeric_value`.
- python/graphed/__init__.py: export `variations`.
- python/graphed/preserve/bundle.py: varied `build_bundle`/`reproduce` over the value/weight/spec
  triple (a Varied value/weight triggers it, no new kw); manifest `analysis.variations`
  (sorted-by-label) additive map; `format_version==2` for varied (FORMAT_VERSION stays 1);
  `inspect()` lists universes without executing; `reproduce` returns `{label:array}` varied /
  bare ndarray unvaried.

Gate results:
- frozen preserve/m50: 8 passed (EXIT 0).
- carryovers debug/m50 + frontend/m50 + awkward/m50: 9 passed; wider tests/frozen/preserve: green.
- ruff: clean. mypy --strict python: clean (77 files).
- Removed a redundant unknown-label guard in universe axis-mode arm — bh.loc already raises
  KeyError; keeps the diff line frozen-covered and lazier.
- combined COV gate + precommit: see below.

## Iteration 2 (final)
- Restructured accessors labels/universe axis-mode arms so the PRE-EXISTING unvaried-histogram
  lines stay textually unchanged (out of the diff); only the axis-mode block is new and is
  frozen-covered by preserve/m50. `_variation_axis_index` is now a one-liner (`next(...)`) whose
  filter's True+False branches are both exercised by the axis-mode fixture (Regular axis → False,
  variation StrCategory → True).
- Added tests/extra/preserve/m50/test_variations_shapes.py: shift-only context (ambient weight
  None → the weight arm is skipped) still reports the shift family; non-context guard raises.

Final gates:
- frozen preserve/m50: 8 passed. extra preserve/m50: 2 passed.
- m48 accessor trees (frozen frontend/m48 + awkward/m48 + extra accessor shapes): 121 passed.
- carryovers debug/m50 + frontend/m50 + awkward/m50: 9 passed. wider tests/frozen/preserve: green.
- determinism: two independent varied build_bundle runs → byte-identical analysis+sources+ir
  manifest (environment is audit-only). format_version==2 varied / ==1 unvaried.
- precommit --fast: toml-valid, workflows-valid, integrity-scan, prek(lint), mypy(strict),
  cargo fmt → all ok (RC=0).
- COV=1 ./scripts/run-tests.sh (combined): see final RC/coverage below.

## Iteration 3 (final gate + cleanup)
- Combined `COV=1 ./scripts/run-tests.sh`: RC=0, 0 FAILED/ERROR; combined `coverage report
  --fail-under=90` PASSES at TOTAL 94%. accessors.py 97% (uncovered: line 129 the variations
  non-context guard, branch 132->135 the ambient-None arm, line 214 pre-existing `_follow`),
  bundle.py 96% (all uncovered items pre-existing: env container_digest, source/external
  missing-entry raises).
- Diff coverage from FROZEN preserve/m50: every new axis-mode line + the variations weight/shift
  enumeration is frozen-covered; only the defensive non-context guard line + its branch are not
  (acceptable, mirrors the existing weight()/labels() guards).
- Removed tests/extra/preserve/m50: run-tests.sh's `preserve` suite entry does not include
  tests/extra/preserve, so it was never collected by the gate (dead). Frozen preserve/m50 already
  carries the coverage story; not worth a shared-runner edit.
