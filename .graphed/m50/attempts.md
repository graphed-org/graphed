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

## D1 — docs (how variations work)
Changed:
- docs/frontend/design.rst: new "How variations work" section — `graphed.vary`/`Varied`, the
  `labels`/`universe`/`nominal`/`variations` verbs, sibling-mode vs axis-mode (analyst view,
  cross-ref to graphed-histogram), varied preservation (build_bundle/reproduce/inspect over the
  value/weight/spec triple, format_version 2 vs 1), and §7.3's THREE invalidation classes each
  with scope: IR-level add/remove = UNCONDITIONAL; label-rename + one-time field churn (m48+m49)
  = ONLY by-value (`OpSpec.from_callable`) journals; the documented `from_ref` idiom is immune.

Executed examples (docs-sweep rule; all run against the shared venv, editable graphed @ G1):
- loose `vary` + labels/nominal/universe on numpy → `('nominal','jes_up','jes_down')`,
  `array([10.,20.,30.])`, `array([10.5,21.,31.5])` (exact reprs pasted).
- `variations(ctx)` on an event context → weight kind w/ `Fraction(1,2)`/`Fraction(-3,2)` and
  `None` for non-numeric `up`; shift kind `('shift', None)`.
- varied `build_bundle`/`reproduce`/`inspect` → `format_version` 2, `{label:array}`, inspect
  lists labels without executing (verified both on the 6-event and the chained 3-event fixture).
- §7.3 claims verified against source (not asserted): `DurablePlan.task_id` =
  sha256(domain, ir, process.identity(), partition) (plan.py); `OpSpec.from_ref` identity =
  `b"ref\0"+ref` (no closure state); `from_callable` non-importable → opaque cloudpickle blob;
  documented idiom in docs/checkpoint/design.rst is `from_ref`.

Gates:
- sphinx -W (docs/ → docs/_build/html): EXIT 0, zero warnings (`grep -c WARNING` = 0).
- precommit --fast: toml/workflows/integrity/prek(ruff+format)/mypy(strict)/cargo fmt → all ok.
- Docs-only change: pytest suite unaffected by an .rst edit (not re-run; the informative legs are
  docs + integrity + lint/types, all green).

## FIX CYCLE 1 (three-lens review) — impl start

Three findings, all graphed-only (histogram READ-ONLY, verified sound):
- INT-1 (BLOCKER): preserve↔histogram content_hash contract broken. H1's `_fill_chash`
  (boost.py) discriminates the fill node id (`spec + "\x00" + disc`, disc = [unweighted? /
  n_weights=N? / variation=<json>?]); preserve's histogram plugin re-derived only `sha256(spec)`,
  so build_bundle integrity rejects unweighted/multi-weight/axis fills. Repro:
  `pytest tests/frozen/preserve/m25 m27 m30` → 3 FAILED "hashes to X not recorded Y".
  Root cause of the conflict I MUST reconcile: two record paths, IDENTICAL params, DIFFERENT
  recorded id — `Histogram.fill` records the DISCRIMINATED id; the legacy base
  `record_external(HISTOGRAM_PLUGIN, spec.encode(), ...)` (frozen m27
  test_histogram_multiple_weights_multiply_on_replay, currently GREEN) records the BARE
  `sha256(spec)` with the same n_weights=2 params. A pure synthesize(params) can serve only one.
  FIX: thread the recorded content_hash into synthesize (bundle.py already holds `ch`); synthesize
  emits whichever canonical form (discriminated | bare) hashes to `ch`. eval reconstructs the fill
  from params["spec"] (+ variation/n_weights), never decoding the now-discriminated payload.
- G1-boost-import (MED): accessors.universe axis arm imported boost_histogram for `bh.loc`;
  replace with the import-free `x.axes[index].index(label)` (measured identical, incl. the
  KeyError "'zzz' not in axis" on an unknown label).
- D1-rename-overclaim (MED): docs/frontend/design.rst "honest limits" — rename "only by-value
  journals" + "immune to the last two" is FALSE for §6.2 axis mode (labels are IR content:
  StrCategory bins → spec/params/content_hash → IR → task_id). Scope rename to sibling; add
  axis-mode carve-out (unconditional, from_ref included). field-churn unchanged.

### Results (all gates green; precommit whole-repo pytest leg is pre-existing-broken, see below)

INT-1 fix (option b, ch-threaded): synthesize(params, recorded_hash) emits whichever canonical
form (discriminated `spec\x00disc` | bare `spec`) hashes to the recorded id; eval reconstructs the
FillEvaluator from params (spec + n_weights + json.loads(variation)), never decoding the payload.
_base.SynthesizePayload gains the hash arg; bundle.py passes `ch`. Why ch is required: two record
paths mint the same params under DIFFERENT ids (fill=discriminated, base record_external=bare) — a
params-only synthesize can serve only one. Adversarial hunt: that discriminated-vs-bare same-spec
pair is the ONLY same-params/different-id case (two real producers); the fix reconciles it via ch,
so no pair the plugin still mis-hashes could be constructed (both frozen — m30 disc + m27 bare —
stay green, and tests/extra witnesses both).

G1: accessors.universe axis arm now `x[{index: x.axes[index].index(label)}]` (no bh import);
measured identical incl. KeyError "'zzz' not in axis" (both bh.loc and StrCategory.index raise it).

D1: rename bullet scoped to sibling; axis-mode carve-out added (labels are IR content →
unconditional, from_ref included). Probe: axis rename → serialized IR CHANGED (1856≠1880);
sibling rename → IR byte-identical (2405=2405).

Gates:
- tests/frozen/preserve m25+m27+m30: 20 passed, 6 skipped, 0 failed (was 3 FAILED).
- tests/extra/preserve/m50 (manual; not in run-tests.sh): 2 passed — axis-mode end-to-end
  build_bundle+reproduce bit-for-bit (guard-free base-record technique) + adversarial disc/bare pair.
- Full `PATH=.venv/bin COV=1 ./scripts/run-tests.sh`: RC=0, combined coverage 94% (≥90). m50 trees
  green: preserve(8) debug(2) frontend(5) awkward(2).
- graphed-histogram tests/frozen: 178, 0 failed (unaffected — nothing changed there).
- ruff + ruff-format clean; mypy --strict clean (4 changed src files + via precommit --fast).
- determinism: canonicalization byte-stable across 2 runs for unweighted/single/multi/axis.
- precommit --fast: RC=0 (toml/workflows/integrity/prek/mypy-strict/cargo-fmt). Full precommit:
  sphinx -W ok; the whole-repo `pytest -q` leg FAILs with 35 collection "import file mismatch"
  errors — PRE-EXISTING repo-wide basename collisions (identical count with/without my file; my
  unique-basename file adds zero). Authoritative test gate is per-subtree run-tests.sh (green).

Pre-existing gap discovered (out of INT-1 scope, NOT fixed): a real `Histogram.fill(...,
variation_axis=True)` records `histogram.weight_guard` externals with no preserve plugin → opaque →
whole-graph reproduce of a seam-active axis fill is blocked. Separate design/review (guard-family
preservability); the INT-1 content_hash contract is fully closed for all three members.
