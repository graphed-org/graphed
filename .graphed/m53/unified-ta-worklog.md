# m53-unified re-freeze — test-author worklog

TOC
- [Scope](#scope)
- [Transform recipe](#transform-recipe)
- [Fixture merge decision](#fixture-merge-decision)
- [VariationError coverage](#variationerror-coverage)
- [Non-vacuity evidence](#non-vacuity-evidence)
- [Determinism + integrity](#determinism--integrity)

## Scope
Re-froze `graphed.vary`'s `points=` call-sites onto the unified `variations=` list grammar
(plan `m53-unified-variations-plan.md` §2/§3/§6) and added `VariationError` (§4) coverage. Base tip
`d090625` (implementation ABSENT — the parser + `VariationError` are not built yet).

Trees with real call-sites: `frontend/{m52,m53,m53b}` ONLY. `corpus/*` and `awkward/*` carry no
`points=` (already auto-fanout) — untouched, re-verified green (`pytest ... corpus/m52 awkward/m52
awkward/m53` → 18 passed, 0 collection errors).

Files edited (12) + 1 new; zero under `python/graphed/**`.

## Transform recipe
Mechanical, at each `graphed.vary(...)` boundary:
`variations={t1:a1, ...}, points=[p1, ...]` → `variations=[(t1,a1), ..., p1, ...]`.
Shift form (`collections=` supplies members, no declare dict): `points=[p]` → `variations=[p]`.
`variations=members` (dict var) → `variations=[*members.items(), p1, ...]`.
Every assertion + discrimination witness preserved verbatim; only input spelling moved.

Two prose repairs forced by the keyword removal (not mere wording — they change a line built/asserted):
- `test_point_registry::test_a_variation_tagged_points_still_registers_through_variations`: docstring
  premise "points leaves BOTH keyword namespaces" is now false; retitled to §2's "`points` is freed
  as a legal tag name". Assertions unchanged.
- `test_composes_as_union::..._construction_error`: assertion `"points" in msg` greps the REMOVED
  keyword → would red once the implementer rewrites the message. Re-anchored to `"composes_as_union"`
  (the flag name survives; discrimination "names the incompatible combination" preserved).

## Fixture merge decision
`m53_policy_fixtures.fanout_weight` / `numeric_fanout` forwarded `points=` through `**vary_kwargs`.
Gave each an explicit keyword-only `points=None` that MERGES into the list ONLY when present:
`variations = members if points is None else [*members.items(), *points]`.
- points=None (pure-declare callers) → passes a DICT → byte-identical vary behavior → PASSES on
  `d090625` (the in-file positive controls: guard/composes pure-declare tests).
- points=[...] → passes a LIST → feature-absence on `d090625`.
Fixture param kept the name `points` (test plumbing, never forwarded to `vary`) → m53 test bodies
calling `fanout_weight(points=...)` stay byte-identical; the re-authoring lives at the fixture's
internal `vary` call. Docstrings updated (points no longer passes through).

## VariationError coverage
New `frontend/m53b/test_variation_error.py`, 6 situations + malformed + 1 in-file positive control.
Each situation: refuse-member asserts `isinstance(exc, graphed.VariationError)` AND
`isinstance(exc, GraphedError)`, `exc.situation == <plan §4 string>`, and a message substring MEASURED
on `d090625` (plan §5 keeps messages when re-typing the raise). Member-at-each-end (all admits proven
non-raising via the equivalent old API):

| test | situation | refuse | admit (adversarial) | substring |
|---|---|---|---|---|
| E1 | unresolved | dep member, `{corr:a, jer:up}` (jer not in jes grid) | `{corr:a, jes:up}` (in grid) | "joint" |
| E3-nuisance | unresolved | `{corr:a, nosuch:up}` | `{corr:a, jes:up}` | "nosuch" |
| E3-tag | unresolved | `{corr:down, jes:up}` (down not declared) | `{corr:up, jes:up}` (up declared) | "down" |
| E4 | unreachable | indep, `{corr:a, jes:up, jer:sideways}` | `...jer:up` (reachable) | "sideways"+"jer" |
| E5 | duplicate | `jes_btag` renders label `jes_btag_up` twice | same label+point twice (idempotent) | "jes_btag_up" |
| E6 | empty | `{corr:a}` own-only | `{corr:a, jes:1, btag:-1, jer:0}` (jer:0 drops, 2 survive) | "central"/"nominal" |
| Ec | conflict | union + placement | union-alone AND placement-alone (neither leg) | "composes_as_union" |
| malformed | (GraphedTypeError) | `variations=[("a",arr), 42]` | valid declare + placement | — |

Discrimination retained by the untouched re-authored tests (own-axis kept for select vs dropped for
re-point; exact re-pointed point read via `graphed.points()`; prune selects-and-drops sibling). A
wrong impl treating every placement as a select keeps the own axis on the re-point tests → fails
`"corr" not in point`; dropping own axis on both → fails the select tests' `"corr" in prune_point`.

## Non-vacuity evidence (on `d090625`)
`pytest frontend/{m52,m53,m53b}`: **33 PASSED / 49 FAILED**, 0 collection errors.
- 33 passes = pure-declare / registry-reader / old-kwargs positive controls (same-run live instrument).
- 49 failures, all feature-absence: 39 × `AttributeError: 'list' object has no attribute 'items'`
  (list-form on the pre-parser router), 7 × `AttributeError: module 'graphed' has no attribute
  'VariationError'` (situations), 1 × list-form on the malformed test.
- Feature was NOT importable/attribute-able: `from graphed import VariationError` → ImportError;
  `graphed.errors` exports only `{GraphedError, GraphedTypeError}`.
`test_variation_error.py`: positive control PASSES, 8 situations FAIL (feature-absence). PLC0415
(repo bans in-body imports) forced `graphed.VariationError` attribute access → the AttributeError
signature above; equally feature-absence, ruff-clean under the repo config.

## Determinism + integrity
- Two full runs → identical PASSED/FAILED set (`diff` empty).
- `ruff check --config pyproject.toml` on all 13 touched files → All checks passed.
- Zero `python/graphed/**` files touched (`git status --porcelain -- python/graphed/` empty).
- Freeze tag `m53-unified-freeze`; `git diff m53-unified-freeze -- tests/frozen` must stay empty.
