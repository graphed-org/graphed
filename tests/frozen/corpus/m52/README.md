# m52 frozen suite — `graphed` / `tests/frozen/corpus/m52`

Milestone m52 (nuisance POINTS). Authority: `systematics-design/nuisance-points-design.md` (FINAL);
§4.4 (`points=` and the R1-c joint set), §4.6 (resolution by projection), §4.10 (`graphed.points()`),
§6-C6. This tree is the arc's **only** check of the physics claim — C3 and C5 freeze *structural*
witnesses (which member node a label resolves to); this one asks whether the number is right. It is
frozen rather than `tests/extra` because §B.3 wants the covering hits from the frozen suite.

Frozen under the §A.7 integrity rule: read-only after the freeze tag. `git diff m52-freeze --
tests/frozen/` must stay empty. A frozen test that looks wrong is a Test Dispute at
`.graphed/m52/disputes/<test_id>.md`, never a repair in place.

Run: `python -m pytest tests/frozen/corpus -q` — m05 and m52 in ONE process, which is why every
basename here is distinct from m05's four and the helper carries the `m52_` prefix. **No
`conftest.py`**: three `corpus/m05` tests do `from conftest import ...` and would import this one.
`graphed_corpus` resolves through `pyproject.toml`'s `pythonpath` entry for the vendored mirror
`tests/_corpus`, never an installed copy.

Every m52-new symbol (`graphed.points`, the `points=` keyword, the two new corpus functions) is
reached from a test BODY — the two corpus functions as module *attributes* — so the tree COLLECTS
against `origin/main` and fails at run time for the RIGHT reason.

| anchor | design § | test |
|---|---|---|
| `vary-m52-C6` | §4.4 / §6-C6 — joint ≠ factorized product; joint == the eager reference; the flat/frozen machine-zero control | `test_joint_factorization.py::test_the_joint_universe_is_not_the_factorized_product_and_equals_the_direct_reference` |
| `vary-m52-C6` | §4.4 / §4.10 — the symmetric four are registered with their points and every universe is distinct | `test_joint_universe_set.py::test_the_joint_set_is_the_symmetric_four_and_every_universe_is_distinct` |
| `vary-m52-C6` | §4.4 — the four joint members are the two one-at-a-time SF expressions, the point choosing the inner universe | `test_joint_member_sharing.py::test_the_four_joint_members_are_the_same_expression_objects_as_the_two_one_at_a_time_ones` |

## Fixture family — `m52_corpus_fixtures.py`

* `EVENTS` — `graphed_corpus.make_events()`, the canonical dataset `corpus/m05`'s 23 stored
  references are built on. `REGION = "4j1b"` (the HT observable); `JES_FACTOR` is
  `systematics._apply_jes`' jet-pT scale.
* `joint_program()` — §4.4's R1-c spelling over the corpus ttbar analysis: a `jes` shift on `Jet`,
  the region's own selection, then one `vary(..., "btag", ..., is_weight=True, variations=…,
  points=JOINT_POINTS)` whose four joint tags take the **same two** `sf_hf_*` objects as the
  one-at-a-time pair. Returns the session, the HT observable, the ambient weight and both SF
  containers.
* `universe_hist` / `reference_hist` — one graphed universe, and the eager reference at one
  coordinate pair, both filled into `Hist.new.Reg(40, 0, 800, name="ht").Double()`.
* `integral` / `factorized` / `reldiff` / `reference_reldiff` — the one instrument all three legs of
  6.1 run through.

## Spellings this freeze pins

The isolated corpus implementer writes against these, and 6.1's legs only reproduce if both sides
write the same analysis.

* **`graphed_corpus.analyses.systematics.btag_sf_rel_uncertainty(pt)`** →
  `0.01 + 0.05 * np.minimum(pt / 100.0, 1.0)` — the per-jet FRACTIONAL b-tag SF uncertainty,
  evaluated at the jet pT **of the universe being computed**; that JES dependence is the cross term.
  `btag_up` multiplies each jet's SF by `1 + rel`, `btag_down` by `1 - rel`.
* **`ttbar_joint_reference(events, *, region, jes, btag, pt_dependent=True,
  freeze_selection=False) -> Hist`** — `jes` in `{"nominal","jes_up","jes_down"}`, `btag` in
  `{"nominal","btag_up","btag_down"}`. Selection and observable are `ttbar_region`'s verbatim (jets
  `pt > 25`, `n_good >= 4`, the region's b-tag count, HT weighted by the per-event SF product).
  `pt_dependent=False` reproduces the existing flat ±3% `_btag_weight` rule. `freeze_selection=True`
  takes **both** the jet-level `pt > 25` mask and the event-level cut from NOMINAL kinematics while
  still evaluating the observable and the SF at the shifted pT — freezing only the event cut leaves
  a residual and voids 6.1's machine-zero control.
* Contents come back **UNROUNDED** (no `_round_hist`); rounding to `STABLE_DECIMALS` is
  `bin_values` / `fingerprint`'s job, exactly as `corpus/m05` compares.
* The graphed-side factor is `0.01 + 0.05 * gak.where(pt < 100.0, pt / 100.0, 1.0)` — `gak` has no
  element-wise minimum, and this is identical arithmetic including at `pt == 100`.
* `TTBAR_FIXTURES` and the 23 stored goldens are untouched: C6 is additive, and
  `corpus/m05 test_fixtures_reproduce.py` must stay green. **No catalog row** — `all_fixtures()` is
  built from `ADL_QUERIES + TTBAR_FIXTURES + TTGAMMA_FIXTURES` and three frozen m05 tests assert the
  count is 23.
* The three `graphed_corpus` mirrors (`graphed-corpus/src`, `graphed/tests/_corpus`,
  `graphed-histogram/tests/_corpus`) stay byte-identical; the `diff -rq` is an implementer gate step,
  not a test — `graphed`'s CI has no copy of the other two to diff against.

## Notes for the implementer

* **The histogram producer here is eager `hist.Hist`, never `graphed_histogram`.** `graphed`'s dev
  extra carries `hist` and `boost-histogram` and no `graphed_histogram`; a tree reaching for it would
  `skip` in CI and this arc's only physics check would contribute zero frozen coverage hits.
* **6.1's three legs are INTEGRALS, not bins.** Per bin, JES migrates events between bins and even
  the flat, frozen-selection leg is nonzero — which would void the control.
* **The ordering assertion alone does not discriminate.** Under the silent-nominal answer the joint
  universe *is* the b-tag-only one, whose reldiff against the factorization is ~0.11 and passes
  `A > 3B` comfortably. What catches it is the weld to the reference and `joint != btag_only`; keep
  all three.
* 6.3 asserts node-id RELATIONS between two named candidate members, never a pinned integer — node
  ids are fixture-dependent.

## Non-vacuity (TEST_SANITY: 3 failed / 60 passed at freeze, m05's 60 untouched)

* 6.1 fails with `AttributeError: module 'graphed_corpus.analyses.systematics' has no attribute
  'ttbar_joint_reference'` — it measures the reference legs before touching graphed.
* 6.2 and 6.3 fail with `GraphedError: a variation member must be an Array or a Varied, got dict`:
  `vary` has `**tags`, so at baseline `points=` is swallowed as a variation TAG and its mapping is
  refused as a member. Feature-absent, at run time, past import — not a `TypeError`.
