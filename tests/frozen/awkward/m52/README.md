# m52 frozen suite — `graphed` / `tests/frozen/awkward/m52`

Milestone m52 (nuisance POINTS). Authority: `systematics-design/nuisance-points-design.md` (FINAL)
as grounded by `m52-decomposition.md` §3.2; this tree carries C3 alone — projection resolution at
combination, at both levels of `context._two_level`, in `accessors._follow`'s row-space follow, and
at REGISTRATION, where `vary.central_universe` is folded into `member_of`.

Every anchor here needs `awkward` and `graphed.awkward`, so none can live in
`tests/frozen/frontend|numpy/m52` (a required free-threaded job collects those under
`pytest hypothesis numpy` alone). `scripts/run-tests.sh` runs this directory in its own process; the
helper still carries the `m52_` prefix because the pytest `pythonpath` publishes cross-dir helpers,
making a bare helper name a global name.

**No node-id INTEGER is pinned anywhere in this tree.** Node ids are fixture-dependent — the same
programme numbers differently under `from_record` and `from_awkward` — so every expectation is a
RELATION between two named candidate nodes (`graphed.member_of(m, "jes_up")` versus
`graphed.nominal(m)`), and each test first asserts that those two candidates ARE distinct nodes, so
the assertion can fail in the direction it guards.

**Freeze rule.** `tests/frozen/**` is read-only after the freeze tag: never edited, deleted,
`skip`ped, `xfail`ed or weakened, and `git diff m52-freeze -- tests/frozen/` must stay empty. A
frozen test that looks wrong is a Test Dispute at `.graphed/m52/disputes/<test_id>.md`.

The `points=` keyword and `graphed.points` are reached in test BODIES and inside fixture functions,
never at module import, so the suite COLLECTS against `origin/main`.

| anchor | design § | test |
|---|---|---|
| `vary-m52-C3` | §4.6/§8-d a joint label resolves to the single-axis container's SHIFTED member (and a one-at-a-time label still to nominal) | `test_projection_resolution.py::test_a_joint_label_resolves_to_the_shifted_member_not_the_nominal_one` |
| `vary-m52-C3` | §4.6 the fast path is kept — a carried label returns the container's own member by IDENTITY | `test_projection_resolution.py::test_the_fast_path_returns_the_containers_own_member_by_identity` |
| `vary-m52-C3` | §4.6/§8-e `_two_level` reaches the true inner cross member; the diagonal stays the diagonal | `test_two_level_projection.py::test_two_level_reaches_the_true_inner_cross_member` |
| `vary-m52-C3` | §4.4/§4.6 a shift ⊗ shift joint point keeps the inner `jes` universe through registration | `test_registration_projection.py::test_a_shift_shift_joint_point_keeps_the_inner_jes_universe_through_registration` |
| `vary-m52-C3` | §4.6/§7.1-2/§8-j a default-point registration keeps the member's own label when it carries it | `test_registration_projection.py::test_a_default_point_registration_keeps_the_members_own_label_when_it_carries_it` |
| `vary-m52-C3` | §4.6/§8-j the boundary half: a different tag, or another family, still reduces to nominal (+ the family-guard control) | `test_registration_projection.py::test_a_default_point_registration_still_reduces_a_different_tag_or_family_to_nominal` |
| `vary-m52-C3` | §4.6 row space — `reindex_to` follows a label by its POINT's own mask | `test_row_space_follow.py::test_reindex_to_follows_a_label_by_its_points_own_mask` |
| `vary-m52-C3` | §4.6 `_follow`'s `project` branch follows the same projection | `test_row_space_follow.py::test_the_project_branch_follows_the_same_projection` |
| `vary-m52-C3` | §4.7's frozen §2.4 union ORDER, unchanged where projection changes WHICH member | `test_union_order_determinism.py::test_the_union_order_is_unchanged` |
| `vary-m52-C3` | §5.3 the joint programme is byte-deterministic across two `PYTHONHASHSEED`s | `test_union_order_determinism.py::test_the_joint_program_is_byte_deterministic_across_two_runs` |

## Spellings this freeze pins

* **`points=`** as `{tag: {nuisance: coordinate}}` on the weight, loose and shift overloads, and
  **`graphed.points(obj)`** — the same surface `tests/frozen/frontend/m52` freezes.
* **`graphed.member_of(container, label)`** is the ONE resolution entry point. `vary.central_universe`
  is gone, so the registration anchors spell the reduction candidate `graphed.nominal(member)`,
  which is the same node today.
* **`context._two_level(container, label)`** keeps its `(container, label)` signature — §4.5's whole
  reason for the Session registry is that no narrowing call site changes signature.
* The four joint labels of §4.4's R1-c comprehension: `btag_jes{up,dn}_hf_{up,down}`, each naming
  the point `{"btag": "hf_<side>", "jes": "<up|down>"}`, with the four joint members the SAME
  expression objects as the two one-at-a-time members.

## Fixture families (`m52_projection_fixtures.py`)

* **`jes_shifted_context`** — `jes` as a lockstep shift of the Jet collection.
* **`btag_scale_factors`** — a pT-DEPENDENT per-event SF product and its two siblings, evaluated on
  the jets of whatever universe they are read from (R1-a propagation). The pT dependence is what
  makes the joint universe differ from the factorized product.
* **`joint_weight_program`** — §4.4's R1-c spelling verbatim, plus the containers the resolution is
  asked about: `jes_only` (one axis, so a joint point restricts onto it), `two_axis` (`jes` + `jer`,
  so a union has labels new to the second operand) and `factor_members` (the nested factor container
  `_two_level` is handed, each member itself `Varied` over `jes`).
* **`selection_program`** — a `jes`-varied event mask and a parent-read carrier holding one joint
  label, for the row-space follow. The two masks select a DIFFERENT number of rows, asserted in the
  test before it can conclude which one was applied.

## Non-vacuity (baseline: 9 failed / 1 passed against `origin/main`)

`test_a_default_point_registration_keeps_the_members_own_label_when_it_carries_it` fails on the
measured INVERTED relation, which is the shape the decomposition's TEST_SANITY table names: today
`graphed.universe(z, "jes_up")` is `graphed.nominal(other)` and not `graphed.member_of(other,
"jes_up")`, and the two are distinct nodes. Its boundary sibling
`test_a_default_point_registration_still_reduces_a_different_tag_or_family_to_nominal` is the one
green test — the half C3 must NOT change — and the two are the design's stated discriminating pair.

The other eight anchors register a MULTI-coordinate point, which no spelling reaches without the
`points=` keyword, so against `origin/main` they fail where the keyword falls through into `**tags`
(`a variation member must be an Array or a Varied, got dict`; in the shift form, `no field named
'points'`). That is a run-time failure for the feature-absent reason, not a collection or import
error, and it is recorded as a measured delta from the decomposition's §5 expected-baseline table in
`.graphed/m52/disputes/awkward_m52_baseline_reason.md`.
