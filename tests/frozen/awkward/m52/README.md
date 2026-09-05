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
| `vary-m53` | §2 a shift ⊗ shift joint is auto-fanned and reaches the inner `jes` universe (the one-at-a-time stays the diagonal) | `test_registration_projection.py::test_a_shift_shift_joint_is_auto_fanned_and_reaches_the_inner_jes_universe` |
| `vary-m52-C3` (survivor) | §4.6/§7.1-2/§8-j a default-point registration keeps the member's own label when it carries it | `test_registration_projection.py::test_a_default_point_registration_keeps_the_members_own_label_when_it_carries_it` |
| `vary-m53` | §2 same family still reduces to nominal, but a FOREIGN one fans out the joint alongside the diagonal (+ the family-guard control) | `test_registration_projection.py::test_a_default_point_registration_reduces_same_family_but_fans_out_a_foreign_one` |
| `vary-m52-C3` | §4.6 row space — `reindex_to` follows a label by its POINT's own mask | `test_row_space_follow.py::test_reindex_to_follows_a_label_by_its_points_own_mask` |
| `vary-m52-C3` | §4.6 `_follow`'s `project` branch follows the same projection | `test_row_space_follow.py::test_the_project_branch_follows_the_same_projection` |
| `vary-m52-C3` | §4.7's frozen §2.4 union ORDER, unchanged where projection changes WHICH member | `test_union_order_determinism.py::test_the_union_order_is_unchanged` |
| `vary-m52-C3` | §5.3 the joint programme is byte-deterministic across two `PYTHONHASHSEED`s | `test_union_order_determinism.py::test_the_joint_program_is_byte_deterministic_across_two_runs` |

## Spellings this freeze pins

* Re-authored for m53: the joints are AUTO-FANNED from dependent members — no `points=` in this tree.
  **`graphed.points(obj)`** reports each label's point, the same surface `tests/frozen/frontend/m52`
  freezes.
* **`graphed.member_of(container, label)`** is the ONE resolution entry point. `vary.central_universe`
  is gone, so the registration anchors spell the reduction candidate `graphed.nominal(member)`,
  which is the same node today.
* **`context._two_level(container, label)`** keeps its `(container, label)` signature — §4.5's whole
  reason for the Session registry is that no narrowing call site changes signature.
* The four machine-minted joint labels: `btag_hf_{up,down}__jes_{up,down}` (`f"{name}_{tag}__{fl}"`),
  each naming the point `{"btag": "hf_<side>", "jes": "<up|down>"}`, with each joint member the SAME
  expression object as its one-at-a-time sibling.

## Fixture families (`m52_projection_fixtures.py`)

* **`jes_shifted_context`** — `jes` as a lockstep shift of the Jet collection.
* **`btag_scale_factors`** — a pT-DEPENDENT per-event SF product and its two siblings, evaluated on
  the jets of whatever universe they are read from (R1-a propagation). The pT dependence is what
  makes the joint universe differ from the factorized product.
* **`joint_weight_program`** — the m53 auto-fanout spelling (a plain weight `vary` over two
  jes-dependent SF members), plus the containers the resolution is asked about: `jes_only` (one axis,
  so a joint point restricts onto it), `two_axis` (`jes` + `jer`, so a union has labels new to the
  second operand) and `factor_members` (the nested factor container `_two_level` is handed, each
  member itself `Varied` over `jes`).
* **`selection_program`** — a `jes`-varied event mask and a parent-read carrier holding one joint
  label, for the row-space follow. The two masks select a DIFFERENT number of rows, asserted in the
  test before it can conclude which one was applied.

## Non-vacuity (re-authored for m53: fails on the current union-collapse tree for the right reason)

The two same-family survivors — `test_the_fast_path_returns_the_containers_own_member_by_identity`
and `test_a_default_point_registration_keeps_the_members_own_label_when_it_carries_it` — PASS on the
current tree: they pin laws m53 preserves, and their passing is the live-harness control.

Every re-authored anchor drives a machine-minted joint (`btag_hf_up__jes_up`, `jer_up__jes_up`,
`jes_up__jer_up`), which the current tree never mints — a dependent member collapses to the union, so
`member_of(container, "<joint>")` silently returns the NOMINAL member. The projection anchors fail on
the measured relation (the joint node ≠ the shifted/cross node; e.g. `member_of(second, JOINT) == 50`
nominal, not `51` jes_up); `test_a_default_point_registration_reduces_same_family_but_fans_out_a_
foreign_one` keeps its same-family assertion green and fails only on the newly-minted foreign joint.
`test_the_joint_program_is_byte_deterministic_across_two_runs` asserts a determinism law m53
preserves and passes on both trees. Feature-absent, at run time, past import.
