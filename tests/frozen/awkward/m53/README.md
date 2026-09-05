# awkward/m53 — dependency-driven fanout headline (traceability)

Milestone m53 (dependency-driven variation fanout). Authority:
`systematics-design/dependency-fanout-design.md` + the m53 plan §2. This tree is the headline
witness: a plain `graphed.vary` over members that DEPEND on another nuisance's varied nodes mints the
full grid automatically, and every joint resolves to the real cross node the graph already holds —
never the nominal one the pre-m53 collapse produced.

Run: `python -m pytest tests/frozen/awkward/m53 -q` (its own process, per the awkward per-milestone
split). The `m53_` helper prefix is load-bearing under prepend import mode. Every m53-new behavior
(the auto-fanout, `composes_as_union=`, `points=`, `max_universes=`) is reached only inside test
bodies, so the tree COLLECTS against a tree with no m53 implementation and fails at RUN time.

Fixture — `m53_fanout_fixtures.py`:
* `fanout_weight(**kw)` — a `jes` shift, the region selection, then a plain weight `vary` over four
  b-tag SF members (`hf_*`/`lf_*`, distinct pT dependence) computed on the jes-varied jets → the
  full jes(3) x btag(5) = 15 grid. Extra keywords pass through to `graphed.vary`.
* `independent_program()` — `jes` x `jer` with the `jer` members built from NOMINAL pt → INDEPENDENT
  → the union, no fanout (the bound m53 preserves).
* `GRID_POINTS`/`JOINT_LABELS`/`UNION_LABELS`/`JOINT_SOURCES` — the expected 15-universe map, the 8
  machine joints, the pre-m53 union, and each joint's (SF source, inner jes universe).

| Work item (plan §) | Test |
|---|---|
| §2 the default is the full grid of fifteen universes | `test_auto_fanout.py::test_the_default_is_the_full_grid_of_fifteen_universes` |
| §2 every joint resolves to the real cross node, not nominal | `test_auto_fanout.py::test_every_joint_resolves_to_the_real_cross_node_not_nominal` |
| §2 the fifteen universes materialize distinctly | `test_auto_fanout.py::test_the_fifteen_universes_materialize_distinctly` |
| §1 an independent family stays the union (positive control) | `test_independent_control.py::test_independent_jes_and_jer_stay_the_union_of_five` |
| §2 the joints follow the one-at-a-time labels, deterministically | `test_joint_order.py::test_joints_follow_all_one_at_a_time_labels_deterministically` |

Byte-determinism across two FRESH interpreters is the re-authored
`awkward/m52::test_the_joint_program_is_byte_deterministic_across_two_runs`; `test_joint_order` here
pins the human-readable ordering RULE (every joint after every one-at-a-time label) that a hash
cannot see.

## Non-vacuity (fails on the current union-collapse tree for the right reason)
- fanout / order / distinctness → `assert 7 == 15` (union, not the grid) and `assert 0 == 8` (no
  joint minted); `member_of(weight, "<joint>")` returns nominal, so the cross-node relation fails.
- `test_independent_jes_and_jer_stay_the_union_of_five` PASSES on the current tree too — the
  live-harness positive control: independent families are never fanned out, then or now.
</content>
