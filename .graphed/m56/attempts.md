# m56 — fan-out composition decided on nodes: implementer attempts

Plan: `graphed-workdir/both-kind-fanout-plan.md` (r4, review-clean after two delta rounds and one whole-artifact pass).

## Iteration 1 (2026-09-08) — d9d256b

- Rule: `AmbientCarrier.resolve(label)` = per lineage factor, nodes of `_two_level(factor, label)` less nodes of its nominal; `_reads_ambient` = that set against the member's input cone; `_foreign` consults it only for a coordinate whose nuisance is in the ambient tag map. `_lineage_factors` walks the ancestry; `_member_nodes` defined once in `vary.py`.
- Prototype history: v1 (literal member keys) leaked joints at joint labels; v2 (`_two_level` over lineage factors) dropped joints when a label-invariant factor (seed weight, unrelated family's central) sat in the member's cone; v3 (minus-nominal narrowing) is the shipped rule.
- Gates on this commit: `scripts/run-tests.sh` exit 0 (66 sections with the frozen m56 tree); ruff/format/mypy strict clean; Sphinx `-W` exit 0; docs example executed (four joints, two-level weight rebuilt by hand).
- m48 dispute: `.graphed/m56/disputes/m48-test_vary_stacking-factor_read_at_the_parent.md` — the row-space control mapped joint labels to nominal rows; re-frozen with the owner's affirmation (452a24a).
