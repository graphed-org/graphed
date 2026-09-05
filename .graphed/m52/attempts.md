# m52 graphed implementer — iteration log

Freeze: `m52-freeze`. Scope C0–C7 (graphed: C0,C1,C2,C3,C4,C6-corpus-fixture consumer,C7).
`git diff m52-freeze -- tests/frozen/` MUST stay empty for the life of the milestone.

## Iteration 0 — baseline (origin/main, no m52 implementation)
- frontend/m52: 43 failed / 2 passed (the 2 green are the §8-b partial-coverage + m50 kind-guard
  backward-compat positive controls).
- awkward/m52:  9 failed / 1 passed (the green is §8-j's boundary half, `test_a_default_point_
  registration_still_reduces_a_different_tag_or_family_to_nominal` — C3 must NOT change it).
- corpus/m52:   3 failed / 60 passed (corpus/m05's 60 untouched and green).
Baseline failure reasons match the (adjudicated) decomposition §5 table: `points=`-fixture anchors
raise `GraphedError: ...got dict` at origin/main (feature absent); C1-row anchors raise
`ModuleNotFoundError: graphed._points`; introspection rows `AttributeError: no 'points'`; the two
default-point awkward anchors 3.5 (fail on inverted relation) / 3.6 (pass).
