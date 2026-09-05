# Test dispute — `awkward/m52`'s expected-baseline reason cannot hold for eight of nine anchors

Filed by `vary-m52-ta-graphed` (test-author) during TEST_AUTHORING. **Not a blocker**: the tree is
authored in full and frozen as specified. This records a measured correction to a TEST_SANITY
criterion, so the reviewer does not read the tree's baseline as vacuous.

## The clause disputed

`m52-decomposition.md` §5-(2):

> For `awkward/m52` and `graphed-histogram/m52` specifically, the baseline failures must be
> **assertion** failures on the node-id RELATIONS and values measured in §3.2/§3.3, not errors.

and the §5-(1) table row for `awkward/m52`, which names 3.1, 3.3 and 3.5 as inverted-relation
failures.

## Why it cannot hold

Rows 3.1, 3.3, 3.4, 3.7, 3.8 and 3.9 all state a property of a label whose point names **two**
nuisances. A multi-coordinate point is registrable only through the `points=` keyword (design §4.4,
§4.5: labels and their points are minted together in `vary.gather_members`), and §4.5 keeps
`graphed.varied.rebuild`'s hand-built labels on exact-match resolution, so a hand-built joint label
is *unreachable by projection* after C3 as well as before it. There is therefore no spelling of
those six properties that runs against a tree with no `points=` keyword.

Measured, against this worktree (`origin/main`, no m52 implementation):

```
$ cd /Users/lgray/vibe-coding/m52/graphed
$ .venv/bin/python -c "
import graphed
from graphed import Session
from graphed.numpy import NumpyBackend, from_record
import numpy as np
s = Session(NumpyBackend()); x = from_record(s,'ev',pt=np.arange(1.,13.))['pt']
graphed.vary(x, 'jes', variations={'up': x*2}, points={'up': {'jes': 1}})"
graphed.errors.GraphedError: a variation member must be an Array or a Varied, got dict
```

`points=` falls through into `**tags` and is rejected as a variation member (and, in the shift form,
as a missing collection: `no field named 'points'`). That is a RUN-time failure for the
feature-absent reason — the criterion §5-(1) states for `frontend/m52` — but it is an error, not an
assertion failure.

Rows 3.5 and 3.6 need no `points=` (both are default-point registrations) and DO behave as §5-(2)
requires: 3.5 fails on the inverted relation, 3.6 is green and must stay green.

## Baseline actually measured (9 failed / 1 passed)

| test | baseline reason |
|---|---|
| `test_a_joint_label_resolves_to_the_shifted_member_not_the_nominal_one` | `GraphedError: a variation member must be an Array or a Varied, got dict` |
| `test_the_fast_path_returns_the_containers_own_member_by_identity` | same (shares the joint fixture) |
| `test_two_level_reaches_the_true_inner_cross_member` | same |
| `test_a_shift_shift_joint_point_keeps_the_inner_jes_universe_through_registration` | `GraphedTypeError: ill-typed op 'field' ... no field named 'points'` (shift form) |
| `test_a_default_point_registration_keeps_the_members_own_label_when_it_carries_it` | **assertion**: `universe(z,"jes_up").node_id == nominal(other).node_id != member_of(other,"jes_up").node_id` |
| `test_a_default_point_registration_still_reduces_a_different_tag_or_family_to_nominal` | **passes** — the half C3 must not change |
| `test_reindex_to_follows_a_label_by_its_points_own_mask` | `GraphedError: a variation member must be an Array or a Varied, got dict` |
| `test_the_project_branch_follows_the_same_projection` | same |
| `test_the_union_order_is_unchanged` | same |
| `test_the_joint_program_is_byte_deterministic_across_two_runs` | same, surfaced through the child process's non-zero exit |

## Proposed correction

Replace §5-(2)'s clause for `awkward/m52` with the §5-(1) table's own weaker and satisfiable
wording — *not an import or collection error* — and restrict the assertion-failure requirement to
the two anchors that can meet it (3.5, and 3.6 which is green by design). The discriminating power
of the six `points=` anchors is established instead by the measured premise each one asserts before
its conclusion: that its two candidate nodes are distinct today. Those premises are recorded in
`tests/frozen/awkward/m52/README.md` and were regenerated with the probes cited there.

No frozen test changes under this correction.

## ADJUDICATION (owner, 2026-09-04) — ACCEPTED
Correct and measured (confirmed independently by all three test-authors). No frozen test changes.
Folded into `m52-decomposition.md` §5: the `points=`-fixture anchors fail as `GraphedError`
(feature-absent, run-time, non-vacuous) at true `origin/main`; the assertion-failure criterion is
restricted to the default-point anchors (3.5 fail / 3.6 pass); discriminating power of the six
`points=` anchors rests on the distinct-candidate-nodes premise recorded in the tree README.
