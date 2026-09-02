# m49 frozen suite — `graphed` / `tests/frozen/core` (raw `GraphStore`, awkward-free)

Milestone m49 (shift path + impact + executor end-to-end). Authority: `systematics-vary-plan.md`
§3.3, whose §10/m49 partition pins this tree to the raw-`GraphStore` scaling benchmark **only** —
every other m49 `graphed` anchor lives in `frontend/m49`, `awkward/m49`, `debug/m49` or
`checkpoint/m49`. The m4 files are frozen and untouched (§B.6); this replicates their pattern in a
new file.

`tests/frozen/core` is collected WHOLE in one process (`scripts/run-tests.sh`; `core` is not a
`SPLIT_PKG`) and by the required free-threaded CI job under `pytest hypothesis numpy` alone, so no
module here may import `graphed.awkward`, `pyarrow`, `hist` or `pandas`, directly or transitively.
This tree imports `graphed.core` and the stdlib and nothing else — not even numpy. It ships no
helper module, so §1's `m49_` helper-basename rule has no subject here.

| anchor | plan clause | test |
|---|---|---|
| `vary-m49-core-shape` | §3.3 exact reduced shape of the variation topology (`stages == N + 1`, `reduced_nodes == 2N + 2`) | `test_variation_benchmark.py::test_variation_topology_reduces_to_the_pinned_shape` |
| `vary-m49-core-shape` | §3.3 bound builder (per-universe terminating reduction, separately marked) | same test — the `Counter` over the reduced store's node kinds and the marked-output count |
| `vary-m49-core-growth` | §3.3 linear-growth bound `time(128)/time(16) < 16.0`, m4 noise floor + best-of-N | `test_variation_benchmark.py::test_reduction_time_grows_linearly_in_the_universe_count` |

## Why the shape assertion is read off the reduced store

The report's `stages`/`reduced_nodes` are the pinned integers; the `Counter` over
`reduced.nodes()` and `len(reduced.outputs())` say what those integers are made of, so a reducer
that hit the right totals with the wrong composition is still red. `reduced_nodes == 2N + 2` is
the with-reduction shape: dropping the terminating reduction makes the correct answer `N + 2`, and
funnelling the universes into one output (m4's `_systematics` shape) makes both assertions
vacuous. Both mutants are red against this file.

## Scope

This is the ONE frozen wall-clock gate in m48–m51 (§3.3), a named carve-out discharged by the
project plan's M4 benchmark mandate and its frozen precedent `core/m4/test_benchmark.py`. It is a
guard, not a witness of new m49 surface: §3.1 binds m49's record→reduced correspondence to change
nothing about what the reducer produces, so this file is green before and after — its job is to go
red if the composed re-indexing changes the reduced shape or costs super-linear time. `reduce()` is
the measured path, per §3.3's replication of `test_benchmark.py`.
