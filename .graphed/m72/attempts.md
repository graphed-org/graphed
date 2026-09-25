# m72 — implementer iterations

Multiout lane (plans in `graphed-workdir/lanes/multiout/`: `plan.md`, `plan-A.md`). Frozen suite
`freeze-m72` = `d2b48e4` (`tests/frozen/frontend/m72`, `tests/frozen/awkward/m72`).

Run: `python -m pytest tests/frozen/frontend/m72 -q -p no:cacheprovider` and the same for
`tests/frozen/awkward/m72`.

## G1 — deferred part writes in `aggregate_plan` (frontend/m72 writes 7/7, awkward/m72 test 5)

`PartWrite` in `write.py`; `aggregate_plan(writes=)` compiles `outputs + write arrays + metadata
Arrays` into one IR, slots every array through `correspondence.node_map`, and ships each write as
`(codec, destination, name, slot, kv)` with `n_values` = the outputs' distinct slot count.
`refuse_chunk_partials(as_outputs=)` takes a collection of compiled ids (the write roots). The
duplicate-part check runs over every (write, task) at build, after the tasks exist.

## G2 — `graphed.awkward.parquet_write` (awkward/m72 tests 1, 2, 6; awkward/m48, m51 green)

`_ArrowParquet(column, arrow_options, parquet_options)` codec: `to_arrow_table(**arrow)`, KV
replaced when given, `pq.write_table(**parquet)`. `_payload` is the wrap rule `_WritePart` now shares.
Registered `"refusing"` in `VERB_DISPOSITIONS`.

## G3 — `gak.num(axis=0)` records a reduction (awkward/m72 test 4)

`record_op(..., reduction=axis == 0)`. No other use in `tests python docs` (the grep's only hit is
the m72 test itself).

## G4 — `graphed.collate` (frontend/m72 tests 8–12, awkward/m72 test 3; all 18 m72 green)

`_Collated(processes, route)` routes by `(uri, tree)` (the route is O(files), in the broadcast
process); `_CollatedCombine` folds per name and returns names in mapping order whatever the tree;
`empty=dict`. Tasks re-keyed `0..N-1` over each plan's key order in mapping order.

## G5 — docs

`docs/frontend/design.rst` (writes in "One pass…", new "Several graphs in one plan", the "Output
groups" bullet replaced by the durable-collate limit), `improvements.rst` (blind part naming),
`docs/awkward/design.rst`, `architecture.rst`, `api.rst`, `changelog.rst` (0.0.7 unreleased). Both
new examples executed with `run_rst_blocks.py` (the 3 frontend and 1 awkward FAILs are the
pre-existing fragment blocks, same on the base); `sphinx-build -W` clean.
