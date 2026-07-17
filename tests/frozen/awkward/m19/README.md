# Frozen acceptance suite — M19 (graphed-awkward): conveniences

dask-awkward parity plan, milestone M19 (P3.8, the last MVP tier). `fields`/`type_of`/
`backend_of` answer from the recorded form/session without recording (node-count witnessed);
`head`/`sample` are eager peeking sugar over the common slice op + reference materialize (a
partitioned head is executor territory, not MVP). Reuses `m16_helpers`.
