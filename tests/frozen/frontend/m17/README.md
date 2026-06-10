# Frozen acceptance suite — M17 (graphed): record-subset getitem

dask-awkward parity plan, milestone M17 (`dask-awkward-parity-plan.md`). One addition to the
COMMON surface: `a[["x", "y"]]` records one fusible `fields` op (canonical comma-joined params;
equal subsets intern; order significant). Tuples, mixed lists, and empty selections stay refused.
The awkward structure-op tier itself is pinned in graphed-awkward's M17 suite.
