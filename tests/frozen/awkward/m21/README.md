# Frozen acceptance suite — M21 (graphed-awkward): the partitioned-source writer dispatch

P3.6 revision (user-confirmed plan, 2026-06-10): the GENERIC `to_parquet` writes any deferred
array partition-wise when its source data implements `graphed.write.PartitionedSource`.
Traceability:

| Test file | Verifies | Item |
|---|---|---|
| `test_partitioned_writer.py` | protocol dispatch with the efficiency WITNESSES (the source's whole-dataset loader never runs; exactly one read per partition tiling the dataset; the disabled plan reads nothing and its partitions are blind); the parquet `_DatasetLoader` rides the same protocol (one code path); the read list merges SYNTACTIC source-field accesses with buffer-projection refinement (offsets-only → one carrier leaf; a zip's replayed-but-untouched leg still read) | P3.6 rev |

The m15 suite remains authoritative for the parquet surface it pins — it must stay green
UNCHANGED through this generalization.
