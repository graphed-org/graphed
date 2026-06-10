# Frozen acceptance suite — M13 (graphed-numpy): manipulation + indexing parity

dask.array user-facing parity, tier P2 (`dask-array-parity-plan.md` in the superproject).
Traceability:

| Test file | Verifies | Plan item |
|---|---|---|
| `test_indexing.py` | axis-0 slices/ints + boolean/fancy getitem evaluate exactly as numpy; tuple subscripts (`a[:, inner…]`) are partition-local fusible ops with interning; indexing the partitioned axis inside a tuple is refused | P2.6 |
| `test_manipulation.py` | reshape (leading dim must be −1) / ravel / squeeze (explicit inner axis) / expand_dims / swapaxes / transpose / `.T` / astype / clip / round / take / where (incl. scalar branches) / diff / isin / searchsorted — all numpy-exact, with the axis-0 geometry rules pinned as record-time refusals; stack/vstack refused (inner unknown dim ⇒ Phase 2 N-D chunks) | P2.6 |
| `test_combination_and_aggregates.py` | concatenate: axis 0 = boundary node (restructures the partition set), inner axes fusible; hstack interns with concatenate; unique/bincount/histogram/histogram2d/histogramdd as boundary reductions matching numpy; histogram keeps the (counts, edges) contract with record-time-concrete edges and requires `range`; determinism over manipulation programs | P2.7 |

Reference semantics are numpy's own throughout. Partition-locality caveats (scans M12; fancy
getitem, diff here) are deliberate MVP semantics — cross-partition variants are Phase 2.
