Improvements
============

What ``graphed.checkpoint`` does not do yet, and what to do instead. :doc:`design` has the longer
version of each, with the reasoning.

Current limitations
-------------------

- **The store is a local directory.** ``Store`` writes to a filesystem path; there is no
  object-store or xrootd backend. Use a shared filesystem every worker can see, or give each node
  its own journal under one root with ``Store(root, node="A")`` — reading replays them all.

- **Recompute is sequential.** ``run_resumable`` processes missing partitions one at a time, in
  order. That is what makes the combine order fixed and the resumed answer bit-for-bit, but it
  means a large recompute is not faster than the work itself. Parallel recompute — a pool doing the
  processing while the final combine stays in order — is the next thing to land here; until then,
  run the analysis through ``graphed-executors`` and use the store for the resume boundary.

- **Everything is combined at the end.** All of a run's per-partition results are held and reduced
  once the last one is in. For a very wide fan-in that is a lot of memory; partial accumulators
  with backpressure would fix it.

- **Nothing prunes the store.** Results accumulate under the store root. Delete the directory when
  a set of results is stale.

- **Results are stored by convention, not self-description.** A result becomes bytes through a
  ``Codec`` — ``numpy.save`` for arrays, pinned-protocol pickle otherwise — and the store does not
  record which codec wrote a given blob. Read a store back with a different codec than you wrote it
  with and you get a decode error, not a helpful one.

- **``RetryElsewhere`` retries locally.** It builds a fresh ``resources`` object for the next
  attempt; it does not move the task to a different host.
