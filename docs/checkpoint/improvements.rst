Improvements
============

What ``graphed.checkpoint`` does not do yet, and what to do instead. :doc:`design` has the longer
version of each, with the reasoning.

Current limitations
-------------------

- **A store at a URL keeps one object per record.** ``FsspecStore`` never compacts its journal,
  so a store that has recorded many tasks takes one listing plus one read per record to resume.
  Records written by different store instances replay in the order of the instances' creation
  times, to the resolution of the clock.

- **``memory://``, ``file://`` and ``s3://`` are the schemes that have been tried**, S3 against a
  local S3 stand-in rather than a real bucket. Any other fsspec scheme (``root://``,
  ``https://``, ...) is a URL plus storage options and should work, but has not been tried.

- **Recompute is sequential.** ``run_resumable`` processes missing partitions one at a time, in
  order, so a large recompute is not faster than the work itself. The ``graphed-executors``
  runners run a plan in parallel but do not skip work a store already holds: pick
  ``run_resumable`` when surviving a crash matters more than wall time, and a parallel runner
  when it does not.

- **Everything is combined at the end.** All of a run's per-partition results are held and reduced
  once the last one is in. For a very wide fan-in that is a lot of memory; partial accumulators
  with backpressure would fix it.

- **Nothing prunes the store.** Results and records accumulate under the store root. Delete the
  directory (or the URL's prefix) when a set of results is stale.

- **Results are stored by convention, not self-description.** A result becomes bytes through a
  ``Codec`` — ``numpy.save`` for arrays, pinned-protocol pickle otherwise — and the store does not
  record which codec wrote a given blob. Read a store back with a different codec than you wrote it
  with and you get a decode error, not a helpful one.

- **``RetryElsewhere`` retries locally.** It builds a fresh ``resources`` object for the next
  attempt; it does not move the task to a different host.
