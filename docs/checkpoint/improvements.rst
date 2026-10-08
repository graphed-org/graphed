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

- ``run_resumable`` **recomputes sequentially.** It processes missing partitions one at a time, in
  order. ``resumable(plan, store)`` resumes the same plan on any fixed-task runner in parallel.

- **Adaptive plans cannot resume.** ``resumable`` refuses a plan with ``next_tasks``, and stores no
  interior combine, so a resumed run recombines every partial.

- **A resumed task still costs store requests.** One journal read on the driver and one blob read
  on a worker per task; on an object store these are separate requests.

- **No lock between drivers.** Two drivers on one store each recompute what they do not see done,
  which is correct for deterministic tasks and duplicate work otherwise.

- **Some keys move when nothing changed.** Data ordered by set iteration, class-level state a task
  fills on a ``__main__`` class, and ``__main__`` numba kernels recompute after a restart; sort the
  data, keep the state on instances or in a module, keep the kernels in a module, or set ``salt``.
  State outside the process (environment variables, files beyond the partition, editable installs)
  is not in the key at all: ``salt`` covers it.

- **Nothing prunes the store.** Results and records accumulate under the store root. Delete the
  directory (or the URL's prefix) when a set of results is stale.

- **Results are stored by convention, not self-description.** A result becomes bytes through a
  ``Codec`` — ``numpy.save`` for arrays, pinned-protocol pickle otherwise — and the store does not
  record which codec wrote a given blob. Read a store back with a different codec than you wrote it
  with and you get a decode error, not a helpful one.

- **``RetryElsewhere`` retries locally.** It builds a fresh ``resources`` object for the next
  attempt; it does not move the task to a different host.
