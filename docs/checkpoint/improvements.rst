Improvements
============

Tracked design improvements and known limitations for ``graphed.checkpoint`` (plan M0 requires this
file in every package).

Delivered
---------

- **Compile once, run on N datasets.** Record + optimize an analysis once
  (``graphed.Session.serialized_ir``), wrap it in a ``graphed.core.DurablePlan``, then re-target it
  at many datasets with ``DurablePlan.with_partitions`` / ``for_dataset`` / ``for_datasets`` (built
  from ``Dataset`` + ``partition_dataset``). The optimized interned IR is shared unchanged across
  datasets, and per-dataset content-addressed ``task_id``\\ s let a single ``Store`` checkpoint them
  all without collision. See ``tests/frozen/checkpoint/m8/test_deployment.py``.

Current limitations
-------------------

- **Local filesystem store only.** The M8 guardrail. A distributed / object-store backend is
  Phase-2; the ``Store`` interface (content-addressed ``put``/``get`` + append-only journal) is
  deliberately backend-shaped so a remote store can slot in later.
- **Sequential, resumable runner.** Resume correctness (skip-completed, no double-count, no lost
  partition, bit-for-bit) is independent of parallelism, so the runner reduces per-task partials in
  deterministic task order. Driving ``process`` through the M7 executors (thread/process pools) for
  parallel recompute is a straightforward, tracked extension — the Store contract is unchanged.
- **Codec-by-convention.** Per-task partials are stored via a ``Codec`` (pickle by default, a
  deterministic numpy ``.npy`` codec for arrays). A typed, self-describing partial format is a
  possible improvement.

Planned
-------

- Parallel recompute of missing tasks via a ``graphed_exec_local`` executor while preserving the
  deterministic final reduction order.
- ``retry_elsewhere`` becomes meaningful once a multi-worker / multi-host executor exists (Phase-2);
  today it re-runs on a fresh local worker context.
- Backpressure / partial-accumulator checkpoints for very large fan-ins (the current design
  recombines all per-task partials at the end, which is correct and simple but holds them in memory).
- This same content-addressed ``Store`` backs M9's preservation-bundle payload references.
