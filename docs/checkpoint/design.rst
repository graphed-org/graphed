How graphed-checkpoint works
============================

``graphed-checkpoint`` is the durability layer for long runs: a content-addressed ``Store`` of
completed work, a ``run_resumable`` driver that skips anything the store already holds, retry
policies for the failures worth retrying, and a dead-letter path for the ones that are not. The
design goal is blunt: **a crashed eight-hour run resumes losing only the partition that was in
flight — and the resumed result is bit-for-bit the uninterrupted one.**

.. contents::
   :local:
   :depth: 2


Content addressing is the whole trick
-------------------------------------

Every piece of work is keyed by *what it computes*, never by when or where it ran. A
``DurablePlan`` carries the analysis IR (the canonical serialized bytes), the partition set,
and its process/combine/empty operations as import-referenced ``OpSpec``\ s; its
``task_id(partition)`` is a SHA-256 over the IR identity, the process spec, and the partition.
Two consequences:

* **Resume is a lookup, not a protocol.** On a fresh ``run_resumable``, every task id is
  missing from the store, so everything executes and each partial is committed as it
  completes. After a crash, the same plan produces the same task ids; whatever was committed
  is loaded instead of recomputed. There is no run manifest to reconcile, no "resume token" —
  the store *is* the state.
* **Cache poisoning is structurally hard.** Change the analysis (different IR bytes), the
  processing function (different OpSpec), or the partition — and the key changes. A stale
  store can never satisfy a different computation; at worst it is ignored.

The ``Store`` itself is a local-filesystem content-addressed blob store with atomic writes
(write-temp-then-rename — a torn write is invisible) and an append-only journal of
``JournalEntry`` records, so "what happened" survives alongside "what was computed". Values
serialize through a pluggable ``Codec`` (``NumpyCodec`` for array partials, ``PickleCodec`` as
the general fallback).

A worked resume (runnable)
--------------------------

The processing functions live in an importable module — that is what lets a plan run on a
machine with no analysis source files, and it is the same convention every graphed writer
follows. Say ``myanalysis.py`` contains::

    import numpy as np

    def hist_chunk(partition, resources):
        rng = np.random.default_rng(partition.entry_start)   # deterministic per partition
        n = partition.entry_stop - partition.entry_start
        return np.histogram(rng.uniform(0, 1, n), bins=4, range=(0, 1))[0]

    def hist_add(a, b): return a + b
    def hist_empty():   return np.zeros(4, dtype=np.int64)

Then::

    from graphed.core import DurablePlan, OpSpec, Partition
    from graphed.checkpoint import Store, run_resumable

    plan = DurablePlan(
        ir=analysis_ir_bytes,                       # session.serialized_ir(out) in real use
        process=OpSpec.from_ref("myanalysis:hist_chunk"),
        combine=OpSpec.from_ref("myanalysis:hist_add"),
        empty=OpSpec.from_ref("myanalysis:hist_empty"),
        partitions=tuple(Partition("toy", "", i*1000, (i+1)*1000) for i in range(6)),
        read_columns=("x",),
    )

    store = Store("checkpoints/")
    res   = run_resumable(plan, store)
    res.report.executed                  # 6 — everything ran, six partials committed

    again = run_resumable(plan, store)   # the same plan, the same store
    again.report.executed                # 0
    again.report.skipped                 # 6 — the run is a pure lookup
    # and again.value == res.value, exactly

The crash case is the same mechanics: the test suite simulates a kill after *k* committed
partitions and asserts the resumed run executes exactly ``n - k``, reports
``did_less_work``, and reproduces the uninterrupted result bit-for-bit. Resumability holds for
any ``k`` because commit is per-partition and atomic — there is no checkpoint "interval" to
tune and no window where completed work is lost.

Retry policies: failures worth distinguishing
---------------------------------------------

Not all failures are alike, and the policy objects encode the differences rather than a single
``retries=N`` knob:

* ``RetryN(n)`` — the transient hiccup: re-run the same partition up to *n* times. Worker
  state (the ``resources`` mapping) survives across attempts, which is also how tests build
  deliberately flaky processes.
* ``RetrySmallerChunk`` — the partition that is *too big* (an OOM): split it and retry the
  halves, recursively, until pieces succeed. The combine tree absorbs the extra partials
  naturally because combine is associative.
* ``RetryElsewhere`` — the *bad worker*: retry on a different worker context rather than
  burning attempts on a poisoned one.

A failure that exhausts its policy does not poison the run: with an ``error_budget`` set, the
partition is **quarantined** — recorded as a dead letter with
``dead_letter_descriptor`` (the partition, the attempts, and the underlying error, which for
graphed analyses is a source-mapped ``StageError`` pointing at the user's line) — and the run
completes on everything else. The harvest is a reproducible to-do list, not a stack of logs:
each descriptor is sufficient to re-run exactly the failing slice after the fix.

Running without source files
----------------------------

Because process/combine/empty are import refs and the IR is canonical bytes, a serialized plan
(``plan.to_bytes()``) executes in a fresh interpreter with *only installed packages* — the
frozen suite literally scrubs the environment and runs from a subprocess with no analysis
files present. Only a genuinely opaque callable falls back to embedded cloudpickle, and it is
flagged ``opaque=True`` so the preservation layer can surface the risk.


Phase 2 (deliberately not built)
--------------------------------

* **Remote/object stores.** The ``Store`` is local-filesystem by scope; S3/xrootd-backed
  stores are the obvious extension once a distributed executor exists.
* **Store garbage collection / retention policies.** Content-addressed blobs accumulate;
  pruning by reachability from live plans is designed but not built.
* **Cross-plan sharing.** Two plans that share sub-computations currently share nothing; task
  ids are per-(plan, partition). Finer-grained (per-stage) addressing is a Phase-2 study.

See :doc:`improvements` for the live tracked list.
