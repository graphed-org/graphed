How graphed.checkpoint works
============================

A batch job dies on partition 3,148 of 5,000. What do you have to redo? With a checkpoint
interval, everything since the last checkpoint, and you had to pick that interval in advance.
Here, one partition — and the answer you finally get is bit-for-bit the answer the run would
have produced if nothing had gone wrong.

This page is the argument for why that holds, and how to use it.

.. contents::
   :local:
   :depth: 2


Why re-running is a lookup
--------------------------

Work is filed by *what it computes*, never by when or where it ran. The key for one unit of work
is a SHA-256 over three things: the recorded analysis, the processing function, and the partition.
That is the whole of :meth:`~graphed.core.DurablePlan.task_id`, and two properties fall out of it.

**Resume needs no protocol.** On a fresh run every key is missing from the store, so everything
executes and each result is written as it completes. After a crash, the same plan produces the
same keys; whatever was written is loaded instead of recomputed. There is no run manifest to
reconcile and no resume token to keep — the store *is* the state, and a run you started on
Tuesday can be finished from a different shell on Thursday.

**A stale result cannot poison a new run.** Change the analysis, change the processing function,
or change the partition, and the key changes with it. Two different computations cannot land on
one key, so at worst an old store is ignored.

The rest is bookkeeping designed not to lie to you after a crash. ``Store`` writes each result to
a temp file in the same directory, ``fsync``\ s it, and renames it into place, so a torn write is
never visible; the name of the file *is* the hash of its contents, so writing the same result
twice is a no-op. Alongside the results is an append-only journal, one ``fsync``\ ed JSON line per
completed unit. Replaying it is how a resumed run learns what is done. If the process died
halfway through appending a line, that line is unparseable and is skipped — a half-written journal
degrades to slightly less recovery, never to a corrupt one. And a journal line whose result file
is missing (the line won the race, the file lost it) is not honoured either.

Values become bytes through a ``Codec``: ``NumpyCodec`` for array results — ``numpy.save``\ 's
fixed layout, so the same array always hashes the same — and ``PickleCodec``, protocol pinned, for
everything else.


A run that actually dies
------------------------

Nothing above depends on a graceful shutdown, so here is an ungraceful one. ``killdemo.py`` is the
same per-partition histogram as before, with a hard exit wired into the fourth partition:

.. code-block:: python

    import os

    import numpy as np

    def hist_chunk(partition, resources):
        if os.environ.get("DEMO_KILL") == partition.uri and partition.entry_start == 3000:
            os._exit(137)  # a hard kill mid-run: no unwinding, no cleanup, nothing flushed
        rng = np.random.default_rng(partition.entry_start)
        n = partition.entry_stop - partition.entry_start
        return np.histogram(rng.uniform(0, 1, n), bins=4, range=(0, 1))[0]

    def hist_add(a, b):
        return a + b

    def hist_empty():
        return np.zeros(4, dtype=np.int64)

The driver runs it in a child process, waits for the corpse, then resumes in the parent. The
``if __name__ == "__main__":`` guard is required — a spawned process re-imports this module.

.. code-block:: python

    import multiprocessing as mp
    import os

    import numpy as np

    from graphed import Session
    from graphed.checkpoint import Store, run_resumable
    from graphed.core import DurablePlan, OpSpec, Partition
    from graphed.numpy import NumpyBackend, from_record

    s = Session(NumpyBackend())
    ev = from_record(s, "events", x=np.zeros(1))
    counts, _edges = np.histogram(ev["x"], bins=4, range=(0, 1))

    plan = DurablePlan(
        ir=s.serialized_ir(counts),
        process=OpSpec.from_ref("killdemo:hist_chunk"),
        combine=OpSpec.from_ref("killdemo:hist_add"),
        empty=OpSpec.from_ref("killdemo:hist_empty"),
        partitions=tuple(Partition("toy", "Events", i * 1000, (i + 1) * 1000) for i in range(6)),
    )


    def doomed_run():
        os.environ["DEMO_KILL"] = "toy"          # kill the worker on the 4th partition
        run_resumable(plan, Store("checkpoints/"))


    if __name__ == "__main__":
        p = mp.get_context("spawn").Process(target=doomed_run)
        p.start()
        p.join()
        print("the run died with exit code", p.exitcode)

        resumed = run_resumable(plan, Store("checkpoints/"))
        print("executed:", resumed.report.executed, " skipped:", resumed.report.skipped)
        print("result:", resumed.value)

::

    the run died with exit code 137
    executed: 3  skipped: 3
    result: [1499 1493 1518 1490]

Three partitions survived the kill, three were redone, and ``[1499 1493 1518 1490]`` is the same
array the uninterrupted run printed on the previous page.

The reason it is the same array and not merely a correct one: what gets stored is each partition's
own result, never a running total. The final combine walks those results in partition order, so
every partition contributes exactly once and the additions happen in the same sequence whether or
not anything crashed. That matters because floating-point addition is not associative — a running
accumulator resumed from a different point would give you a different last digit. It also means
there is no checkpoint interval to tune: a commit happens per partition, so the loss from a kill
at any moment is the one partition in flight.

``resumed.report`` is a ``ResumeReport``: ``executed``, ``skipped``, ``dead``, ``stopped``,
``dead_letters``, and ``did_less_work`` (true when anything was skipped). It is the receipt for a
resume actually having resumed.


What invalidates a cached result
--------------------------------

Everything the key is built from, and nothing else. Recording the analysis a second time gives the
same key; re-binning the histogram does not. (``myanalysis`` here is the module from
:doc:`index`.)

.. code-block:: python

    import numpy as np

    from graphed import Session
    from graphed.core import DurablePlan, OpSpec, Partition
    from graphed.numpy import NumpyBackend, from_record


    def analysis_ir(bins):
        s = Session(NumpyBackend())
        ev = from_record(s, "events", x=np.zeros(1))
        counts, _edges = np.histogram(ev["x"], bins=bins, range=(0, 1))
        return s.serialized_ir(counts)


    def make_plan(ir, process):
        return DurablePlan(
            ir=ir,
            process=process,
            combine=OpSpec.from_ref("myanalysis:hist_add"),
            empty=OpSpec.from_ref("myanalysis:hist_empty"),
            partitions=(Partition("toy", "Events", 0, 1000),),
        )


    part = Partition("toy", "Events", 0, 1000)
    chunk = OpSpec.from_ref("myanalysis:hist_chunk")
    four = analysis_ir(4)

    base = make_plan(four, chunk).task_id(part)
    print("same analysis again:  ", make_plan(analysis_ir(4), chunk).task_id(part) == base)
    print("re-binned to 8:       ", make_plan(analysis_ir(8), chunk).task_id(part) == base)
    print("different partition:  ", make_plan(four, chunk).task_id(Partition("toy", "Events", 0, 500)) == base)

    carried = OpSpec.from_callable(lambda partition, resources: 0)
    print("carried by value:     ", carried.opaque)

::

    same analysis again:   True
    re-binned to 8:        False
    different partition:   False
    carried by value:      True

The last line is the one that will bite you. ``OpSpec`` has two forms. ``OpSpec.from_ref("mypkg.jets:select")``
records an import path, so the key depends on the *name* of your function — edit its body and the
cached results survive, which is what you want when you are fixing a log message and not what you
want when you are fixing physics. ``OpSpec.from_callable`` falls back to embedding the function's
bytes when it is not importable (a lambda, a closure, something defined in ``__main__``); those
plans report ``opaque`` and their keys change every time the function changes. Prefer import paths,
and when you change what a function computes, either rename it or start a fresh store.

Variations follow the same rule: adding or removing a systematic variation changes the recorded
analysis and so invalidates the affected work, while renaming a label usually does not.


Compile once, run over many datasets
------------------------------------

Recording and optimizing an analysis costs something; running it over another sample should not
cost it again. ``DurablePlan.with_partitions`` returns a sibling plan with the same computation and
a new set of work units, sharing the recorded analysis unchanged. ``for_dataset`` and
``for_datasets`` are the convenient spellings: hand them a ``Dataset`` (a uri, an event count, and
a tree name) and a chunk size, and they do the splitting.

Because the key includes the partition, every dataset's work lands in the same store without
colliding — so a combined run over samples you have already processed individually is free:

.. code-block:: python

    import numpy as np

    from graphed import Session
    from graphed.checkpoint import Store, run_resumable
    from graphed.core import Dataset, DurablePlan, OpSpec
    from graphed.numpy import NumpyBackend, from_record

    s = Session(NumpyBackend())
    ev = from_record(s, "events", x=np.zeros(1))
    counts, _edges = np.histogram(ev["x"], bins=4, range=(0, 1))

    analysis = DurablePlan(                      # recorded and optimized once
        ir=s.serialized_ir(counts),
        process=OpSpec.from_ref("myanalysis:hist_chunk"),
        combine=OpSpec.from_ref("myanalysis:hist_add"),
        empty=OpSpec.from_ref("myanalysis:hist_empty"),
    )

    samples = [
        Dataset("ttbar.root", n_events=4000, tree="Events", name="ttbar"),
        Dataset("wjets.root", n_events=2000, tree="Events", name="wjets"),
    ]

    store = Store("checkpoints/")
    for ds in samples:
        plan = analysis.for_dataset(ds, chunk_size=1000)
        res = run_resumable(plan, store)
        print(f"{ds.name}: {len(plan.partitions)} partitions, executed {res.report.executed} -> {res.value}")

    both = analysis.for_datasets(samples, chunk_size=1000)   # one run over everything
    res = run_resumable(both, store)
    print("combined: executed", res.report.executed, "skipped", res.report.skipped, "->", res.value)

::

    ttbar: 4 partitions, executed 4 -> [1012  987 1015  986]
    wjets: 2 partitions, executed 2 -> [495 490 515 500]
    combined: executed 0 skipped 6 -> [1507 1477 1530 1486]


When a partition fails
----------------------

Failures are not interchangeable, so the recovery policies are not a single ``retries=N`` knob.
Pass one as ``retry=``:

``RetryN(n)``
    The transient hiccup — a flaky mount, a timed-out read. Runs the same partition again, up to
    *n* times. The ``resources`` object you passed to the run survives across attempts.

``RetrySmallerChunk(splits=2)``
    The partition that is simply too big. Splits its entry range and processes the pieces,
    recursing until they fit or until it can't split further, then combines the sub-results — safe
    to do because combine is associative. A chunk that ran out of memory whole succeeds split.

``RetryElsewhere(attempts=1, new_resources=...)``
    The bad worker. Retries with a fresh ``resources`` object from ``new_resources`` rather than
    burning attempts against a poisoned one.

``Quarantine()``
    Don't retry at all; go straight to the failure list.

A partition that exhausts its policy does not take the run down with it. Set ``error_budget`` and
it is recorded as a *dead letter* — the partition, the error type and message, and, when the
failure is a :class:`~graphed.debug.StageError`, the operation that failed and the line of your
analysis it came from. The run then carries on with everything else, and stops only when the dead
count exceeds the budget.

``flaky.py`` here has one file that always fails:

.. code-block:: python

    import numpy as np

    def count_chunk(partition, resources):
        if partition.uri == "corrupt.root":
            raise ValueError("unexpected EOF in basket 3")
        n = partition.entry_stop - partition.entry_start
        return np.full(4, n // 4, dtype=np.int64)

    def add(a, b):
        return a + b

    def empty():
        return np.zeros(4, dtype=np.int64)

.. code-block:: python

    import numpy as np

    from graphed import Session
    from graphed.checkpoint import RetryN, Store, run_resumable
    from graphed.core import Dataset, DurablePlan, OpSpec
    from graphed.numpy import NumpyBackend, from_record

    s = Session(NumpyBackend())
    ev = from_record(s, "events", x=np.zeros(1))
    counts, _edges = np.histogram(ev["x"], bins=4, range=(0, 1))

    plan = DurablePlan(
        ir=s.serialized_ir(counts),
        process=OpSpec.from_ref("flaky:count_chunk"),
        combine=OpSpec.from_ref("flaky:add"),
        empty=OpSpec.from_ref("flaky:empty"),
    ).for_datasets(
        [Dataset("good.root", n_events=2000), Dataset("corrupt.root", n_events=1000)],
        chunk_size=1000,
    )

    res = run_resumable(plan, Store("checkpoints/"), retry=RetryN(2), error_budget=1)

    print("executed:", res.report.executed, " dead:", res.report.dead, " stopped:", res.report.stopped)
    print("partial result:", res.value)
    for d in res.report.dead_letters:
        print(f"  {d['uri']} [{d['entry_start']}:{d['entry_stop']}] {d['error_type']}: {d['error_message']}")

::

    executed: 2  dead: 1  stopped: None
    partial result: [500 500 500 500]
      corrupt.root [0:1000] ValueError: unexpected EOF in basket 3

The failure list is a to-do list, not a pile of logs: each entry names the exact entry range to
re-run once the file is replaced. Fix the file, run the same plan against the same store, and the
two good partitions are skipped while the repaired one is processed.


Running where your analysis files aren't
----------------------------------------

A plan is bytes. ``plan.to_bytes()`` is a canonical JSON document — sorted keys, no incidental
whitespace, binary blobs base64'd — so identical plans serialize identically and a round trip
gives you the same bytes back. What it carries is the recorded analysis itself, not a pickle of
your session, plus import paths for your process/combine/empty functions. Ship those bytes to a
machine that has your packages installed and it runs there, with none of your analysis scripts on
disk:

.. code-block:: python

    import numpy as np

    from graphed import Session
    from graphed.core import DurablePlan, OpSpec, Partition
    from graphed.numpy import NumpyBackend, from_record

    s = Session(NumpyBackend())
    ev = from_record(s, "events", x=np.zeros(1))
    counts, _edges = np.histogram(ev["x"], bins=4, range=(0, 1))

    plan = DurablePlan(
        ir=s.serialized_ir(counts),
        process=OpSpec.from_ref("myanalysis:hist_chunk"),
        combine=OpSpec.from_ref("myanalysis:hist_add"),
        empty=OpSpec.from_ref("myanalysis:hist_empty"),
        partitions=(Partition("toy", "Events", 0, 1000),),
    )
    print("carries an embedded callable:", plan.opaque)
    print("computation fingerprint:", plan.ir_fingerprint()[:16])

    with open("plan.bin", "wb") as f:
        f.write(plan.to_bytes())

    # elsewhere, later, on another machine
    with open("plan.bin", "rb") as f:
        shipped = DurablePlan.from_bytes(f.read())

    print("same computation:", shipped.ir_fingerprint() == plan.ir_fingerprint())
    print("recovered graph:", shipped.graph().node_count(), "nodes")
    print("same task ids:", shipped.task_id(shipped.partitions[0]) == plan.task_id(plan.partitions[0]))

::

    carries an embedded callable: False
    computation fingerprint: 8d5fc3d15dc8a71a
    same computation: True
    recovered graph: 3 nodes
    same task ids: True

``plan.opaque`` is the one to watch. False means every callable in the plan is an import path and
the plan is fully portable. True means at least one was embedded by value, which still runs but
ties the plan to a matching interpreter and library set — :doc:`../preserve/index` surfaces that as
a reproducibility risk when you build a bundle.

Two fingerprints are available: ``ir_fingerprint()`` identifies the computation alone, so two plans
over different datasets share it; ``fingerprint()`` covers the whole plan including its partitions
and metadata.


Resuming a shuffle
------------------

A join or a repartition is not a single map-then-reduce: every task writes one block per output
partition, then every output partition collects the blocks addressed to it. ``run_shuffle_resumable``
resumes that shape, taking a ``graphed.core.DurablePlanV2`` — a plan of stages, where a gather
stage names the stages it reads. Every block is content-addressed and journaled with its stage and
the hashes of the blocks it consumed, so a crash anywhere resumes from the last durable block and
the result is byte-identical to an uninterrupted run.

The keys here also fold in the stage's routing — which key goes to which output partition, and
which exchange backend decided that. Two backends that route the same key to different destinations
therefore never file different content under the same key.

It returns a ``ShuffleResumeResult``: the gather blocks' content hashes in destination order, plus
the same ``ResumeReport`` you get from ``run_resumable``.


Not supported yet
-----------------

**The store is a local directory.** Results go to a local filesystem path; there is no S3 or xrootd
backend. Point the store at a shared filesystem your workers can all see, or checkpoint per node —
a ``Store(root, node="A")`` writes its own journal so several writers under one root never contend
on the same append, and reading replays all of them.

**Recompute is sequential.** Missing partitions are processed one at a time, in order. Resume
correctness does not depend on that, but a big recompute takes as long as the work does; for
parallel execution drive the same analysis through ``graphed-executors`` and use the store for the
resume boundary.

**No garbage collection.** Results accumulate under the store root and nothing prunes them. Delete
the directory when a set of results is stale; there is no reachability sweep.

**Results are per-plan.** Two plans that share a sub-computation share nothing in the store, since
the key covers the whole analysis. Splitting a long analysis into stages you checkpoint separately
is the way to get reuse today.

**Retrying elsewhere means a fresh local context.** ``RetryElsewhere`` builds a new ``resources``
object; it does not move the task to another host.

See :doc:`improvements` for the shorter list of what is on the way.
