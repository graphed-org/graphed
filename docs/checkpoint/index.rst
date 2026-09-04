graphed.checkpoint
==================

Your eight-hour run died at hour seven. Restart it and only the partition that was in flight is
redone — the rest are read back from disk, and the total you get is the one the uninterrupted run
would have produced, down to the last bit.

That works because every piece of work is filed under a hash of *what it computes*: the analysis,
the processing function, and the partition (content addressing). Re-running is then a lookup —
same inputs, same key, result already there. Change any of them and the key changes, so a stale
result can never be handed back for a computation you didn't ask for.

Three things come out of this package:

- a ``Store`` — a directory of results on your local filesystem, written atomically, so a crash
  mid-write leaves nothing half-finished behind;
- ``run_resumable`` — a runner that skips whatever the store already holds and combines the rest in
  a fixed order, so no partition is counted twice and none is lost;
- retry policies for the failures worth retrying, and a list of the ones that aren't — each entry
  carrying enough detail to re-run exactly that slice once you've fixed it.

Sixty seconds of it. The per-partition work lives in an importable module, ``myanalysis.py``:

.. code-block:: python

    import numpy as np

    def hist_chunk(partition, resources):
        rng = np.random.default_rng(partition.entry_start)
        n = partition.entry_stop - partition.entry_start
        return np.histogram(rng.uniform(0, 1, n), bins=4, range=(0, 1))[0]

    def hist_add(a, b):
        return a + b

    def hist_empty():
        return np.zeros(4, dtype=np.int64)

and the plan that drives it:

.. code-block:: python

    import numpy as np

    from graphed import Session
    from graphed.checkpoint import Store, run_resumable
    from graphed.core import DurablePlan, OpSpec, Partition
    from graphed.numpy import NumpyBackend, from_record

    # record the analysis once: this is what the plan carries
    s = Session(NumpyBackend())
    ev = from_record(s, "events", x=np.zeros(1))
    counts, _edges = np.histogram(ev["x"], bins=4, range=(0, 1))

    plan = DurablePlan(
        ir=s.serialized_ir(counts),
        process=OpSpec.from_ref("myanalysis:hist_chunk"),
        combine=OpSpec.from_ref("myanalysis:hist_add"),
        empty=OpSpec.from_ref("myanalysis:hist_empty"),
        partitions=tuple(Partition("toy", "Events", i * 1000, (i + 1) * 1000) for i in range(6)),
        read_columns=("x",),
    )

    store = Store("checkpoints/")

    first = run_resumable(plan, store)
    print("first  executed:", first.report.executed, " skipped:", first.report.skipped)
    print("result:", first.value)

    again = run_resumable(plan, store)
    print("second executed:", again.report.executed, " skipped:", again.report.skipped)
    print("identical:", bool((again.value == first.value).all()))

::

    first  executed: 6  skipped: 0
    result: [1499 1493 1518 1490]
    second executed: 0  skipped: 6
    identical: True

The second run touched no data at all.

:doc:`design` goes on to a real kill-and-resume, what invalidates a cached result, compiling an
analysis once and running it over many datasets, and what happens to a partition that keeps
failing.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   design
   improvements

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
