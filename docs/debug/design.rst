How graphed-debug works
=======================

``graphed-debug`` exists because of one specific, painful failure mode of distributed array
analysis: a computation fails inside a worker, and what reaches the user is an opaque wall of
framework internals — their actual mistake nowhere in it. This package makes the opposite
guarantee: **a runtime failure points at the user's analysis line**, with the failing operation,
its input types, and the partition that tripped it — and that diagnosis survives optimization,
pickling, and process boundaries.

Three pieces deliver it: *opt-level lowering* (a debuggable view of what executes),
*source-mapped tracebacks* (``StageError``), and a small *graph visualizer*.

.. contents::
   :local:
   :depth: 2


Lowering: choosing how literally to run
---------------------------------------

``lower(session, array, opt_level=...)`` walks the recorded graph and produces a
``LoweredGraph`` — stages of ``LoweredOp`` members, each member carrying its operator token,
its input references, its inferred form, and **its own** ``SourceFrame`` (the user line that
recorded it).

* ``opt_level=0`` — **one op per stage, 1:1 with the user's code**, no fusion. The canonical
  mode for pinpointing where a value first goes wrong; the debug runner additionally checks
  inter-op consistency as it goes. What you debug is literally what you wrote.
* ``opt_level>=1`` — maximal op runs between boundaries are **fused** into multi-member stages,
  the same boundary rule the real optimizer uses: the structure you debug matches the structure
  that actually executes.

The provenance design choice worth noticing: source frames attach to *members*, not stages. A
fused stage of four ops carries four frames. That is why optimization never costs the
traceback — the arrow can land on the exact line even when the failing op is buried mid-stage::

    lowered = gd.lower(session, result, opt_level=1)
    for stage in lowered.stages:
        for m in stage.members:
            print(m.op, f"{m.provenance.filename}:{m.provenance.lineno}")


StageError: the diagnosis as a real exception
---------------------------------------------

``run(session, array, opt_level=..., partition=...)`` executes the lowered graph on the
session's source data and, on the first failing op, raises :class:`~graphed_debug.StageError`
carrying:

* ``op`` — the failing operator;
* ``frames`` — the user's analysis frames, with ``user_frame`` the closest one;
* ``input_forms`` — the inferred types feeding the op (often the diagnosis by itself);
* ``partition`` — *which* chunk of data tripped it (data-dependent bugs fail on some
  partitions and not others; this field is how you find the culprit events);
* ``cause_type`` / ``cause_message`` — the underlying exception, as data;
* ``opt_level`` — which structure was executing.

Two properties are load-bearing and pinned by tests rather than promised:

* **It pickles intact.** ``StageError`` round-trips ``pickle`` with every field — so a worker
  process can raise it, a futures executor can transport it, and the driver re-raises the
  *same* diagnosis. This is the difference between "remote error: see worker logs" and an
  arrowed traceback at the driver.
* **It renders for humans.** ``format_traceback(err)`` produces the user-code traceback with a
  ``-->`` arrow on the faulty line, the op, forms, partition, and cause — the thing you print
  in an ``except StageError`` handler::

    Traceback (most recent call last) — user analysis frames:
          File "analysis.py", line 27, in <module>
      --> File "analysis.py", line 29, in <module>
    ValueError: structure imposed by 'counts' does not fit ...
      [stage op 'ak.unflatten', partition skim@16384:32768, opt_level=1]

A worked failure
~~~~~~~~~~~~~~~~

::

    import numpy as np, graphed_debug as gd
    from graphed import Session
    from graphed_numpy import NumpyBackend, from_array

    s   = Session(NumpyBackend())
    x   = from_array(s, "x", np.arange(4.0))
    bad = (x * 2.0).map(lambda a: a[100], name="oob")   # an opaque op with a latent bug

    try:
        gd.run(s, bad, opt_level=1, partition="demo@0:4")
    except gd.StageError as err:
        print(err.op, err.cause_type)        # external IndexError
        print(err.user_frame)                # the `bad = ...` line
        print(gd.format_traceback(err))

The pattern scales unchanged to real executors: a worker task records and ``run``\ s its chunk
(or evaluates compiled IR and wraps), the ``StageError`` crosses the pool, and the driver's
handler prints the same rendering — demonstrated end-to-end (spawned ``ProcessExecutor``, real
ROOT partitions, a data-dependent bug invisible to record-time typing) in the ADL benchmarks
notebook.

Record-time vs run-time
~~~~~~~~~~~~~~~~~~~~~~~

This package handles *run-time* failures. Record-time type errors never get this far: the
frontend's form inference raises ``GraphedTypeError`` at the offending line before any data is
read. The division gives two distinct safety nets — typos and type mismatches die instantly at
recording; data-dependent failures (the off-by-one that only wrong-counts on real events) die
at execution *with the same quality of source mapping*.


Visualization
-------------

``visualize(lowered, fmt="mermaid"|"graphviz", projection=...)`` renders the lowered graph as
diagram source — stages with their member ops, boundaries distinct, and (optionally) each
source annotated with its projected columns. Deterministic text output, suitable for docs,
notebooks, and diffing two lowerings of the same analysis.


Live execution dashboard
------------------------

``Dashboard`` (M37) is an **opt-in, passive** live view of a running ``Plan``. While work runs on
*any* executor, a daemon thread serves a webpage showing task progress, throughput, per-worker
activity, a statistical-sampling flamegraph, and any ``StageError`` mapped to the user's analysis
line. Unlike dask — which starts a Bokeh server on every client — nothing runs until you ask:

.. code-block:: python

   from graphed_debug import Dashboard
   from graphed_exec_local.executors import ProcessExecutor

   with Dashboard(port=8787, profile=True) as dash:   # prints http://127.0.0.1:8787/
       result = ProcessExecutor(monitor=dash, persistent=True).run(plan)
   # the server thread, collector, and worker samplers are torn down on exit

How it fits together, in three layers with strict boundaries:

* **The seam (``graphed_core.execution``).** ``TaskEvent`` (a frozen, picklable, *display-only*
  record), ``TaskPhase``, and the ``Monitor`` / ``WorkerProfiler`` protocols are pure data — core
  gains no web or profiler dependency. The event vocabulary is shared by every executor, so it
  lives at the layer it serves.
* **Emit (``graphed-exec-local``).** Each executor takes an optional ``monitor=``. A thread pool
  calls the monitor in-process; a process pool forwards worker events over a bounded
  ``Manager().Queue()`` drained by a driver-side collector thread. Per task: one ``SUBMITTED``
  (driver-side), then ``STARTED``, then exactly one of ``FINISHED`` / ``ERRORED`` (worker-side).
* **Consume + render (``graphed-debug``).** The ``Dashboard`` *is* the ``Monitor`` and owns the web
  server. Python emits JSON over **Server-Sent Events**; a static single-page app (uPlot for the
  time series, d3-flame-graph for the profile) renders in the browser — there is no Python
  rendering framework. The error panel reuses the same source-mapped ``StageError`` rendering as
  the rest of this package.

The headline guarantee is **passivity**: attaching a dashboard (even with ``profile=True``) leaves
the reduced result, the combine count, and the serialized plan byte-identical. Emission is
best-effort and drops on a full queue rather than back-pressuring a worker, and a monitor that
raises is swallowed — the determinism gate is green attached-or-not.

The statistical sampler is `pyinstrument <https://github.com/joerick/pyinstrument>`_ (an optional
``dashboard`` extra). Each worker runs its own sampler; ``graphed-exec-local`` drives it through the
abstract ``WorkerProfiler`` protocol without importing it. Per-worker sessions are serialized, sent
over the same channel as task events, and merged driver-side into one flamegraph.


Phase 2 (deliberately not built)
--------------------------------

* **Interactive/time-travel debugging** (stepping through stage members against live data) is
  explicitly out of scope for the MVP.
* **Value-level probes** (capturing intermediate arrays at a chosen node during ``run``) —
  the consistency checks at ``opt_level=0`` are structural, not value snapshots.
* **Richer viz** (cost overlays, diffing renderers) beyond the deterministic mermaid/graphviz
  text.

See :doc:`improvements` for the live tracked list.
