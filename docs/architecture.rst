How a run fits together
=======================

Between the awkward code you type and the filled histogram you plot, ``graphed`` records what
you wrote and reduces it on your machine, then runs it — and the running happens a chunk at a
time, inside whatever worker got that chunk. This page is the map: who does what, which parts
you install, and what actually crosses a process boundary when the work goes to a cluster. Read
:doc:`quickstart` first if you want the code; this page explains the shape it has.

From your code to a result
--------------------------

Nothing builds a large graph and then simplifies it. Each operation you write is folded into the
reduced form as it arrives, so the only graph that ever exists is the concise one.

.. code-block:: text

    on your machine                              on each worker
    ---------------                              --------------
    graphed        records what you write
      |
      v
    graphed.core   reduces it as it arrives,
      (optimizer)  builds the plan
      |
      |  ---- plan + partitions ---->            graphed.awkward / graphed.numpy
      |                                          evaluate one stage of the plan
      |  <--- partial results -------            on one chunk of data
      v
    the answer, combined in an order the plan fixes

Even on your laptop the right-hand column is real: with one process it is the same code running
in the same interpreter, but it is still a stage at a time on a chunk at a time.

**Record.** ``graphed`` itself. ``gak.max(...)``, ``jets.pt > 25.0``, ``h.fill(...)`` — each one
returns a proxy and records a node. Two things are settled at this moment, on your machine,
before any file is opened: the type and shape of the result (awkward calls this a *form*, and
computes it from metadata alone), and the source line you wrote it on. That is why a shape
mistake is reported at the line that made it rather than three hours into a cluster job.

**Reduce.** ``graphed.core``, a compiled Rust extension. Write the same cut in two places and
you get one node, not two: identical expressions collapse on the way in. Anything no output
depends on is dropped. Then the optimizer looks at the equivalent ways your expression could be
written — all at once, rather than applying rewrites in a fixed order — and keeps the cheapest,
fusing maximal runs of array operations into groups that execute as a single pass over the data
(a *stage*). A group ends where the data has to change shape or leave: a read, a cross-event
reduction, a repartition, a checkpoint, a call out to a correction or a model. The consequence
is that the Python interpreter is touched once per group instead of once per operation, and
that the reduced form of a given analysis is byte-identical every time you produce it.

**Evaluate.** A backend, *inside the worker*. ``graphed.awkward`` is the one HEP analyses live
on — ``gak`` carries the ``ak.*`` names — and ``graphed.numpy`` is the same idea for flat
arrays. A backend turns one stage into real array calls on one chunk of data. Nothing about
awkward or numpy reaches the frontend or the optimizer, which is why a plan compiled with one is
not tied to it.

**Run.** A runner. It receives a plan — a list of partitions, a function to run on each, and a
way to combine partial results — hands the tasks out, and gives back one answer. ``graphed.core``
defines that contract and ships one runner of its own: ``SequentialRunner``, in-process, no extra
dependencies, which is what the examples in these docs use. ``graphed-executors`` supplies the
pooled and cluster runners — thread and process pools, dask, parsl — for the same plan.

**Debug, restart, preserve** attach to the boundary between the two columns rather than sitting
in the line. ``graphed.debug`` re-raises a failure that happened inside a fused stage on a remote
worker as an exception on your machine that points at your analysis line, and shows task events
live as they cross back. ``graphed.checkpoint`` remembers partial results as they arrive, so a
killed run resumes without changing the answer. ``graphed.preserve`` exports a directory someone
else can reproduce or inspect. They are three independent things you can attach; none feeds
another.

What crosses a process boundary
-------------------------------

When a run goes to workers, what travels is the *recording* — a versioned, self-describing byte
string — plus the partition each task should read. Your Python objects do not travel; neither
does the session you built them in. That has three consequences worth knowing before you
debug a cluster job:

* A worker needs to be able to import whatever your analysis imports. If a helper module is on
  your laptop but not on the cluster, the failure is at import time on the worker.
* Genuinely opaque callables — a lambda you passed to ``apply``, say — are the one exception:
  they are serialized by value, and the recording marks them as a reproducibility risk. Refer to
  a function by import path instead and it stays readable, and its cached results survive.
* Partial results come back and are combined in an order fixed by the plan, not by which worker
  finished first — so your totals do not wobble between runs or between worker counts.

Errors travel intact in the same way: a failure inside a stage arrives at your machine as a
:class:`~graphed.debug.StageError` carrying the operation, the input shapes, the partition and
your source frame, not as an opaque string from another process.

When rows have to move between partitions
-----------------------------------------

Most analysis is partition-local: each task reads its own chunk and never needs anyone else's.
Two things are not.

``graphed.join`` and ``graphed.repartition`` (with ``join_plan`` and ``shuffle_plan`` for the
plan-level form) match or redistribute rows across partitions by key. Because that means every
task potentially sending data to every other, the exchange is a boundary — a stage cannot fuse
across it — and the runner batches the transfers rather than opening a file per pair. On a
cluster the workers exchange blocks with each other where the backend can address them, and
route through your submit node where it cannot; the choice is made from the plan, so every
worker agrees on it and two runs of the same plan take the same route.

``graphed.aggregate_plan`` goes the other way: several outputs that share a sub-expression — one
selection feeding two histograms, a sum and a count over the same cut — compile into one
recording, so the shared part is read and evaluated once rather than once per output. That is
what ``graphed_histogram``'s ``plan({...})`` is built on, and what makes hundreds of histograms
with systematic variations one pass over the data instead of hundreds.

What you install
----------------

The base install is light: the recording frontend, the compiled core, and two small pure-Python
dependencies (``executing``, for the source line, and ``cloudpickle``, for opaque callables).
Everything heavier is an extra.

.. list-table::
   :header-rows: 1
   :widths: 22 58 20

   * - Import path
     - What it gives you
     - Install extra
   * - ``graphed``
     - the recording surface: ``Session``, ``Array``, ``vary``, provenance, projection,
       compilation, joins and repartition, multi-output aggregation plans
     - (base)
   * - ``graphed.core``
     - the compiled optimizer, the durable formats — a saved plan, its content hashes — the
       runner and monitor contracts, and ``SequentialRunner``, the in-process runner
     - (base)
   * - ``graphed.awkward``
     - ragged analysis: the ``gak`` namespace at ``ak.*`` parity, ``gnano.events``, vector
       behaviours, column and buffer projection, parquet I/O (add ``[parquet]``), and
       correctionlib/ONNX calls
     - ``[awkward]``
   * - ``graphed.numpy``
     - deferred numpy for flat arrays: ufuncs, array functions, reductions
     - ``[numpy]``
   * - ``graphed.debug``
     - the unfused 1:1 view, source-mapped tracebacks, and the live run dashboard
     - (base); ``[dashboard]`` for the live view
   * - ``graphed.checkpoint``
     - results filed by what they compute, so a restart redoes only what was in flight; plus
       retry policies and a dead-letter queue
     - (base)
   * - ``graphed.preserve``
     - a self-contained bundle that reproduces or is inspected elsewhere, plus plugins for
       torch, TensorFlow, XGBoost, JAX, ONNX, correctionlib and Triton payloads
     - (base); ``[preserve]`` for correctionlib/ONNX payloads

A missing extra behaves in one of two ways. The array backends import their array library
eagerly, so ``import graphed.awkward`` without ``[awkward]`` fails immediately with a clear
message. ``graphed.debug``, ``graphed.checkpoint`` and ``graphed.preserve`` import fine on the
base install and pull their heavy dependencies only when a feature reaches for one — the
dashboard, or a correctionlib/ONNX payload. ``graphed.checkpoint`` needs nothing extra at all;
its ``[checkpoint]`` marker exists so an old install line keeps working.

``pip install "graphed[all]"`` pulls every extra including ``[parquet]``, but not the heavy ML
frameworks — those are ``[ml]``. Import paths from before the packages were combined are mapped
in ``MIGRATION.md`` in the repository.

The pieces that ship separately
-------------------------------

Two packages you will want are their own installs, because executors and histograms are things
you swap:

* `graphed-executors <https://github.com/graphed-org/graphed-executors>`_ — the runners past
  ``SequentialRunner``. Thread and process pools on one machine, with straggler-tolerant tree
  reduction, file-handle
  reuse across the operations in a partition, work stealing, and worker-to-worker exchange;
  dask and parsl backends under ``[dask]`` and ``[parsl]`` for batch clusters. Every one of them
  takes the same plan.
* `graphed-histogram <https://github.com/graphed-org/graphed-histogram>`_ — deferred
  ``boost-histogram`` and ``hist`` fills, the dask-histogram analogue. A ``.fill()`` records;
  ``plan()`` exports the task graph a runner aggregates. It also backs ``hist.graphed``, so the
  ``Hist.new.Reg(...).Double()`` builder you already use works on deferred arrays.

``graphed``'s numbers are checked against the same analyses written in plain awkward — exactly,
with no tolerances. :doc:`corpus/index` describes those reference analyses and what they cover.
