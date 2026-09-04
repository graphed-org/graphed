API reference
=============

These pages are generated from the installed package, so they always match the code you are
running. This page groups them by what you are trying to do. Follow a module link for its full
contents.

If you are writing an analysis, ``graphed.awkward`` is where nearly all of your time goes:
``gak`` is the working surface. ``graphed.core`` is mostly internal — you reach into it for the
plan and partition types when you drive a runner yourself.

Record and run
--------------

The recording surface. ``Session`` owns the recording and ``Array`` is the proxy you manipulate;
``compile_ir`` and ``evaluate_ir`` turn a recording into something a worker evaluates, and
``aggregate_plan`` builds the task graph that computes several outputs in one pass over the
data. ``vary`` declares a systematic variation, and ``labels`` / ``nominal`` / ``universe`` /
``variations`` read the results back. ``join``, ``repartition``, ``join_plan`` and
``shuffle_plan`` move rows between partitions; ``read_columns`` and ``impact_by_label`` tell you
what a recording will actually read off disk.

.. autosummary::
   :toctree: generated
   :recursive:

   graphed

Ragged analysis
---------------

The backend HEP analyses live on. ``gak`` mirrors ``ak.*`` name for name and signature for
signature; ``gnano.events`` wraps a record as an event context so ``vary`` can shift a whole
collection; ``from_awkward`` and ``from_parquet`` are the sources; ``project`` and
``project_buffers`` show which columns and which pieces of them a recording needs.

.. autosummary::
   :toctree: generated
   :recursive:

   graphed.awkward

Flat arrays
-----------

Deferred numpy, for rectilinear data: your ``np.*`` idiom records instead of computing.
``from_array``, the constructors (``zeros``, ``arange``, ``linspace``, …), ``apply_gufunc``, and
``default_rng`` for randomness that stays reproducible across runs.

.. autosummary::
   :toctree: generated
   :recursive:

   graphed.numpy

Debugging a run
---------------

``StageError`` is what you catch: it carries the operation, the input shapes, the partition and
your source frame, and it survives the trip back from a worker process. ``format_traceback``
renders it with an arrow at your line. ``lower`` gives you the unfused, one-operation-at-a-time
view of a recording. ``Dashboard`` streams a live run.

.. autosummary::
   :toctree: generated
   :recursive:

   graphed.debug

Resuming and preserving
-----------------------

``Store`` and ``run_resumable`` (``run_shuffle_resumable`` for a run containing an exchange)
restart a killed job without redoing finished work; ``RetryN`` and friends decide what to do
with a partition that keeps failing, and a dead-letter queue holds what is left.
``build_bundle``, ``inspect`` and ``reproduce`` are the export side: a directory someone else
can run, or read without running. ``register_plugin`` adds your own payload kind.

.. autosummary::
   :toctree: generated
   :recursive:

   graphed.checkpoint
   graphed.preserve

Plans, partitions and contracts
-------------------------------

The types you touch when you drive a runner yourself or save a compiled analysis: ``Plan``,
``Task``, ``Partition``, ``SequentialRunner``, the ``Executor`` and ``Monitor`` protocols, and
``DurablePlan`` with ``with_partitions`` / ``for_dataset`` / ``for_datasets`` for re-aiming one
compiled analysis at many datasets. The rest of this module is the compiled optimizer.

.. autosummary::
   :toctree: generated
   :recursive:

   graphed.core
