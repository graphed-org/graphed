How graphed works
=================

``graphed`` is the recording frontend of the ecosystem. A user writes ordinary array
expressions; this package turns each operation into a node in the Rust-backed
:mod:`graphed_core` store and hands back a lightweight proxy. Nothing computes until something
asks — and when something asks, what runs is the *reduced* graph, never a replay of the user's
operations one by one.

The package is strictly **backend-agnostic**: it knows nothing about numpy or awkward. Array
semantics (type inference, evaluation, column projection) arrive through a small ``Backend``
protocol, implemented by ``graphed-numpy`` and ``graphed-awkward``. This document explains what
the frontend itself contributes: the recording session, the proxy, forms and provenance, the
projection machinery, compilation/evaluation, and the two shared I/O bases.

.. contents::
   :local:
   :depth: 2


Recording: Session and the Array proxy
--------------------------------------

A :class:`~graphed.Session` owns one ``graphed_core.GraphStore`` plus the side tables the core
deliberately does not hold: per-node **forms** (backend type/shape descriptions), **provenance**
(the user source line that recorded each node), and source/external metadata. Recording an op is
three steps::

    in_forms = [forms[a.node_id] for a in inputs]
    form     = backend.op_form(op, in_forms, params)   # typetracer-style inference, NO data
    node_id  = store.add_op(op, input_ids, params)     # interned: duplicates return the old id

The important property: **type errors surface at the recording line.** ``op_form`` runs the
backend's inference on metadata only; if the user wrote something ill-typed (a missing field, a
shape mismatch), the resulting :class:`~graphed.GraphedTypeError` carries the captured user
frame — before any data is read. Try it::

    import numpy as np
    from graphed import Session
    from graphed_numpy import NumpyBackend, from_array

    s = Session(NumpyBackend())
    x = from_array(s, "x", np.arange(6.0))
    y = (x * 2.0 + 1.0)[x > 2.0]      # records 4 nodes; computes nothing
    s.materialize(y)                   # -> array([ 7.,  9., 11.])
    s.form(y).describe()               # -> 'vector[float64]'
    s.provenance(y)                    # -> the file:line of the `y = ...` statement

What the user holds is an :class:`~graphed.Array` — a proxy carrying only ``(session,
node_id)``. The proxy implements the *common* surface of deferred arrays: arithmetic and
comparison dunders, ``__array_ufunc__`` (so ``np.sqrt(x)`` records instead of executing),
boolean/slice/integer/field-list ``__getitem__``, and the shared helpers backends build on
(axis normalization, the reduction/scan recording rule). Everything *idiomatic to one array
library* lives outside this class: a backend may supply a richer proxy via its
``array_type()`` factory (graphed-numpy's ``NumpyArray`` adds ``.shape``, ``.sum()``,
``__array_function__`` and friends), while graphed-awkward deliberately keeps the base proxy
and exposes its idiom as free functions. The split keeps one library's conventions from
leaking into another's.

Two recording details with outsized consequences:

* **Interning means recording is idempotent.** Writing the same subexpression twice — directly
  or via a helper — yields the same node id. Sessions can be long-lived and exploratory; the
  graph holds the *set* of distinct computations, not the history of statements.
* **Incremental reduction is opt-in at the session.** ``Session(backend, incremental=True)``
  maintains the reduced canonical form *as the graph is built* (a ``graphed_core``
  ``IncrementalReducer`` consuming deltas), so a large un-reduced graph never exists; the
  one-shot and incremental paths are pinned byte-identical.

The Backend protocol
--------------------

Five methods are the entire seam between the frontend and an array library::

    op_form(op, input_forms, params) -> Form        # record-time inference (metadata only)
    eval_stage(op, inputs, params)   -> value       # evaluation of one op / fused member
    boundary_ops()                   -> frozenset   # which op names are stage boundaries
    project(op, used, params)        -> used'       # reporting-tracer step for projection
    external_payload(op, params)     -> descriptor  # M3-family Externals (corrections, models)

A backend never sees the graph; the frontend never sees an array. ``Form`` is likewise a
protocol (``describe() -> str``) — the frontend stores and forwards forms, it does not
interpret them.

For External nodes recorded *by other packages* (histogram fills are the canonical example),
``Session.record_external`` accepts explicit ``descriptor=`` and ``form=`` arguments, skipping
the backend entirely — the mechanism that lets ``graphed-histogram`` exist without teaching any
backend about histograms.


Projection: what would this result actually read?
--------------------------------------------------

Before reading anything, an executor wants the *minimal* input. The frontend supplies the
machinery; backends supply the semantics. ``Session.walk`` is a generic cached graph traversal
with caller-supplied handlers for sources, ops, and externals — ``materialize`` is just ``walk``
with evaluating handlers; projection is ``walk`` with *reporting tracers* flowing through the
backend's ``project``.

Two granularities, and the distinction matters:

* :class:`~graphed.Projection` — the **column** view: which named columns of each source are
  touched. Right for "what should I read for *this value*".
* :class:`~graphed.BufferProjection` — the **buffer** view: per column, whether its *data* is
  needed or only its *offsets* (list structure). ``len(jets.pt)`` per event needs Jet offsets
  but no leaf data; a column view either over-reads or under-specifies that. Writers translate
  an offsets-only need into the cheapest carrier their format allows.

One projection lesson is encoded as API rather than prose: **evaluation read lists are
syntactic, not buffer-projected.** Compiled-IR evaluation replays *every* recorded node — a
zip's untouched legs included — so consumers that will re-evaluate a graph must cover every
source field the graph mentions, then refine leaves by the buffer view. The buffer projection
answers "what data is needed"; only the syntactic walk answers "what must exist".

Opaque ops (a cloudpickled ``map``) cannot be projected through. The ``on_fail`` policy
(``pass`` — optimistically assume nothing extra, ``warn`` — conservative full read with a
warning, ``raise``) is explicit at every projection entry point, mirroring dask-awkward's
choice but never silently.


Compile once, evaluate anywhere
-------------------------------

``compile_ir(session, *outputs)`` reduces the graph **for exactly those outputs** and returns a
:class:`~graphed.CompiledGraph` — the serialized reduced bytes plus source names. Outputs are a
property of the compile request: compiling ``a`` then ``b`` from one session yields two
independent single-output artifacts, byte-identical to fresh-session compiles (and a deliberate
multi-output ``compile_ir(s, a, b)`` carries both, evaluated in one pass).

``evaluate_ir(compiled, backend, sources, externals=...)`` walks the *reduced* node list once:
one backend dispatch per reduced node, fused stage members run inline. ``sources`` binds source
names to data (or zero-arg loaders); ``externals`` resolves External payloads **by content
hash**, failing loudly when one is missing — an opaque payload is never silently skipped.
Continuing the example above::

    from graphed import compile_ir, evaluate_ir

    compiled = compile_ir(s, y)
    evaluate_ir(compiled, NumpyBackend(), {"x": np.arange(6.0)})
    # -> [array([ 7.,  9., 11.])]   (a list: one entry per requested output)

This is the deployment seam: the bytes inside ``compiled.ir`` are the durable artifact.
``Session.serialized_ir(*outputs)`` exposes them directly (``optimize=False`` gives the 1:1
auditable form); identical analyses serialize byte-identically, which is what checkpoint
stores, preservation bundles, and the determinism CI gate all build on.


The shared I/O bases
--------------------

Two small modules host what every I/O integration shares, with **no array-library content**:

:mod:`graphed.parquet`
    Deterministic dataset discovery (directories/globs sorted; explicit lists keep caller
    order — the list is part of the dataset's identity), metadata-only row counts, blind
    partitioning, and the deferred-source recording convention. The array codecs live in the
    backends.

:mod:`graphed.write`
    The format-agnostic partitioned-write skeleton: ``write_plan`` builds a task graph whose
    tasks write one part each and *report their paths* up a deterministic combine tree;
    ``graphed_core.execution.SequentialRunner`` is the dependency-free reference runner (any real executor accepts the
    same plan); ``file_bases``/``blind_part_index``/``step_of``/``part_path`` let a worker
    derive its own part name from its partition plus an O(#files) table. The module also
    defines :class:`~graphed.write.PartitionedSource` — the read-side protocol (``partitions()``
    blind, ``read_partition(partition, columns, resources)``) that lets *generic* consumers
    (the parquet writer, the histogram aggregator) process any source partition-by-partition
    without ever invoking its whole-dataset loader.

Partitions are **blind** wherever possible: planning opens no files; a worker resolves its
entry range against the file it already opened. This is both a performance property and a
correctness one — a plan built on machine A is valid on machine B whose files it has never
seen.


Errors and provenance
---------------------

``capture()`` records the nearest user frame at every recording call; ``GraphedTypeError``
formats it into the message. Runtime errors are the next package up
(``graphed-debug``'s source-mapped ``StageError``) — the frontend's contribution is that the
provenance *exists* for every node, cheaply, from the moment it was recorded.


Phase 2 (deliberately not built)
--------------------------------

* **Predicate pushdown.** Projection covers columns/buffers; pushing *filters* into readers is
  explicitly out of scope for the MVP.
* **Behavior methods with arguments through the proxy.** Behavior *properties* record (the
  ragged backend resolves them); ``a.deltaR(b)``-style method calls do not — analyses write the
  explicit formula today.
* **Output isolation conveniences.** Compile-request scoping is done; higher-level helpers
  (e.g. compiling output *groups* with shared sub-plans) are future work.
* **Non-local sources for the parquet base.** Discovery and row counts are local-filesystem
  (and fsspec-compatible only incidentally); remote-store-aware planning is Phase 2.

See :doc:`improvements` for the live tracked list.
