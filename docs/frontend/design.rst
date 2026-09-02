How graphed works
=================

``graphed`` is the recording frontend of the ecosystem. A user writes ordinary array
expressions; this package turns each operation into a node in the Rust-backed
:mod:`graphed.core` store and hands back a lightweight proxy. Nothing computes until something
asks — and when something asks, what runs is the *reduced* graph, never a replay of the user's
operations one by one.

The package is strictly **backend-agnostic**: it knows nothing about numpy or awkward. Array
semantics (type inference, evaluation, column projection) arrive through a small ``Backend``
protocol, implemented by ``graphed.numpy`` and ``graphed.awkward``. This document explains what
the frontend itself contributes: the recording session, the proxy, forms and provenance, the
projection machinery, compilation/evaluation, and the two shared I/O bases.

.. contents::
   :local:
   :depth: 2


Recording: Session and the Array proxy
--------------------------------------

A :class:`~graphed.Session` owns one ``graphed.core.GraphStore`` plus the side tables the core
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
    from graphed.numpy import NumpyBackend, from_array

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
``array_type()`` factory (graphed.numpy's ``NumpyArray`` adds ``.shape``, ``.sum()``,
``__array_function__`` and friends), while graphed.awkward deliberately keeps the base proxy
and exposes its idiom as free functions. The split keeps one library's conventions from
leaking into another's.

Two recording details with outsized consequences:

* **Interning means recording is idempotent.** Writing the same subexpression twice — directly
  or via a helper — yields the same node id. Sessions can be long-lived and exploratory; the
  graph holds the *set* of distinct computations, not the history of statements.
* **Incremental reduction is opt-in at the session.** ``Session(backend, incremental=True)``
  maintains the reduced canonical form *as the graph is built* (a ``graphed.core``
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
    ``graphed.core.execution.SequentialRunner`` is the dependency-free reference runner (any real executor accepts the
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
(``graphed.debug``'s source-mapped ``StageError``) — the frontend's contribution is that the
provenance *exists* for every node, cheaply, from the moment it was recorded.


How variations work
-------------------

A systematic *variation* is the same analysis re-run with one knob moved — a jet-energy scale
shifted, a weight scaled up/down. ``graphed.vary(target, name, ...)`` records that intent and
returns a :class:`~graphed.Varied`: a proxy that behaves like the ``target`` but carries a
**family of labelled universes**, ``"nominal"`` plus one per tag. It never mutates — the target
stays valid — and, per §1.2, the labels live **only in the frontend**: each universe lowers to
an ordinary marked output in the IR (the *sibling* lowering), so the core, optimizer, and
executor never learn the word "variation". Interning still deduplicates whatever the universes
share::

    import numpy as np
    from graphed import Session, vary, labels, universe, nominal
    from graphed.numpy import NumpyBackend, from_array

    s = Session(NumpyBackend())
    pt  = from_array(s, "pt", np.array([10.0, 20.0, 30.0]))
    jes = vary(pt, "jes", up=pt * 1.05, down=pt * 0.95)   # a Varied: three universes

    labels(jes)                            # -> ('nominal', 'jes_up', 'jes_down')
    s.materialize(nominal(jes))            # -> array([10., 20., 30.])
    s.materialize(universe(jes, "jes_up")) # -> array([10.5, 21. , 31.5])

Three read-only verbs narrow a ``Varied`` (they also accept a plain value, returning it
unchanged): ``labels`` lists the family, ``nominal`` is the central universe, and ``universe``
selects one by label. ``compile_ir`` deliberately *refuses* a bare ``Varied`` — you compile the
universes you name, not an ambiguous family.

Weight and shift variations register into an **event context** (``graphed.awkward``'s
``gnano.events``), and ``graphed.variations`` reports a context's registry as
``{name: {tag: (kind, value)}}``. The *kind* is a two-word vocabulary — ``"weight"`` for a
weight factor, ``"shift"`` for a collection shift — and numeric tags parse to an ordering value
(the σ handle for envelope plots) under both the canonical e-form (``5em1`` → ½) and the
datacard p-form (``2p5`` → 2½); a non-numeric tag carries ``None``::

    import awkward as ak
    import graphed.awkward as ga
    from graphed import variations
    from graphed.awkward import AwkwardBackend, from_awkward, gak

    events = ak.Array({
        "MET": ak.zip({"pt": [10.0, 20.0, 30.0]}),
        "Jet": ak.zip({"pt": ak.Array([[40.0, 25.0], [55.0], [30.0, 60.0, 20.0]])}),
    })
    s   = Session(AwkwardBackend())
    ev  = from_awkward(s, "events", events)
    ctx = ga.gnano.events(ev)
    w   = ev.MET.pt
    ctx = vary(ctx, "pu", w, is_weight=True,
               variations={"5em1": w, "m15em1": w * 1.1, "up": w * 1.3})
    jets = ctx.Jet
    ctx = vary(ctx, "jes",
               collections={"Jet": {"up": gak.with_field(jets, jets.pt * 1.05, "pt"),
                                    "down": gak.with_field(jets, jets.pt * 0.95, "pt")}})

    variations(ctx)["pu"]
    # -> {'5em1': ('weight', Fraction(1, 2)), 'm15em1': ('weight', Fraction(-3, 2)),
    #     'up': ('weight', None)}
    variations(ctx)["jes"]
    # -> {'up': ('shift', None), 'down': ('shift', None)}

**Sibling mode vs axis mode.** By default every universe is its own output — its own histogram,
its own column. When the sink is a histogram fill, the analyst can instead opt a *single* fill
into an **axis** that carries the universes as a ``StrCategory("variation")`` axis inside one
histogram (``h.fill(..., variation_axis=True)``); weight-label universes then collapse into an
evaluator-side loop rather than N separate fills. That machinery lives in ``graphed-histogram``
(see its design doc); from the frontend's side the two modes are interchangeable — the same
``labels``/``universe``/``nominal`` verbs read a result histogram's variation axis, and the
plan-level ``{output: [labels]}`` listing answers identically regardless of which mode produced
each output.

Varied preservation
~~~~~~~~~~~~~~~~~~~~

A preservation bundle over a **varied** value/weight reproduces *every* universe from one
bundle. ``build_bundle`` takes the ``value``/``weight``/``{name,bins,lo,hi}`` triple it already
accepted; hand it a ``Varied`` and ``reproduce`` returns ``{label: counts}`` instead of a bare
array, each universe bit-for-bit against build time. The manifest gains a per-label output map
(sorted, so ``canonical_bytes`` stays deterministic) and its ``format_version`` bumps to ``2``;
an **unvaried** bundle keeps today's singular shape and version ``1``. ``inspect`` lists the
labels without executing::

    from graphed.preserve import build_bundle, reproduce, inspect
    from graphed.awkward import gak

    value  = gak.sum(ev.Jet.pt, axis=1)
    weight = vary(ev.MET.pt, "sf", up=ev.MET.pt * 1.1, down=ev.MET.pt * 0.9)
    HIST   = {"name": "ht", "bins": 5, "lo": 0.0, "hi": 200.0}

    bundle = build_bundle(root, session=s, value=value, weight=weight,
                          datasets={"events": events}, histogram=HIST)
    bundle.manifest["format_version"]        # -> 2
    reproduce(bundle)                        # -> {'nominal': array(...), 'sf_down': ..., 'sf_up': ...}
    all(l in inspect(bundle) for l in labels(weight))   # -> True  (no execution)

Checkpoints and variation churn (honest limits)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Within one plan, checkpoint resume works per partition exactly as for an unvaried run — the
N-universe composite partial is the journal unit. **Across** plan revisions there are three ways
a variation edit invalidates the checkpoint cache, and their scopes differ:

* **Adding or removing a variation — unconditional.** A variation is sibling nodes in the IR
  (§1.2), so toggling one changes the serialized graph, and every ``task_id`` is a hash over that
  graph (``DurablePlan.task_id`` = ``sha256(domain, ir, process.identity(), partition)``). No
  journal survives it. This is the exemplar's own "``skip_obj_systematics``" switch: turning the
  expensive shift class on or off between runs rebuilds the IR and invalidates the whole cache.
* **Renaming a label — only by-value journals.** A pure rename leaves the IR byte-identical
  (labels are not in it), so §1.2's *no-recompute at the interning level* still holds. But if the
  journal's ``process`` embeds its worker closure **by value** (``OpSpec.from_callable`` on a
  non-importable function — an ``"opaque"`` spec whose ``identity()`` is the cloudpickle blob
  itself), the label *strings* travel inside that blob and the rename changes every ``task_id``.
* **The one-time field churn — only by-value journals, twice.** Landing the variation machinery
  added a field to the worker/artifact dataclasses (once at m48, once at m49); a dataclass field
  is in every pickled instance whatever its value, so any by-value journal's ``task_id`` churned
  once each time, unvaried programs included.

The documented checkpoint idiom is immune to the last two. It references worker functions **by
import path** — ``OpSpec.from_ref("myanalysis:hist_chunk")`` (see :doc:`../checkpoint/design`) —
whose ``identity()`` is ``b"ref\0" + ref`` and carries no closure state, so neither a rename nor a
field addition can perturb it. By-value journals arise only where a caller wrapped a closure with
``OpSpec.from_callable`` by hand. The scope is deliberately narrow, and the general fix
(stage-granular content addressing) is named Phase 2.


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
