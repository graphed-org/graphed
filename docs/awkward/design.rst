How graphed-awkward works
=========================

``graphed-awkward`` is the **ragged backend** — the one HEP analyses actually live on. It
teaches the graphed frontend awkward-array's type system by *reusing awkward's own typetracer*
(never reimplementing type inference), exposes a deferred ragged-array library whose interface
mirrors ``ak.*`` function-for-function and default-for-default, and contributes the two pieces
of machinery unique to ragged data: buffer-granular projection and the structural reduction
rule that decides what an executor can fuse.

.. contents::
   :local:
   :depth: 2


The backend in one example
--------------------------

::

    import awkward as ak
    from graphed import Session
    from graphed_awkward import AwkwardBackend, from_awkward, gak, project, project_buffers

    events = ak.Array({"Jet": [[{"pt": 50.0, "eta": 0.1}, {"pt": 30.0, "eta": 2.2}],
                               [],
                               [{"pt": 70.0, "eta": -0.5}]],
                       "MET": [12.0, 35.0, 8.0]})

    s    = Session(AwkwardBackend())
    g    = from_awkward(s, "events", events)
    good = g.Jet[abs(g.Jet.eta) < 1.0]         # jagged boolean mask — recorded, not computed
    keep = gak.num(good, axis=1) >= 1          # per-event multiplicity
    out  = g.MET[keep]

    s.form(out).describe()                      # '## * float64' (event-axis float)
    ak.Array(s.materialize(out)).tolist()       # [12.0, 8.0]

    project(out).columns_for("events")          # frozenset({'MET', 'Jet.eta'})
    project_buffers(out).buffers_for("events")  # {'Jet.eta': DATA, 'MET': DATA}

Note what the projections say: this result needs ``Jet.eta`` and ``MET`` — and **not**
``Jet.pt``, even though the analysis sliced whole-Jet records. That precision is the point of
this backend's projection machinery.


Forms via the typetracer — reuse, don't reimplement
---------------------------------------------------

A form here is a *typetracer array*: a real awkward array whose buffers are length-typed
placeholders. ``op_form`` evaluates the actual awkward operation on tracer inputs — ``ak.num``
on a tracer produces a tracer of integer counts, a jagged boolean getitem produces a tracer with
option/jagged structure, a behavior property derives through vector's own code — so type
inference is **exactly** awkward's semantics, version-for-version, because it *is* awkward
executing. The backend never encodes typing rules of its own; when awkward changes, the forms
change with it.

The same trick caught a class of bug early and keeps catching it: ill-typed expressions (a
missing field, an axis out of range) raise inside the tracer evaluation at the recording line,
wrapped with the user's source frame.

Evaluation is the same code path with real arrays: ``eval_stage`` looks the op name up in
tables that pair every recorded op with its ``ak.*``/ufunc call. The record-side and eval-side
tables are built from shared constants so they cannot drift apart.


``gak``: the function idiom and the parity contract
---------------------------------------------------

This backend deliberately supplies **no enriched proxy**. Users hold the frontend's plain
``Array`` (field access, operators, ufuncs, getitem) and reach everything ragged-specific
through free functions: ``gak.num``, ``gak.flatten``, ``gak.combinations``, ``gak.cartesian``,
``gak.zip``, ``gak.sort``, ``gak.with_name``, ``gak.where`` — the working set of a real
analysis (the eight ADL benchmark queries run on it end to end).

The contract is **interface parity with awkward**: every ``gak`` function that mirrors an
``ak.*`` operation surfaces the same parameters and the *same defaults* — and that is enforced
by an inspection-based anti-drift test comparing signatures against installed awkward across
the whole surface, because defaults are interface too (``ak.concatenate`` defaults to axis 0;
so does ``gak.concatenate``). Weighted moments take their weight as a second *graph input*,
never a parameter. Three things are deliberately absent and are not gaps: ``highlevel`` and
``attrs`` (eager-construction concerns with no recorded meaning) and ``behavior`` as a per-call
kwarg — behaviors are backend-owned.

Behaviors (vector, and friends)
-------------------------------

``AwkwardBackend(behavior=...)`` registers an awkward behavior dict for the whole session:
``gak.zip(..., with_name="Momentum4D")`` then makes records whose *properties* (``.pt``,
``.mass``) record through plain attribute access — the typetracer derives them, and projection
sees exactly the buffers a property touches (``.pt`` of a pt/eta/phi/mass vector reads only
``pt``; ``.px`` reads ``pt`` and ``phi``). Two operational rules follow from behavior dicts
containing lambdas: they never pickle to workers — executors receive backends **by import
reference** — and losing behaviors must be loud (a worker built without them fails on the
property; it never silently computes something else).

The structural reduction rule
-----------------------------

Which recorded operations are *stage boundaries*? The rule is structural, not name-based:
a reduction over the **event axis** (``axis=None`` or ``axis=0``) crosses partitions, so it is
a boundary the executor tree-reduces; a reduction over an **inner axis** (``axis=1`` and
deeper, per-event work) is partition-local and **fusible** — it rides inside a stage like any
elementwise op. Scans are always fusible. The practical consequence is large: an analysis full
of per-event ``sum``/``max``/``num`` calls still reduces to a handful of stages, because none
of those are boundaries.

Projection: columns, and then buffers
-------------------------------------

``project`` walks the graph with a **reporting typetracer** — awkward's tracer instrumented to
record which buffers each operation touches — giving the column view. ``project_buffers``
refines it to the buffer view: per column path, ``DATA`` (leaf values needed) or ``OFFSETS``
(only list structure needed). The distinction is unique to ragged data and load-bearing:
``gak.num(g.Jet, axis=1)`` touches *no leaf data at all*, only Jet's offsets — a column view
must either over-read a leaf or under-specify; the buffer view says precisely "offsets of
Jet". Writers translate an offsets-only need into the cheapest carrier the format allows (any
single leaf under the path, for parquet/ROOT).

One hard-won rule is documented here because consumers keep needing it: **read lists for
compiled-IR evaluation are syntactic, not buffer-projected** — evaluation replays every
recorded node (a zip's untouched legs included), so the read list is the union of source
fields the graph mentions, leaf-refined by the buffer view. The buffer projection is the
answer to "what data does this *result* need", not "what must exist to *replay* the graph".

Parquet I/O
-----------

``from_parquet`` records a deferred dataset source: the file list is part of the source's
identity; partitions are blind; the form comes from the parquet schema via
``ak.from_arrow_schema`` with ``Form.select_columns`` projecting it. ``to_parquet`` is the
generic writer over the shared :mod:`graphed.write` base: it dispatches on the
``PartitionedSource`` protocol (so *any* partitioned source — parquet datasets, a ROOT reader's
source — writes through this one entry point, without its whole-dataset loader ever running),
evaluates the compiled IR per partition with the merged syntactic+buffer read list, and writes
one part per partition with worker-derived names.

Externals: corrections and models
---------------------------------

``gak.apply_correction`` (correctionlib) and ``gak.onnx_inference`` record External nodes whose
``PayloadDescriptor`` carries the payload's *content hash* — the analysis is durable while the
multi-megabyte payload lives in a content-addressed store. Evaluation resolves evaluators by
that hash, loudly. (Histogram fills are the third member of this family, contributed by
``graphed-histogram`` through the frontend's descriptor-override seam — this backend knows
nothing about them, by design.)


Phase 2 (deliberately not built)
--------------------------------

* **``__awkward_function__``-style dispatch** (calling ``ak.num(g_array)`` directly instead of
  ``gak.num``): largely syntactic sugar; the function module is the supported surface.
* **Behavior methods with arguments** (``a.deltaR(b)``): properties record; method calls do
  not — analyses write the explicit formula.
* **Niche parameter tails**: ``zip``'s ``right_broadcast``/``optiontype_outside_record``,
  ``broadcast_arrays``' rule controls, ``concatenate``/``where``'s ``mergebool``,
  ``including_unknown``, ``unzip(how=)`` — tracked, deliberately deferred.
* **Form reconstruction for counter-only reads**: an offsets-only need currently reads a
  carrier leaf; reconstructing the jagged form from a counter branch alone (no leaf) would
  shave the last unnecessary bytes.

See :doc:`improvements` for the live tracked list.
