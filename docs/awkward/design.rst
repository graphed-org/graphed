How graphed.awkward works
=========================

You have a NanoAOD file with four hundred branches and an analysis that uses six of them. You
want the type errors at the line you wrote, the systematic variations for free, and the run to
read six branches — not four hundred, and not six whole branches when three of them were only
counted.

That is what this page is about: how a deferred ragged analysis is typed, what it reads, what it
fuses, and what each of those buys you.

What your analysis actually reads
---------------------------------

Start with the smallest thing that shows the point — an analysis that counts jets and never
looks at one:

.. code-block:: python

    import awkward as ak
    from graphed import Session
    from graphed.awkward import AwkwardBackend, from_awkward, gak, project, project_buffers

    events = ak.Array({"Jet": [[{"pt": 50.0, "eta": 0.1}, {"pt": 30.0, "eta": 2.2}],
                               [],
                               [{"pt": 70.0, "eta": -0.5}]],
                       "MET": [12.0, 35.0, 8.0]})

    s = Session(AwkwardBackend())
    g = from_awkward(s, "events", events)

    njet = gak.num(g.Jet, axis=1)            # how many jets, not which jets

    print(sorted(project(njet).columns_for("events")))
    print(project_buffers(njet).buffers_for("events"))
    print(ak.Array(s.materialize(njet)).tolist())

Printed output:

.. code-block:: text

    []
    {'Jet': <BufferNeed.OFFSETS: 'offsets'>}
    [2, 0, 1]

``project`` answers at the granularity you are used to: *which columns does this result need?*
Here the answer is **none** — the multiplicity of a jagged column lives in its list
structure, not in any leaf. A column-level reader has to choose between over-reading a leaf it
does not need and reporting an empty set it cannot act on.

``project_buffers`` answers one level finer: inside a ragged column, which *pieces*. ``OFFSETS``
means the list structure alone; ``DATA`` means the leaf values. A reader that understands this
serves ``{'Jet': OFFSETS}`` from a counter branch in a ROOT file or an index column in an
RNTuple, and never touches the payload.

The same machinery is what keeps ordinary analyses cheap. In the selection example on
:doc:`index`, the result reports ``Jet.eta`` and ``MET`` and *not* ``Jet.pt``, even though the
code sliced whole ``Jet`` records — record-shaped syntax does not force a record-shaped read.

Both answers are computed by replaying your recorded operations with awkward tracing types
instead of computing values, and reporting what it touched — so no event data is read to work
out what event data is needed. They are also
correct whether or not the graph has been optimized: the replay runs the operations, and a fused
group contains them unchanged.

Why the types are right before a file is opened
-----------------------------------------------

A form here is a real awkward array whose buffers are length-typed placeholders — awkward's
typetracer. ``graphed.awkward`` does not implement type rules of its own; it runs the actual
awkward operation on tracer inputs. ``ak.num`` on a tracer gives a tracer of integer counts, a
jagged boolean ``getitem`` gives a tracer with option-and-list structure, and a four-vector
property derives through ``vector``'s own code. Type inference is awkward's semantics
version-for-version, because it *is* awkward running. When awkward changes, this changes with
it, at no cost to you.

The immediate payoff is that a mistyped expression fails where you wrote it:

.. code-block:: python

    import inspect

    import awkward as ak
    from graphed import GraphedTypeError, Session
    from graphed.awkward import AwkwardBackend, from_awkward, gak

    # gak carries awkward's own signatures, defaults included
    print(inspect.signature(gak.concatenate))
    print(inspect.signature(gak.sort))

    s = Session(AwkwardBackend())
    g = from_awkward(s, "events", ak.Array({"Jet": [[{"pt": 50.0}], []]}))

    try:
        g.Jet.energy          # no such field - and no file has been opened yet
    except GraphedTypeError as exc:
        print(exc)

Printed output — the third line names the absolute path of the file you ran, shortened here:

.. code-block:: text

    (arrays: 'Sequence[Array]', axis: 'int' = 0) -> 'Array'
    (arr: 'Array', axis: 'int' = -1, *, ascending: 'bool' = True, stable: 'bool' = True) -> 'Array'
    ill-typed op 'field' at /home/you/analysis.py:15: no field named 'energy'

Evaluation is the same code path with real arrays instead of tracers: each recorded operation
name maps to its ``ak.*`` or ufunc call, and the record-side and evaluate-side tables are built
from one set of shared constants, so an operation cannot mean one thing at record time and
another at run time.

Porting an analysis: what maps, what changes
--------------------------------------------

``gak`` carries ``ak``'s function names, and for a parameter the two share it carries ``ak``'s
own default. That matters more than it sounds: ``ak.concatenate`` defaults to ``axis=0``, so
``gak.concatenate`` does too, and for the reducers and the structure functions that agreement is
checked against the installed awkward rather than transcribed by hand — when awkward moves a
default, the check says so. Your muscle memory for what an argument *means* transfers intact.

What does not transfer is argument *position*.

**Optional arguments are keyword-only.** Past the arrays, every parameter is passed by name:
``gak.zip({"pt": pt}, depth_limit=1)``, ``gak.cartesian([a, b], axis=1)``,
``gak.isclose(x, y, rtol=1e-4)``, ``gak.nan_to_num(x, nan=0.0)``. ``ak`` takes those
positionally; here a positional argument in that slot raises ``TypeError`` rather than binding
silently to a neighbouring parameter.

**The statistics functions lead with** ``axis``. ``gak.mean``, ``gak.std``, ``gak.var`` and
``gak.moment`` put ``axis`` where ``ak`` puts ``weight``, and take ``weight=`` and ``ddof=`` by
name — so ``ak.mean(x, w)`` ports to ``gak.mean(x, weight=w)``, never to ``gak.mean(x, w)``.
``gak.softmax`` defaults to ``axis=1``, per event, where ``ak.softmax`` defaults to ``axis=-1``.

**Free functions, not methods.** You hold graphed's plain deferred ``Array`` — field access,
operators, ufuncs, ``getitem`` — and everything ragged-specific comes from the module:
``gak.num``, ``gak.flatten``, ``gak.combinations``, ``gak.cartesian``, ``gak.zip``, ``gak.sort``,
``gak.with_name``, ``gak.where``, and the rest of the working set. ``gak.with_field`` wants the
field name every time — ``gak.with_field(jets, jets.pt * 1.05, "pt")`` — where
``ak.with_field`` lets you leave it out.

**No** ``behavior=`` **per call.** Behaviors belong to the session, not to a call site — see the
next section.

**No** ``highlevel=`` **or** ``attrs=``. Both control how an array is constructed eagerly, and
nothing here is constructed eagerly. They are absent rather than ignored, so a typo is an error
instead of a silent no-op.

**Weights are inputs, not parameters.** ``gak.moment`` and its neighbours take ``weight=`` as
another deferred array, which is what lets a weight be a recorded quantity — varied, reused,
projected — rather than a value baked into the graph. ``gak.corr``, ``gak.covar`` and
``gak.linear_fit`` are the exception: they are unweighted.

Four-vectors and other behaviors
--------------------------------

Register a behavior dict once on the backend and behavior *properties* work through plain
attribute access everywhere downstream:

.. code-block:: python

    import awkward as ak
    import vector
    from graphed import Session
    from graphed.awkward import AwkwardBackend, from_awkward, gak, project

    vector.register_awkward()

    s = Session(AwkwardBackend(behavior=vector.backends.awkward.behavior))
    g = from_awkward(s, "events", ak.Array({"Jet": [
        [{"pt": 50.0, "eta": 0.1, "phi": 0.3, "mass": 5.0},
         {"pt": 30.0, "eta": 2.2, "phi": -1.1, "mass": 4.0}],
        [{"pt": 70.0, "eta": -0.5, "phi": 2.0, "mass": 6.0}],
    ]}))

    jets = gak.with_name(g.Jet, "Momentum4D")

    print(ak.to_list(s.materialize(jets.px)))
    print(sorted(project(jets.px).columns_for("events")))
    print(sorted(project(gak.sum(jets.pt, axis=1)).columns_for("events")))

Printed output:

.. code-block:: text

    [[47.7668244562803, 13.60788364276732], [-29.130278558299967]]
    ['Jet.phi', 'Jet.pt']
    ['Jet.pt']

Projection sees straight through the property. ``.px`` of a pt/eta/phi/mass vector is
``pt·cos(phi)``, so it reads ``pt`` and ``phi`` and nothing else; ``.pt`` reads ``pt`` alone.
Four-vector convenience costs you no extra bytes off disk.

Two consequences follow from behavior dicts holding lambdas, which do not pickle to a worker
process. First, a worker is given the backend by *import reference*, not by value — which is why
``to_parquet`` takes ``behavior="vector.backends.awkward:behavior"`` as well as a dict. Second,
a worker that was built without the behaviors fails loudly on the property rather than quietly
computing something else.

What a per-event reduction costs, and what a cross-event one does
-----------------------------------------------------------------

Whether a reduction interrupts the flow of work is decided by its axis, not its name. A
reduction over the event axis — ``axis=None`` or ``axis=0`` — crosses partitions, so it becomes a
combine step that a runner performs as a tree over partial results. A reduction over an inner
axis — ``axis=1`` and deeper, per-event work — stays inside one partition and rides along in the
same pass as the arithmetic around it. Scans always ride along.

.. code-block:: python

    import collections

    import awkward as ak
    from graphed import Session, compile_ir
    from graphed.awkward import AwkwardBackend, from_awkward, gak
    from graphed.core import GraphStore

    events = ak.Array({"Jet": [[{"pt": 50.0, "eta": 0.1}, {"pt": 30.0, "eta": 2.2}],
                               [],
                               [{"pt": 70.0, "eta": -0.5}]],
                       "MET": [12.0, 35.0, 8.0]})


    def node_kinds(session, out):
        store = GraphStore.deserialize(bytes(compile_ir(session, out).ir))
        return collections.Counter(n["kind"] for n in store.nodes())


    s = Session(AwkwardBackend())
    g = from_awkward(s, "events", events)

    jets = g.Jet[abs(g.Jet.eta) < 2.4]
    lead = gak.max(jets.pt, axis=1)                  # per-event: inside the pass
    ht = gak.sum(jets.pt, axis=1)                    # per-event: inside the pass
    njet = gak.num(jets, axis=1)                     # per-event: inside the pass
    per_event = (lead > 40.0) & (ht > 60.0) & (njet >= 1)
    print(node_kinds(s, g.MET[per_event]))

    total = gak.sum(g.MET[per_event], axis=None)     # across every event: a combine step
    print(node_kinds(s, total))
    print(float(s.materialize(total)))

Printed output:

.. code-block:: text

    Counter({'stage': 4, 'source': 1})
    Counter({'stage': 4, 'source': 1, 'reduction': 1})
    20.0

Three reductions, a jagged cut, four comparisons and two boolean combinations, and the compiled
graph is four groups of fused work with nothing to synchronise on. Adding the cross-event sum
adds exactly one combine step. This is why an analysis dense in per-event ``sum``/``max``/``num``
calls — which is to say, every HEP analysis — does not turn into a graph with a boundary every
few lines.

Joining two datasets on run/lumi/event
--------------------------------------

``gak.join`` matches rows across two recorded sources on a shared key, with the relational
semantics of ``pandas.merge``: a probe row with *k* matches on the build side produces *k* output
rows, and ``how`` takes ``inner``, ``left``, ``right`` or ``outer``.

.. code-block:: python

    import awkward as ak
    from graphed import Session
    from graphed.awkward import AwkwardBackend, from_awkward, gak

    s = Session(AwkwardBackend())
    events = from_awkward(s, "events", ak.Array([
        {"event": 1, "MET": 12.0}, {"event": 2, "MET": 35.0}, {"event": 3, "MET": 8.0}]))
    trig = from_awkward(s, "trig", ak.Array([
        {"event": 1, "path": 0}, {"event": 1, "path": 1}, {"event": 3, "path": 2}]))

    flat = gak.join(events, trig, on=["event"], how="inner")
    print(ak.to_list(s.materialize(gak.without_field(flat, "__joinkey__"))))

    grouped = gak.join(events, trig, on=["event"], how="inner", grouped=True)
    print(ak.to_list(s.materialize(gak.without_field(grouped, "__joinkey__"))))

Printed output:

.. code-block:: text

    [{'event': 1, 'MET': 12.0, 'path': 0}, {'event': 1, 'MET': 12.0, 'path': 1}, {'event': 3, 'MET': 8.0, 'path': 2}]
    [[{'event': 1, 'MET': 12.0, 'path': 0}, {'event': 1, 'MET': 12.0, 'path': 1}], [{'event': 3, 'MET': 8.0, 'path': 2}]]

``grouped=True`` is the ragged convenience the flat form cannot express: one sublist per matching
left-hand row, holding that row's matches. Put your events on the left and a one-to-many table —
trigger paths, generator-level matches — on the right, and you get back the shape you would have
written by hand. It exists only here; a rectilinear backend has nowhere to put the sublists.

The output carries an extra ``__joinkey__`` column: the packed integer the match ran on. Drop it
with ``gak.without_field`` when you are done, as above.

Underneath, a join packs the key fields into one unsigned 64-bit column, exchanges both sides so
that equal keys land on the same partition, and then matches locally. The exchange route is a
hash of the key's big-endian bytes rather than Python's ``hash()``, so a key lands on the same
destination in every worker process and two runs of the same plan move the same rows the same
way. The moving itself is done with awkward's own primitives: blocks are split, concatenated and
serialized through ``ak.to_buffers``/``ak.from_buffers``, so jaggedness survives the trip across
the network intact rather than being flattened and rebuilt.

``graphed.repartition`` gives you that exchange on its own — by key, by target partition count,
or by target bytes — when you want to reshape partitioning without a join, and
``graphed.shuffle_plan`` turns such a graph into a plan a cluster runner can execute.

Corrections and models are recorded, not inlined
------------------------------------------------

Some steps are calls out to something graphed does not look inside: a correctionlib scale
factor, an ONNX model, a histogram fill. Those record as external calls — the runner performs
them, the graph just knows they happen, what goes in, and what comes out.

What makes them reproducible is that an external call is identified by the *content* of its
payload, not by a path on your filesystem:

.. code-block:: python

    import json

    import awkward as ak
    import correctionlib
    import correctionlib.schemav2 as cs
    from graphed import Session
    from graphed.awkward import AwkwardBackend, from_awkward, gak, payloads

    # a two-bin pt scale factor, in correctionlib's own JSON - no graphed format invented
    payload = cs.CorrectionSet(
        schema_version=2,
        corrections=[cs.Correction(
            name="jet_sf",
            version=1,
            inputs=[cs.Variable(name="pt", type="real")],
            output=cs.Variable(name="sf", type="real"),
            data=cs.Binning(nodetype="binning", input="pt", edges=[0.0, 40.0, 1000.0],
                            content=[0.95, 1.05], flow="clamp"),
        )],
    ).model_dump_json(exclude_unset=True).encode()

    evaluator = correctionlib.CorrectionSet.from_string(payload.decode())["jet_sf"]

    s = Session(AwkwardBackend())
    g = from_awkward(s, "events", ak.Array({"Jet": [[{"pt": 50.0}, {"pt": 30.0}], [{"pt": 70.0}]]}))

    sf = gak.apply_correction(payload, "jet_sf", [gak.flatten(g.Jet.pt)],
                              lambda pt: evaluator.evaluate(pt), args=["$0"])
    print(ak.to_list(s.materialize(sf)))

    # reformatting the JSON does not change what the correction IS
    reformatted = json.dumps(json.loads(payload), indent=4).encode()
    print(len(payload), len(reformatted))
    print(payloads.correctionlib_contents_hash(payload)
          == payloads.correctionlib_contents_hash(reformatted))

This example needs ``pip install "graphed[preserve]"`` for ``correctionlib``. Printed output:

.. code-block:: text

    [1.05, 0.95, 1.05]
    248 736
    True

The two payloads differ by 488 bytes and hash the same, because the hash is taken over the
correction set's canonical contents rather than its file bytes. Pretty-print your correction
JSON, or move it to another site, and your cached results and preserved bundles stay valid;
change a bin edge and they do not. ``gak.onnx_inference`` does the same for models, hashing the
weights and the graph structure rather than the ``.onnx`` file.

The payload itself never rides in the graph. The graph carries the hash; the multi-megabyte
correction set or model lives in a content-addressed store, and a run resolves it by hash — and
says so loudly if it cannot.

Reading and writing parquet
---------------------------

``from_parquet`` records a deferred source over a file, directory, glob or list. Its type comes
from the parquet schema; ``to_parquet`` writes one part per partition, running the same plan a
cluster would run.

.. code-block:: python

    import tempfile

    import awkward as ak
    from graphed import Session
    from graphed.awkward import AwkwardBackend, from_parquet, gak, project_buffers, to_parquet

    with tempfile.TemporaryDirectory() as tmp:
        ak.to_parquet(ak.Array({
            "Jet_pt":  [[50.0, 30.0], [], [70.0]],
            "Jet_eta": [[0.1, 2.2], [], [-0.5]],
            "Jet_phi": [[0.3, -1.1], [], [2.0]],
            "MET":     [12.0, 35.0, 8.0],
        }), f"{tmp}/events.parquet")

        s = Session(AwkwardBackend())
        g = from_parquet(s, "events", f"{tmp}/events.parquet")

        keep = gak.num(g.Jet_pt[g.Jet_pt > 40.0], axis=1) >= 1
        out = g.MET[keep]

        print(s.form(out).describe())
        print(project_buffers(out).buffers_for("events"))

        parts = to_parquet(out, f"{tmp}/skim")
        print(len(parts))
        print(ak.to_list(ak.from_parquet(parts[0])))

This example needs ``pip install "graphed[awkward,parquet]"``. Printed output:

.. code-block:: text

    ## * float64
    {'Jet_pt': <BufferNeed.DATA: 'data'>, 'MET': <BufferNeed.DATA: 'data'>}
    1
    [{'data': 12.0}, {'data': 8.0}]

``Jet_eta`` and ``Jet_phi`` are on disk and never read. A bare array is written under a single
``data`` column; pass ``column=`` to name it something else, and write a record if you want
named fields.

Three details worth knowing before you point this at a real dataset:

*You can skip the metadata pass entirely.* ``steps_per_file`` sets how finely each file is split;
``open_files=False`` makes the split blind, so no file is opened to plan the work and the first
task starts without waiting on a scan of the whole dataset. Each task then resolves its own
partition when it reads.

*Any partitioned source writes through the same entry point.* ``to_parquet`` dispatches on the
protocol a source implements rather than on the source's type, so a parquet dataset, a ROOT
reader's source, or one you wrote yourself all write through it — without ever running the
source's whole-dataset loader.

*Writing reads more than the final result does.* A write replays every recorded operation,
including the legs of a ``zip`` whose values the final result never used, so the per-task read
list is the union of source fields the graph *mentions*, refined at the leaf level by the buffer
view. The buffer projection answers "what does this result need", which is a smaller question
than "what must exist to replay this graph".

``compute=False`` returns the plan instead of running it, so you can hand the identical write to
a cluster runner rather than to the in-process one.

One skim file that holds every systematic universe
--------------------------------------------------

Writing a skim with systematics usually means writing it once per universe. Here it is one file
and one pass: ``to_parquet(record, select=…)`` stores every universe's post-selection data
together, and ``read_varied`` gives them back as ``{label: array}``.

.. code-block:: python

    import tempfile

    import awkward as ak
    import graphed
    import graphed.awkward as ga
    from graphed import Session
    from graphed.awkward import AwkwardBackend, from_awkward, gak

    session = Session(AwkwardBackend())
    events = ga.gnano.events(from_awkward(session, "events", ak.Array(
        [{"Jet": [{"pt": 40.0}, {"pt": 12.0}]}, {"Jet": [{"pt": 55.0}]}, {"Jet": [{"pt": 8.0}]}])))

    jets = events.Jet
    up = gak.with_field(jets, jets.pt * 1.05, "pt")
    down = gak.with_field(jets, jets.pt * 0.95, "pt")
    ctx = graphed.vary(events, "jes", collections={"Jet": {"up": up, "down": down}})
    vjets = ctx.Jet

    evt = gak.any(vjets.pt > 30.0, axis=1)   # event mask - migrates with the shift
    jet = vjets.pt > 25.0                    # per-jet mask

    with tempfile.TemporaryDirectory() as tmp:
        paths = ga.to_parquet(vjets, f"{tmp}/jes_skim", select={0: evt, 1: jet})
        universes = ga.read_varied(paths[0])
        print(list(universes))
        for label, arr in universes.items():
            print(label, ak.to_list(arr.pt))

Printed output:

.. code-block:: text

    ['nominal', 'jes_up', 'jes_down']
    nominal [[40.0], [55.0]]
    jes_up [[42.0], [57.75]]
    jes_down [[38.0], [52.25]]

``select=`` is the selection, given per level: a single row mask, or a mapping keyed by the
record's own depth (``0`` for the event mask) and by ``(field, depth)`` for a cut scoped to one
collection. ``to_parquet`` still returns an ordinary list of part paths; leave ``select=`` out and
you get a plain skim, with none of the extra columns below.

**What the file holds.** One pass over one compiled graph writes: the superset of rows — the OR
over every universe's event mask, so no universe loses a row it selected; the nominal values on
those rows; for each leaf that a variation actually changes, a same-dtype XOR delta against
nominal; each universe's per-level mask as a bit-packed column; and a small JSON manifest in the
parquet key-value metadata naming what each stored column is.

A leaf that is equal in every universe contributes no delta column at all, which is why this is
cheaper than *N* skims rather than merely tidier: only the values a systematic actually moves are
stored twice. Reading is the exact inverse — nominal XOR delta rebuilds a universe's values,
then the masks select its rows and objects — so ``read_varied(path)[label]`` is bit-for-bit the
data that universe would have produced in memory.

Two things are refused rather than guessed. A field-name collision between a nested ``Jet.pt``
and a flat ``Jet_pt`` in the same record is an error, because the on-disk names flatten. And a
variation that changes a field's *multiplicity* — different numbers of objects per universe — is
refused at run time, because a same-shaped XOR delta cannot represent it. Write those universes
as separate skims.

The manifest is serialized with sorted keys and an explicit level order, so the same analysis
produces the same file on every run and on every machine.

Not supported yet
-----------------

- **Calling** ``ak.*`` **on a deferred array** (``ak.num(g.Jet)`` instead of
  ``gak.num(g.Jet)``). Use the ``gak`` module; it is the supported surface.
- **Behavior methods that take arguments** (``a.deltaR(b)``). Properties record; method calls do
  not. Write the formula, or reach for the ``vector`` components you need and combine them with
  ``gak``.
- **A handful of parameter tails**: ``zip``'s ``right_broadcast`` and
  ``optiontype_outside_record``, ``broadcast_arrays``' rule controls, ``mergebool`` on
  ``concatenate``/``where``, ``including_unknown`` on the ``*_like`` constructors and
  ``values_astype``, ``nan_to_num``'s ``copy``, ``unzip(how=)``, and the weighted forms of
  ``corr``, ``covar`` and ``linear_fit``. ``inspect.signature(gak.f)`` is the answer for any
  function you are unsure of; it reports what ``gak`` actually accepts.
- **Reading a counter branch with no leaf at all.** An offsets-only need is served today by
  reading one cheap leaf under the path and taking its structure. Correct and already far
  cheaper than reading the payload, but not yet the last possible byte.

See :doc:`improvements` for limitations in the behaviour of what does work.
