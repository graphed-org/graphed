How graphed works
=================

You write the analysis you always write: fields, cuts, sums, a histogram fill. None of it runs as
you write it. Each operation becomes a node in a graph carrying the type of its result and the
line of your file that made it; when you ask for a value, what executes is the *reduced* graph —
duplicate work collapsed, unused branches dropped, the survivors fused into a few passes over the
data.

This page is how that works and how to drive it: why the graph stays small while you build it,
how a systematic variation rides through the whole analysis, what actually gets read off disk,
and how a recorded analysis becomes a plan a cluster can run.


Your code, kept instead of run
------------------------------

.. code-block:: python

    import numpy as np
    from graphed import Session, compile_ir, evaluate_ir
    from graphed.numpy import NumpyBackend, from_array

    s = Session(NumpyBackend())
    pt = from_array(s, "pt", np.arange(6.0))
    sel = (pt * 2.0 + 1.0)[pt > 2.0]        # four operations recorded, nothing computed

    p = s.provenance(sel)
    print(s.form(sel).describe())
    print(f"{p.lineno}: {p.source}")
    print(s.materialize(sel))

    compiled = compile_ir(s, sel)           # reduce the graph for this one output
    print(evaluate_ir(compiled, NumpyBackend(), {"pt": np.arange(6.0)}))

Prints::

    vector[float64]
    7: (pt * 2.0 + 1.0)[pt > 2.0]
    [ 7.  9. 11.]
    [array([ 7.,  9., 11.])]

A :class:`~graphed.Session` owns the graph plus the three side tables that make the rest of this
page possible:

* the **form** of every node — the type and shape of its result, computed from metadata with no
  data touched (awkward calls this a form; ``graphed.numpy`` reports dtype and shape the same
  way);
* the **provenance** of every node — file, line, enclosing function, and the exact
  sub-expression text, captured as you record;
* the **sources** — what each input name will be bound to at run time.

What you hold is a :class:`~graphed.Array`: a proxy carrying a session and a node id, nothing
else. It implements the surface every deferred array shares — arithmetic and comparison
operators, ``__array_ufunc__`` (so ``np.sqrt(x)`` records instead of computing), boolean, slice,
integer and field-list indexing. Anything idiomatic to *one* array library lives in that
library's package: ``graphed.numpy`` hands you a richer proxy with ``.shape``, ``.sum()`` and
``__array_function__``; ``graphed.awkward`` keeps the plain proxy and exposes its idiom as free
functions, exactly as ``ak.*`` does. One library's conventions never leak into the other's.

Two ways to get a value out. ``materialize`` walks the graph in this process and is right for
small, local data and for poking at things interactively. ``compile_ir`` + ``evaluate_ir`` is the
path everything else takes: reduce once, then evaluate the reduced node list wherever the data
is. ``evaluate_ir`` returns one entry per requested output.


Why the graph stays small while you build it
--------------------------------------------

The failure mode this design exists to avoid is the one you have already met: a graph that grows
for an hour, then an optimizer whose own runtime dominates the analysis. So the collapsing happens
on the way in, not at the end.

.. code-block:: python

    import numpy as np
    from graphed import Session, compile_ir
    from graphed.core import GraphStore
    from graphed.numpy import NumpyBackend, from_array

    s = Session(NumpyBackend())
    pt = from_array(s, "pt", np.arange(6.0))

    def tight(a):                 # a helper you call from two places
        return a[a > 2.0]

    first, second = tight(pt), tight(pt)
    print("same node:", first.node_id == second.node_id)

    scaled = pt * 1.0             # a no-op left behind by somebody's helper
    unused = pt * 99.0            # nothing you asked for depends on this
    print("recorded:", s.node_count())

    compiled = compile_ir(s, first, scaled)
    for node in GraphStore.deserialize(bytes(compiled.ir)).nodes():
        print(node["kind"], [m["name"] for m in node.get("members", [])])

Prints::

    same node: True
    recorded: 5
    source []
    stage ['gt', 'getitem']

Read that from the bottom. Five nodes were recorded and two survive: the source, and one fused
run of operations. The cut written twice was one node the second time you wrote it — recording
the same expression on the same inputs returns the node that already exists (*interning*), so
a session can be long-lived and exploratory and still hold the set of distinct computations
rather than the history of your statements. ``pt * 1.0`` collapsed into ``pt`` itself, so
``scaled`` is the source node. ``pt * 99.0`` fed nothing you asked for and is gone. The two
operations that remained were fused into one *stage* — a run of operations executed together as a
single pass over the data — so the interpreter is entered once for the group instead of once per
operation.

That collapsing is not a fixed list of rewrites applied in a fixed order. The optimizer holds the
equivalent forms of your expression together and picks the cheapest (equality saturation over
e-graphs, in a compiled Rust extension). A stage ends where the data has to change shape or leave:
reading a source, a reduction, a repartition, a join, or a call out to something
graphed does not look inside — a correction, an ML model, a histogram fill.

Reduction can also run continuously rather than at compile time. ``Session(backend,
incremental=True)`` maintains the reduced form as you build, so the un-reduced graph never
exists at all; both paths produce byte-identical output for the same analysis, so the choice
costs you nothing in reproducibility.


Where a mistake shows up
------------------------

At the line you wrote it, before any file is opened:

.. code-block:: python

    import awkward as ak
    from graphed import GraphedTypeError, Session
    from graphed.awkward import AwkwardBackend, from_awkward

    s = Session(AwkwardBackend())
    ev = from_awkward(s, "events", ak.Array({"Jet": ak.zip({"pt": [[40.0, 25.0], [55.0]]})}))

    try:
        ht = ev.Jet.et                       # there is no `et` field — only `pt`
    except GraphedTypeError as exc:
        print(exc.detail)
        print(f"line {exc.provenance.lineno}: {exc.provenance.source}")

Prints::

    no field named 'et'
    line 9: ev.Jet.et

Type inference runs on the recorded forms, so a missing field, a non-boolean mask or a shape
mismatch is a :class:`~graphed.GraphedTypeError` raised while you are still recording — not an
exception from worker 47 four hours in. The exception carries the operation, the detail, and the
provenance record shown above.

Failures that can only happen at run time are the job of ``graphed.debug``, which re-raises them
on your machine pointing at your analysis line; see :doc:`../debug/design`. The frontend's
contribution is that the provenance is there for every node, cheaply, from the moment it was
recorded.


Vary once, get every universe
-----------------------------

Each label **names a point** in nuisance space. A label registered without ``points=`` carries the
default point ``{name: tag}`` and so differs from ``nominal`` on exactly one axis — a jet-energy
scale shifted, a scale factor scaled up and down; that axis-aligned set is the default and is what
most analyses need. A universe displaced on two or more axes at once is registered explicitly with
``points=`` (below) and is never produced implicitly. Either way you write the analysis once:
``graphed.vary`` attaches the knob, and everything downstream carries the whole family.

.. code-block:: python

    import numpy as np
    from graphed import Session, compile_ir, labels, nominal, universe, vary
    from graphed.numpy import NumpyBackend, from_array

    s = Session(NumpyBackend())
    pt = from_array(s, "pt", np.array([10.0, 20.0, 30.0]))

    jes = vary(pt, "jes", up=pt * 1.05, down=pt * 0.95)

    print(labels(jes))
    print(s.materialize(nominal(jes)))
    print(s.materialize(universe(jes, "jes_up")))

    try:
        compile_ir(s, jes)
    except Exception as exc:
        print(type(exc).__name__, exc)

Prints::

    ('nominal', 'jes_up', 'jes_down')
    [10. 20. 30.]
    [10.5 21.  31.5]
    GraphedError graphed.compile_ir does not accept a Varied output; pass one universe with graphed.universe(v, label), or build the varied plan through the histogram group API

``vary`` returns a :class:`~graphed.Varied`: a handle that behaves like the value you varied but
carries a labelled family — ``"nominal"`` plus one universe per tag. It never mutates anything, so
``pt`` stays exactly as valid as it was. Three read-only verbs narrow a family, and each also
accepts a plain value and returns it unchanged, so helper code does not need to care whether its
input is varied: ``labels`` lists the family, ``nominal`` is the central universe, and
``universe`` picks one by label. Compiling refuses an ambiguous family on purpose — you compile
the universes you name.

The labels live in the frontend only. Each universe lowers to an ordinary marked output, so the
optimizer, the plan format and the executor never learn the word "variation" — and interning
still shares whatever the universes have in common, which is usually almost everything.

Variations that ride an event context
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Varying one array by hand is fine for one knob. A real analysis varies a whole collection, or a
weight that every histogram must pick up, and does it once at the top. That is what an **event
context** is for: wrap your event record in one, read your collections through it, and the
context remembers which variations are in play and which weight is ambient. ``graphed.awkward``
supplies the NanoEvents-flavoured constructor, ``gnano.events``.

.. code-block:: python

    import awkward as ak
    import graphed.awkward as ga
    from graphed import Session, labels, universe, variations, vary, weight
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
               variations={"1p0": w * 1.1, "m1p0": w * 0.9, "extreme": w * 1.3})

    jets = ctx.Jet
    ctx  = vary(ctx, "jes", collections={"Jet": {
        "up":   gak.with_field(jets, jets.pt * 1.05, "pt"),
        "down": gak.with_field(jets, jets.pt * 0.95, "pt")}})

    ht  = gak.sum(ctx.Jet.pt, axis=1)   # written once, computed in every jet-scale universe
    sel = ctx[ht > 70.0]                # the cut follows the variations too

    print(labels(ht))
    print(labels(sel.Jet.pt))
    print(labels(weight(ctx)))
    print(variations(ctx)["jes"])
    print(variations(ctx)["pu"])
    print(s.materialize(universe(ht, "jes_up")))

Prints::

    ('nominal', 'jes_up', 'jes_down')
    ('nominal', 'jes_up', 'jes_down')
    ('nominal', 'pu_1p0', 'pu_m1p0', 'pu_extreme')
    {'up': ('shift', None), 'down': ('shift', None)}
    {'1p0': ('weight', Fraction(1, 1)), 'm1p0': ('weight', Fraction(-1, 1)), 'extreme': ('weight', None)}
    [68.2, 57.8, 116]

``ht`` was written once and is varied because it reads a varied collection through the context.
The cut is written once and applies inside every universe. The pile-up weight does not appear in
``ht``'s labels because a weight is not part of the value — it is applied where the fill happens,
automatically, to every universe.

Two kinds of knob, and the distinction is the one that matters for cost. A ``"shift"`` changes
the *values* — a shifted jet collection — so every universe needs its own pass over the data. A
``"weight"`` changes only the multiplicative factor, so all its universes can share one pass.
``graphed.variations`` reports a context's registry as ``{name: {tag: (kind, ordering)}}``.
Numeric tags parse to an ordering value — the σ handle you want for envelope plots — under both
the exponent form (``5em1`` is ½) and the datacard form (``1p0`` is 1, ``m1p0`` is −1); a
non-numeric tag such as ``"extreme"`` carries ``None`` and is simply unordered.

Three ways two things can be correlated
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"These two systematics are correlated" means three different things, and graphed keeps them apart
because they cost different amounts and only one of them is new:

.. list-table::
   :header-rows: 1
   :widths: 38 46 16

   * - What you mean
     - How you say it
     - New?
   * - Two registrations are the *same fit parameter*
     - **Share the nuisance name.** Two ``vary`` calls with one ``name``.
     - no
   * - A correction *depends on* a quantity a nuisance moves
     - **Propagation.** Compute the correction from the varied quantity, with
       ``gak.apply_correction`` or ordinary ops.
     - no
   * - One universe is displaced on *two or more axes at once*
     - **``points=``.** Register the universe with its coordinate map.
     - yes

Propagation is a recording detail worth spelling out: record a correction with
``graphed.awkward.gak.apply_correction``, not with ``Session.record_external`` directly.
``apply_correction`` goes through the ``gak`` dispatch layer, so a ``Varied`` input fans out and the
correction is *re-evaluated* in each universe — a scale factor whose jet crosses a binning edge
under the shift gets the other bin's value. ``Session.record_external`` is the raw seam underneath;
it takes plain ``Array`` inputs and knows nothing about labels.

.. code-block:: python

    import awkward as ak
    import correctionlib
    import correctionlib.schemav2 as cs
    import graphed.awkward as ga
    from graphed import Session, labels, universe, vary
    from graphed.awkward import AwkwardBackend, from_awkward, gak

    payload = cs.CorrectionSet(schema_version=2, corrections=[cs.Correction(
        name="jet_sf", version=1, inputs=[cs.Variable(name="pt", type="real")],
        output=cs.Variable(name="sf", type="real"),
        data=cs.Binning(nodetype="binning", input="pt", edges=[0.0, 40.0, 1000.0],
                        content=[0.95, 1.05], flow="clamp"))],
    ).model_dump_json(exclude_unset=True).encode()
    evaluator = correctionlib.CorrectionSet.from_string(payload.decode())["jet_sf"]

    s   = Session(AwkwardBackend())
    ev  = from_awkward(s, "events", ak.Array({"Jet": [[{"pt": 38.0}], [{"pt": 70.0}]]}))
    ctx = ga.gnano.events(ev)

    jets = ctx.Jet
    ctx  = vary(ctx, "jes", collections={"Jet": {
        "up": gak.with_field(jets, jets.pt * 1.10, "pt")}})

    sf = gak.apply_correction(payload, "jet_sf", [gak.flatten(ctx.Jet.pt)],
                              lambda pt: evaluator.evaluate(pt), args=["$0"])
    print(labels(sf))
    print(s.materialize(universe(sf, "nominal")).to_list())
    print(s.materialize(universe(sf, "jes_up")).to_list())

This example needs ``pip install "graphed[preserve]"`` for ``correctionlib``. Prints::

    ('nominal', 'jes_up')
    [0.95, 1.05]
    [1.05, 1.05]

The 38 GeV jet crosses the 40 GeV edge under the shift, so its scale factor changes from 0.95 to
1.05 — you never wrote a second correction call. :doc:`../awkward/design` covers the recording
itself, content hashing and all.

A universe at two coordinates at once
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The b-tag scale factor above is itself computed from jets, so it already varies with the jet scale.
What you cannot get from name-sharing or propagation alone is a *universe* that sits at
``jes = up`` **and** ``btag = hf_up`` together — the fit's correlated template. That universe is
registered, not inferred: ``points=`` maps a tag to the coordinates its universe occupies, and
resolution projects that point onto whatever axes each container downstream happens to know.

.. code-block:: python

    import awkward as ak
    import graphed.awkward as ga
    from graphed import Session, points, universe, vary, weight
    from graphed.awkward import AwkwardBackend, from_awkward, gak

    events = ak.Array({
        "MET": ak.zip({"pt": [10.0, 20.0, 30.0]}),
        "Jet": ak.zip({"pt": ak.Array([[40.0, 25.0], [55.0], [30.0, 60.0, 20.0]])}),
    })

    s   = Session(AwkwardBackend())
    ctx = ga.gnano.events(from_awkward(s, "events", events))

    jets = ctx.Jet
    ctx  = vary(ctx, "jes", collections={"Jet": {
        "up":   gak.with_field(jets, jets.pt * 1.05, "pt"),
        "down": gak.with_field(jets, jets.pt * 0.95, "pt")}})

    # a pT-dependent b-tag scale factor, propagated: recomputed on each universe's jets
    jets = ctx.Jet
    rel  = 0.05 * jets.pt / 100.0
    sf_c, sf_up, sf_dn = (gak.prod(1.0 + k * rel, axis=1) for k in (0.0, 1.0, -1.0))

    ctx = vary(ctx, "btag", sf_c, is_weight=True,
               variations={"hf_up": sf_up, "hf_down": sf_dn,
                           "jesup_hf_up": sf_up, "jesdn_hf_up": sf_up},
               points={"jesup_hf_up": {"btag": "hf_up", "jes": "up"},
                       "jesdn_hf_up": {"btag": "hf_up", "jes": "down"}})

    w = weight(ctx)
    print(points(ctx)["btag_hf_up"])
    print(points(ctx)["btag_jesup_hf_up"])
    print([round(x, 4) for x in s.materialize(universe(w, "btag_hf_up")).to_list()])
    print([round(x, 4) for x in s.materialize(universe(w, "btag_jesup_hf_up")).to_list()])
    print([round(x, 4) for x in s.materialize(universe(w, "btag_jesdn_hf_up")).to_list()])

Prints::

    {'btag': 'hf_up'}
    {'btag': 'hf_up', 'jes': 'up'}
    [1.0328, 1.0275, 1.0559]
    [1.0344, 1.0289, 1.0587]
    [1.0311, 1.0261, 1.0531]

Three things to read off that. The four scale-factor universes are the **same expression objects**
— ``sf_up`` is passed twice — because the point, not the object, decides which inner universe a
label reads; building an arithmetically equal but distinct expression instead only adds a node the
optimizer may merge back. ``btag_hf_up`` names no ``jes`` coordinate, so it keeps nominal
kinematics, exactly as it does today. ``btag_jesup_hf_up`` names one, so it reads the shifted jets,
and the three numbers differ — which is the whole point, and is what silently taking the nominal
would hide.

The label grammar does not change: a point is metadata attached to an ordinary ``name_tag`` label,
never rendered into one. ``graphed.points(obj)`` is the authoritative coordinate view — label-sorted,
each map nuisance-sorted, ``"nominal"`` mapping to ``{}`` — and it answers only on record-time
shapes (a ``Varied``, an event context), because points are not carried on disk and a label cannot be
parsed back into a point. ``graphed.variations`` keeps reporting a point family as a family.

Numbers reach numeric tags, and zero is asymmetric
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A coordinate is an ordinary tag, and tags come in two flavours that do not mix. An identifier tag
(``up``, ``hf_up``) is opaque and is *not* promoted to ±1σ — that reading is a stats-export
convention, not a frontend fact. A numeric tag is canonicalised by value, so ``"0p5"``, ``"0.5"``,
``0.5`` and ``Fraction(1, 2)`` are one coordinate. Writing a point with numbers therefore reaches
only families registered with numeric tags, and the mismatch is refused rather than silently
resolved to nominal:

.. code-block:: python

    import numpy as np
    from graphed import Session, labels, points, vary
    from graphed.numpy import NumpyBackend, from_array

    s  = Session(NumpyBackend())
    pt = from_array(s, "pt", np.array([10.0, 20.0, 30.0]))

    named  = vary(pt, "jes", up=pt * 1.05, down=pt * 0.95)
    sigmas = vary(pt, "jes", **{"1": pt * 1.05, "0p5": pt * 1.025})
    print(points(sigmas))

    try:                                    # numbers against identifier tags
        vary(named, "corr", variations={"up": pt}, points={"up": {"jes": 1}})
    except Exception as exc:
        print(type(exc).__name__, exc)

    zero = vary(pt, "shift", **{"0": pt * 1.01})    # a tag that happens to be "0"
    print(labels(zero), points(zero)["shift_0"])

    try:                                    # a point that IS the origin
        vary(sigmas, "corr", variations={"c": pt}, points={"c": {"jes": 0}})
    except Exception as exc:
        print(type(exc).__name__, exc)

Prints::

    {'jes_0p5': {'jes': '5em1'}, 'jes_1': {'jes': '1'}, 'nominal': {}}
    GraphedError points= on graphed.vary('corr'): '1' is not a registered tag of nuisance 'jes', whose tags are ['down', 'up']
    ('nominal', 'shift_0') {'shift': '0'}
    GraphedError points= entry 'c' names the central universe — every coordinate sits at 0, which is what absence already says; nominal is not a variation

Every nuisance and coordinate a ``points=`` entry names must already be registered somewhere the
call can see; otherwise a joint point written before its ``jes`` axis exists would quietly produce a
b-tag-only universe wearing a joint name. Labels *inherited* from upstream keep the ordinary silent
fallback to nominal — partial coverage is a legitimate pattern; a coordinate you typed is not.

Note the asymmetry in the last two cases, which is deliberate. In an explicit ``points=`` map a
coordinate of 0 means "this axis sits at its central value", which is what leaving it out already
says, so ``{jes: 1, btag: 0}`` and ``{jes: 1}`` are one point and an entry that canonicalises to the
empty point is refused — that is ``nominal``, not a variation. A *default* point is never
zero-dropped: its coordinate is the tag you registered, a name for a universe rather than a
displacement, so the legal tag ``0`` mints the ordinary label ``shift_0`` sitting at ``{shift: 0}``,
distinct from ``nominal``, exactly as it does today. To name a universe that sits at zero on some
axis, register it as a tag; ``points=`` cannot spell it.

Also worth knowing: ``points=`` earns a new label only for a genuinely multi-coordinate universe. A
single-coordinate entry is refused, because the plain tag already names that universe and two labels
for one point would mean two slots, two category bins and two templates for a fit to reconcile.
There is no ``explode=`` verb either — a factorial grid falls out of a dict comprehension over
``points=``, and the full grid is a closure study rather than something a fit ingests.

One histogram per variation, or one variation axis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You have jet-energy up/down and a scale factor up/down. Do you want five histograms, or one
histogram with a ``variation`` axis? By default each universe is its own output — its own
histogram, its own written column. Passing ``variation_axis=True`` to a fill gives you the second
shape instead: one histogram carrying its universes on a ``StrCategory("variation")`` axis, with
the weight-only universes collapsing into a loop inside the fill rather than into separate fills.
That machinery lives in ``graphed-histogram``; from here the two are interchangeable, because
``labels``, ``nominal`` and ``universe`` read a result histogram's variation axis exactly as they
read a family of separate outputs.

Every universe out of one preservation bundle
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A preservation bundle built over a varied value or weight reproduces *every* universe from the
one directory. Hand ``build_bundle`` a ``Varied`` and ``reproduce`` gives you ``{label: counts}``
instead of a single array, each universe identical to what you got at build time; ``inspect``
lists the labels without executing anything.

.. code-block:: python

    import tempfile

    import awkward as ak
    from graphed import Session, labels, vary
    from graphed.awkward import AwkwardBackend, from_awkward, gak
    from graphed.preserve import build_bundle, inspect, reproduce

    events = ak.Array({
        "MET": ak.zip({"pt": [10.0, 20.0, 30.0]}),
        "Jet": ak.zip({"pt": ak.Array([[40.0, 25.0], [55.0], [30.0, 60.0, 20.0]])}),
    })

    s  = Session(AwkwardBackend())
    ev = from_awkward(s, "events", events)

    value  = gak.sum(ev.Jet.pt, axis=1)
    weight = vary(ev.MET.pt, "sf", up=ev.MET.pt * 1.1, down=ev.MET.pt * 0.9)
    HIST   = {"name": "ht", "bins": 5, "lo": 0.0, "hi": 200.0}

    root = tempfile.mkdtemp()
    bundle = build_bundle(root, session=s, value=value, weight=weight,
                          datasets={"events": events}, histogram=HIST)

    print(bundle.manifest["format_version"])
    print(reproduce(bundle))
    print(all(lbl in inspect(bundle) for lbl in labels(weight)))

Prints::

    2
    {'nominal': array([ 0., 30., 30.,  0.,  0.]), 'sf_down': array([ 0., 27., 27.,  0.,  0.]), 'sf_up': array([ 0., 33., 33.,  0.,  0.])}
    True

The manifest carries a per-label output map and declares format version ``2``; an unvaried bundle
keeps its single-output shape and version ``1``, so older bundles stay readable. See
:doc:`../preserve/design` for what else goes in the directory.

Writing every universe to one skim file is the same idea on the I/O side —
``graphed.awkward.to_parquet(..., select=...)`` writes the superset of rows any universe keeps,
plus enough to reconstruct each one; :doc:`../awkward/design` covers it.

What a variation edit does to cached results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Within one run, resuming from a checkpoint works per partition exactly as for an unvaried
analysis: the N-universe result for a partition is one journal entry. Between runs, three edits
behave differently, and you can predict all three from one rule — a cached result is keyed by the
recorded graph plus the worker function, so anything that changes either invalidates it:

* **Adding or removing a variation invalidates everything.** Universes are nodes, so toggling one
  changes the graph and therefore every cached entry. Flipping an expensive shift class on or off
  between runs is a full recompute.
* **Renaming a label usually costs nothing.** With one output per universe the labels are not part
  of the recorded graph, so a rename leaves it byte-identical and the cache survives. The
  exception is the variation-axis shape above: there the labels *are* axis categories inside the
  recorded fill, so renaming one invalidates the cache just as adding a variation would.
* **How you name your worker function decides the rest.** Refer to it by import path —
  ``OpSpec.from_ref("myanalysis:hist_chunk")`` — and its identity is that path, so editing
  unrelated code nearby cannot disturb the cache. Carry it by value instead (a lambda, a closure,
  anything defined in ``__main__``) and its identity is its pickled bytes, so editing the body —
  or anything it closes over, label strings included — invalidates its cached results. The
  import-path form is the documented idiom for this reason; see :doc:`../checkpoint/design`.


What actually gets read off disk
--------------------------------

Before reading anything, a runner wants the minimal input. The frontend walks the graph; the
backend supplies the semantics — the same generic walk that ``materialize`` uses, but with
reporting tracers flowing through instead of values. There are two granularities and the
difference is worth money:

.. code-block:: python

    import awkward as ak
    from graphed import Session
    from graphed.awkward import AwkwardBackend, from_awkward, gak
    from graphed.awkward.projection import project, project_buffers

    events = ak.Array({
        "Jet": ak.zip({"pt": [[40.0, 25.0], [55.0]], "eta": [[0.1, 2.0], [1.0]]}),
        "MET": ak.zip({"pt": [10.0, 20.0]}),
    })

    s  = Session(AwkwardBackend())
    ev = from_awkward(s, "events", events)

    njet = gak.num(ev.Jet, axis=1)          # how many jets, never their values

    print(sorted(project(njet).read_columns["events"]))
    print(sorted(project_buffers(njet).offsets_only_for("events")))
    print(sorted(project_buffers(gak.sum(ev.Jet.pt, axis=1)).columns_for("events")))

Prints::

    []
    ['Jet']
    ['Jet.pt']

:class:`~graphed.Projection` is the column view: which named columns of each source are touched.
It is the right answer for "what should I read to get this value", and for a jet-count analysis
it says *no columns* — true, and useless to a reader, which would either read nothing or give up
and read everything. :class:`~graphed.BufferProjection` is finer: per column, whether its leaf
*data* is needed or only its list structure. Counting jets needs the ``Jet`` offsets and none of
the payload, and a writer can serve that from a counter branch or an index column. Ask for
columns and you get the coarse view back (``to_projection``); ask for buffers and you get the
cheapest correct read.

One more distinction, which is API rather than prose. ``read_columns`` reports what a compiled
graph *syntactically* mentions — including the untouched legs of a ``zip`` — because evaluating a
reduced graph replays every recorded node, and a field the replay mentions has to exist in the
chunk. Buffer projection answers "what data is needed"; ``read_columns`` answers "what must be
present". Plans pass the latter to the reader and let buffer projection narrow the leaves.

Some things cannot be projected through: a callable graphed cannot look inside is opaque by
definition, so no replay can say which columns it reads. Every
projection entry point takes an explicit policy, so this is your choice and never a silent one:
``on_fail="pass"`` assumes the callable adds nothing, ``"warn"`` falls back to reading everything
and tells you, ``"raise"`` refuses.


Compile once, evaluate anywhere
-------------------------------

``compile_ir(session, *outputs)`` reduces the graph *for exactly those outputs* and returns a
:class:`~graphed.CompiledGraph` — the serialized reduced bytes plus the source names it needs.
Outputs are a property of the request, not of the session: compiling ``a`` and then ``b`` from one
session gives two independent single-output artifacts, each byte-identical to what a fresh session
would have produced, while ``compile_ir(s, a, b)`` carries both and evaluates them in one pass.

``evaluate_ir(compiled, backend, sources, externals=...)`` walks the reduced node list once — one
backend dispatch per reduced node, fused members inline. ``sources`` binds source names to data or
to zero-argument loaders. ``externals`` resolves each call-out payload by its content hash and
fails loudly when one is missing, so a correction or a model is never silently skipped.

The bytes in ``compiled.ir`` are the durable artifact, and ``Session.serialized_ir(*outputs)``
hands them to you directly (``optimize=False`` gives the unfused, one-node-per-operation form,
which is what you want when you are auditing what an analysis does). The same analysis always
serializes to the same bytes. Checkpoint keys, preservation bundles and cross-run comparison all
rest on that.


One pass over the dataset, many outputs
---------------------------------------

Most analyses want several results out of one read. ``aggregate_plan`` compiles the outputs you
name into one graph, then builds a plan that reads each partition once, evaluates all of them, and
folds partial results with the reduce/combine functions you supply. The example writes two small
parquet files first so that it runs anywhere:

.. code-block:: python

    import tempfile

    import awkward as ak
    import numpy as np
    from graphed import Session, aggregate_plan, read_columns
    from graphed.awkward import AwkwardBackend, from_parquet, gak
    from graphed.core.execution import SequentialRunner

    # Two small parquet files standing in for a dataset.
    root = tempfile.mkdtemp()
    for i in range(2):
        ak.to_parquet(
            ak.Array({
                "Jet_pt": ak.Array([[40.0, 25.0], [55.0], [30.0, 60.0, 20.0]]),
                "MET_pt": np.array([10.0, 20.0, 30.0]) + 100.0 * i,
                "run": np.array([1, 1, 2]),
            }),
            f"{root}/part{i}.parquet",
        )

    s = Session(AwkwardBackend())
    ev = from_parquet(s, "events", root, steps_per_file=1)

    ht = gak.sum(ev.Jet_pt, axis=1)          # shared by both outputs
    n_high = gak.sum(ht > 60.0, axis=None)
    total_ht = gak.sum(ht, axis=None)

    plan = aggregate_plan(
        n_high,
        total_ht,
        reduce=lambda outs: [float(o) for o in outs],
        combine=lambda a, b: [x + y for x, y in zip(a, b, strict=True)],
        empty=lambda: [0.0, 0.0],
    )
    print(SequentialRunner().run(plan).value)
    print(sorted(read_columns([n_high, total_ht], s.source_ids()[0])))

Prints::

    [4.0, 460.0]
    ['Jet_pt']

Both outputs share the ``ht`` sub-expression, so it is read and evaluated once per partition, not
once per output — and of the three columns in each file, only ``Jet_pt`` is read at all.
``SequentialRunner`` runs the plan here with no extra dependencies; the same plan object is what
you hand to a process pool, a dask cluster or a parsl pool from ``graphed-executors`` (``pip
install graphed-executors``). ``backend=`` names the workers' evaluation backend, ``partitions=``
overrides the partitioning, and ``steps_per_file=`` splits each file into more tasks.
``graphed-histogram``'s ``gh.plan(...)`` is the same entry point specialised to histograms.


Joining and repartitioning datasets
-----------------------------------

Joins and repartitions are neither an awkward idiom nor a numpy one, so they are module verbs
rather than array methods. ``repartition(array, by=)`` records a hash exchange on a field,
``n=`` targets a partition count and ``target_bytes=`` coalesces by measured size. ``join`` gives
you relational, row-duplicating semantics — a probe row with *k* matches on the build side yields
*k* output rows — with ``how`` in ``{"inner", "left", "right", "outer"}``, matching
``pandas.merge`` rather than an awkward broadcast. ``pack_key`` is the same key-packing step the
join uses internally, public so you can pre-key a source.

Because a join or a repartition is a barrier, the plan for it has stages:

.. code-block:: python

    import tempfile

    import awkward as ak
    import numpy as np
    from graphed import Session, join, join_plan
    from graphed.awkward import AwkwardBackend, from_parquet

    root = tempfile.mkdtemp()
    ak.to_parquet(ak.Array({"run": np.array([1, 1, 2]), "MET_pt": np.array([10.0, 20.0, 30.0])}),
                  f"{root}/events.parquet")
    ak.to_parquet(ak.Array({"run": np.array([1, 2]), "lumi_w": np.array([0.9, 1.1])}),
                  f"{root}/lumi.parquet")

    s = Session(AwkwardBackend())
    events = from_parquet(s, "events", f"{root}/events.parquet")
    lumi = from_parquet(s, "lumi", f"{root}/lumi.parquet")

    joined = join(events, lumi, on=["run"], how="inner")
    plan = join_plan(joined)
    print([(st.kind, len(st.tasks)) for st in plan.stages])

Prints::

    [('map_write', 1), ('map_write', 1), ('gather_join', 1)]

Each side is routed and written by its own map stage; one gather stage depends on both and does
the matching. ``shuffle_plan`` is the single-source counterpart for a plain repartition: a
map-write stage and a gather stage, with the barrier edge between them. Both builders produce a
durable, byte-deterministic plan; running one across processes is ``graphed-executors``' job, and
its shuffle documentation covers where the blocks actually travel.


Reading and writing partitioned files
-------------------------------------

Two modules hold what every I/O integration shares, with no array-library content of their own:

:mod:`graphed.parquet`
    Dataset discovery that is deterministic (directories and globs sorted; an explicit list keeps
    your order, because the list is part of the dataset's identity), row counts from metadata,
    partitioning, and the convention for recording a deferred source. The array codecs live in the
    backends.

:mod:`graphed.write`
    The format-agnostic partitioned write: ``write_plan`` builds a plan whose tasks each write one
    part and report their paths up a deterministic combine tree, so the returned file list does
    not depend on which task finished first. ``file_bases``, ``blind_part_index``, ``step_of`` and
    ``part_path`` let a worker derive its own part name from its partition alone. The module also
    defines :class:`~graphed.write.PartitionedSource`: implement ``partitions()`` and
    ``read_partition(partition, columns, resources)`` and any generic consumer — the parquet
    writer, the histogram aggregator, ``aggregate_plan`` — can drive your source partition by
    partition without ever calling its whole-dataset loader.

Partitions are **blind** wherever possible: planning opens no files, and a worker resolves its own
entry range against the file it has just opened. You do not pay for a metadata scan of the whole
dataset before work starts, and a plan built on one machine stays valid on another that has never
seen the files.


Using another array library
---------------------------

Five methods are everything a backend has to provide::

    op_form(op, input_forms, params) -> Form        # record-time type/shape inference, no data
    eval_stage(op, inputs, params)   -> value       # evaluate one operation or fused member
    boundary_ops()                   -> frozenset   # which operations end a stage
    project(op, used, params)        -> used'       # narrow the read set through this operation
    external_payload(op, params)     -> descriptor  # identify a correction, model, or other call-out

A backend never sees the graph and the frontend never sees an array. ``Form`` is likewise minimal
— anything with ``describe() -> str``; the frontend stores and forwards forms, it does not
interpret them.

A package that records its *own* call-outs — histogram fills are the example — passes
``descriptor=`` and ``form=`` to ``Session.record_external`` and the backend is not consulted at
all. That is how ``graphed-histogram`` exists without teaching either backend what a histogram is,
and it is the path to follow for your own deferred operation.


Not supported yet
-----------------

* **Predicate pushdown.** Projection narrows what is read to the columns and buffers you touch,
  but a cut is applied after the read, not handed to the reader.
* **Behavior methods that take arguments.** Behavior *properties* record — ``jets.pt`` resolves
  through a registered behavior — but ``a.deltaR(b)`` does not. Write the formula, or wrap it in a
  function of plain arrays.
* **Output groups.** Compiling a chosen set of outputs together is done; there is no higher-level
  helper for organising many such groups with shared sub-plans.
* **Remote stores in the parquet base.** Discovery and row counts assume local filesystem paths.

:doc:`improvements` lists the limits you are most likely to hit, with the workaround for each.
