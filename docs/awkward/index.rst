graphed.awkward
===============

This is awkward-array, deferred. ``gak.num`` is ``ak.num``, ``gak.combinations`` is
``ak.combinations`` — the same parameters under the same names, carrying awkward's own defaults —
so an analysis you already have ports by changing where the events come from, which module the
functions live in, and a handful of arguments you now pass by name. Nothing computes
when you write it. ``graphed`` records what you asked for, works out the type and shape of every
result from metadata alone, and when you finally run it reads only the parts of your files the
result actually depends on.

Install
-------

.. code-block:: bash

   pip install "graphed[awkward]"           # ragged analysis
   pip install "graphed[awkward,parquet]"   # ... plus parquet datasets

Running across a cluster additionally needs ``graphed-executors``; filling histograms needs
``graphed-histogram``. Neither is required for anything on this page.

Your first deferred analysis
----------------------------

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
    good = g.Jet[abs(g.Jet.eta) < 1.0]       # jagged boolean mask - recorded, not computed
    keep = gak.num(good, axis=1) >= 1        # per-event jet multiplicity
    out = g.MET[keep]

    print(s.form(out).describe())
    print(ak.Array(s.materialize(out)).tolist())
    print(sorted(project(out).columns_for("events")))
    print(project_buffers(out).buffers_for("events"))

Printed output:

.. code-block:: text

    ## * float64
    [12.0, 8.0]
    ['Jet.eta', 'MET']
    {'Jet.eta': <BufferNeed.DATA: 'data'>, 'MET': <BufferNeed.DATA: 'data'>}

Two lines there are worth a second look.

``s.form(out).describe()`` printed the result's type before a single value was touched — the
mask, the multiplicity cut and the selection were all typed from metadata. Get a field name or
an axis wrong and you find out at the line you wrote it, not an hour into a batch job.

``project(out)`` says this result needs ``Jet.eta`` and ``MET`` — and **not** ``Jet.pt``, even
though the analysis sliced whole ``Jet`` records. Point that at a file with 400 branches and you
pay for the two you used.

Events, weights and systematics
-------------------------------

``gnano.events`` wraps a root record the way NanoEvents does, so systematic variations attach to
your events rather than to individual arrays. Vary once and every downstream quantity carries
every universe:

.. code-block:: python

    import awkward as ak
    import graphed
    import graphed.awkward as ga
    from graphed import Session
    from graphed.awkward import AwkwardBackend, from_awkward, gak

    s = Session(AwkwardBackend())
    events = ga.gnano.events(from_awkward(s, "events", ak.Array(
        {"Jet": [[{"pt": 31.0}, {"pt": 12.0}], [{"pt": 55.0}], [{"pt": 8.0}]]})))

    jets = events.Jet
    scaled = graphed.vary(events, "jes", collections={"Jet": {
        "up":   gak.with_field(jets, jets.pt * 1.05, "pt"),
        "down": gak.with_field(jets, jets.pt * 0.95, "pt"),
    }})

    njet = gak.num(scaled.Jet[scaled.Jet.pt > 30.0], axis=1)
    print(graphed.labels(njet))
    for label in graphed.labels(njet):
        print(label, ak.to_list(s.materialize(graphed.universe(njet, label))))

Printed output:

.. code-block:: text

    ('nominal', 'jes_up', 'jes_down')
    nominal [1, 1, 0]
    jes_up [1, 1, 0]
    jes_down [0, 1, 0]

The 31 GeV jet survives the nominal and upward shifts and fails the downward one — the cut
migrates with the variation, because the cut was recorded once against varied jets rather than
copy-pasted three times. :doc:`../frontend/design` covers the whole systematics story;
:doc:`design` covers what it costs to read and write on this backend.

What changes when you port
--------------------------

- **Free functions, not methods.** ``gak.num(jets, axis=1)``, never ``jets.num()``. Field access,
  operators, ufuncs and ``getitem`` all work on the deferred array as usual.
- **Optional arguments go by name.** Everything after the arrays is keyword-only, where ``ak``
  would also take it positionally: ``gak.zip({"pt": pt}, depth_limit=1)``,
  ``gak.cartesian([a, b], axis=1)``, ``gak.isclose(x, y, rtol=1e-4)``,
  ``gak.nan_to_num(x, nan=0.0)``. Passing one positionally is a ``TypeError``, not a silent
  mis-binding.
- **The statistics functions lead with** ``axis``. ``gak.mean``, ``gak.std``, ``gak.var`` and
  ``gak.moment`` take ``axis`` in the slot where ``ak`` takes ``weight``, so write
  ``gak.mean(x, axis=1, weight=w)`` rather than porting ``ak.mean(x, w)`` verbatim.
  ``gak.corr``, ``gak.covar`` and ``gak.linear_fit`` are unweighted, and ``gak.softmax``
  defaults to ``axis=1`` — per event — where ``ak.softmax`` defaults to ``axis=-1``.
- ``gak.with_field`` **needs the field name.** ``gak.with_field(jets, jets.pt * 1.05, "pt")``:
  ``where`` is required here, and optional in ``ak.with_field``.
- **No** ``behavior=`` **per call.** Register behaviors once on the backend —
  ``AwkwardBackend(behavior=vector.backends.awkward.behavior)`` — and ``.pt``, ``.mass`` and
  friends work through plain attribute access from then on.
- **No** ``highlevel=`` **or** ``attrs=``. Both describe how an array is built eagerly, and
  nothing is built eagerly here.
- **Behavior properties record; behavior methods do not.** ``a.pt`` is fine, ``a.deltaR(b)`` is
  not — write the formula.
- **Reductions carry their axis into the plan.** ``axis=1`` is per-event work that rides along
  with everything else in the same pass; ``axis=None`` or ``axis=0`` combines across events and
  becomes a step of its own. Nothing to configure; it just changes what a run costs.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   design
   improvements

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
