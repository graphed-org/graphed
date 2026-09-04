Port your awkward analysis
==========================

.. contents::
   :local:

You already have an analysis: read a dataset, cut on jets, fill a histogram, and do the whole
thing again for every systematic variation. This page runs that shape end to end. Everything
below is one program — copy it, run it, then read the walkthrough underneath.

Two installs get you through the first program. The base package records your analysis and runs
it in-process; the histogram package gives you deferred ``boost-histogram`` fills:

.. code-block:: bash

    pip install "graphed[awkward,parquet]" graphed-histogram

The whole thing
---------------

.. code-block:: python

    import os
    import tempfile

    import awkward as ak

    # Two small parquet files standing in for a dataset.
    data_dir = tempfile.mkdtemp()
    for i, chunk in enumerate([
        ak.Array({"Jet": [[{"pt": 40.0}, {"pt": 12.0}], [{"pt": 55.0}], [{"pt": 8.0}]]}),
        ak.Array({"Jet": [[{"pt": 70.0}, {"pt": 31.0}], [], [{"pt": 26.0}, {"pt": 9.0}]]}),
    ]):
        ak.to_parquet(chunk, os.path.join(data_dir, f"part-{i}.parquet"))

    import boost_histogram as bh
    import graphed
    import graphed_histogram as gh
    from graphed import Session
    from graphed.awkward import AwkwardBackend, from_parquet, gak, gnano
    from graphed.core.execution import SequentialRunner

    session = Session(AwkwardBackend())
    events = gnano.events(from_parquet(session, "events", data_dir))

    jets = events.Jet
    events = graphed.vary(events, "jes", collections={"Jet": {
        "up": gak.with_field(jets, jets.pt * 1.05, "pt"),
        "down": gak.with_field(jets, jets.pt * 0.95, "pt"),
    }})

    selected = events.Jet.pt[events.Jet.pt > 25.0]
    lead_pt = gak.max(selected, axis=1)

    h = gh.boost.Histogram(bh.axis.Regular(5, 0.0, 100.0), storage=bh.storage.Int64())
    h.fill(lead_pt)

    plan = gh.plan({"lead_pt": h})
    result = gh.unpack(SequentialRunner().run(plan).value)

    for label, hist in result["lead_pt"].items():
        print(label, hist.view(), hist.sum())

.. code-block:: text

    nominal [0 1 2 1 0] 4.0
    jes_up [0 1 2 1 0] 4.0
    jes_down [0 1 1 1 0] 3.0

Three histograms came out of one *read* of the data: the files are opened once and ``Jet.pt`` is
read once. A shift changes the values, so the selection and the fill still run once per
universe — but the expensive part, getting the bytes off disk, happens once. The numbers differ
in exactly the way they should: a 26 GeV jet scaled down by 5% falls under the 25 GeV cut, so
``jes_down`` has one entry fewer.

What just happened
------------------

**Nothing ran until the last four lines.** ``from_parquet`` opened the dataset's metadata to
learn its schema, not its contents. Every ``gak`` call after it recorded an operation and handed
back a proxy. ``h.fill(lead_pt)`` recorded a fill; it did not fill anything. There is no
``.compute()`` here on purpose — you build a plan and give it to a runner.

**The selection was written once, not three times.** ``graphed.vary`` registers a variation
family on the event context, so ``events.Jet`` downstream of it means *the varied jets*, in
every universe at once. You write the cut, the ``max``, and the fill exactly as you would for
the nominal analysis, and each histogram comes back labelled — ``nominal``, ``jes_up``,
``jes_down``. Weight-only variations are cheaper still: because only the multiplicative factor
differs, all of a weight family's universes share one pass over the values, not just one read.

**The plan is the unit of work.** ``gh.plan({...})`` compiles every fill into one recording,
works out which columns that recording actually reads, and produces a task per partition of the
dataset. The runner calls each task on a chunk and adds the partial histograms together.
Histograms add, so the partials merge in any order — which is why the total does not depend on
how many workers you used. ``gh.unpack`` turns the runner's flat result into
``{name: {label: hist}}``.

**Only ``Jet.pt`` was read.** The recording knows which fields it touches, so a file with four
hundred branches costs you the ones you used.

Run it on more than one core
----------------------------

The plan does not know or care what executes it. ``SequentialRunner`` above came with the base
install; the pooled and cluster runners are a second package:

.. code-block:: bash

    pip install graphed-executors

Here is the same analysis on a process pool — the only differences are the runner and the
``if __name__ == "__main__":`` guard a spawned worker requires:

.. code-block:: python

    import os
    import tempfile

    import awkward as ak
    import boost_histogram as bh
    import graphed
    import graphed_histogram as gh
    from graphed import Session
    from graphed.awkward import AwkwardBackend, from_parquet, gak, gnano
    from graphed_executors.local import ProcessPoolExecutor


    def build_plan(data_dir: str):
        session = Session(AwkwardBackend())
        events = gnano.events(from_parquet(session, "events", data_dir))
        jets = events.Jet
        events = graphed.vary(events, "jes", collections={"Jet": {
            "up": gak.with_field(jets, jets.pt * 1.05, "pt"),
            "down": gak.with_field(jets, jets.pt * 0.95, "pt"),
        }})
        lead_pt = gak.max(events.Jet.pt[events.Jet.pt > 25.0], axis=1)
        h = gh.boost.Histogram(bh.axis.Regular(5, 0.0, 100.0), storage=bh.storage.Int64())
        h.fill(lead_pt)
        return gh.plan({"lead_pt": h})


    def main() -> None:
        data_dir = tempfile.mkdtemp()
        for i, chunk in enumerate([
            ak.Array({"Jet": [[{"pt": 40.0}, {"pt": 12.0}], [{"pt": 55.0}], [{"pt": 8.0}]]}),
            ak.Array({"Jet": [[{"pt": 70.0}, {"pt": 31.0}], [], [{"pt": 26.0}, {"pt": 9.0}]]}),
        ]):
            ak.to_parquet(chunk, os.path.join(data_dir, f"part-{i}.parquet"))

        plan = build_plan(data_dir)
        result = gh.unpack(ProcessPoolExecutor(max_workers=2).run(plan).value)
        for label, hist in result["lead_pt"].items():
            print(label, hist.view(), hist.sum())


    if __name__ == "__main__":
        main()

.. code-block:: text

    nominal [0 1 2 1 0] 4.0
    jes_up [0 1 2 1 0] 4.0
    jes_down [0 1 1 1 0] 3.0

Bin for bin, that is the single-process answer. It stays that way for any worker count, because
the order the partial histograms are added in comes from the plan rather than from whichever
worker happened to finish first.

``graphed-executors`` ships dask and parsl runners under its ``[dask]`` and ``[parsl]`` extras
for the same plan on a batch cluster; ``build_plan`` above does not change.

Where to go from here
---------------------

* :doc:`awkward/index` — the porting guide: what maps one-to-one from ``ak.*``, what changes,
  and how ``gnano.events`` gives you the NanoEvents-flavoured entry point used above.
* :doc:`frontend/index` — the recording surface: sessions, forms, provenance, and the full
  ``vary`` grammar including weight variations and per-collection shifts.
* :doc:`debug/index` — when the run fails on a worker and you want the arrow at your line.
* :doc:`checkpoint/index` — when the run is long enough that you want to survive losing it.
* :doc:`architecture` — the mental model of everything above.
