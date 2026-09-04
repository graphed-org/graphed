graphed
=======

Write your analysis in ordinary awkward code. ``graphed`` records what you wrote instead of
running it, works out the type and shape of every result before a file is opened, and collapses
the redundancy as you go — the cut you wrote twice, the helper that multiplies by 1.0 — so a big
unreduced graph never has to exist. When you are ready, you hand the recording to a runner: a
pool on your laptop, a dask cluster, a parsl pool.

.. code-block:: python

    import awkward as ak
    from graphed import Session
    from graphed.awkward import AwkwardBackend, from_awkward, gak

    session = Session(AwkwardBackend())
    jets = from_awkward(session, "jets", ak.Array([[40.0, 12.0], [55.0], [8.0]]))
    leading = gak.max(jets[jets > 25.0], axis=1)     # nothing has run yet

    print(session.form(leading).describe())          # the result type, without reading any data
    print(ak.to_list(session.materialize(leading)))  # now it runs

.. code-block:: text

    ## * ?float64
    [40.0, 55.0, None]

``gak`` carries ``awkward``'s names, and for the parameters it takes, ``awkward``'s own defaults:
``gak.concatenate`` defaults to ``axis=0`` because ``ak.concatenate`` does. What changes is that
``gak.max`` hands you a recording instead of an array — and a short list of argument-passing
rules, which :doc:`awkward/index` walks through under *What changes when you port*.

What you get past the deferred array
------------------------------------

* **Systematics from one read.** Declare a variation once and every histogram downstream of it
  comes back for every universe, out of a single read of the data.
* **Errors that point at your line.** A failure deep inside a fused step on a remote worker is
  re-raised on your machine with an arrow at the line you wrote.
* **Restartable runs.** A job that died at hour seven of eight restarts and redoes only the
  partitions that were in flight, and the result is identical to an uninterrupted run.
* **Cheap reads.** Only the columns your analysis touches are read off disk — and inside a
  ragged column, only the pieces it touches.
* **A directory a colleague can run.** Export a bundle that reproduces your histograms
  bit-for-bit on a clean machine, or that someone can inspect without running anything.

Install
-------

.. code-block:: bash

    pip install "graphed[awkward,parquet]"   # ragged analysis + parquet I/O
    pip install graphed-executors            # runners: laptop pools, dask, parsl
    pip install graphed-histogram            # deferred boost-histogram / hist fills

Wheels ship for Linux, macOS and Windows; installing from a source checkout also needs a Rust
toolchain. :doc:`architecture` lists the other extras — ``[numpy]``, ``[dashboard]``,
``[preserve]`` — and what each one buys.

Where to go next
----------------

**Write an analysis.** :doc:`quickstart` takes a parquet dataset through a selection, a
systematic variation and a histogram in one program. :doc:`awkward/index` is the porting guide
for ``gak`` — what maps one-to-one from ``ak.*`` and what does not. :doc:`frontend/index`
covers the recording surface itself: sessions, arrays, forms, provenance, ``vary``.
:doc:`numpy/index` is the same idea for flat arrays.

**Understand what the recording does.** :doc:`architecture` is the mental model of a run — who
does what, what crosses a process boundary, which pieces you install. :doc:`core/index` goes
under it: how duplicate work collapses, how operations get fused into runs that execute as one
pass, and why the reduced form is byte-identical every time.

**Run it, debug it, keep it.** :doc:`debug/index` is what to do when an analysis fails on a
worker. :doc:`checkpoint/index` is restarting a killed run. :doc:`preserve/index` is handing
the whole analysis to someone else.

.. toctree::
   :maxdepth: 1
   :caption: Getting started

   quickstart
   architecture

.. toctree::
   :maxdepth: 1
   :caption: Writing an analysis

   awkward/index
   frontend/index
   numpy/index

.. toctree::
   :maxdepth: 1
   :caption: Running, debugging, keeping

   debug/index
   checkpoint/index
   preserve/index

.. toctree::
   :maxdepth: 1
   :caption: Reference

   api
   core/index

.. toctree::
   :maxdepth: 1
   :caption: How the results are checked

   corpus/index

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
