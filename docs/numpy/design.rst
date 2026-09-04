How graphed.numpy works
=======================

You have ordinary numpy code over flat columns, and you want it deferred: recorded now,
optimized, and run later — on one core or many — without rewriting it. ``graphed.numpy``
does that. A large slice of the ``np.*`` surface records into the graph through numpy's own
dispatch protocols, so the code below is the code you already write.


The backend in one example
--------------------------

Install with ``pip install graphed[numpy]``.

.. code-block:: python

    import numpy as np
    from graphed import Session
    from graphed.numpy import NumpyBackend, from_record

    s = Session(NumpyBackend())
    ev = from_record(s, "events",
                     pt=np.array([25.0, 55.0, 10.0, 80.0]),
                     eta=np.array([0.1, -1.2, 2.0, 0.4]))

    sel = ev["pt"][abs(ev["eta"]) < 1.5]     # records; nothing computes
    print(sel.shape, sel.dtype)              # known WITHOUT data
    print(s.materialize(sel.mean()))

    counts, edges = np.histogram(ev["pt"], bins=4, range=(0, 100))
    print(type(counts).__name__, edges)
    print(s.materialize(counts))

.. code-block:: text

    (None,) float64
    53.333333333333336
    NumpyArray [  0.  25.  50.  75. 100.]
    [1 1 1 1]

Three things to notice. Shapes and dtypes are known at record time — the leading axis is the
event axis, whose length is unknown until data is read, written ``None``. Numpy *functions*,
not just operators, record: ``np.histogram`` dispatched through the deferred array. And a
call with mixed results returns each part in its natural form — the counts depend on data,
so they are deferred; the edges depend only on the binning, so they are concrete.


How your result's type is known before any data is read
-------------------------------------------------------

Every recorded array carries a description of its type and shape (its *form*):
dtype, shape, and — for a record source like ``ev`` above — its field names. When you record
an operation, the backend applies the real numpy callable to zero-length arrays of the input
forms, and the result's dtype and shape become the new form. Dtype promotion, broadcasting,
and error behaviour are therefore numpy's own, not a reimplementation that could drift.

The payoff is that mistakes surface on the line you wrote, before any file is opened:

.. code-block:: python

    import numpy as np
    from graphed import GraphedTypeError, Session
    from graphed.numpy import NumpyBackend, from_array

    s = Session(NumpyBackend())
    x = from_array(s, "x", np.array([1.0, 2.0, 3.0]))

    # The full message is "ill-typed op 'add' at <your_file.py>:<line>: ..."
    try:
        x + "a string"
    except GraphedTypeError as err:
        print("caught at record time:", str(err).rsplit(": ", 1)[-1])

.. code-block:: text

    caught at record time: ufunc 'add' did not contain a loop with signature matching types (dtype('float64'), dtype('<U8')) -> None

The same applies to geometry: a ``np.reshape`` or ``np.concatenate`` whose shapes cannot
work fails when you write it, with numpy's own error text and your source line attached.


How much of numpy records
-------------------------

The frontend's base ``Array`` carries only the surface every backend shares. Numpy's idiom —
``.shape`` / ``.dtype`` / ``.ndim``, method-style reductions (``.sum()``, ``.mean()``,
``.cumsum()``, ...), tuple subscripts like ``arr[:, 0]`` — lives on ``NumpyArray``, the
deferred array type this backend hands to the session. Two numpy protocols make the surface
broad:

* ``__array_ufunc__`` routes every ufunc — ``np.sqrt``, ``np.maximum``, ``np.logaddexp``,
  the full table — into recorded operations.
* ``__array_function__`` routes fifty numpy functions: ``np.where``, ``np.concatenate``,
  ``np.stack``, shape manipulation (``reshape`` / ``transpose`` / ... with record-time
  geometry checks), ``np.histogram`` / ``np.histogram2d`` / ``np.histogramdd``, statistics
  (``mean`` / ``std`` / ``var`` and their ``nan*`` variants), and ``np.unique``,
  ``np.searchsorted``, ``np.isin``, ``np.take``.

Each table entry maps the numpy callable to a recorded operation; evaluation calls real
numpy. Record-time inference runs the *same* callables on zero-length arrays, so what
records and what runs cannot disagree.

For a function outside the table, ``apply_gufunc`` applies your own callable elementwise
over the event axis with a declared signature, so even custom code stays deferred (the graph
records only the name you give it, never the function body — see :doc:`improvements` for what
that costs you).


Sources, creation, and reproducible randomness
----------------------------------------------

``from_array`` records a single-array source; ``from_record`` a named-field table of
equal-length columns (the events-table shape). Creation routines (``arange``, ``linspace``,
``full``, ``zeros``, ...) record the *recipe* rather than the bytes, so two identical calls
are one node.

Randomness is graph-friendly: ``default_rng(session, seed)`` returns a generator whose
draws are keyed by the seed and the draw order. The same seed and the same program produce
the same values — and the same recorded graph — every run:

.. code-block:: python

    import numpy as np
    from graphed import Session
    from graphed.numpy import NumpyBackend, default_rng

    s = Session(NumpyBackend())
    r1 = default_rng(s, seed=42)
    r2 = default_rng(s, seed=42)
    a = s.materialize(r1.normal(size=3))
    b = s.materialize(r2.normal(size=3))
    print(a)
    print(np.array_equal(a, b))

.. code-block:: text

    [-0.3736199   1.04616118  0.45588016]
    True

So a toy study or a smearing step is exactly as reproducible as the rest of your analysis.


Why the answer doesn't depend on the worker count
-------------------------------------------------

Data is split along the event axis into partitions, and every reduction knows how to
compute a partial result per partition and how to combine partials: ``sum`` adds,
``mean`` accumulates (sum, count) pairs, ``var`` carries its sufficient statistics,
``histogram`` adds bin-wise. Partial results merge in any order, so the total doesn't
depend on how many workers you used — and the runner, sequential on your laptop or a
cluster combining results in a tree, needs to know nothing about the specific reduction.


Parquet in and out
------------------

Reading and writing parquet needs the ``[parquet]`` extra:
``pip install graphed[numpy,parquet]``.

``from_parquet`` records a deferred dataset source: the column types come from the file
schema, so no data is read until the runner needs it. ``to_parquet`` writes one file per
partition.

.. code-block:: python

    import os
    import tempfile

    import pyarrow as pa
    import pyarrow.parquet as pq

    from graphed import Session
    from graphed.numpy import NumpyBackend
    from graphed.numpy.io import from_parquet, to_parquet

    d = tempfile.mkdtemp()
    path = os.path.join(d, "events.parquet")
    pq.write_table(pa.table({"pt": [25.0, 55.0, 10.0], "njet": [2, 3, 1]}), path)

    s = Session(NumpyBackend())
    ev = from_parquet(s, "events", path)     # schema read; no data yet
    print(ev["pt"].dtype, s.materialize(ev["pt"].sum()))

    out = to_parquet(ev["pt"] * 2.0, os.path.join(d, "scaled"))
    print([os.path.basename(p) for p in out])
    print(pq.read_table(out[0]).to_pydict())

.. code-block:: text

    float64 90.0
    ['part-00000.parquet']
    {'data': [50.0, 110.0, 20.0]}

Two behaviours worth knowing. Only fixed-width primitive columns — integers, floats,
booleans — are read; a nested or string column is refused by name at record time, and for
ragged data the error points you at ``graphed.awkward`` rather than half-supporting it
here. And the read path decodes arrow columns
straight to numpy without importing pandas, so a worker environment does not need pandas
installed.


Only the columns you touch are read
-----------------------------------

``project(array)`` walks the recorded graph and returns, per source, the set of fields the
computation actually uses. A parquet table with forty columns costs you the two your
analysis touched. For flat data the column list is exact — there are no offsets or nested
buffers to account for, so no finer granularity is needed.


Not supported yet
-----------------

* ``einsum``, FFTs, and linear algebra beyond ``apply_gufunc``-style application do not
  record. Wrap the call with ``apply_gufunc``, or materialize the inputs and run it
  eagerly.
* The event axis is the only partitioned axis. Trailing axes are fine (``(-1, 3)``
  vectors, say), but dask-style multi-axis chunking is not available; for large
  non-event-axis structure, use dask.array.
* The parquet reader does not push filters into the file scan. Read the columns and cut
  with a boolean mask; only the columns you touch are read either way.
