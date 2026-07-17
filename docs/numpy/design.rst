How graphed-numpy works
=======================

``graphed-numpy`` is the **rectilinear backend**: it teaches the graphed frontend numpy's type
system and numpy's idiom. It exists for two reasons. First, it is the seam-prover — the
simplest possible real backend, demonstrating that the frontend's five-method ``Backend``
protocol is sufficient for an actual array library. Second, it is a genuinely usable deferred
numpy: a large slice of the ``np.*`` surface records into the graph through numpy's own
dispatch protocols, so ordinary numpy code runs deferred with few changes.

.. contents::
   :local:
   :depth: 2


The backend in one example
--------------------------

::

    import numpy as np
    from graphed import Session
    from graphed.numpy import NumpyBackend, from_record

    s  = Session(NumpyBackend())
    ev = from_record(s, "events",
                     pt=np.array([25.0, 55.0, 10.0, 80.0]),
                     eta=np.array([0.1, -1.2, 2.0, 0.4]))

    sel = ev["pt"][abs(ev["eta"]) < 1.5]    # records; nothing computes
    sel.shape, sel.dtype                     # ((None,), float64) — known WITHOUT data
    s.materialize(sel.mean())                # -> 53.33...

    counts, edges = np.histogram(ev["pt"], bins=4, range=(0, 100))
    # counts is a DEFERRED NumpyArray (np.histogram dispatched through the proxy);
    # edges are concrete (they depend only on the binning, not the data)
    s.materialize(counts)                    # -> array([1, 1, 1, 1])

Three things to notice: shapes/dtypes are inferred at record time (the leading axis is the
unknown event axis, written ``None``); numpy *functions* — not just operators — record through
the proxy; and a function with mixed deferred/concrete results (``np.histogram``) returns each
part in its natural representation.


Forms: shape/dtype inference without data
-----------------------------------------

``NumpyForm`` carries ``(dtype, shape, kind, fields)``. ``op_form`` answers "what does this op
produce?" from forms alone: ufuncs follow numpy promotion rules on dtypes; reductions and
slices transform the shape tuple; record-field access selects a field's form. The leading axis
is ``None`` — the partitioned event axis, unknown until execution and irrelevant to typing.
This is the same architectural move the ragged backend makes with awkward's typetracer; here
the "tracer" is just dtype/shape arithmetic, which is the point: for rectilinear data, type
inference is simple enough to hand-roll, and the backend proves the protocol doesn't *require*
a tracer library.

Inference failures raise at the recording line (the frontend wraps them with the captured user
frame): ``ev["pt"] + "a string"`` or a shape-incompatible ``np.cross`` fail before any data is
read, with numpy's own error text.


The NumpyArray proxy: numpy idiom, supplied by the backend
----------------------------------------------------------

The frontend's base ``Array`` carries only the *common* deferred-array surface. Numpy's idiom —
``.shape``/``.dtype``/``.ndim`` properties, method-style reductions (``.sum()``, ``.mean()``,
``.cumsum()``...), tuple subscripts (``ev_array[:, 0]``), and the two dispatch protocols —
lives on ``NumpyArray``, which the backend supplies via its ``array_type()`` factory. The
session consults that factory when wrapping nodes, so *numpy users get numpy ergonomics without
a single numpy-ism leaking into the shared proxy* (the ragged backend, by contrast, supplies no
proxy at all and exposes its idiom as free functions).

The dispatch protocols are where the surface gets broad:

* ``__array_ufunc__`` (on the base proxy) routes every ufunc — ``np.sqrt``, ``np.maximum``,
  ``np.logaddexp``, the full table — into recorded ops.
* ``__array_function__`` (on ``NumpyArray``) routes ~50 numpy *functions*: ``np.where``,
  ``np.concatenate``, ``np.stack``, manipulation ops (``reshape``/``transpose``/...with
  record-time geometry checks), ``np.histogram``/``np.histogram2d``, statistics, and the
  ``*_like`` constructors.

Mechanically, each table entry maps the numpy callable to a recorded op name plus a parameter
encoding; evaluation looks the op up in the matching table and calls real numpy. The two tables
(record-side and eval-side) are generated together so they cannot drift.

Sources, creation, randomness
-----------------------------

``from_array`` records a single-array source; ``from_record`` a named-field record of equal-
length arrays (the "events table" shape). Creation routines (``arange``, ``linspace``,
``full``...) record *parameterized* sources — the array is its recipe, so identity comes from
the parameters, not from data bytes.

Randomness is deliberately graph-friendly: ``default_rng(seed)`` returns a ``GraphedRNG`` whose
draws record ``(seed, draw_index)``-keyed source nodes. The same seed and draw order produce
the same node — and therefore the same bytes in the serialized IR — making "random" analyses
exactly as reproducible as everything else.

Reductions as monoids
---------------------

Each reduction op carries a ``Monoid`` (empty / per-chunk map / combine / finalize): ``sum`` is
the obvious one; ``mean`` accumulates ``(sum, count)`` pairs; ``var`` carries the three-term
sufficient statistics; ``histogram`` combines by bin-wise addition. This is what makes every
reduction **tree-reducible by the M7 executors** — a partition-local map followed by an
associative combine — without the executor knowing anything about the specific reduction.

Parquet I/O (rectilinear only)
------------------------------

``graphed.numpy.io`` specializes the shared :mod:`graphed.parquet` base: ``from_parquet``
records a deferred dataset source (blind partitions; schema from metadata via an explicit
arrow→numpy dtype map), ``to_parquet`` rides the shared write plan. Two deliberate constraints:
**rectilinear refusal** — a nested/list-typed parquet column is rejected with a pointer to the
ragged backend, rather than half-supported — and a **pandas-free decode path** (arrow's
convenience converters route through a pandas shim; this backend reads columns without it, and
a frozen test imports-blocks pandas to keep it that way).

Projection
----------

``project(array)`` returns the frontend's ``Projection`` by walking the graph with a
field-touch tracker: a source's record fields are wrapped, ops mark what they touch, and the
union per source is the read list. Rectilinear data needs no buffer-level view (there are no
offsets); the column view is exact here.


Phase 2 (deliberately not built)
--------------------------------

* **Broader ``np.*`` coverage on demand.** The dispatch tables grow by use case (the dask.array
  parity plan tracked the tiers); ``einsum``, FFTs, and linear algebra beyond ``gufunc``-style
  application are out of MVP scope.
* **Chunked/rechunked layouts.** The leading axis is the only partitioned axis; dask-style
  multi-axis chunking is explicitly not a goal for this backend.
* **Predicate pushdown** in the parquet reader (shared Phase-2 line with the frontend).

See :doc:`improvements` for the live tracked list.
