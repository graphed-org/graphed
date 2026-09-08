Current limits and workarounds
==============================

Four things you are likely to run into while porting an analysis, and what to do about each.
:doc:`design` explains the machinery behind them.

An opaque callable has no result type
-------------------------------------

``Array.map(fn)`` and ``graphed.apply(fn, *arrays)`` record a call to a function graphed cannot
look inside, so the result's type is unknown — it prints as ``vector[object]`` — and it cannot
feed an operation that needs a type, such as a reduction:

.. code-block:: python

    import numpy as np
    from graphed import Session
    from graphed.numpy import NumpyBackend, from_array

    s = Session(NumpyBackend())
    x = from_array(s, "x", np.arange(6.0))
    doubled = x.map(lambda a: a * 2, name="double")

    print(s.form(doubled).describe())
    print(s.materialize(doubled))
    try:
        doubled.reduce("sum")
    except Exception as exc:
        print(type(exc).__name__, exc.detail)

Prints::

    vector[object]
    [ 0.  2.  4.  6.  8. 10.]
    GraphedTypeError sum requires a numeric array, got vector[object]

The same opacity blocks projection: graphed cannot tell which columns the callable reads. Every
projection entry point takes ``on_fail``, which defaults to ``"raise"`` — ``"warn"`` falls back to
reading every column and says so, ``"pass"`` assumes the callable adds nothing.

**Instead:** give the callable a type. On flat data, ``graphed.numpy.apply_gufunc(fn, signature,
*arrays, output_dtype=...)`` takes a gufunc signature such as ``"(i),(i)->()"``, which is enough
to infer the result form and keep it usable downstream. For anything else, record the call
yourself with ``Session.record_external(op, fn, inputs, descriptor=..., form=...)`` and declare
the form you are producing. Typed transformations are better expressed as operations than as opaque callables.

Behavior methods with arguments
-------------------------------

Recorded since m54: ``a.deltaR(b)``, ``jets.scaled(2.0, offset=1.0)`` and any other callable
attribute of the record's behavior class record one node per call, arguments and all. See
:doc:`../awkward/design` ("Behavior *methods* record the same way") for the constant rules, the
refusals and the per-partition limitation.

Cuts are not pushed into the reader
-----------------------------------

Projection narrows a read to the columns you touch, and inside a ragged column to the buffers you
touch, so a file with four hundred branches costs you the six you used. It does not push your
*filter* down: the rows are read, then cut.

**Instead:** if a dataset is routinely used through one selection, skim it once with
``graphed.awkward.to_parquet`` — including every systematic universe in the one file, via
``select=`` — and analyse the skim.

Parquet datasets are local, and the first file sets the schema
--------------------------------------------------------------

``from_parquet`` resolves a file, directory, glob or explicit list through ordinary filesystem
paths; remote object stores are not supported. The dataset's fields are taken from the first file
in the resolved list — sorted for a directory or a glob, your own order for an explicit list — so
a dataset whose files disagree about their schema is described by just one of them.

**Instead:** pass an explicit list of paths when you want a specific order or a specific first
file, and pass ``columns=`` to pin the fields you rely on so a mismatch surfaces at recording time
rather than mid-run.
