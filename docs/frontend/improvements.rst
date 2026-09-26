Current limits and workarounds
==============================

Five things you are likely to run into while porting an analysis, and what to do about each.
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

**Instead:** give the callable a type. ``output_type=`` on ``map`` and ``graphed.apply`` declares
the type of each element of the result, and the recorded form is that type:

.. code-block:: python

    import numpy as np
    from graphed import Session
    from graphed.numpy import NumpyBackend, from_array

    s = Session(NumpyBackend())
    x = from_array(s, "x", np.arange(6.0))
    doubled = x.map(lambda a: a * 2, name="double", output_type="float64")
    print(s.form(doubled).describe())
    print(s.materialize(doubled.reduce("sum")))


    def pair(a):
        out = np.zeros(len(a), dtype=[("pt", "f8"), ("eta", "f8")])
        out["pt"], out["eta"] = a, -a
        return out


    rec = x.map(pair, name="pair", output_type=[("pt", "f8"), ("eta", "f8")])
    print(s.form(rec).describe(), s.form(rec["eta"]).describe())

Prints::

    vector[float64]
    30.0
    record[pt,eta] vector[float64]

The declaration is part of the node's identity and is never a cast: the callable must return
what it declares. On flat data, ``graphed.numpy.apply_gufunc(fn, signature, *arrays,
output_dtype=...)`` also takes a gufunc signature such as ``"(i),(i)->()"``, which types the
core dimensions too. Typed transformations are better expressed as operations than as opaque
callables.

A behavior method runs per partition
------------------------------------

``a.deltaR(b)``, ``jets.scaled(2.0, offset=1.0)`` and any other callable attribute of a record's
behavior class record one node per call, arguments and all. The body runs on one chunk at a time,
the same rule ``map_partitions`` has, so a body that consumes the event axis — ``self[:1]``, an
``axis=0`` reduction — answers per partition rather than globally. It still records, because the
tracer has forgotten the outer length; only a scalar result is refused.

**Instead:** keep a method row-wise, and do anything that crosses events on the recorded array it
returns. :doc:`../awkward/design` ("Behavior *methods* record the same way") has the rules for
arguments, what a worker needs to resolve the behavior, and what is refused at the call.

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

A part is named before its partition is read
--------------------------------------------

A write's ``name`` sees the task's partition as the plan holds it. A blind partition (the default
of ``steps_per_file``, which opens no file to plan) has no entry range until a worker reads it, so
its ``entry_start`` and ``entry_stop`` are both 0, and a name built from them is the same for every
step of a file. ``aggregate_plan`` refuses such a plan when it is built.

**Instead:** name blind parts by ``partition.blind_step``, or pass explicit ``partitions=`` (for
example from the dataset's entry counts) and name them by their range.
