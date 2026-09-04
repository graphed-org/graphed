Current limitations
===================

Two behaviours of ``graphed.awkward`` will surprise you if you meet them without warning. Both
are about what the recorded graph *claims*, not about whether it computes the right answer.

Missing API surface is listed separately, under "Not supported yet" on :doc:`design`.

The recorded type of a correction or model call is approximate
--------------------------------------------------------------

Everything else on this backend is typed by running the real awkward operation on tracing types
instead of on data (:doc:`design` shows how), so the recorded type is exact. An external call is
the exception: graphed does not
look inside a correctionlib correction or an ONNX model, so it cannot derive their output type.
It uses the **first input's** type instead.

For a scale factor that is what you want — a correction is shape-preserving, and one value comes
back per input value. For a model that reshapes its input, ``s.form(...)`` on the result will
describe the input rather than the output until the model actually runs.

**What to do.** If something downstream of a model depends on the exact output type — an axis on
a reduction, a projection you are reading off — materialize a small slice once and check the
type against what you expected, rather than trusting the recorded form.

A join key that is null on both sides matches
---------------------------------------------

``gak.join`` follows ``pandas.merge``, not strict SQL. In SQL, ``NULL`` matches nothing,
including another ``NULL``. In pandas — and here — two rows whose key is missing on *both* sides
are paired:

.. code-block:: python

    import awkward as ak
    from graphed import Session
    from graphed.awkward import AwkwardBackend, from_awkward, gak

    s = Session(AwkwardBackend())
    left = from_awkward(s, "L", ak.Array({"k": [1, None, 3], "l": [10, 20, 30]}))
    right = from_awkward(s, "R", ak.Array({"k": [1, None], "r": [100, 200]}))

    for how in ("inner", "left"):
        out = gak.join(left, right, on=["k"], how=how)
        print(how, ak.to_list(s.materialize(gak.without_field(out, "__joinkey__"))))

Printed output:

.. code-block:: text

    inner [{'k': 1, 'l': 10, 'r': 100}, {'k': None, 'l': 20, 'r': 200}]
    left [{'k': 1, 'l': 10, 'r': 100}, {'k': None, 'l': 20, 'r': 200}, {'k': 3, 'l': 30, 'r': None}]

A null on only *one* side still matches nothing, and the row survives with its null key intact,
which is what SQL does too. It is only the both-null case that differs.

Option-typed keys are easy to arrive at by accident in awkward — a column is option-typed after
a cut or a mask — so this is worth knowing even though a real event key (``run``, ``lumi``,
``event``) is never missing.

**What to do.** If you need SQL's behaviour, drop the null-keyed rows before joining, with
``gak.drop_none`` or an explicit ``~gak.is_none(key)`` mask.

A related, smaller effect: when either input's key is option-typed, the joined key is recorded as
option-typed for every ``how``, even where a run can prove no null survives. The recorded type is
a safe over-statement, never an under-statement, and the values you get back carry the exact type
the run produced.
