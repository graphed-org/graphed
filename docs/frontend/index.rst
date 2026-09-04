Recording an analysis
=====================

Write the array code you would write anyway. Nothing runs: ``graphed`` records each operation,
works out the type and shape of its result from metadata alone, and remembers the line of your
file that created it. You get back a placeholder you can keep building on — and when you finally
ask for a value, what executes is the reduced graph, not a replay of your statements one at a
time.

.. code-block:: python

    import numpy as np
    from graphed import Session
    from graphed.numpy import NumpyBackend, from_array

    s = Session(NumpyBackend())
    pt = from_array(s, "pt", np.arange(6.0))
    sel = (pt * 2.0 + 1.0)[pt > 2.0]     # nothing has run yet

    p = s.provenance(sel)
    print(s.form(sel).describe())
    print(p.lineno, p.source)
    print(s.materialize(sel))

Prints::

    vector[float64]
    7 (pt * 2.0 + 1.0)[pt > 2.0]
    [ 7.  9. 11.]

Three things happened before any number was computed. The result's type was checked and is
available as its *form*; the node knows which line and which sub-expression produced it, which is
what puts an arrow on your own source when something fails on a worker later; and ``materialize``
computed the value here, in this process, because the data is small and local.

For a real dataset you hand the recorded graph to a runner instead — that is the same graph,
reduced once and evaluated partition by partition. :doc:`/architecture` shows where each piece
runs, and :doc:`../awkward/index` is the entry point for ragged HEP data (``gak`` mirrors
``ak``, so existing analysis code ports nearly unchanged).

:doc:`design` is the full walkthrough: why the graph stays small as you build it, how a
systematic variation rides through the whole analysis, what actually gets read off disk, and how
a recorded analysis becomes a plan a cluster can run.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   design
   improvements

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
