graphed.numpy
=============

Deferred numpy: your ``np.*`` code records into a graph instead of computing, and a runner
computes it later. Shapes and dtypes are known the moment you write each line — before any
data is read — so type mistakes surface at your line, not deep in a batch job. Reductions
split across partitions and their partial results merge in any order, so the answer does not
depend on how many workers ran. Seeded randomness reproduces exactly, and parquet columns
read without pandas.

Use it for flat, tabular data — a trigger-rate scan, a calibration table, an ntuple that is
really just columns of numbers. For ragged HEP event data (jets per event, muons per event),
use :doc:`graphed.awkward <../awkward/index>` instead; the two share the same session and
runners.

Install::

    pip install graphed[numpy]

A first taste — the same numpy you already write, deferred:

.. code-block:: python

    import numpy as np
    from graphed import Session
    from graphed.numpy import NumpyBackend, from_record

    s = Session(NumpyBackend())
    ev = from_record(s, "events",
                     pt=np.array([25.0, 55.0, 10.0, 80.0]),
                     eta=np.array([0.1, -1.2, 2.0, 0.4]))

    sel = ev["pt"][abs(ev["eta"]) < 1.5]     # records; nothing computes
    print(sel.shape, sel.dtype)              # known without data
    print(s.materialize(sel.mean()))

.. code-block:: text

    (None,) float64
    53.333333333333336

:doc:`design` walks through how recording, type inference, randomness, and parquet I/O work.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   design
   improvements

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
