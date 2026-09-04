graphed.preserve
================

Hand someone a directory. They get your histograms back, bit for bit — with none of your
analysis code, none of your input files, and no idea what your environment looked like. Or they
read the whole analysis, operation by operation, without running any of it.

That directory is a **preservation bundle**: your recorded analysis, the data it read, and every
correction and model it called, all stored under a hash of their contents so nothing can be
quietly swapped underneath them.

Install
-------

.. code-block:: bash

   pip install "graphed[preserve]"

That pulls in awkward, correctionlib, and ONNX Runtime. The TensorFlow / PyTorch / XGBoost / JAX
/ Triton plugins need ``pip install "graphed[ml]"`` on top.

Build one, ship it, run it
--------------------------

.. code-block:: python

   import awkward as ak

   from graphed import Session
   from graphed.awkward import AwkwardBackend, from_awkward, gak
   from graphed.preserve import Bundle, build_bundle, inspect, reproduce

   events = ak.Array({"Jet": ak.zip({"pt": ak.Array([[40.0, 25.0], [55.0], [30.0, 60.0, 20.0]])})})

   s = Session(AwkwardBackend())
   ev = from_awkward(s, "events", events)
   ht = gak.sum(ev.Jet.pt, axis=1)

   build_bundle(
       "ttbar_bundle",
       session=s,
       value=ht,
       weight=gak.num(ev.Jet, axis=1) * 1.0,
       datasets={"events": events},
       histogram={"name": "ht", "bins": 4, "lo": 0.0, "hi": 200.0},
   )

   # ttbar_bundle/ now holds everything the run needs. Ship the directory; on the
   # other machine, with no analysis code and no input files present:
   elsewhere = Bundle.open("ttbar_bundle")
   print(reproduce(elsewhere))
   print(inspect(elsewhere).splitlines()[-1])

Prints::

    [0. 3. 3. 0.]
      no opaque nodes (every node is durable IR or a content-addressed payload)

Three calls, three jobs:

* ``build_bundle`` writes the directory — the recorded analysis, the input data, and every
  correction or model it calls, each filed under a hash of its own contents.
* ``reproduce`` runs it from those contents alone. If anything the analysis needs is missing, it
  raises; it never quietly returns a different number.
* ``inspect`` prints the analysis — every operation, the line of your source that recorded it,
  and every external dependency with its identity — without opening a file or evaluating
  anything. That last line is the risk report: it tells you whether some part of the analysis is
  an opaque Python callable that no bundle can preserve.

Read :doc:`design` next for what is actually in the directory, why identifying a correction by
its contents is not the same as identifying it by its bytes, and how to teach the bundle about a
model format it does not already know.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   design
   improvements

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
