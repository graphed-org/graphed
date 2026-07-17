graphed-core
============

The Rust+PyO3 spine of the graphed ecosystem: a **thread-safe interned graph IR**, the
**optimizer** that reduces recorded analyses to a few fused stages (DCE + CSE +
equality-saturation canonicalization + stage fusion — with an incremental mode so the
un-reduced graph never has to exist), and the **deterministic durable codec** (``GIR1``) plus
the ``DurablePlan`` layer executors and checkpoint stores consume.

Structurally identical nodes share one ``NodeId`` (``node_count()`` equals the number of
distinct structural keys); identical graphs serialize to identical bytes; reduction is
deterministic and benchmarked against super-linear scaling in CI. This package must not import
awkward — array semantics live in the backends.

Start with :doc:`design` for the engineering walkthrough (the optimizer section is written to
be read, not skimmed), then the :doc:`/api` reference for the surface.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   design
   improvements

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
