graphed
=======

A schedulable, serializable, debuggable HEP task-graph system. ``graphed`` records ordinary
array expressions into an interned Rust IR, reduces the graph to a concise set of fused stages via
equality saturation as the user builds it — so a large un-reduced graph never has to exist — and
executes, checkpoints, debugs, and preserves the result.

The distribution is a single pip package, ``graphed``, consolidated from what were formerly
separate prototype repositories (``graphed_core`` → ``graphed.core``, and likewise for
``awkward``/``numpy``/``debug``/``checkpoint``/``preserve``). Each subpackage below keeps its own
design writeup; the :doc:`api` page is the generated reference for the whole distribution.

.. toctree::
   :maxdepth: 1
   :caption: Packages

   frontend/index
   core/index
   awkward/index
   numpy/index
   debug/index
   checkpoint/index
   preserve/index
   corpus/index

.. toctree::
   :maxdepth: 1
   :caption: Reference

   api

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
