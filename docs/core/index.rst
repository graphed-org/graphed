graphed-core
============

Rust+PyO3 **thread-safe interned graph IR** for ``graphed`` (milestone M1). Structurally identical
nodes share one ``NodeId``, so ``node_count()`` equals the number of distinct structural keys. The
graph lives in Rust; this package **must not import awkward**, and there is **no optimization** here
(that is M4).

.. toctree::
   :maxdepth: 2
   :caption: Contents

   design
   api
   improvements

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
