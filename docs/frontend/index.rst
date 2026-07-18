graphed (the frontend)
======================

The top-level ``graphed`` package is the deferred-array **recording frontend**: ordinary array
expressions become nodes in the Rust-backed ``graphed.core`` store through a pluggable, five-method
``Backend`` protocol.
Type errors surface at the recording line (form inference on metadata only); results
materialize through the reference walker or — the real path — compile to a reduced IR any
executor evaluates. The frontend is strictly backend-agnostic: numpy and awkward idioms live
in their backends, common machinery (projection, compilation, the parquet/write I/O bases)
lives here.

Start with :doc:`design` for the engineering walkthrough, then the :doc:`/api` reference.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   design
   improvements

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
