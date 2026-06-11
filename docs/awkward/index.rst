graphed-awkward
===============

The **ragged backend** — the one HEP analyses live on: awkward typetracer form inference
(metadata only, reusing awkward's own tracer) + real-array evaluation; the ``gak`` function
namespace with enforced interface parity against ``ak.*`` (an inspection-based anti-drift pin);
session-owned behaviors (vector four-vectors); the structural reduction rule (event-axis
reductions are boundaries, per-event reductions fuse); buffer-granular projection; generic
parquet I/O over the partitioned-source protocol; and content-hashed correctionlib/ONNX
External payloads.

Start with :doc:`design` for the engineering walkthrough.

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
