graphed-preserve
================

The analysis **preservation bundle** for ``graphed`` (milestone M9): a self-contained,
content-addressed export of an analysis that reproduces its histograms **bit-for-bit on a clean
machine** with no access to the original user code, environment, author, or input files — inputs are
resolved only via the bundle's content-addressed references.

- ``build_bundle`` captures the canonical serializable IR (M8), the M6 provenance/sourcemap, every
  ``External`` payload descriptor + the correction/model bytes (in the M8 content-addressed Store),
  the input datasets by content hash, the software environment, config, and seeds, into a top-level
  **manifest** whose hash is the bundle fingerprint.
- ``reproduce`` re-instantiates and runs the analysis from references alone (raising on any missing
  payload — never a silent wrong result).
- ``inspect`` renders the IR + provenance + payload inventory + opaque-node risk flags WITHOUT
  executing anything.

Reuses HEP standards (correctionlib / ONNX / UHI) — invents no formats. Builds on M8's determinism
and Store; this is the reproducibility requirement of A.3.1.

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
