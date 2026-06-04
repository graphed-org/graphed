graphed
=======

The deferred-array frontend for ``graphed`` (milestone M2). A user writes ordinary array
expressions; ``graphed`` records them into the Rust-backed ``graphed-core`` store via a pluggable
``Backend``. The recorded graph is **backend-agnostic** — backends supply only form inference and
evaluation. No fusion (M4); no awkward (M3); provenance is a stub (M3).

.. toctree::
   :maxdepth: 2
   :caption: Contents

   api
   improvements

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
