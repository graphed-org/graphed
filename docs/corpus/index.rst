graphed-corpus
==============

Ground-truth **requirements + runnable canonical-analysis fixtures** for ``graphed`` (milestone
M0.5), distilled from the A.8 reference corpus. The fixtures run on a deterministic synthetic
NanoAOD-like dataset and emit stored reference histograms so later milestones can assert
``graphed`` reproduces plain awkward bit-for-bit.

Unlike the subpackages, ``graphed-corpus`` is **not** part of the consolidated distribution — there
is no ``graphed.corpus`` import. It is a separate package (``graphed_corpus``), vendored under
``tests/_corpus/`` for this repository's own suite and published on its own for downstream
repositories that consume the fixtures. It is documented here because the requirements it encodes
are the specification the whole project is measured against.

Start with :doc:`design` for the engineering walkthrough.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   requirements/ops_catalog
   graph_bloat_note
   design
   improvements

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
