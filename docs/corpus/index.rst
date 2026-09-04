How graphed's results are checked
=================================

Every claim ``graphed`` makes about correctness reduces to one check: the same analysis,
written in plain awkward, must produce the same histogram — exact bin counts, no
tolerances. The reference suite (the *corpus*) is where that plain-awkward side lives: a
set of standard HEP analyses, a deterministic synthetic dataset to run them on, and the
stored histograms they produce.

The corpus is a validation fixture, not a library. It ships inside the source checkout
under ``tests/_corpus/`` — there is no ``graphed.corpus`` to import, and you never need it
to run your own analysis.

:doc:`design` describes what the suite contains and what a check looks like.
:doc:`graph_bloat_note` is worth reading on its own — it quantifies the graph-size problem
that motivates ``graphed``, in terms a dask-awkward user will recognise.

.. toctree::
   :maxdepth: 1
   :caption: Contents

   design
   graph_bloat_note
   requirements/ops_catalog
   improvements
