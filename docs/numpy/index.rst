graphed-numpy
=============

The **rectilinear backend** for ``graphed``: numpy's type system and idiom over the deferred
graph. Shapes and dtypes are inferred at record time; a broad ``np.*`` surface records through
``__array_ufunc__``/``__array_function__`` on the backend-supplied ``NumpyArray`` proxy;
reductions carry monoids for tree reduction; randomness is (seed, draw)-keyed and reproducible;
parquet I/O is rectilinear-refusing and pandas-free. Originally the M2 seam-prover, it is now a
usable deferred numpy.

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
