graphed-debug
=============

Debugging for ``graphed`` (milestone M6): opt-level lowering, source-mapped **picklable**
tracebacks, and visualization. The headline guarantee is that a runtime error — even one raised deep
inside a fused stage on a remote worker process — is re-raised in the driver as a formatted
traceback pointing at the **user's analysis line**, never a raw opaque worker traceback (plan A.3 #8).

.. toctree::
   :maxdepth: 2
   :caption: Contents

   api
   improvements

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
