graphed-debug
=============

Debugging for ``graphed`` (milestone M6): opt-level lowering, source-mapped **picklable**
tracebacks, and visualization. The headline guarantee is that a runtime error — even one raised deep
inside a fused stage on a remote worker process — is re-raised in the driver as a formatted
traceback pointing at the **user's analysis line**, never a raw opaque worker traceback.

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
