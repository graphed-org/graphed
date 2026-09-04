graphed.debug
=============

Your analysis died on worker 47 of a batch job. What you get back is not a wall of framework
internals from another process — it is an exception on your machine that names the operation
that failed, the chunk of data that tripped it, the types that went in, and the line of your
analysis that wrote it.

The same package gives you two more things for when the error is not obvious: a mode that runs
your operations one at a time, exactly as you wrote them, and a live view of a running job in
your browser.

.. code-block:: python

   import numpy as np
   import graphed.debug as gd
   from graphed import Session
   from graphed.numpy import NumpyBackend, from_array

   s = Session(NumpyBackend())
   pt = from_array(s, "pt", np.arange(4.0))
   leading = pt.map(lambda a: a[100], name="leading")

   try:
       gd.run(s, leading, opt_level=1, partition="skim@0:4")
   except gd.StageError as err:
       print(err.user_frame.source)      # the line you wrote
       print(err.cause_type, "|", err.partition)

Which prints::

   pt.map(lambda a: a[100], name='leading')
   IndexError | skim@0:4

:doc:`design` is the how-to: reading the error, dropping to the one-operation-at-a-time view,
printing the arrowed traceback, and watching a run as it happens.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   design
   improvements

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
