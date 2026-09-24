graphed.debug
=============

Your analysis died on worker 47 of a batch job. What you get back is not a wall of framework
internals from another process — it is an exception on your machine that names the operation
that failed, the chunk of data that tripped it, the types that went in, and the line of your
analysis that wrote it.

The rest of the package is for everything around that error:

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

.. list-table::
   :header-rows: 1
   :widths: 55 45

   * - You want to
     - Reach for
   * - know which line, which chunk of data and which input types a failure came from
     - ``StageError``, and ``format_traceback`` to print it
   * - step through the values, one operation at a time, to see where they first go wrong
     - ``opt_level=0`` in ``run`` and ``lower``
   * - re-run the one task that failed, on your own machine, with the input it actually read
     - ``replay``
   * - watch a long run in your browser, and pause, resume or cancel it there
     - ``Dashboard`` (``control=True`` adds the buttons)
   * - keep a record of how a run went — timings, failures, environment — next to the analysis
     - ``RunRecorder``, then ``graphed.preserve.attach_run_report``

:doc:`design` walks through each of these with a program you can run.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   design
   improvements

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
