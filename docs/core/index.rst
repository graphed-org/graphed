graphed.core
============

Between the analysis you write and the workers that run it there has to be something small,
fixed and portable: a description of the work that the driver can hold in memory without
strain, that two runs agree on exactly, and that a worker with none of your source files can
still execute. ``graphed.core`` is that middle piece — the recording of your analysis, the
optimizer that shrinks it, and the plan a runner consumes.

Most of the time you never import it. You write ``graphed.awkward`` code, hand the result to a
runner, and ``graphed.core`` does its work out of sight. You reach for it directly when you are

* writing or adapting a runner (the ``Plan`` / ``Task`` / ``Partition`` contract is here),
* checkpointing a long job, or shipping a compiled analysis to a machine that has no copy of
  your code, or
* trying to find out why your graph is the size it is.

Here is the whole idea in eight lines — record an expression twice, get it once, and reduce
what you recorded to what a runner has to dispatch:

.. code-block:: python

    import graphed.core as gc

    s = gc.GraphStore()
    src = s.add_source("events", {"uri": "skim.root"})
    pt = s.add_op("pt", [src])
    pt_again = s.add_op("pt", [src])          # the same expression, written somewhere else
    total = s.add_reduction("sum", [pt])

    print("recorded twice, stored once:", pt == pt_again, "-- nodes so far:", s.node_count())
    reduced, report = s.reduce(outputs=[total])
    print("an executor sees", report["reduced_nodes"], "nodes in", report["stages"], "stage")

which prints::

    recorded twice, stored once: True -- nodes so far: 3
    an executor sees 3 nodes in 1 stage

:doc:`design` is the walkthrough: what the optimizer removes and why it cannot be wrong, why
reduction stays fast as your analysis grows, and why the thing you save to disk is a plan
rather than a pickle. :doc:`/api` is the generated reference.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   design
   improvements

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
