API reference
=============

Lowering
--------

.. autofunction:: graphed_debug.lower

.. autoclass:: graphed_debug.LoweredGraph
   :members:

.. autoclass:: graphed_debug.LoweredStage
   :members:

.. autoclass:: graphed_debug.LoweredOp
   :members:

Errors + tracebacks
-------------------

.. autoclass:: graphed_debug.StageError
   :members:

.. autoclass:: graphed_debug.SourceFrame
   :members:

.. autofunction:: graphed_debug.format_traceback

Execution + visualization
-------------------------

.. autofunction:: graphed_debug.run

.. autofunction:: graphed_debug.visualize

Live dashboard (M37)
--------------------

The opt-in live dashboard: FINOS Perspective tables served over Tornado, fed by a websocket network
transport. See :doc:`design` for the full picture. Requires the ``dashboard`` extra.

.. autoclass:: graphed_debug.Dashboard
   :members:

.. autoclass:: graphed_debug.DashboardServer
   :members:

.. autoclass:: graphed_debug.NetworkMonitor
   :members:
