graphed-checkpoint
==================

Content-addressed checkpoint store, deterministic resume, and error harvesting for ``graphed``
(milestone M8). On top of the M8 ``graphed_core.DurablePlan`` (versioned, byte-identical IR +
content-addressed ``task_id``), this package adds the durable side of M8:

- a crash-safe, content-addressed ``Store`` (atomic blob writes + an append-only manifest/journal +
  a dead-letter set);
- a ``run_resumable`` runner that **skips already-completed tasks** and recombines per-task partials
  in deterministic order — so a killed-and-resumed run does measurably less work and still matches an
  uninterrupted run **bit-for-bit** (no double-count, no lost partition);
- error harvesting with retry policies (``retry_n`` / ``retry_smaller_chunk`` / ``retry_elsewhere`` /
  ``quarantine``) and an error budget as a stopping condition.

Local-filesystem and single-machine only (the M8 guardrail). Analysis *preservation* is M9.

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
