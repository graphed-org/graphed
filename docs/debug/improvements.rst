Current limits
==============

What debugging does not do yet, and what to do instead.

**No stepping or time travel.** You cannot pause a run at an operation, inspect it, and step
forward, and you cannot replay a finished run against captured data. Debugging is static: lower
the graph, read it, run it, read the error. To narrow down where a value first goes wrong, run
with ``opt_level=0`` (one operation at a time) and bisect by materializing intermediate results
yourself.

**No value capture.** The extra checks at ``opt_level=0`` are structural — they catch an
operation that produced nothing, not one that produced the wrong number. If you need to see an
intermediate array, split the analysis and materialize it.

**Per-operation contracts are coarse.** Beyond "this operation produced something", there are no
per-operation dtype and shape assertions at ``opt_level=0``. The recorded type and shape of every
node are still there for you to check yourself: ``lower(...)`` gives you ``.form`` on each member.

**The dashboard cannot resubmit.** It can pause, resume and cancel a run (``control=True``), but
it cannot retry a failed task or resubmit a run. The live view is in memory only and is not
replayable after the process exits; to keep a run's per-task record, attach a ``RunRecorder``
and keep its report in a preservation bundle (``attach_run_report``), or keep
``dash.snapshot()``.

**Lean dashboard figures are estimates.** With ``NetworkMonitor(lean=True)`` the server derives
start times and in-flight counts from terminal events: durations are upper bounds only for worker
names that run one task at a time, in-flight can read low during the first wave or high on a reused
dashboard, and a rerun can leave a task that a cancelled or failed run left at ``SUBMITTED`` with a
``submitted`` row. Leave ``lean`` off when
you need the exact figures.

**Graph pictures need your own renderer.** ``visualize`` emits Mermaid or Graphviz source text;
turning it into a PNG or an SVG is your toolchain's job (``mmdc``, the Mermaid live editor,
``dot``). There are no cost overlays or diffing renderers.
