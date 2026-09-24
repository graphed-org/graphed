Current limits
==============

What debugging does not do yet, and what to do instead.

**No live breakpoints.** You cannot pause a running task at an operation and inspect it on the
worker. ``replay`` steps through one task afterwards on your machine, from the input the run
captured (``aggregate_plan(store=)``) or by re-reading its partition, and only for
``aggregate_plan`` plans.

**Replay needs the recording session.** The unfused graph and your source lines come from the
``Session`` the analysis was recorded in, so a replay in a fresh interpreter, or from a
preservation bundle, is not possible; ``reproduce`` re-runs a bundle whole.

**No intermediate values from the run itself.** A capture keeps each task's input and partial,
not the values in between; ``replay`` recomputes those. The extra checks at ``opt_level=0`` are
structural: they catch an operation that produced nothing, not one that produced the wrong number.

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
