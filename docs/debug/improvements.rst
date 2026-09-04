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

**The dashboard is observe-only.** There is no pause, cancel, or resubmit from the browser; stop
a run the way you would stop any Python job. The live view is in memory only — it is not written
into a preservation bundle or replayable after the process exits, so take a screenshot or keep
``dash.snapshot()`` if you want the numbers afterwards.

**Worker events reach a remote dashboard through the driver.** In a process pool, workers forward
their events to the driver process, which relays them to the server; a worker does not open its
own connection. For a run whose driver is behind a firewall from your browser, put the
``DashboardServer`` where the browser is and point the driver's ``NetworkMonitor`` at it.

**Graph pictures need your own renderer.** ``visualize`` emits Mermaid or Graphviz source text;
turning it into a PNG or an SVG is your toolchain's job (``mmdc``, the Mermaid live editor,
``dot``). There are no cost overlays or diffing renderers.
