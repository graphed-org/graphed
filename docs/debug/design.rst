How debugging works
===================

A computation fails inside a worker. In most deferred-array frameworks what comes back is an
opaque wall of framework internals with your actual mistake nowhere in it, or worse, a string
that says "remote error: see worker logs". ``graphed.debug`` makes the opposite promise: **the
failure arrives on your machine as an exception that points at your analysis line**, carrying
the operation that failed, the types feeding it, and which chunk of data tripped it — and that
diagnosis survives optimization, pickling, and a process boundary.

Here is the whole thing in one program. The bug is a lambda that indexes past the end of an
array, which no amount of type checking can see before real data arrives:

.. code-block:: python

   import numpy as np
   import graphed.debug as gd
   from graphed import Session
   from graphed.numpy import NumpyBackend, from_array

   s = Session(NumpyBackend())
   x = from_array(s, "x", np.arange(4.0))
   pt = x * 2.0
   bad = pt.map(lambda a: a[100], name="oob")     # only fails once real data arrives

   try:
       gd.run(s, bad, opt_level=1, partition="skim@0:4")
   except gd.StageError as err:
       print(err.op, "|", err.cause_type, "|", err.partition)
       print(err.user_frame)
       print(gd.format_traceback(err))

Transcripts on this page show your analysis file by name; you will see its full path.

::

   external | IndexError | skim@0:4
   analysis.py:9
   Traceback (most recent call last) — user analysis frames:
         File "analysis.py", line 7, in <module>
             from_array(s, 'x', np.arange(4.0))
         File "analysis.py", line 8, in <module>
             x * 2.0
     --> File "analysis.py", line 9, in <module>
             pt.map(lambda a: a[100], name='oob')
   IndexError: index 100 is out of bounds for axis 0 with size 4  [stage op 'external', partition skim@0:4, opt_level=1]
     input forms: vector[float64]

The arrow is on the line you wrote. Above it is the trail of how that value was built, oldest
first — the same "most recent call last" convention Python uses, but every frame is your code.


Reading the error
-----------------

``StageError`` is a plain exception with plain fields, so you can branch on it in a handler
rather than parse a message:

``op``
   The operation that failed.

``user_frame``
   The closest analysis frame — file, line, function, and the sub-expression text. ``frames``
   is the whole trail behind it, failing line first.

``input_forms``
   The types and shapes going into the failing operation. Often this is the diagnosis by
   itself: a float where you expected an integer index, a flat array where you expected a
   ragged one.

``partition``
   *Which* chunk of data tripped it. Data-dependent bugs fail on some partitions and not
   others, and this is how you find the events to look at.

``cause_type`` / ``cause_message``
   The underlying exception, as data — the type name and the message, not a live traceback
   object.

``opt_level``
   Which structure was executing when it blew up (see the next section).

``variation``
   Which systematic universe failed, when the run carries variation labels. See
   `Which variation failed`_.

``format_traceback(err)`` renders all of it as the block above: the user-code frames with a
``-->`` on the faulty line, then the cause, the operation, the partition, and the input forms.
That is the thing to print in an ``except graphed.debug.StageError`` handler.


Running exactly what you wrote
------------------------------

By default your operations are fused: a run of array operations between boundaries executes as
one pass over the data, so the Python interpreter is touched once per group instead of once per
operation. That is what you want for speed, and it is what actually runs on a cluster.

When you are chasing where a value first goes wrong, you want the opposite — every operation
on its own, in the order you wrote them, nothing rearranged. That is ``opt_level=0``:

.. code-block:: python

   import numpy as np
   import graphed.debug as gd
   from graphed import Session
   from graphed.numpy import NumpyBackend, from_array

   s = Session(NumpyBackend())
   x = from_array(s, "x", np.arange(4.0))
   scaled = x * 2.0
   shifted = scaled + 1.0

   for level in (0, 1):
       lowered = gd.lower(s, shifted, opt_level=level)
       print(f"opt_level={level}: {len(lowered.stages)} stages, one_to_one={lowered.one_to_one}")
       for stage in lowered.stages:
           for m in stage.members:
               print(f"  stage {stage.index}  {m.op:<6} line {m.provenance.lineno}  form {m.form}")

::

   opt_level=0: 3 stages, one_to_one=True
     stage 0  x      line 7  form vector[float64]
     stage 1  mul    line 8  form vector[float64]
     stage 2  add    line 9  form vector[float64]
   opt_level=1: 2 stages, one_to_one=False
     stage 0  x      line 7  form vector[float64]
     stage 1  mul    line 8  form vector[float64]
     stage 1  add    line 9  form vector[float64]

At ``opt_level=0`` each operation is its own group and what you debug is literally what you
wrote; the debug runner additionally flags an operation that produced nothing. At
``opt_level>=1`` the same two operations share one group, using the same grouping rule the real
optimizer uses, so the structure you are looking at is the structure that executes.

Notice what did **not** change between the two listings: every operation keeps its own line
number, even the one buried in the middle of a fused group. That is why turning optimization on
never costs you the arrow — a fused group of four operations carries four source lines, and the
error names the right one. ``opt_level=0`` is for when you want to *step through* the values,
not for getting a correct traceback; you get that either way.

``lower()`` hands you the whole listing as data — ``lowered.stages``, each with ``.members``
carrying ``op``, ``input_ids``, ``form``, ``provenance``, and whether it is a boundary — so you
can inspect what will run without running it.


When the failure happens on a worker
------------------------------------

``StageError`` carries structured fields rather than a live Python traceback object, which is
exactly what makes it survive a process boundary intact: it pickles with every field, so a
worker can raise it, a pool can ship it, and your driver re-raises *the same* diagnosis.

.. code-block:: python

   import multiprocessing as mp
   from concurrent.futures import ProcessPoolExecutor

   import numpy as np
   import graphed.debug as gd
   from graphed import Session
   from graphed.numpy import NumpyBackend, from_array


   def analyse(chunk_id):
       s = Session(NumpyBackend())
       x = from_array(s, "x", np.arange(4.0))
       counts = x.map(lambda a: a.astype("int64") - 1, name="counts")
       unflat = counts.map(lambda a: np.repeat(a, a), name="unflatten")
       return gd.run(s, unflat, opt_level=1, partition=f"skim@{chunk_id}")


   if __name__ == "__main__":
       with ProcessPoolExecutor(max_workers=2, mp_context=mp.get_context("spawn")) as pool:
           try:
               pool.submit(analyse, 16384).result()
           except gd.StageError as err:
               print("re-raised in the driver:", type(err).__name__)
               print("partition:", err.partition, "| op:", err.op, "| opt_level:", err.opt_level)
               print("your line:", err.user_frame.lineno, "|", err.user_frame.source)
               print("cause:", err.cause_type, "-", err.cause_message)

::

   re-raised in the driver: StageError
   partition: skim@16384 | op: external | opt_level: 1
   your line: 14 | counts.map(lambda a: np.repeat(a, a), name='unflatten')
   cause: ValueError - repeats may not contain negative values.

A spawned pool needs the ``if __name__ == "__main__":`` guard, and the function the workers run
has to be importable — a lambda or a locally-defined function will fail to pickle before your
analysis ever starts.

The shape of this example is the shape of a real run. A worker task records its chunk and runs
it (or evaluates a compiled plan and wraps the failure), the ``StageError`` crosses the pool,
and your handler prints the same rendering it would have printed on your laptop.


Which variation failed
----------------------

When a run carries systematic variations, "it crashed" is not enough — you need to know which
universe crashed. A plan built through
:func:`graphed.aggregate_plan <graphed.aggregate.aggregate_plan>` can carry the variation labels
along with the compiled graph (``aggregate_plan(on_compiled=...)``), and a worker fills
``err.variation`` from them:

- the empty string means nominal — there is no ``"nominal"`` label to test against;
- several universes sharing one failing operation come back sorted and comma-joined,
  e.g. ``"jes_down,jes_up"``;
- a plan carrying no labels, or a failure at an operation the labels do not cover, propagates
  the original exception untouched rather than wrapping it in a half-filled diagnosis.

``err.summary()`` — the message you see if you never catch the error — includes the variation
when there is one.


Mistakes that never reach the data
----------------------------------

This page is about *run-time* failures. Most mistakes never get that far: the type and shape of
every result is worked out from metadata before a single file is opened, so a typo, a wrong
axis, or a type mismatch raises :exc:`GraphedTypeError <graphed.errors.GraphedTypeError>` at the
offending line while you are still recording — in a second, on your laptop, before any job is
submitted.

That leaves exactly one class of failure for this package: the data-dependent bug, the
off-by-one that only produces a negative count on real events. Those die at execution, with the
same quality of source mapping.


Drawing what will actually run
------------------------------

``visualize(lowered, fmt="mermaid" | "graphviz")`` renders a lowered graph as diagram source —
one node per group, its member operations, boundaries marked, and each node annotated with the
line that recorded it. Pass ``projection=`` (a mapping of source name to the set of columns
read) and each source is annotated with the columns it will actually read off disk.

.. code-block:: python

   import numpy as np
   import graphed.debug as gd
   from graphed import Session
   from graphed.numpy import NumpyBackend, from_array

   s = Session(NumpyBackend())
   x = from_array(s, "x", np.arange(4.0))
   lowered = gd.lower(s, (x * 2.0) + 1.0, opt_level=1)
   print(gd.visualize(lowered, fmt="mermaid"))

::

   flowchart TD  %% opt_level=1
     n0["source x\n@ analysis.py:7"]
     n2["stage[mul → add]\n@ analysis.py:8"]
     n0 --> n2

The output is deterministic text, which is the useful part: paste it into a notebook or a docs
page, or diff two lowerings of the same analysis to see what a change did to the grouping.
Turning that source into a PNG or an SVG is your toolchain's job — ``mmdc``, the Mermaid live
editor, or ``dot``.


Watching a run while it happens
-------------------------------

A long job that has not failed is its own kind of debugging problem: is it making progress, is
one worker carrying everything, where is the time going? ``Dashboard`` is a live view of a
running plan in your browser — task progress overall and per worker, a merged CPU profile, and
any ``StageError`` mapped back to your analysis line.

Unlike a scheduler that starts a diagnostics server on every client, nothing runs until you ask
for it. It needs the ``dashboard`` extra (``pip install "graphed[dashboard]"``) and, for a real
run, the executors package (``pip install graphed-executors``):

.. code-block:: python

   from graphed.core import Partition, Plan, Task
   from graphed.debug import Dashboard
   from graphed_executors.local import ProcessPoolExecutor


   def count_entries(partition, resources):
       return partition.entry_stop - partition.entry_start


   def add(a, b):
       return a + b


   def zero():
       return 0


   if __name__ == "__main__":
       tasks = [Task(i, Partition("skim.root", "Events", i * 1000, (i + 1) * 1000)) for i in range(8)]
       plan = Plan(process=count_entries, combine=add, empty=zero, tasks=tasks)

       with Dashboard(profile=True) as dash:
           print("dashboard at", dash.url)
           result = ProcessPoolExecutor(max_workers=2, monitor=dash.monitor).run(plan)
           snap = dash.wait_for(finished=len(tasks))

       print("result:", result.value)
       print("tasks finished:", snap["stats"]["finished"], "errored:", snap["stats"]["errored"])

::

   dashboard at http://127.0.0.1:52993/
   result: 8000
   tasks finished: 8 errored: 0

Your port will differ: without ``port=`` the dashboard takes a free one, which keeps it clear of a
Jupyter server already on 8888. Open ``dash.url`` while that runs and the page updates live. ``dash.attach(executor)`` is the
same wiring for an executor you built elsewhere; ``dash.snapshot()`` gives you the current
counters from Python, and ``dash.wait_for(finished=n)`` blocks until ``n`` tasks have landed —
events cross a websocket, so a run can finish microseconds before its last event arrives.

Attaching it does not change your answer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The monitor only observes. With a dashboard attached — profiling included — the reduced result,
the number of combines, and the serialized plan are byte-identical to the same run without one.
It cannot slow your run down either: events go onto a bounded buffer drained by a background
sender, and if that buffer fills (the oldest go first) or the connection drops, events are **dropped** rather than
blocking the executor, and a monitor that raises is swallowed rather than killing your job. A
dashboard is never a reason a job fails. The one exception is a dashboard you build with
``control=True`` and then press pause or cancel on: that changes the run, as the next section says.

What the browser shows you
~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Progress**, in the shape a scheduler dashboard uses rather than a scrolling event log a
  human cannot read mid-run: one overall bar tiling finished / in-flight / errored / pending,
  then per worker a chronological strip with one cell per task, coloured by state. The number
  of cells per row is your load balance at a glance; hover any cell — finished or still
  running — for that task's key, partition, entry count, state, and duration.
- **A merged CPU profile**, as a flamegraph, when you pass ``profile=True``.
- **A live table** of run counters you can pivot and filter in place.

Why profiling is cheap enough to leave on
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A per-call profiler hooks every Python call on the very thread doing your data work, which taxes
call-heavy analysis code regardless of the sample rate you ask for. Instead, each worker samples
its own task thread's stack from a *separate* thread every 10 ms and folds the samples into a
call tree. The data path is never hooked — and array kernels release the GIL while they work, so
the sampler mostly runs in time that would otherwise be idle. The worker trees ride the same
connection as the task events and the server merges them into one tree for the browser.

Pausing and cancelling from the browser
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``Dashboard(control=True)`` adds **pause**, **resume** and **cancel** buttons to the page and a
``RunControl`` at ``dash.control``; ``dash.attach(runner)`` hands the runner both the monitor and
the control. A pause lets the tasks already running finish and starts no more until you resume.
A cancel starts no more tasks, lets the running ones finish, and makes ``run`` return the reduction
of exactly the tasks that completed, with ``stopped`` set to ``StopReason.CANCELLED`` — the rules
are in :doc:`../core/design`, and every runner that takes a ``control`` follows them. The buttons post to
``/api/control``; this does the same from Python:

.. code-block:: python

   import json
   import time
   import urllib.request

   from graphed.core import Partition, Plan, RunState, SequentialRunner, Task
   from graphed.debug import Dashboard


   def press(dash, cmd):  # what the page's buttons send
       req = urllib.request.Request(
           dash.url + "api/control",
           data=json.dumps({"cmd": cmd}).encode(),
           headers={"Content-Type": "application/json"},
       )
       return json.loads(urllib.request.urlopen(req).read())


   with Dashboard(control=True) as dash:
       while dash.snapshot()["control_listeners"] == 0:  # the monitor connects in the background
           time.sleep(0.01)

       def process(partition, resources):
           if partition.entry_start == 2:
               print(press(dash, "cancel"))
               while dash.control.state is not RunState.CANCELLED:
                   time.sleep(0.01)
           return 1

       tasks = [Task(k, Partition(f"f{k}.root", "Events", k, k + 1)) for k in range(6)]
       plan = Plan(process=process, combine=lambda a, b: a + b, empty=lambda: 0, tasks=tasks)
       result = dash.attach(SequentialRunner()).run(plan)
       print(result.value, result.stopped, dash.control.state)

which prints::

   {'cmd': 'cancel', 'delivered': 1}
   3 cancelled running

``delivered`` counts the monitors that took the command, so 0 means no run is listening. The page
shows the state last *requested*, not an acknowledgement from the run. A control is reset to
running when a cancelled run ends, so a cancel stops one run. ``attach`` refuses, with a
``TypeError``, an executor that has no ``monitor`` attribute, or no ``control`` attribute on a
``control=True`` dashboard, since setting a new attribute would wire up something nothing reads.

The commands travel down the same websocket the events come up. A ``NetworkMonitor`` built with
``control=ctl`` connects as soon as it starts, keeps that connection open while the run is quiet —
a paused run emits nothing — and reconnects on its own if the server restarts. For a remote run,
start ``DashboardServer(control=True)``, and give the job ``NetworkMonitor(ingest_url,
control=ctl)`` and a runner built with ``control=ctl``. Anyone who can reach the server's port can
pause or cancel what it steers. The route answers 404 on a server without ``control=True``, 415
unless the body is sent as ``application/json`` (which a form on another site cannot do), and 400
for anything but ``pause``, ``resume`` or ``cancel``.

Watching a job on another machine
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``Dashboard`` is a server plus a loopback client bundled together. Split them and the same view
works across machines: run the server where your browser is, and point the executor's monitor at
its ingest address. This is a recipe — the address and the executor are yours to fill in:

.. code-block:: python

   # on your laptop
   from graphed.debug import DashboardServer

   server = DashboardServer(host="0.0.0.0", port=8899).start()
   print(server.url, server.ingest_url)

   # on the submit node, in the job that runs the plan
   from graphed.debug import NetworkMonitor
   from graphed_executors.local import ProcessPoolExecutor

   monitor = NetworkMonitor("ws://your-laptop:8899/ingest", profile=True).start()
   result = ProcessPoolExecutor(monitor=monitor).run(plan)
   monitor.close()

Lean events and a connection per worker
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A run of thousands of short tasks sends three events per task through your driver, and every
worker formats a partition label for each one. Two opt-in switches on ``NetworkMonitor`` cut that
cost; each trades away some precision in what the page shows.

``lean=True`` asks each worker for one event per task — the ``FINISHED`` or ``ERRORED`` — with no
partition label, so no worker ever formats one. The driver's ``SUBMITTED`` still carries the label,
and the dashboard server works out the rest: a task started when the same worker's previous task on
the same connection ended, and it is in flight from its ``SUBMITTED`` until its terminal event.
What that costs you:

* **Durations are estimates.** For a worker that runs one task at a time (every local executor's
  workers) a derived duration is an upper bound: it includes any queue, idle and paused time since
  that worker's previous task, which on a reused dashboard can be an earlier run's, and a worker's
  first task on a connection shows zero length. Where several task threads share one worker name
  (a multi-threaded dask worker's address, a parsl thread pool's ``host:pid``) a duration has no
  bound either way.
* **In-flight is approximate.** It is the number of submitted tasks with no terminal event yet,
  capped at the number of worker names seen so far. So it counts a name running several tasks at
  once as one, reads low until every running worker has finished a task, and can read high on a
  reused dashboard whose worker names change from run to run. There is no per-worker in-flight
  count.
* **An interrupted run leaves rows behind.** A task that a cancel kept from starting, or that a
  failure stopped, stays submitted, so after such a run in-flight stays above zero. On a rerun on
  the same dashboard, a task's row can stay at ``submitted`` if the rerun's terminal event for it
  arrives before the rerun's ``SUBMITTED``: worker and driver use separate connections, and the
  server sees no run boundary.

``per_worker=True`` takes the driver out of the event path. ``monitor.worker_monitor_factory()``
then returns a picklable factory, and an executor whose workers are separate processes builds one
monitor per worker process from it, so each worker sends its task events and profile trees over
its own connection — the workers have to be able to reach the dashboard's address. The driver
keeps the ``SUBMITTED`` events and the combine counts, and thread workers share the driver's
monitor. A worker's connection opens on its first event, and at exit a worker waits at most half a
second for it to drain. That exit flush runs through ``atexit``, which a ``fork`` child skips
before Python 3.13; the executors start their workers with ``spawn``.

.. code-block:: python

   import time

   from graphed.core import Partition, Plan, SequentialRunner, Task
   from graphed.debug import DashboardServer, NetworkMonitor

   server = DashboardServer().start()
   monitor = NetworkMonitor(server.ingest_url, lean=True)
   tasks = [Task(k, Partition(f"f{k}.root", "Events", 0, 10)) for k in range(4)]
   plan = Plan(process=lambda p, r: p.n_entries, combine=lambda a, b: a + b, empty=lambda: 0, tasks=tasks)
   print(SequentialRunner(monitor=monitor).run(plan).value)
   monitor.close()
   while server.snapshot()["stats"]["finished"] < 4:
       time.sleep(0.01)
   stats = server.snapshot()["stats"]
   print(stats["submitted"], stats["started"], stats["finished"], stats["inflight"])
   print(NetworkMonitor(server.ingest_url, per_worker=True).worker_monitor_factory() is not None)
   server.stop()

which prints::

   40
   4 4 4 0
   True

Keeping a record of a run
-------------------------

The dashboard shows a run while it happens and keeps nothing. ``RunRecorder`` is a monitor that
records every task event of a run at the driver; ``report()`` folds them into a ``RunReport``:
the outcome, per task its partition, worker, state, duration and error, the ``StageError`` the
run raised, and the environment digest a preservation bundle would record for this interpreter.
A report is plain data, and ``to_json()`` / ``RunReport.from_json()`` round-trip it.

.. code-block:: python

   from graphed.core import Partition, Plan, SequentialRunner, Task
   from graphed.debug import RunRecorder


   def entries(partition, resources):
       if partition.entry_start == 1000:
           raise ValueError("bad counts in this chunk")
       return partition.entry_stop - partition.entry_start


   tasks = [Task(i, Partition("skim.root", "Events", i * 1000, (i + 1) * 1000)) for i in range(3)]
   plan = Plan(process=entries, combine=lambda a, b: a + b, empty=lambda: 0, tasks=tasks)

   rec = RunRecorder()
   try:
       SequentialRunner(monitor=rec).run(plan)
   except ValueError as exc:
       report = rec.report(error=exc)

   print(report.outcome, report.error, report.failed_keys)
   for t in report.tasks:
       print(t.key, t.partition, t.state, t.duration_s is not None)

::

   failed ValueError: bad counts in this chunk (1,)
   0 skim.root:Events:0-1000 finished True
   1 skim.root:Events:1000-2000 errored True
   2 skim.root:Events:2000-3000 submitted False

Pass ``result=`` the ``ExecResult`` of a run that returned, or ``error=`` the exception it raised —
exactly one. A duration is the task's terminal time minus its start time, both stamped by the
worker that ran it; a task that never started has none. A retried task reports its last attempt.

``RunRecorder(inner=dash_monitor)`` forwards every call to another monitor, so a dashboard can
watch the run it records. The recorder takes the full event stream even when that monitor is lean
or pushes from its workers: the report needs every ``STARTED``.

**One run at a time.** Each ``report()`` takes the events that arrived since the previous one, so
call it after each run. A recorder shared by concurrent runs, or by plans queued with
``executor.submit()`` whose results you have not awaited in turn, mixes their events.

**Completeness is opt-in.** ``RunRecorder`` carries ``complete_events = True``, so a runner that
honours it delivers a run's events by the time ``run()`` returns or raises. A run failed by a
raising task reports that task ``errored``; a crashed worker ships nothing. On a peer route or a
``SubmitRunner`` adaptive run, other tasks in flight when the run failed can read ``started`` or
``submitted``.


Replaying a task
----------------

One task failed, or returned a partial you do not believe, and you want to watch it happen on
your own machine rather than read about it in a log. After a run, finished or failed,
``replay(plan, key, *outputs)`` re-executes one task of an
``aggregate_plan`` plan on your machine, one operation at a time at ``opt_level=0``, with the
exact input that task read. Build the plan with ``store=`` (a directory every worker shares, or
an fsspec URL) and each task keeps its input chunk and its partial there as it runs; without it,
replay re-reads the task's partition. ``outputs`` are the Arrays you gave ``aggregate_plan``, in
the same order: the session they were recorded in supplies the unfused graph and your source
lines, so replay works while that session is alive.

.. code-block:: python

   import tempfile

   import awkward as ak
   import numpy as np
   import graphed.debug as gd
   from graphed import Session, aggregate_plan
   from graphed.awkward import AwkwardBackend, from_parquet, gak
   from graphed.core import SequentialRunner

   root = tempfile.mkdtemp()
   ak.to_parquet(ak.Array({"x": np.arange(1000.0)}), f"{root}/d.parquet")


   def calibrate(x):
       if float(x[0]) == 500.0:
           raise ValueError("no calibration for this block")
       return x * 1.01


   s = Session(AwkwardBackend())
   ev = from_parquet(s, "events", f"{root}/d.parquet")
   total = gak.sum(ev.x.map(calibrate, name="calibrate"), axis=None)

   plan = aggregate_plan(
       total,
       reduce=lambda outs: float(outs[0]),
       combine=lambda a, b: a + b,
       empty=lambda: 0.0,
       steps_per_file=4,
       store=f"{root}/capture",  # keep each task's input and partial
   )
   try:
       SequentialRunner().run(plan)
   except gd.StageError as exc:
       print("run failed:", exc.cause_message)

   r = gd.replay(plan, 2, total)
   print("input from:", r.input_source)
   try:
       for step in r.steps():
           print(f"  {step.node.op:<8} line {step.node.provenance.lineno}")
   except gd.StageError as err:
       print(f"  failed at {err.op} on line {err.user_frame.lineno}: {err.cause_message}")

   d = gd.replay(plan, 1, total).diff()
   print(d.reference, d.equal, d.replayed)

::

   run failed: no calibration for this block
   input from: store
     events   line 21
     field    line 22
     failed at external on line 22: no calibration for this block
   recorded True 94561.25

``steps()`` is lazy: each ``Step`` carries the lowered operation (its op, kind, form and your
source frame), its value, and the seconds it took, and nothing runs until you ask for the next
one. A step that fails raises the same ``StageError`` ``gd.run`` would, pointing at your line.
``value`` is the task's partial, the plan's ``reduce`` over the replayed outputs. ``diff()``
compares it with what the run recorded (``reference == "recorded"``), or, when the run kept no
output for that task, with the plan's fused graph run again on the same input
(``"re-evaluated"``), so a difference between the fused and the one-at-a-time evaluation shows up
too. The comparison is by value: dicts by key, tuples and lists by position, arrays with
``numpy.array_equal``, which counts NaN as unequal to itself.

External evaluators, including ``aggregate_plan(externals=)`` overrides, are the plan's own, so
the replay calls what the run called. A capture root holds one run: see "Keeping each task's
input for replay" in the checkpoint guide.


Sending the events somewhere else
---------------------------------

The dashboard is one consumer of a stream any executor emits. Anything with the four methods
below is a monitor: your own progress bar, a log line per task, a metrics push. Per task you get
exactly one ``SUBMITTED``, then one ``STARTED``, then exactly one of ``FINISHED`` or
``ERRORED`` — except that a task a cancel kept from starting stops at ``SUBMITTED``.

.. code-block:: python

   from graphed.core import Partition, Plan, Task
   from graphed.core.execution import SequentialRunner, TaskEvent, TaskPhase


   class PrintMonitor:
       def on_task(self, event: TaskEvent) -> None:
           if event.phase is not TaskPhase.SUBMITTED:
               print(f"{event.phase.value:<8} key={event.key} {event.partition} error={event.error}")

       def on_profile(self, worker: str, payload: bytes) -> None:
           pass

       def on_combine(self, leaves_done: int) -> None:
           pass

       def worker_profiler_factory(self):
           return None


   def entries(partition, resources):
       if partition.entry_start == 1000:
           raise ValueError("bad counts in this chunk")
       return partition.entry_stop - partition.entry_start


   tasks = [Task(i, Partition("skim.root", "Events", i * 1000, (i + 1) * 1000)) for i in range(3)]
   plan = Plan(process=entries, combine=lambda a, b: a + b, empty=lambda: 0, tasks=tasks)

   try:
       SequentialRunner(monitor=PrintMonitor()).run(plan)
   except ValueError as exc:
       print("run failed:", exc)

::

   started  key=0 skim.root:Events:0-1000 error=None
   finished key=0 skim.root:Events:0-1000 error=None
   started  key=1 skim.root:Events:1000-2000 error=None
   errored  key=1 skim.root:Events:1000-2000 error=ValueError: bad counts in this chunk
   run failed: bad counts in this chunk

``TaskEvent`` is an immutable, picklable, display-only record — phase, task key, worker,
timestamp, partition label, entry count, bytes read, error string. It is not a handle on the run:
a monitor cannot reorder tasks, change the reduction tree, or touch a result, which is what lets
you attach one to a production job without thinking about it.


Not supported yet
-----------------

See :doc:`improvements` for the current limits and what to do instead.
