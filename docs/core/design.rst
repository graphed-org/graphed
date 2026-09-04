How graphed.core works
======================

.. contents::
   :local:
   :depth: 2

You wrote two hundred lines of array code and fifty systematic variations of it. Somewhere
between your keyboard and a thousand batch workers that has to become a small, fixed,
portable description of work — small enough that your driver process is not the bottleneck,
fixed enough that two runs agree exactly, and portable enough that a worker holding none of
your source files can still execute it.

``graphed.core`` is that middle. It records what you wrote, shrinks the recording as you build
it, and hands a runner a plan it can serialize, hash, ship and resume. This page is the
walkthrough: what gets removed, why removing it can never change your answer, why the removal
stays cheap as your analysis grows, and why what you save to disk is a plan rather than a
pickle of live objects.

Everything below uses the low-level recording calls directly, because that is the clearest way
to see the machinery. In real analysis code the frontend makes these calls for you.

Start with the whole pipeline on one small graph — the kind of accumulation that happens
naturally as analysis code grows:

.. code-block:: python

    import graphed.core as gc

    s = gc.GraphStore()
    src = s.add_source("events", {"uri": "data.root"})
    pt = s.add_op("pt", [src])
    w = s.add_op("weight", [src])
    a = s.add_op("add", [pt, w])
    b = s.add_op("add", [w, pt])                                  # the same sum, written the other way round
    scaled = s.add_op("mul", [a], {"scalar": 1.0, "side": "r"})   # a helper that multiplies by 1.0
    s.add_op("cut", [pt])                                         # tried, abandoned, never an output
    out = s.add_reduction("sum", [s.add_op("mul", [scaled, b])])

    reduced, report = s.reduce(outputs=[out])
    for k in sorted(report):
        print(k, report[k])
    for n in reduced.nodes():
        print(n["id"], n["kind"], n.get("n_members", ""))

which prints::

    boundary_nodes 2
    canonical_nodes 6
    input_nodes 9
    reachable_nodes 8
    reduced_nodes 4
    stages 2
    0 source
    1 stage 3
    2 stage 1
    3 reduction

Nine recorded operations became four things a runner dispatches, and two of those four are the
read and the sum you asked for. The abandoned cut is gone; the sum written both ways round is
one sum; the multiply-by-1.0 helper left no trace. The report is not decoration — it is how you
find out what your own graph did, at every step.

The rest of this page explains each of those numbers.


Why writing the same expression twice costs nothing
---------------------------------------------------

A recorded node is one of six kinds: a read of an input dataset (``source``), an array
operation (``op``), an aggregation (``reduction``), a call out to something graphed does not
look inside such as a correction set or an ONNX model (``external``), a data exchange between
partitions (``exchange``), and a relational join (``join``). The optimizer adds a seventh,
``stage``, which you will meet below; nothing that records an analysis can create one.

A node **is** its structure: its kind, its name, its parameters, its inputs, and — for an
external call — the full descriptor of the payload it invokes, content hash included. The store
keeps a table keyed on exactly that, so recording a structurally identical node hands you back
the node that already exists (*interning*). Three consequences are worth carrying with you:

* **Duplicate expressions collapse on the way in.** Recording ``events.pt * 2`` in two places
  produces one node. There is no later pass hunting for common subexpressions in what you
  recorded; the duplicate never became two nodes.
* **Identity is content, not history.** Two sessions that record the same analysis produce
  graphs that serialize to identical bytes, whatever order you wrote things in.
* **External payloads are part of that identity.** Swap an ONNX model for a retrained one and
  its content hash changes, so it is a *different* node. Nothing downstream can confuse the
  two, and nothing cached under the old model is served for the new one.

Floats in parameters
~~~~~~~~~~~~~~~~~~~~

Parameter values are ``int``, ``float``, ``bool`` or ``str``. Hashing floats needs a total
order, so a float is keyed by its IEEE bit pattern with every ``NaN`` collapsed to one
canonical pattern. The user-visible effect: recording the same ``NaN``-parameterised cut twice
gives you one node, while ``0.0`` and ``-0.0`` stay *distinct*. Treating those two as equal is
a claim about values, which is the optimizer's job, not the recording table's.

Recording from many threads
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Interning is a read-modify-write — look the structure up, and if it is absent append a node and
remember its id — so the store's inner state sits behind a single lock that makes that step
atomic. This is correct under the GIL and under free-threaded CPython alike; the critical
section is a hash-map probe plus a vector push. The locking discipline is model-checked and
stress-tested from many threads in both Rust and Python.

One property matters more than it looks: **which outputs you want is a property of the request,
not of the store.** ``reduce``, ``serialize`` and the incremental reducer's ``finalize`` all
take the output set as an argument and use exactly that set — there is no output mutator on the
store. So compiling two different expressions from one session gives byte-identical results to
compiling each in a fresh session, and two threads compiling at once cannot interfere, because
compiling only ever reads.


What the optimizer removes, and why it cannot be wrong
------------------------------------------------------

``reduce`` turns what you recorded into a graph of *stages* — the unit a runner dispatches. It
is four passes around a swappable rewrite engine::

    reduce(nodes, outputs):
      1. reachable = drop everything no output depends on
      2. canonical = engine.canonicalize(reachable)     # equality saturation
      3. deduped   = collapse newly-identical nodes
      4. stages    = fuse maximal runs of ops between boundaries
         rebuild into a fresh interned store

Dropping dead code and collapsing duplicates are plain passes *outside* the engine. The engine
does exactly one thing — decide which nodes are **semantically equivalent** — because that is
the only part that benefits from the heavy machinery, and keeping it to that one job is what
makes it replaceable. The engine boundary traffics only in an engine-neutral graph, so no
detail of the current rewrite library escapes into the passes around it.

Dropping what you did not ask for
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pass 1 is plain reachability from the outputs you requested, followed by a compaction that
preserves topological order. Nothing on a path to an output is ever dropped; everything else is.
A cut you tried and abandoned costs one entry in the recording table and nothing more — it is
never read, never scheduled, never shipped.

Recognising two ways of writing the same thing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Interning collapses expressions that are *structurally* identical, but people write the same
thing in different shapes: ``a + b`` here and ``b + a`` there; a convenience helper that
multiplies by ``1.0``; a weight of ``x + 0.0``. Structure cannot see through those. A rewrite
system can.

The technique is *equality saturation over e-graphs*. An e-graph stores a set of terms together
with an equivalence relation over them, compactly: each *e-class* holds all the terms known to
be equal, and terms name e-classes — not concrete terms — as their children, so one e-class of
"things equal to ``a + b``" serves every parent expression at once. Saturation seeds the e-graph
with your program and then applies rewrite rules not as destructive edits but as *additions*:
each match merges or extends e-classes. Run to saturation, the e-graph encodes every way of
writing your program under the rules, and the equivalence classes are exactly the answer wanted.

Loading your graph into it is one linear pass in topological order.

**The rule set is small on purpose, and every rule is provably sound:**

* *commutativity* for the symmetric operations — ``add``, ``mul``, ``and``, ``or``, ``eq``,
  ``ne``, ``maximum``, ``minimum`` — where argument order genuinely cannot matter, so
  ``add(a, b)`` and ``add(b, a)`` merge into one class. Asymmetric operations (``sub``, ``div``,
  ordered comparisons) are absent, because for those the order is the meaning.
* *identity elimination* for ``x + 0.0`` and ``x * 1.0``, both operand orders, matched on the
  scalar's exact bit pattern.

Rewrites that depend on array semantics — mask fusion, field collapse — are excluded, because
their soundness depends on things this layer cannot see. That restraint is not modesty: an
unsound rule corrupts *every* analysis that trips it, silently, and you would find out from a
disagreeing number months later.

The saturation budget is an iteration limit and a node limit, never a wall-clock limit. A time
budget would make your optimized graph depend on how loaded the machine was, and the whole
point is that identical input gives a byte-identical reduced graph on every machine, every run.

Collapsing what the rewrites made identical
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Canonicalization rewrites inputs, so nodes that were distinct can become identical afterwards:
once ``add(a, b)`` and ``add(b, a)`` share a representative, ``inc(add(a, b))`` and
``inc(add(b, a))`` are the same operation on the same inputs — but they are still two entries in
the list. One linear pass collapses them. Interning did this at record time for structural
duplicates; this pass re-establishes it for the ones that only just became duplicates.


Why reduction does not slow down as your analysis grows
-------------------------------------------------------

This is the failure that motivates the whole package. A graph system whose optimizer costs more
than linear time in graph size eventually spends more wall time optimizing than computing, and
a systematics-heavy analysis — thousands of variations hanging off one deep shared selection
chain — is precisely the shape that provokes it. Two things keep that from happening here.

Picking a representative without searching for one
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After saturation you have to pick one concrete term per e-class — *extraction*. The general
answer is a recursive cost-based search costing depth × nodes, which on a deep shared chain
degrades to quadratic. That would relocate the very failure this project exists to avoid into
the optimizer.

The escape is a property of *this* rule set: every rule only ever equates a node with an
**already-existing, earlier** node — the commuted twin of something recorded before, or the
input of an identity operation. No rule invents a term that would have to be materialized. So
extraction does not need to search at all: quotient the original node list by the e-graph's
equivalence and keep the earliest member of each class. One linear pass. Keeping the earliest
member is topologically safe — it can only refer to even-earlier representatives — and it *is*
the cost function, because for these rules the earliest form is the simplest form.

Reduction time is benchmarked across graphs of 1k, 2k, 4k and 8k nodes, and the benchmark
**fails on super-linear growth**, so this property cannot quietly regress.

Reducing as you build, not after
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The pipeline above sees the whole recording at once. The stronger goal is that the large
un-reduced graph never exists at all. ``IncrementalReducer`` consumes the recording delta by
delta: each ``step`` processes only the nodes added since the last one, maintaining a canonical
form as you build.

A single pass per node is enough because every rule in the sound set is *constructor-local* —
whether an operation is an identity, and what its canonical orientation is, depends only on the
operation itself and its inputs' already-canonical ids, never on its consumers or on nodes that
do not exist yet. Canonicalizing each node once, on arrival, in topological order therefore
reaches the same fixpoint as saturating the whole graph. The engine's rules and the incremental
canonicalizer are generated from the same shared constants, so the two paths cannot drift apart,
and ``finalize`` is required to produce byte-identical output to a one-shot ``reduce``.

You can watch the work stay proportional to the delta:

.. code-block:: python

    from graphed.core import GraphStore, IncrementalReducer

    s = GraphStore()
    reducer = IncrementalReducer()

    src = s.add_source("events", {"uri": "skim.root"})
    pt = s.add_op("pt", [src])
    print("first batch, new nodes reduced:", reducer.step(s))

    for i in range(50):                       # fifty systematic variations of one weight
        s.add_op("mul", [pt], {"scalar": 1.0 + i / 100.0})
    print("second batch, new nodes reduced:", reducer.step(s))

    out = s.add_reduction("sum", [pt])
    print("third batch, new nodes reduced: ", reducer.step(s))

    print("nodes recorded:", s.node_count(), "-- total nodes reduced:", reducer.total_work())

    incremental, _ = reducer.finalize(s, outputs=[out])
    one_shot, _ = s.reduce(outputs=[out])
    print("same bytes as reducing all at once:", incremental.serialize() == one_shot.serialize())

which prints::

    first batch, new nodes reduced: 2
    second batch, new nodes reduced: 50
    third batch, new nodes reduced:  1
    nodes recorded: 53 -- total nodes reduced: 53
    same bytes as reducing all at once: True

Fifty-three nodes recorded, fifty-three nodes reduced. A reducer that secretly re-scanned its
history each time would report far more, and it would not agree byte-for-byte with the one-shot
path.


What a runner actually receives
-------------------------------

A *stage* is a maximal run of operations between *boundaries*. Everything that is not a plain
array operation is a boundary: the read, the reduction, the external call, the exchange, the
join — the points where data enters, leaves, changes shape, or crosses into code graphed cannot
see through. Fusion never crosses a boundary, and boundaries survive reduction as themselves.

Two fusion policies, chosen per compile:

* **Single-use** (the default): an operation fuses into its consumer when it has exactly one
  consumer, that consumer is an operation, and it is not itself a requested output. The effect
  you will notice is that a value feeding two different stages stays its own single-operation
  stage — so it is computed once, not inlined and recomputed twice.
* **Maximal** (``maximal_fusion=True``): additionally fuses a fan-out whose consumers *all* land
  in one stage, so a diamond contained inside a region of plain operations becomes a single
  stage. On the opening example this collapses the two stages into one, giving
  ``source → stage(4 members) → sum``.

Each fused component becomes one stage node recording its **members** — the operations in
topological order, each referring either to an earlier member's result or to one of the stage's
external inputs. The last member is the stage's result. A runner therefore dispatches once per
*stage*, runs the members as a tight loop with no graph interpretation between them, and never
sees how many intermediate variables your code accumulated.

Walking the opening example's numbers, in order:

1. **Dropping dead code**: nine recorded, eight reachable. The abandoned cut is gone.
2. **Canonicalization**: eight to six. The sum written the other way round merged with the
   first one, and the multiply-by-1.0 was equated with its own input; consumers were re-pointed
   at the survivors, so the final multiply became ``mul(a, a)``.
3. **Fusion**: two stages. After the rewrites, ``a`` is consumed twice by ``mul(a, a)``, so it
   is a fan-out and heads its own stage — the three-member stage ``[pt, weight, add]`` that
   computes ``a`` once. The one-member stage is the multiply that uses it twice.
4. The two boundaries survive as themselves. **Four nodes**, two dispatches of actual array
   work.

Fusion is deterministic by construction, not by luck: the single-use pass unions components
keeping the smaller root, so component identity does not depend on the order unions happened,
and the maximal pass is one descending walk over the topologically ordered list, so every
decision is local and the whole pass is linear.


Why a plan is not a pickle
--------------------------

Two runs of the same analysis have to agree exactly, and a worker that has never seen your
source tree has to be able to run it. A pickle of live Python objects fails both: it embeds
whatever your process happened to contain, it is not stable across library versions, and it
tells you nothing about *what* it computes.

So the durable artifact is the reduced graph itself. ``serialize`` writes a canonical byte
encoding — a version tag, nodes in interned id order, parameters key-sorted, stage members with
their references, output ids at the tail. Two graphs with the same structure produce the same
bytes; a round trip rebuilds the same ids and re-serializes byte-identically. ``deserialize`` is
the entry point for everything downstream: runners execute these bytes, checkpoint stores key
work by their content, preservation bundles embed them, and the debugger maps them back to your
source lines.

On top of that, :class:`~graphed.core.DurablePlan` packages those bytes — its ``ir`` field, for
the recorded and reduced form of your analysis — with what a runner needs: the partitions, the
columns to read, the per-chunk ``process``, the ``combine`` that merges partials, and the
``empty`` that starts the fold, plus stopping conditions, file locality and resource hints. Each of the three callables is carried as an ``OpSpec``, and this
is where the durability argument gets concrete:

* ``OpSpec.from_ref("mypackage.analysis:process")`` carries an **import path**. The plan
  contains no code, so it runs anywhere the package is installed, and editing an unrelated part
  of your repository does not disturb it.
* ``OpSpec.from_callable(fn)`` prefers a reference and falls back to embedding the function
  **by value** when it cannot be imported — a lambda, a closure, or anything defined in
  ``__main__``. Those plans set ``plan.opaque``, which is your warning that this plan carries
  bytes rather than a name, and will not survive as cleanly.

This example is two files, because that split is the lesson. Put the analysis functions in a
module so they can be imported by name:

.. code-block:: python

    # analysis.py
    from graphed.core import Partition, WorkerResources


    def process(partition: Partition, resources: WorkerResources) -> float:
        return float(partition.n_entries)


    def combine(left: float, right: float) -> float:
        return left + right


    def empty() -> float:
        return 0.0

then compile, save, reload and run:

.. code-block:: python

    import analysis
    from graphed.core import Dataset, DurablePlan, GraphStore, OpSpec, Plan, SequentialRunner, Task

    s = GraphStore()
    src = s.add_source("events", {"uri": "skim.root"})
    total = s.add_reduction("sum", [s.add_op("pt", [src])])

    plan = DurablePlan(
        ir=s.serialize(outputs=[total]),
        process=OpSpec.from_callable(analysis.process),
        combine=OpSpec.from_callable(analysis.combine),
        empty=OpSpec.from_callable(analysis.empty),
    ).for_dataset(Dataset(uri="skim.root", n_events=1000), chunk_size=400)

    print("process is carried as:", plan.process.kind, plan.process.ref)
    print("anything embedded by value?", plan.opaque)
    print("partitions:", len(plan.partitions))

    blob = plan.to_bytes()
    reloaded = DurablePlan.from_bytes(blob)
    print("round trip is byte-identical:", reloaded.to_bytes() == blob)

    result = SequentialRunner().run(
        Plan(
            process=reloaded.process.resolve(),
            combine=reloaded.combine.resolve(),
            empty=reloaded.empty.resolve(),
            tasks=[Task(key=i, partition=p) for i, p in enumerate(reloaded.partitions)],
        )
    )
    print("result:", result.value, "over", result.n_partitions, "partitions")

which prints::

    process is carried as: ref analysis:process
    anything embedded by value? False
    partitions: 3
    round trip is byte-identical: True
    result: 1000.0 over 3 partitions

``blob`` is the whole job — 658 bytes of canonical JSON here, with the graph base64'd inside it.
Write it next to your results, mail it to a colleague, or hand it to a batch system.

Compile once, run on many datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``for_dataset`` above did two things: it chunked a dataset into partitions and attached them to
the plan. The graph is untouched by that, which is the point — ``with_partitions``,
``for_dataset`` and ``for_datasets`` all return a sibling plan with the *same* computation over
a different input set, without recording, optimizing or serializing your analysis again. Compile
your analysis once at the start of a campaign, then stamp out one plan per dataset:

.. code-block:: python

    # continues the block above: `plan` and `Dataset` are already in scope
    per_sample = {
        name: plan.for_dataset(Dataset(uri=f"{name}.root", n_events=n), chunk_size=400)
        for name, n in [("ttbar", 1000), ("wjets", 1000)]
    }
    print({name: len(p.partitions) for name, p in per_sample.items()})
    print("same computation:", per_sample["ttbar"].ir_fingerprint() == plan.ir_fingerprint())

which prints::

    {'ttbar': 3, 'wjets': 3}
    same computation: True

``ir_fingerprint`` is the hash of the computation alone, so it is how you check that two plans
really are the same analysis over different inputs. ``fingerprint`` hashes the whole plan,
partitions included.

Shuffles and joins need more than one stage — every task writes its blocks, and then every
output partition collects the blocks addressed to it. ``DurablePlanV2`` carries that as an
explicit list of stages with their dependencies, using a distinct format tag so a reader can
never mistake one kind of plan for the other.


What invalidates a cached result
--------------------------------

``plan.task_id(partition)`` is a SHA-256 over the graph's identity, the ``process`` spec, and
the partition. That is what makes a checkpoint store safe: work is keyed by *what* it computes,
not by when or where it ran, so two different computations can never collide onto one id and a
result can never be served for a question it did not answer.

Reading that definition backwards tells you exactly when your cached results survive. Using the
``analysis`` module from the previous section:

.. code-block:: python

    from graphed.core import DurablePlan, GraphStore, OpSpec, Partition


    def plan_with_cut_at(threshold: float) -> DurablePlan:
        s = GraphStore()
        src = s.add_source("events", {"uri": "skim.root"})
        kept = s.add_op("greater", [s.add_op("pt", [src])], {"threshold": threshold})
        total = s.add_reduction("sum", [kept])
        return DurablePlan(
            ir=s.serialize(outputs=[total]),
            process=OpSpec.from_ref("analysis:process"),
            combine=OpSpec.from_ref("analysis:combine"),
            empty=OpSpec.from_ref("analysis:empty"),
        )


    first_chunk = Partition("skim.root", "Events", 0, 400)
    baseline = plan_with_cut_at(25.0)

    print("recompiled, nothing changed:", plan_with_cut_at(25.0).task_id(first_chunk) == baseline.task_id(first_chunk))
    print("cut moved to 30 GeV:        ", plan_with_cut_at(30.0).task_id(first_chunk) == baseline.task_id(first_chunk))
    print("a different chunk:          ", baseline.task_id(Partition("skim.root", "Events", 400, 800)) == baseline.task_id(first_chunk))

which prints::

    recompiled, nothing changed: True
    cut moved to 30 GeV:         False
    a different chunk:           False

In user terms: recompiling an unchanged analysis reuses everything, changing anything the graph
records — a cut value, an added variable, a swapped model's content hash — invalidates the steps
that depend on it, and each chunk is cached independently. And because a by-value ``process``
hashes its own bytes, editing the body of a function the plan carries by value invalidates its
results, while one referred to by import path does not.


Why the same plan gives the same answer on any runner
-----------------------------------------------------

The runtime side of the contract is thin. A ``Plan`` is a ``process`` that turns
one ``Partition`` into a partial result, an associative ``combine`` that merges two partials, an
``empty`` that starts the fold, and a list of ``Task`` s — each a partition plus an integer
``key``. That key is what fixes the shape of the reduction tree, so a fixed set of partitions
combines in the same order regardless of which worker finished first. Floating-point addition is
not associative; without a fixed order your totals would wobble from run to run.

``SequentialRunner`` ships in ``graphed.core`` itself: it runs the tasks in key order,
in-process, with no worker pool and no extra dependencies. Every layer that needs to execute a
plan can therefore do so without pulling in an executor package, and it is the baseline a real
runner has to match.

This example needs the sibling package: ``pip install graphed-executors``. It also needs the
``analysis`` module from above, and ``if __name__ == "__main__":`` because the process pool
spawns.

.. code-block:: python

    import analysis
    from graphed.core import Dataset, Plan, SequentialRunner, Task, partition_dataset
    from graphed_executors.local import ProcessPoolExecutor, ThreadExecutor

    plan = Plan(
        process=analysis.process,
        combine=analysis.combine,
        empty=analysis.empty,
        tasks=[
            Task(key=i, partition=p)
            for i, p in enumerate(partition_dataset(Dataset(uri="skim.root", n_events=1000), chunk_size=250))
        ],
    )

    if __name__ == "__main__":
        for runner in (SequentialRunner(), ThreadExecutor(max_workers=4), ProcessPoolExecutor(max_workers=4)):
            result = runner.run(plan)
            print(f"{type(runner).__name__:19} {result.value} over {result.n_partitions} partitions")

which prints::

    SequentialRunner    1000.0 over 4 partitions
    ThreadExecutor      1000.0 over 4 partitions
    ProcessPoolExecutor 1000.0 over 4 partitions

Two other pieces of the contract are worth knowing about before you write a runner of your own.
``open_once(uri, opener)`` hands a worker a cached file handle so a file is opened once per
worker rather than once per chunk, with the number of simultaneously open handles bounded so a
long-lived worker grinding through thousands of files does not accumulate them all. And a
``StopCondition`` ends submission early on a target event count, a wall-clock limit or an error
budget, reporting which one fired.

The obligation running the other way is on the runner: **a worker failure must reach the driver
intact.** In particular the ``StageError`` from ``graphed.debug`` must not degrade into an
opaque string in transit, because that error is what lets the driver print a traceback pointing
at the line you actually wrote.


Watching a run without changing it
----------------------------------

The vocabulary a live dashboard consumes lives here rather than in the dashboard, because every
runner emits it and this is the layer they all share. It is pure data: a ``TaskEvent`` — an
immutable, picklable, display-only record of one task transition, carrying the task key, the worker,
a timestamp, a partition label, and a pre-rendered error summary rather than an exception object
— plus the ``Monitor`` and ``WorkerProfiler`` protocols an observer implements. Per task the
contract is exactly one ``SUBMITTED``, then one ``STARTED``, then exactly one ``FINISHED`` or
``ERRORED``.

Two properties make it safe to attach one to a production run:

* **It knows nothing about rendering or transport.** A ``TaskEvent`` is data. How it reaches a
  screen — an in-process call, a websocket, something not yet written — is entirely the
  consumer's business, and no web or profiler dependency enters here.
* **A monitor cannot change your result.** Emission is a no-op when no monitor is attached and
  swallows any exception a monitor raises. ``SequentialRunner`` reduces to an identical value
  whether or not one is attached, and so must every other runner.

The concrete monitors and the browser side live in ``graphed.debug``; the runners that emit
through this contract live in ``graphed-executors``.


How workers exchange data
-------------------------

Repartitioning, joins, workers combining their partials with each other rather than shipping
everything back to your submit node, and idle workers stealing work from busy ones all need
workers to be able to address one another. ``WorkerTransport`` is that channel: ``send``,
``broadcast``, ``poll``, ``recv``, ``peers``, ``close``, carrying arbitrary picklable messages
between the driver and workers and between workers.

It is best-effort and non-blocking on purpose. ``send`` enqueues and returns ``True``, or drops
the message and returns ``False`` when the destination's inbox is full, so back-pressure in the
coordination channel can never reach the data path. It makes **no ordering or
delivery guarantee**, because determinism is not this layer's job: the reduction protocol above
it keys every combine by leaf index and never by arrival time or worker identity, which is why
workers combining among themselves gives bit-for-bit the same total as routing everything
through the driver.

The data-movement primitives a repartition or join engine calls — route rows to sub-blocks by a
hash of a key field, concatenate, slice, measure, serialize, deserialize, and for joins match
and gather rows — are likewise declared here as protocols and implemented in the array backends,
so this layer never depends on an array library. Each backend declares a versioned format token
that is folded into the multi-stage task ids, so two different backends can never journal the
same id for different content.


Not supported yet
-----------------

* **Rewrites that understand array semantics.** Mask fusion, field collapse, associativity
  regrouping and constant folding would all need revisiting the linear-extraction argument (a
  rule that invents a new term forces a real cost search) and the one-pass argument behind
  incremental reduction. The byte-identity requirement between incremental and one-shot
  reduction is what will catch it if either is broken.
* **Cost-model-driven fusion.** Single-use and maximal are structural policies. Choosing fusion
  per stage from a cost model — kernel size, memory residency — is not implemented; if fusion
  is hurting a particular analysis, switching ``maximal_fusion`` is the only lever.
* **Finer-grained locking while recording.** One lock guards the recording table. It has been
  sufficient under free-threaded stress testing, but a workload that genuinely contends on
  recording will serialize there.
* **One durable format version.** Saved graphs carry a version tag and only the current one is
  accepted; anything else is rejected with an error rather than misread. The tag is the hook for
  a future format, and plans and content hashes you write today stay readable.

See :doc:`improvements` for the tracked list.
