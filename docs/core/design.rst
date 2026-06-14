How graphed-core works
======================

``graphed-core`` is the spine of the graphed ecosystem: a thread-safe, interned intermediate
representation (IR) for array analyses, an optimizer that reduces that IR to a handful of fused
*stages*, and a deterministic, versioned codec that makes the reduced graph the durable artifact
executors run. The graph lives in Rust (PyO3 bindings on top); Python supplies the recording
frontends and executors, but never holds a large graph of Python objects.

The package exists to avoid two failure modes observed in earlier HEP task-graph systems:

* building a complete, operator-by-operator graph **before** optimizing it, so the driver pays
  memory and interpreter time proportional to every recorded operation; and
* an optimizer whose cost grows super-linearly in graph size, so optimization itself dominates
  wall time on realistic (systematics-heavy) analyses.

Everything below is organized around how those two are avoided: identity is established at
construction time (interning), reduction can run incrementally as the graph is built, and every
optimizer pass is linear or near-linear, with a CI benchmark that fails on super-linear scaling.

.. contents::
   :local:
   :depth: 2


The IR: nodes, structural identity, interning
---------------------------------------------

A node is one of five kinds::

    Source | Op | Reduction | External | Stage

``Source`` is an input dataset; ``Op`` is an elementwise/array operation; ``Reduction`` is an
aggregation boundary; ``External`` is a call into an outside payload (a correction set, an ONNX
model, a histogram fill) identified by a :class:`~graphed_core.PayloadDescriptor`; ``Stage`` is
a fused run of ops and is **only ever constructed by the optimizer** — frontends cannot record
one.

A node *is* its structural key (``NodeKey``): kind, name, parameters (key-sorted), input ids,
and — for ``External`` — the full payload descriptor, including its content hash. The store
keeps an intern table keyed on that structure, so constructing a structurally identical node
returns the existing ``NodeId``. Three consequences are worth internalizing:

* **CSE falls out of construction.** Recording ``events.pt * 2`` twice produces one node. There
  is no separate common-subexpression pass over user recordings — deduplication happened the
  moment the duplicate was recorded.
* **Identity is content, not history.** Two sessions that record the same analysis produce
  graphs that serialize to identical bytes (the determinism gate builds on this).
* **External payloads participate in identity.** Swapping an ONNX model for a retrained one (a
  different content hash) is a *different node*; nothing downstream can confuse the two.

Parameters and floats
~~~~~~~~~~~~~~~~~~~~~

Parameter values are ``int | float | bool | str``. Hashing floats requires a total order:
``ParamValue`` keys a float by its IEEE bit pattern, with every ``NaN`` collapsed to one
canonical bit pattern. So ``NaN`` interns to itself (recording the same NaN-parameterized cut
twice gives one node), while ``0.0`` and ``-0.0`` remain *distinct* — treating them as equal is
a value-level canonicalization, which is the optimizer's job, not the intern table's.

Every node also has an **operator token**: a string encoding kind, name, and parameters — but
*not* inputs. Two nodes with the same token are the same operator applied to possibly different
operands. The token is the node's symbol in the e-graph (inputs become e-graph children) and the
key for rebuilding full nodes after canonicalization. Parameter strings inside tokens are
escaped so the encoding stays injective; float parameters appear as their hex bit pattern
(``f3ff0000000000000`` is 1.0), which is how the optimizer's identity rules can name "multiply
by exactly 1.0" as a plain string match.

Thread safety
~~~~~~~~~~~~~

Interning is a read-modify-write (look up the key; if absent, append a node and record its id),
so the store's inner state sits behind a single ``Mutex`` making that step atomic — correct
under the GIL and under free-threaded CPython alike. The critical section is a hash-map probe
plus a vector push; finer sharding is tracked as a future improvement but has not been needed.
The discipline is model-checked with ``loom`` and stress-tested from many threads in both Rust
and Python.

One subtlety, fixed by design: **graph outputs are a property of a compile request, not store
state.** ``reduce``, ``serialize``, and ``IncrementalReducer.finalize`` take the output set as
an argument and use exactly that set; there is no output mutator on the store. Sequential
compiles of different expressions from one session are therefore independent (byte-identical to
fresh-session compiles), and concurrent compiles cannot interfere — compiling is a read-only
operation.


The optimizer
-------------

``reduce`` turns the recorded arena into a graph of *stages* — the schedulable unit an executor
dispatches. The pipeline is four passes around a swappable rewrite engine::

    reduce(nodes, outputs):
      1. reachable = dead_code_elimination(nodes, outputs)   # plain reachability
      2. canonical = engine.canonicalize(reachable)          # equality saturation (egg)
      3. deduped   = cse(canonical)                          # hash-consing, again
      4. stages    = stage_fusion(deduped, mode)             # maximal op runs between boundaries
         rebuild into a fresh interned store

A deliberate division of labor: DCE and CSE are *plain passes outside the engine*. The engine
does exactly one thing — compute which nodes are **semantically equivalent** — because that is
the part that benefits from equality saturation, and keeping it minimal is what makes the
engine swappable (the ``RewriteEngine`` trait traffics only in an engine-neutral
``EngineGraph``; no ``egg`` type crosses the boundary).

Pass 1 — dead code elimination
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Plain reachability from the requested outputs over the input edges, followed by compaction that
preserves ascending-id (topological) order. Nothing on a path to an output is ever dropped;
everything else is. A recorded-but-unused branch (a cut the user tried and abandoned) costs one
intern-table entry and nothing more.

Pass 2 — canonicalization by equality saturation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is the heart of the optimizer, and the part most worth understanding precisely.

**The problem.** Interning dedups *structurally identical* nodes, but users write semantically
identical expressions in different shapes: ``a + b`` here, ``b + a`` there; a convenience helper
that multiplies by ``1.0``; a weight of ``x + 0.0``. Structural identity cannot see through
these; a rewrite system can.

**The e-graph idea.** An e-graph stores a set of terms together with an equivalence relation
over them, compactly: each *e-class* holds all terms known to be equal, and terms refer to
e-classes (not concrete terms) as children — so one e-class of "things equal to ``a + b``"
serves every parent expression at once. *Equality saturation* seeds the e-graph with your
program, then applies rewrite rules not as destructive edits but as *additions*: each rule
match merges or extends e-classes. Run to saturation (or a budget), the e-graph encodes every
way of writing the program under the rules, and the equivalence classes are exactly what we
want to know.

**Loading the IR.** Each node's token becomes a symbol; its inputs become children. Loading is
one linear pass in topological order, recording the e-class of every original node.

**The rule set is deliberately tiny and provably sound:**

* *commutativity* for the symmetric operator vocabulary (``add``, ``mul``, ``and``, ``or``,
  ``eq``, ``ne``, ``maximum``, ``minimum``) — argument order is semantically irrelevant for
  these, so ``(add a b)`` and ``(add b a)`` merge into one class. Asymmetric ops (``sub``,
  ``div``, ordered comparisons) are deliberately absent.
* *identity elimination* for ``x + 0.0`` and ``x * 1.0`` (both operand orders), expressed as
  exact token matches on the scalar's bit pattern — equating the op's class with its input's
  class.

Domain-dependent rewrites (mask fusion, field collapse) are intentionally excluded: their
soundness depends on array semantics the core cannot see. Soundness here is not a slogan — an
unsound rule corrupts *every* analysis that triggers it, silently.

**Determinism over speed-feel.** The saturation budget is an iteration limit and a node limit —
never a wall-clock limit, because a time budget makes the optimized graph depend on machine
load. Identical input must give a byte-identical reduced graph on every machine, every run;
that property is gated in CI.

**Extraction — the O(N²) trap and the O(N) escape.** After saturation you must pick one
concrete representative per e-class ("extraction"). egg ships a recursive cost-based
``Extractor``; its cost is O(depth × nodes), and a systematics-style analysis — thousands of
variations hanging off one deep shared selection chain — is exactly the deep-chain shape that
degrades it to O(N²). This is the same failure class as the legacy systems this project
replaces, just relocated into the optimizer.

The escape comes from a property of *our* rule set: every rule only ever equates a node with an
**already-existing, earlier** node (the commuted twin of something recorded before; the input
of an identity op). No rule invents a term that must be materialized. Extraction therefore does
not need to search: **quotient the original node list by the e-graph's equivalence and keep the
earliest member of each class.** One linear pass. Picking the earliest member is topologically
safe (it can only refer to even-earlier representatives) and is itself the cost function — for
these rules the earliest form is the simplest form (the identity-free, first-recorded
orientation). A CI benchmark runs reduction across graphs of 1k/2k/4k/8k nodes and **fails on
super-linear growth**, so this property cannot silently regress.

Pass 3 — CSE, again, and why
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Canonicalization rewrites inputs: once ``(add a b)`` and ``(add b a)`` share a representative,
two previously-distinct parents like ``inc(add(a,b))`` and ``inc(add(b,a))`` become the *same*
``(token, inputs)`` pair — but they are still two entries in the list. A linear hash-consing
pass collapses them. (At record time interning did this for structurally identical nodes; this
pass re-establishes the property for nodes that became identical only after canonicalization.)

Pass 4 — stage fusion
~~~~~~~~~~~~~~~~~~~~~

A *stage* is a maximal run of ops between *boundaries*. Boundary-ness is a property of node
kind: everything except a plain ``Op`` is a boundary (sources, reductions, externals — the
points where data enters, leaves, or crosses into foreign code). Fusion never crosses a
boundary; boundaries survive reduction as themselves.

Two modes, selected per compile:

* **SingleUse** (the default, and the long-standing pinned behavior): an op fuses into its
  consumer when it has *exactly one* consumer, that consumer is an op, and the op is not itself
  a requested output. Implemented with a union-find over op nodes; the union always keeps the
  smaller root so component identity is independent of union order (determinism, again). The
  consequence users notice: a fan-out op — one feeding two different stages — stays its *own*
  single-op stage, so its value is computed once, not inlined twice.
* **Maximal**: additionally fuses a fan-out op when **all** of its consumers land in one stage,
  so a diamond contained within an op region becomes one stage. The implementation is a single
  descending pass over the (topologically ordered) node list: by the time node *i* is visited,
  every consumer has a final component assignment, so the decision "do all my consumers share a
  component?" is local — making the whole pass linear and, being a pure function of the ordered
  graph, deterministic.

Each op-component becomes one ``Stage`` node. A stage records its **members** — the fused ops in
topological order, each holding its operator token and references that are either
``Member(j)`` (the j-th member's result, an intra-stage edge) or ``Input(k)`` (the k-th external
stage input, deduplicated in first-seen order). The stage's last member is its result. An
executor therefore dispatches once per *stage*, runs the members as a tight loop with no
graph-interpretation overhead between them, and never sees the original op count.

A worked example (runnable)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Record this (typical of how analysis code accumulates) and reduce it::

    import graphed_core as gc

    s = gc.GraphStore()
    src  = s.add_source("events", {"uri": "data.root"})
    pt   = s.add_op("pt", [src])
    w    = s.add_op("weight", [src])
    a    = s.add_op("add", [pt, w])
    b    = s.add_op("add", [w, pt])                      # the commuted twin, written elsewhere
    one  = s.add_op("mul", [a], {"scalar": 1.0, "side": "r"})  # a helper that multiplied by 1.0
    dead = s.add_op("cut", [pt])                         # tried, abandoned, never an output
    out  = s.add_reduction("sum", [s.add_op("mul", [one, b])])

    reduced, report = s.reduce(outputs=[out])
    print(report)
    for n in reduced.nodes():
        print(n["id"], n["kind"], n.get("n_members", ""))

which prints (exactly — this is deterministic)::

    {'input_nodes': 9, 'reachable_nodes': 8, 'canonical_nodes': 6,
     'stages': 2, 'boundary_nodes': 2, 'reduced_nodes': 4}
    0 source
    1 stage 3
    2 stage 1
    3 reduction

Walking the counts:

1. **DCE**: 9 recorded → 8 reachable (``dead`` is gone).
2. **Canonicalization**: 8 → 6 — ``b`` merged with ``a`` (commutativity) and ``one`` equated
   with ``a`` (multiplicative identity); extraction kept the earliest member of each class and
   re-pointed consumers, so the final ``mul`` became ``mul(a, a)``.
3. **Stage fusion** (SingleUse): the surviving ops form **two** stages, and the reason is
   instructive — after the rewrites, ``a`` is consumed *twice* by ``mul(a, a)``, so it is a
   fan-out and heads its own single-member stage; ``pt``/``w``/``add`` would each feed it...
   in fact the three-member stage is ``[pt, weight, add→a]`` computing ``a`` once, and the
   one-member stage is the ``mul`` that uses it twice. The fan-out value is computed once,
   never inlined twice — exactly the SingleUse guarantee.
4. The two boundaries (``source``, ``sum``) survive as themselves: **4 nodes total**, two
   executor dispatches for the op work, regardless of how many intermediate variables the
   user's code accumulated.

(Under ``maximal_fusion=True`` the fan-out *would* fuse — all of ``a``'s consumers are ops in
one stage — giving ``source → Stage[4] → sum``.) The ``reduction_report`` makes the collapse
observable at every step rather than folklore; re-running ``reduce`` produces byte-identical
output (``reduced.serialize()`` equality is part of the frozen suite).

The incremental reducer
~~~~~~~~~~~~~~~~~~~~~~~

The pipeline above is one-shot: it sees the whole arena. The plan's architectural requirement
is stronger — *the large un-reduced graph should never exist*. ``IncrementalReducer`` achieves
this by consuming the arena **delta by delta**: each ``step`` processes only the nodes recorded
since the last step, maintaining a canonical arena (identity-eliminated, symmetry-deduped,
hash-consed) as the graph is built.

The pedagogically important point is *why a single pass per node suffices*. Every rule in the
sound set is **constructor-local**: whether an op is an identity, and what its canonical
orientation is, depends only on the op itself and its inputs' already-canonical ids — never on
consumers or on future nodes. So canonicalizing each node once, at arrival, in topological
order, reaches the same fixpoint that running equality saturation over the whole graph reaches
*for this rule set*. The two paths are kept provably aligned by construction: the engine's rule
set and the incremental canonicalizer are generated from the **same** shared constants
(``SYMMETRIC_OPS``, ``IDENTITY_TOKENS``), and ``finalize`` (which runs DCE + fusion over the
maintained canonical form) is pinned byte-identical to a one-shot ``reduce``.

Incrementality is not taken on faith: the reducer exposes a cumulative work counter, and the
frozen suite asserts that per-step work equals the delta size and total work equals the node
count — a test an "incremental" alias that secretly re-scanned history could not pass. If a
rule that is *not* constructor-local is ever added to the engine (one whose canonical form
depends on consumers), the one-pass argument breaks — and the byte-identity pin between
``finalize`` and ``reduce`` is the tripwire that will say so.


The durable form (GIR1)
-----------------------

``serialize`` writes the canonical byte encoding: a version magic (``GIR1``), nodes in interned
id order, parameters key-sorted, stage members with their ``Member``/``Input`` references, and
the output ids at the tail. Two graphs with the same structure produce the same bytes; a round
trip rebuilds the same ids and re-serializes byte-identically. This is the **canonical durable
representation** of an analysis — never a pickle of live objects. ``deserialize`` is the entry
point for everything downstream: executors evaluate these bytes, checkpoint stores key work by
their content, preservation bundles embed them, and debuggers map them back to source.


The plan layer
--------------

On top of the codec, the pure-Python :class:`~graphed_core.DurablePlan` packages a reduced IR
with what an executor needs: partitions, read columns, and the process/combine/empty operations
as ``OpSpec``\ s. An ``OpSpec`` references a callable **by import path** so a plan can run on a
machine with no analysis source files; only a genuinely opaque callable is embedded by value
(cloudpickle) and is flagged ``opaque=True`` as a preservation risk. ``task_id``
content-addresses (plan, process, partition) with SHA-256, which is what makes checkpoint
stores safe against cache poisoning: work is keyed by *what* it computes, not by when or where
it ran.


The monitor seam (live observability)
-------------------------------------

``graphed_core.execution`` also carries the M37 **observability seam** — the contract a live
dashboard plugs into. It is deliberately tiny and pure data: ``TaskEvent`` (a frozen, picklable,
*display-only* record of one task transition), ``TaskPhase`` (``SUBMITTED`` / ``STARTED`` /
``FINISHED`` / ``ERRORED``), and two protocols, ``Monitor`` (``on_task`` / ``on_profile`` /
``on_combine`` / ``worker_profiler_factory``) and ``WorkerProfiler`` (``start`` / ``flush`` /
``stop``).

Why it lives in ``graphed-core`` and not in the dashboard: the event vocabulary is shared by *every*
executor, so it belongs at the layer they all depend on — a shared primitive at the layer it serves.
Keeping it here also keeps it honest about its boundaries:

* **Render- and transport-agnostic.** Core gains no web, websocket, or profiler dependency. A
  ``TaskEvent`` is just data; *how* it reaches a screen — an in-process call, a websocket to a
  Perspective server, something not yet written — is entirely the consumer's business. (This is why
  the seam survived ``graphed-debug``'s switch from an SSE prototype to Perspective unchanged.)
* **Passive by construction.** ``emit_task(monitor, event)`` is a no-op for a ``None`` monitor and
  swallows any exception a monitor raises. An executor emits best-effort; a misbehaving or absent
  monitor can never change a result. ``SequentialRunner`` is the observable baseline that proves it:
  its reduced value is identical with or without a monitor attached.

Concrete monitors (and the websocket/Perspective rendering) live in ``graphed-debug``; the reference
executors that *emit* through this seam live in ``graphed-exec-local``.


Reading map
-----------

====================================  ===========================================================
File                                  What lives there
====================================  ===========================================================
``src/param.rs``                      ``ParamValue``/``ParamMap``: total-order float hashing, token escaping
``src/node.rs``                       ``NodeKey`` (structural identity), tokens, ``PayloadDescriptor``, ``Stage``
``src/store.rs``                      the ``Mutex``-guarded intern table; ``reduce*`` entry points; ``to_dot``
``src/optimizer/engine.rs``           the ``RewriteEngine`` boundary; ``EggEngine``; the shared rule constants
``src/optimizer/mod.rs``              DCE, CSE, stage fusion (both modes), the pipeline, the toy-interpreter tests
``src/optimizer/incremental.rs``      the delta-consuming canonicalizer + work counters
``src/serialize.rs``                  the GIR1 codec
``src/lib.rs``                        PyO3 bindings (thin; everything above is plain Rust)
``python/graphed_core/plan.py``       ``DurablePlan`` + ``OpSpec`` + content-addressed ``task_id``
``python/graphed_core/execution.py``  the executor-facing ``Plan``/``Task``/``ExecResult`` contract
====================================  ===========================================================


Phase 2 (deliberately not built)
--------------------------------

The MVP draws its scope lines explicitly; these are the known next steps, in this package's
territory, that are *intentionally absent* rather than forgotten:

* **The egglog engine swap.** ``RewriteEngine`` exists precisely so a second engine
  implementation (egglog) can replace ``EggEngine`` without touching DCE/CSE/fusion; the MVP
  ships egg only.
* **A richer (still sound) rule vocabulary.** Associativity regrouping, constant folding, and
  any domain-informed rules would require revisiting both the O(N) extraction argument (rules
  must keep the equate-with-an-earlier-node property or extraction needs a real cost search)
  and the incremental reducer's constructor-locality argument — the byte-identity pin between
  ``finalize`` and ``reduce`` is the tripwire either way.
* **Finer store sharding.** The single ``Mutex`` has been sufficient under free-threaded
  stress; sharding the intern table is tracked for genuinely contended recording workloads.
* **Cost-model-driven fusion.** ``SingleUse``/``Maximal`` are structural policies; a
  Volcano/Cascades-style cost model choosing fusion per stage (kernel size, memory residency)
  is a Phase-2 optimizer evolution.

See :doc:`improvements` for the live tracked list.
