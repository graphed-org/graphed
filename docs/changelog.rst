What changed
============

Newest release first. Numbers in parentheses are the pull requests on
`graphed-org/graphed <https://github.com/graphed-org/graphed>`_.

0.0.2
-----

Systematics are an axis of the graph
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The headline of this release. You declare a variation once and every histogram downstream comes
back for every universe, out of one read of the data — instead of running the analysis again per
systematic. :doc:`frontend/design` is the grammar and :doc:`notebooks/systematics-tour` walks it
one declaration at a time.

* ``graphed.vary`` shifts an array, a whole collection on an event context, or the event weight —
  and the selection, the ``max`` and the fill downstream of it are written once (#3, #4).
* ``graphed.labels``, ``universe``, ``nominal``, ``variations`` and ``member_of`` read the result
  back per universe, and a preservation bundle holds every universe in one directory (#5).
  ``graphed.variations`` reports each tag's kind as a ``graphed.Kind`` flag, so a nuisance that is
  both a weight and a shift reports both rather than a third word (#24).
* ``graphed.awkward.to_parquet(record, dest, select=...)`` writes a varied skim — every universe
  in one file — and ``read_varied`` reads each one back bit-for-bit (#7).
* A label **names a point** in nuisance space. ``points=`` declares a universe that differs on
  more than one axis, and a label resolves to its own member first (#12).
* **A scale factor evaluated on shifted objects now gets its joint universes.** A b-tag weight
  computed over jet-energy-shifted jets used to lose the cross term silently; it now fans out over
  the nuisance it read. ``composes_as_union=True`` collapses back to the one-at-a-time datacard
  union and ``max_universes=`` is a loud guard on a runaway grid (#14, #25).
* A collection that is a *function* of a varied one — Type-1 MET of the varied jets — moves in
  lockstep: pass the varied collection itself where the shift form wants a tag map (#23).
* Variation tags may be numbers: ``{+2.5: pt * 1.1, -2.5: pt * 0.9}`` mints ``jes_25em1`` and
  ``jes_m25em1``, and ``2.5`` and ``"2.5"`` are one tag rather than two universes (#19, #26).
* **A second family on a scale factor you already registered no longer squares it.** The weight
  form reads its ``nominal`` to decide what is being varied: naming a live factor joins that
  factor, naming a composition you read back replaces it at that family's labels, anything else is
  a new factor. ``graphed.explain(ctx)`` prints, one line per item, how each family entered, what
  the weight is made of here and where every universe came from (#28).
* The unified surface is spelled ``points=`` and its error is ``PointError`` (#15).
* Fan-out is fast enough to use at analysis scale: recording no longer reads the whole Python
  stack per operation, and the event weight composes lazily — 128 weight families cost 635
  multiply nodes instead of 16,637 (#16, #20). ``graphed.weight`` on a projected universe stays
  the bare member for the life of the session (#21).

Joining rows across partitions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``graphed.join`` and ``graphed.awkward.gak.join`` match rows by key across partitions, with
``join_plan`` for the plan-level form; ``repartition`` already redistributed them. An exchange is a
stage boundary, and the route is decided from the plan rather than at run time, so every worker
agrees on it and two runs of the same plan move the blocks the same way: workers exchange directly
where the runner can address them, through your submit node where it cannot. ``join`` follows
``pandas.merge`` rather than SQL on missing keys — two rows whose key is null on *both* sides are
paired — which :doc:`awkward/improvements` spells out.

Behavior methods take arguments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``a.deltaR(b)`` and ``jets.scaled(2.0, offset=1.0)`` record, arguments and all, so a Δ-quantity is
written the way you write it in awkward instead of spelled out as a property-safe formula. Array
arguments become graph inputs; everything else has to be a JSON constant, which keeps the plan
durable. Column projection replays the method, so it reads exactly the fields the body touches
(#22).

Corrections and models reach a worker
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A plan reading a ``correctionlib`` correction or an ONNX model used to die the moment it left the
main process, which meant the central "hundreds of histograms with systematics" path could not run
on a pool, a checkpoint, or any out-of-process runner. Those calls now travel (#13), a correction
upstream of a histogram fill is wired into the plan and the universes off one correction set no
longer collide on one evaluator (#9), and an in-process correction takes the same flat-buffer path
the executor does (#18).

Wheels that install on more Linux machines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

0.0.1's Linux wheels needed glibc 2.34, so importing ``graphed`` on an EL8 / Rocky 8 cluster failed
with ``version 'GLIBC_2.30' not found``. Linux wheels are now built against glibc 2.28 (#6), and
musl builds ship alongside them, so Alpine-based images get a binary wheel too (#8).

Documentation
~~~~~~~~~~~~~

* Every page is written for someone porting an analysis rather than for someone who built the
  package, and :doc:`quickstart` is a new on-ramp: a parquet dataset through a selection, a
  systematic and a histogram in one program (#10, #11).
* :doc:`notebooks/systematics-tour` is an executed notebook — every label set, point and yield
  under a cell is what the code printed (#29).
* The Read the Docs build installs the package it documents, so the published API reference is
  generated from real code (#1).

Also in this release: the merge queue runs the same checks a pull request does (#17), and the test
tree is type-checked along with the package (#27).

0.0.1
-----

The first release: the recording frontend, the compiled core that reduces your analysis as you
build it, the awkward and numpy backends, projection down to the buffers a column actually needs,
source-mapped errors from inside a worker, the live run dashboard, content-addressed checkpoints,
preservation bundles, and ``repartition`` for moving rows between partitions. ``graphed-executors``
and ``graphed-histogram`` shipped their first releases alongside it.
