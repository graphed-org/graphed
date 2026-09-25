What changed
============

Newest release first. Numbers in parentheses are the pull requests on
`graphed-org/graphed <https://github.com/graphed-org/graphed>`_.

0.0.7 (unreleased)
------------------

Services an analysis calls
~~~~~~~~~~~~~~~~~~~~~~~~~~

* ``graphed.services.ServiceSpec`` (with an optional ``Launch`` recipe) declares a service an
  analysis calls: ``Session.declare_service``, ``Session.services()``, ``Session.service_for``. An
  External names one through ``params["service"]``; an undeclared name is refused at record time.
* ``Plan.services`` and ``DurablePlan.services`` carry the specs the recording names
  (``aggregate_plan(services=...)`` adds names no node carries), and so does the awkward
  ``to_parquet`` write plan. ``DurablePlan.to_bytes`` writes the key only when non-empty, so a plan
  without services keeps its bytes.
* ``graphed.services.bind_services(plan, {name: "scheme://host:port"})`` binds run endpoints into a
  plan's process without changing the recording; ``split_endpoint`` checks the form (``tcp``,
  ``http``, ``https``, ``grpc``, ``grpcs``). An unbound service raises ``UnboundService``.
* A Triton External names ``service=`` or a literal ``url=``, not both. The endpoint's scheme picks
  ``tritonclient.http`` or ``tritonclient.grpc`` (TLS on ``https``/``grpcs``); a url without a
  scheme keeps the HTTP client and ``params["transport"]`` still overrides. The ``ml`` extra
  installs ``tritonclient[grpc,http]``.
* The per-process External resource cache keys on the endpoint and the params ``load`` reads
  (``ExternalPlugin.load_params``), so a correction set is still loaded once across systematics.
* Preservation bundles list the referenced specs in ``manifest["services"]`` (absent when there are
  none) and ``inspect()`` prints them. ``RunReport.endpoints`` records where a run reached each
  service, outside the fingerprint; ``reproduce`` refuses an External that calls a service.

Declared output types for external calls
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ``Array.map``, ``graphed.apply`` and ``Session.record_external`` take ``output_type=``, the type
  of each element of the call's value, and record that type instead of the first input's (awkward)
  or ``vector[object]`` (numpy). A boolean mask declared ``"bool"`` indexes as a mask at build
  time, and a declared named record resolves its behavior. awkward takes type strings, type
  objects, forms, numpy dtypes and Python types; numpy takes dtypes of every kind and Python
  types; every numerical dtype, ``float16`` included, works on both. The declaration is a node
  param, so it is identity; an undeclared call records the same bytes as 0.0.6 (#57).
* ``graphed.preserve.externals.record_external`` takes ``output_type=`` too, so a plugin
  External such as a golden-JSON lumi mask records ``bool`` and indexes as a mask (#57).
* ``ExternalPlugin.output_dtype`` declares a plugin's static leaf dtype. The seven float64
  built-in plugins (correctionlib, ONNX, TensorFlow, PyTorch, XGBoost, JAX, Triton) and
  ``gak.apply_correction(..., args=)`` now record float64 leaves instead of their first input's
  dtype. Their plan bytes are unchanged.
* ``gak.apply_correction`` and ``gak.onnx_inference`` with ``args=`` refuse an input from another
  ``Session`` with a ``GraphedTypeError`` at the call, not a plain ``TypeError``.
* A numpy ``gufunc`` External refuses ``output_type=``: its signature and ``output_dtype=`` type it.

0.0.6
-----

A checkpoint store contract
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ``graphed.checkpoint.CheckpointStore`` is the runtime-checkable protocol that
  ``run_resumable`` and ``run_shuffle_resumable`` take: ``put``, ``get``, ``record_done``,
  ``completed``, ``record_dead`` and ``dead_letters``. ``Store`` meets it unchanged on disk.
* ``Store.get`` returns ``None`` for a blob whose bytes do not hash to its name, and the next
  ``put`` rewrites it, so resume recomputes a corrupted partial instead of failing to decode it.
* Concurrent ``Store.put`` of the same bytes from threads of one process no longer fails with
  ``FileNotFoundError`` on a shared temp file.
* ``graphed.checkpoint.FsspecStore(url, node=None, **storage_options)`` is a checkpoint store at
  an fsspec URL, so workers on different machines can share one store and a run resumes anywhere
  given the URL. Results keep ``Store``'s names and bytes, and each journal or dead-letter record
  is its own object holding ``Store``'s line. It needs the ``[checkpoint]`` extra, which now
  installs ``fsspec``; ``import graphed.checkpoint`` still does not load it.
* ``FsspecStore`` reads listings fresh (``use_listings_cache`` is always off) so that on S3 an
  instance sees records another process wrote after it first listed the store. Its tests run on
  ``s3://`` against a moto server as well as on ``memory://`` and ``file://``.
* ``Store`` writes its journal with ``\n`` line endings on Windows too, so a local store and one
  at a URL hold the same records; journals already written with ``\r\n`` still replay (#48).

Pause, resume and cancel a run
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ``graphed.core.RunControl`` pauses, resumes or cancels a run from any thread, and
  ``SequentialRunner(control=...)`` honours it. A cancel lets running tasks finish and returns the
  fold of the tasks that completed, with ``stopped=StopReason.CANCELLED``; a run never cancelled
  is unchanged (#49).
* ``Dashboard(control=True)`` puts pause, resume and cancel buttons on the page, and
  ``dash.attach(runner)`` wires them to the runner. For a remote run,
  ``NetworkMonitor(url, control=ctl)`` receives the commands over the connection it sends events
  on (#49).

Cheaper to watch
~~~~~~~~~~~~~~~~

* ``NetworkMonitor(lean=True)`` asks workers for one event per task with no partition label, and
  ``per_worker=True`` lets each worker process send its events straight to the dashboard instead
  of through the driver. The sender now batches events into one message per batch (#50).
* With no monitor attached, ``SequentialRunner`` builds no task events at all (#50).

A record of each run
~~~~~~~~~~~~~~~~~~~~

* ``graphed.debug.RunRecorder`` records a run's task events, and ``report()`` folds them into a
  ``RunReport``: the outcome, each task's partition, worker, state, duration and error, the
  ``StageError`` the run raised, and the environment digest. It round-trips through JSON (#51).
* ``graphed.preserve.attach_run_report`` keeps a report in a preservation bundle without changing
  the bundle's fingerprint; ``Bundle.run_reports()`` reads them back and ``inspect()`` lists them
  (#51).

Replay one task
~~~~~~~~~~~~~~~

* ``aggregate_plan(store=...)`` keeps each task's input chunk and partial result in a checkpoint
  store (a directory or an fsspec URL), and ``graphed.debug.replay(plan, key, *outputs)`` re-runs
  that task on your machine one operation at a time: ``.steps()`` walks it, ``.value`` is the
  replayed partial, ``.diff()`` compares it with what the run recorded, and a failing step raises
  the run's own ``StageError`` at your line (#52).
* ``graphed.debug.lower`` and ``run`` handle a whole-array reduction such as
  ``gak.sum(x, axis=None)`` (#52).
* A capture made from a projected ROOT read keeps the chunk as read: the buffers the read skipped
  reload as unread placeholders and a behavior holding lambdas survives, so ``aggregate_plan(store=...)``
  over uproot or coffea input no longer fails while recording (#55).

Failures at your line
~~~~~~~~~~~~~~~~~~~~~

* A worker failure at any framed key, labelled or not, is raised as a ``StageError`` pointing at
  your analysis line; an unvaried program reports ``variation == ""`` (#47).

Wheels
~~~~~~

* Wheels for Windows on ARM64 (``win_arm64``) ship with each release, and CI runs the suite on
  ``windows-11-arm`` (#46).


0.0.5
-----

Deferred arrays answer their own metadata, and a plan projects all its outputs in one replay
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Two findings from reviewing coffea's graphed mode against the upstream mains.

* ``arr.fields``, ``arr.ndim``, ``arr.type``, ``arr.typestr``, ``arr.is_tuple`` and
  ``arr.positional_axis`` on an awkward-backed deferred array are answered at once from the
  record-time typetracer instead of being recorded as a ``field`` op — ``"pt" in arr.fields`` is a
  bool and ``arr.ndim == 2`` compares, with nothing added to the graph. The metadata a typetracer
  cannot answer (``nbytes``, ``layout``, ``mask``, ``attrs``, ``behavior``, ``named_axis``) raises
  ``AttributeError`` naming what to do instead. A record field or a behavior that defines the same
  name still wins (graphed-org/graphed#42).
* ``graphed.awkward.project_buffers_many(arrays)`` projects several outputs of one session with a
  single reporting typetracer per source, where calling ``project_buffers`` per output replayed the
  whole mapped form each time (about 27 ms per output on NanoAOD). The answer applies the module's
  covering rule across outputs: ``DATA`` on a path absorbs ``OFFSETS`` on it. ``project_buffers``
  is unchanged (graphed-org/graphed#43).
* Wheels for CPython 3.15: the ``cp311-abi3`` wheel is smoke-tested on 3.15 and a dedicated
  free-threaded ``cp315t`` wheel is built, on PyO3 0.29 and maturin 1.14+.


0.0.4
-----

Partition-wise drivers are sound
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A plan and a writer evaluate the compiled IR once per chunk. What per-chunk replay cannot compute
is now refused where the plan is built, and what it needs is shipped with the task
(graphed-org/graphed#38).

* A reduction on the partitioned axis — ``x[2:8]``, ``x[0]``, ``gak.sum(x)`` — inside a
  partitioned plan is a per-chunk partial. ``aggregate_plan`` refuses one that feeds another node
  (``gak.sum(x[2:8])`` summed a partial per chunk), and both ``to_parquet`` writers refuse one as
  their output (each part held its own chunk's slice). An output reduction that the plan's
  ``combine`` folds is unchanged. ``graphed.refuse_chunk_partials(compiled, as_outputs=...)`` is
  the check, for a writer of your own.
* ``to_parquet`` wires the session's External evaluators into its write tasks, as
  ``aggregate_plan`` always did, and projects through an opaque node conservatively instead of
  refusing it — ``graphed.apply(f, x)`` writes. ``graphed.aggregate.external_evaluators(session,
  compiled)`` is that wiring, public for other writers.
* ``graphed.write.file_bases`` refuses an input listed twice: a worker derives its part index from
  the partition alone, so two partitions of one file and step wrote the same part path twice.


0.0.3
-----

A library can record into graphed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The headline of this release: the hooks a package needs to hand its users deferred arrays the way
it hands them dask ones — what wrapping graphed in coffea's NanoEvents, fastjet's
``ClusterSequence`` and uproot's form mappings turned up. :doc:`frontend/design` has a section on
wrapping graphed in a library of your own.

* ``graphed.provenance.register_internal("mylib")`` takes a library's frames out of the search for
  the analyst's line, so a recorded op — and the error a worker raises for it — points at the code
  that called the library rather than into it (#34).
* ``graphed.expand`` is public: the mapping that runs a verb answering one array once per
  universe of a varied operand, and passes an unvaried call straight through (#34).
* A source may declare its own read list with ``projected_columns(outputs)``. A form-mapped source
  — a NanoAOD schema over a flat tree — has field names that are not file columns; its answer
  ships to the workers verbatim, and a source without the hook reads exactly what it read before
  (#32).
* A node recorded with ``record_external(descriptor=, form=)`` types what follows it in column
  projection by the form it declared, not by its first input — jets out of particles (#32).

Awkward idioms the frontend used to refuse
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ``array[:, :2]``, ``a[:, 0]``, ``a[..., 0]`` and ``a[:, None, :]``: a tuple key that leaves the
  partitioned axis whole records, fuses like any per-row op and gives a partitioned run the same
  answer as an unpartitioned one. A key that would cut that axis is refused where you wrote it,
  naming the chained spelling (``a[1:3][:, 0]``) (#33).
* A numpy scalar operand keeps its dtype and its exact value: ``mask * np.uint64(1 << bit)``
  records ``uint64``, as awkward computes it, where it recorded ``float64`` (#33).
* A behavior method may return a nested tuple of arrays —
  ``metric_table(..., return_combinations=True)`` — and you get the same nesting back (#33).
* ``from_parquet`` keeps the record names and parameters awkward wrote into the file, so a
  behavior keyed on a record name resolves on the deferred array (#34).

Two silent wrong answers, now refused or fixed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **A Session refuses an array it did not record.** Asking one Session to read, materialize or
  compile an array recorded in another used to answer with that Session's *own* node of the same
  id — another analysis's result, without a word. Every entry now raises, recording nothing (#34).
* **Two different un-named callables no longer share a node.** ``x.map(lambda v: v * 2)`` and
  ``x.map(lambda v: v * 100)`` were both named ``<lambda>`` and interned to one node, so the second
  returned the first's result. A second distinct callable of a name now records as ``name#1``; the
  same bound method asked for twice is still one node, and a program whose callables have distinct
  names records the same bytes as before. ``name=`` stays your own declaration of identity (#34).

Packaging and documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* The MIT ``LICENSE`` file ships in the repository and in the sdist (#31).
* :doc:`architecture` says what is true of an opaque callable: it travels by value in the durable
  plan only — on a process pool it must be importable, so define it at module level (#34).

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
