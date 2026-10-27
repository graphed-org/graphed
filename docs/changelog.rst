What changed
============

Newest release first. Numbers in parentheses are the pull requests on
`graphed-org/graphed <https://github.com/graphed-org/graphed>`_.

0.0.10 (unreleased)
-------------------

Several datasets in one Session
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ``aggregate_plan``, ``shuffle_plan``, ``join_plan``, the awkward and numpy ``to_parquet`` and
  the varied write count only the sources their outputs read, so one Session can record several
  datasets (a reader's ``session=``) and join them (#103).
* ``Session.source`` refuses a name reused with another form or another data object: the second
  binding was silently dropped, so ``from_awkward(s, "x", B)`` after ``from_awkward(s, "x", A)``
  evaluated ``A`` (#103).
* ``graphed.materialize(arrays)`` evaluates Arrays from any number of Sessions in this process,
  alone or nested in lists, tuples and dicts, as ``dask.compute`` takes them. It reads only the
  source columns those Arrays use, as a plan over them would. A plan over several Sessions is
  ``graphed.collate``, which ``aggregate_plan``'s refusal now names (#103).

Interrupts stop a resumable run
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ``run_resumable`` lets a ``KeyboardInterrupt`` or ``SystemExit`` raised inside a task stop the
  run: it recorded them as dead letters and carried on with the next partition (#111).

A plan finalizes its value on the driver
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ``Plan.finalize`` (and ``DurablePlan.finalize`` / ``DurablePlanV2.finalize``, an ``OpSpec``)
  turns a run's value into its final form once, on the driver, after the last combine — also for
  an empty, stopped, cancelled or resumed run. ``SequentialRunner``, ``graphed.checkpoint``'s
  ``run_resumable`` and ``resumable`` apply it; ``aggregate_plan(finalize=)`` sets it and
  ``collate`` applies each plan's to its own name. A runner of its own returns
  ``graphed.core.finalize_result(plan, result)``. A durable plan without one keeps its bytes.

Exceptions are named ``…Error``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* PEP 8 names: ``StoreUnavailableError``, ``EnvironmentChangedError`` (``graphed.checkpoint``),
  ``UnboundServiceError`` (``graphed.services``) and ``UnresolvedPayloadError``
  (``graphed.preserve``). The first three keep their old names as aliases.

The systematics internals, reorganized
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ``graphed.systematics`` is split by owner, and its imports run one way: a Session's variation
  registries are one ``Session.systematics`` object, an event context's ambient weight is one
  record that ``graphed.systematics.ambient`` alone builds, and a value's context stamping lives
  in ``graphed.systematics.handles``. What ``vary``, the readers and the plans produce is
  unchanged (#113).
* ``graphed.vary`` on a Session shared between threads is atomic: a failing ``vary`` in one
  thread rolled back the registry to its own snapshot, erasing a ``vary`` another thread had
  minted meanwhile, and ``graphed.explain`` could raise while another thread minted (#113).
* ``Session.reachable_sources(node_id)`` and ``Session.recorded_op(node_id)`` answer which sources
  a node reads and what op recorded it.
* ``graphed.write.sources_read`` refuses outputs from more than one Session with a
  ``TypeError``; it answered with the first output's Session's sources alone.
* ``graphed.systematics.registration.vary`` is ``graphed.systematics.verb.vary``;
  ``graphed.vary`` and ``graphed.systematics.vary`` are unchanged. The compatibility modules
  (``graphed.vary``, ``graphed.varied``, ``graphed.accessors``, ``graphed.context`` and the rest)
  keep every public name and stop re-exporting private ones.

0.0.9
-----

Fixes
~~~~~

* ``gak.apply_correction`` takes ``compound=True`` to apply an entry of the set's
  ``compound_corrections``: a compound correction that shares its name with a regular one
  evaluated the regular one. The flag is part of the node's identity; a regular correction's node
  and plan bytes are unchanged. Without the flag a regular correction still wins and a
  compound-only name still falls back to ``compound_corrections`` (#96).
* ``gak.apply_correction`` with ``args=``, and an External recorded through a preserve plugin,
  record the broadcast of their array inputs' types instead of the first input's: a flat argument
  before a jagged one recorded a flat result, so ``gak.sum``/``gak.num``/``gak.flatten`` at
  ``axis=1`` on it raised ``GraphedTypeError`` (#97).
* ``PartWrite`` takes ``storage_options``, given to the fsspec filesystem its part directory is
  created on; that filesystem was built from the URL alone, so a destination needing options to
  connect (S3 with a custom endpoint, sftp) failed before its codec ran. A write without them ships
  as before, so its plan's bytes do not change (uproot5#1758).

0.0.8
-----

Field assignment on deferred awkward arrays
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ``AwkwardArray`` and ``AwkwardVaried`` take ak's field assignment, ``events["Photon"] = photons``
  and ``events["Photon", "pt"] = pt``, recorded as ``with_field`` and rebinding only that proxy.
  A key that is not a field name or a tuple of them raises ``GraphedTypeError``, as does assigning a
  ``Varied`` value into a plain array (rebind with ``gak.with_field`` instead) (#80).
* ``gak.with_field`` (and so field assignment) takes a scalar value, which awkward broadcasts in the
  scalar's own dtype (``np.float32(0.5)`` stays ``float32``); it raised ``AttributeError``. Any other
  value (``None``, a complex or datetime scalar, a list or concrete array) raises ``GraphedTypeError``.
  A varied ``to_parquet`` of a record holding such a field writes it (it raised ``IndexError``), with
  no delta column when the scalar is the same in every universe (#80).

Less time and memory per plan and task
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ``project_buffers_many`` replays a sub-graph its outputs share once instead of once per output, so
  planning many outputs over common selections no longer costs the shared graph once per output.
  ``Session.walk`` takes a ``memo`` that walks of several outputs share (#81).
* ``gak.apply_correction`` called again with the same payload object no longer re-digests it, and
  its ``float64`` output type is parsed once per process: recording a 14 MB correction set drops
  from about 10 ms to 0.5 ms per call (#86).
* A task drops each intermediate value once the last step that reads it has run, inside a fused
  stage as well as between nodes, instead of holding every value until the task returns: a
  whole-file task's peak memory no longer grows with the length of the analysis (#90).

Fixes
~~~~~

* An op recorded or refused inside installed third-party code (``ak.from_iter``, a ``boost_histogram``
  or ``hist`` fill, a correctionlib ``evaluate``) names the analysis line that called it, not the
  library's file: provenance skips frames in site-packages, unless every frame outside graphed is
  there, as for an analysis installed non-editable (#92).
* ``gak.values_astype``, ``zeros_like``, ``ones_like`` and ``full_like`` take a dtype as numpy reads
  one (``np.float64``, ``float``, ``np.dtype(...)``), not only its name (#83).
* ``gak.cartesian``/``argcartesian`` (and ``ak.cartesian`` on a deferred array) take a dict of named
  slots, nesting by key, as awkward does (#83).
* numpy functions that are not ufuncs (``np.clip``, ``np.where``, ``np.nan_to_num``, ...) record on a
  deferred awkward array or ``Varied`` the way ``ak.Array`` answers them; one awkward implements but
  ``gak`` lacks (``np.round``) is refused naming the missing ``ak`` function (#83).
* ``hasattr``/``getattr(x, name, default)`` on a deferred array report a missing field as absent: the
  ``GraphedTypeError`` it raises is also an ``AttributeError`` (#83).
* A ``Varied`` answers a form attribute (``.fields``, ``.ndim``, ...) with the value, not another
  ``Varied``, and refuses one its universes disagree on (#83).
* A scalar operand keeps its type, as eagerly: a numpy scalar in ``gak.where``/``fill_none``
  (``where(m, x_float32, np.float32(1))`` stays ``float32``), an integer ``initial`` of a reducer
  (it crashed on an integer array) or ``full_like`` value, a numpy scalar or 0-d array passed to a
  behavior method, and any scalar in the numpy backend's ``np.where``/``np.clip``/``full_like``
  (``np.where(m, x_int32, 0)`` stays ``int32``). Scalars of different dtypes no longer share one
  node (#83).
* A leaf that is equal in every universe no longer gets an all-zero delta column per universe when
  it reaches the written record through ``without_field``, a field selection (``rec[["x", "y"]]``),
  or a field projection (a sibling of ``rec["Jet", "w"] = w``), as under ``with_name`` and
  ``with_parameter`` since 0.0.7 (#80).
* A plan whose ``graphed.apply``/``map`` functions, ``reduce``, ``combine`` or ``empty`` are lambdas,
  closures, partials of them or ``__main__`` functions runs on process and remote runners: the plan
  pickles each such function through cloudpickle (by value unless importable), and task keys are
  unchanged. A backend given to ``aggregate_plan(backend="module:factory")`` gives its behavior to
  the arrays it evaluates, and to what ``reduce``, ``combine`` and ``empty`` return, as that import
  ref, so a partial carrying it (vector's behaviors hold lambdas) pickles too, also after an op or a
  combine over several arrays (#82).
* ``graphed.preserve.record_external`` on a ``Varied`` input records one External per universe, as
  ``graphed.apply`` does; it raised ``AttributeError`` (#85).
* An ML plugin's (n, k) output keeps one row of k per event, as ``ak.from_numpy`` gives it; it was
  flattened to n·k values. An (n,) or (n, 1) output is still one value per event. Declare the row
  with ``output_type="k * float64"`` (#85).
* ML plugin inputs keep their dtype: torch and onnxruntime models are fed the dtype the model
  declares (a float64 model runs in float64, bit-equal to calling it), xgboost, keras and jax
  convert as they do when called directly, and Triton requests stay FP32 (#85).
* A gzipped JSON payload (correctionlib's ``.json.gz``, an xgboost or Triton JSON) hashes, loads and
  evaluates as its plain twin; its content hash raised ``UnicodeDecodeError`` (#85).
* A preservation bundle holds a histogram fill with scalar constants (``h.fill(x, 1.5,
  weight=[w, 2.0])``, graphed-histogram's constant fills): the histogram plugin rebuilds the
  fill's content hash and evaluator with its constants, so ``build_bundle`` no longer reports a
  hash mismatch and ``reproduce`` replays the fill (#89).
* A varied ``to_parquet`` whose ``select=`` mask is option-typed (``gak.mask(e.m > 100, e.n > 0)``)
  selects as ak indexing does: a ``None`` row reads back from ``read_varied`` as a ``None`` record in
  every universe whose mask holds it, and a ``None`` object as ``None``, with the leaves' own types.
  It raised ``ValueError`` at a level-1 mask and dropped the row at level 0, making the leaves
  optional (#91).


0.0.7
-----

A deferred array has no truth value
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ``if x:``, ``bool(x)``, ``np.asarray(x)`` and ``np.array(x)`` on a deferred ``graphed.Array`` or
  ``Varied`` raise ``GraphedTypeError`` at the calling line. They answered ``True`` and a 0-d object
  array, so an ``if`` on a deferred condition always ran its body and ``np.where`` on an
  awkward-backed condition took one branch for every row. Select elementwise with
  ``np.where`` (numpy) or ``ak.where``/``gak.where`` (awkward) or ``&``/``|``/``~``, or
  ``materialize`` first.

``ak.*`` on deferred awkward arrays
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ``graphed.awkward.AwkwardArray`` (and ``AwkwardVaried`` for its ``Varied`` containers) answers
  awkward's ``__awkward_function__`` hook: ``ak.<fn>(x)`` records exactly what ``gak.<fn>`` records,
  binding ``ak``'s own argument positions and names. ``AwkwardBackend`` sessions return it; a
  backend subclass can return its own subclass of it from ``array_type``. ``ak`` functions with no
  ``gak`` counterpart raise ``GraphedError``.

Resuming any plan from a store
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ``graphed.checkpoint.resumable(plan, store, *, storage_options=None, codec=None, salt="",
  accept_environment=False)`` returns a plan (a runtime ``Plan``, a ``DurablePlan`` as a runtime
  ``Plan``, or a ``DurablePlanV2``) whose process records each finished task under ``store`` and
  decodes the ones it already holds, so any fixed-task runner resumes. ``check_resumable(plan)``
  runs its refusals without I/O. Workers raise ``StoreUnavailable`` for a store they cannot reach.
* A store keeps an environment record per ``salt`` (installed distributions and the interpreter's
  cache tag); a run in another environment raises ``EnvironmentChanged`` unless
  ``accept_environment=True``. ``run_resumable`` and ``run_shuffle_resumable`` check it too and take
  ``accept_environment=``. ``Store`` and ``FsspecStore`` gain ``record_environment``, and
  ``completed()`` sets those records aside on its ``environments`` attribute.
* ``run_resumable`` folds each partial into the result as it is produced, in task order, instead of
  holding every partial until the end; values are unchanged bit for bit.

Writes to an fsspec URL
~~~~~~~~~~~~~~~~~~~~~~~

* A write destination containing ``://`` (``root://``, ``s3://``, ``memory://``, ...) is an fsspec
  URL: ``aggregate_plan(writes=...)`` with ``parquet_write``, ``graphed.awkward.to_parquet``
  (``select=`` too) and ``graphed.numpy.to_parquet`` write each part, and its parent directory, on
  that URL's filesystem instead of creating a local ``root:`` tree or failing in pyarrow. A part is
  ``destination`` with the trailing slashes after its ``://`` dropped, then ``/`` and the name, on
  every OS. A URL needs ``graphed[checkpoint]`` (an ``ImportError`` names it) plus the protocol's
  fsspec driver; ``parquet_write`` and the numpy writer import no fsspec for a local destination.
  ``graphed.write`` gains ``is_url``, ``join_part``, ``part_fs``, ``prepare_part`` and ``url_fs``.
* A leading ``dir::`` on a write destination or a ``FsspecStore`` URL roots at the wrapped URL,
  with a ``UserWarning`` to pass that URL instead, and the store's storage options reach the
  wrapped filesystem. A ``dir::`` store written by an
  earlier release sits at the wrapped path repeated under itself and is not found at the new root.

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
  ``http``, ``https``, ``grpc``, ``grpcs``). A part bound earlier keeps its endpoint.
* ``SequentialRunner`` refuses a plan with an unbound service before its first task:
  ``graphed.services.require_bound(plan)`` raises ``UnboundService`` naming every missing service
  (``.names``), so a collated plan writes no parts first. A plan without services skips the check.
  A runner with its own task loop must make the same call before its first task.
* A ``reduce`` that calls a service must give itself a ``bind_services`` hook that raises
  ``UnboundService`` when the endpoints lack its name and it holds none, as an External does; the
  check then names it too. A reduce without such a hook is not checked.
* ``graphed.services.resolve_services(plan, value)`` resolves a run's value through the parts
  ``bind_services`` reaches: a process or ``reduce`` with a ``resolve_services`` hook
  (``graphed.services.Resolvable``) returns the resolved value, and ``collate`` hands each plan's
  process its own entry. A part without the hook keeps its value.
* A Triton External names ``service=`` or a literal ``url=``, not both. The endpoint's scheme picks
  ``tritonclient.http`` or ``tritonclient.grpc`` (TLS on ``https``/``grpcs``); a url without a
  scheme keeps the HTTP client and ``params["transport"]`` still overrides. The ``ml`` extra
  installs ``tritonclient[grpc,http]``.
* The per-process External resource cache keys on the endpoint and the params ``load`` reads
  (``ExternalPlugin.load_params``), so a correction set is still loaded once across systematics.
* Preservation bundles list the referenced specs in ``manifest["services"]`` (absent when there are
  none) and ``inspect()`` prints them. ``RunReport.endpoints`` records where a run reached each
  service, outside the fingerprint; ``reproduce`` refuses an External that calls a service.

Join and repartition plans run their recording
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ``join_plan`` and ``shuffle_plan`` build stages that evaluate the recorded graph: each
  ``map_write`` task reads its partition and evaluates up to the join or exchange, each gather
  task joins or concatenates its destination and runs what was recorded after it. Before, the
  stages were block kernels that no runner could run, and operations after a join were dropped.
  ``shuffle_plan`` now ends in a one-task ``reduce`` stage that folds the destinations; the
  plans' bytes and task ids change with their stages.
* ``join_plan(reduce=, combine=, empty=)`` (all or none) folds the per-destination values as
  ``aggregate_plan`` folds partitions; without them the plan's value is one output per
  destination. Each ``map_write`` stage reads the source of the matching join input, so the left
  side stays on the left whatever order the sources were registered in.
* ``SequentialRunner().run`` takes a ``DurablePlanV2`` and returns ``plan.value`` of its last
  stage (``DurablePlanV2.value``); a cancel returns ``value=None``. ``run_shuffle_resumable`` runs
  builder output too. ``graphed.shuffle.split``/``pick`` read and slice a map-write payload.
* ``DurablePlanV2.services`` carries the services a join or repartition plan's operations call
  (``services=`` adds names), written only when non-empty, outside the task ids.
  ``bind_services``, ``require_bound`` and ``resolve_services`` take a ``DurablePlanV2``; binding
  sets ``OpSpec.live`` and changes neither the bytes nor any task id.
* ``evaluate_ir`` evaluates ``exchange`` and ``join`` nodes through the backend, as
  ``Session.materialize`` does, and takes ``outputs=``/``given=`` to evaluate one side of a
  barrier.
* A per-destination awkward join has the whole join's type, also where one side has no rows.
* The builders refuse, when the plan is built, a ``target_bytes=`` repartition (run it with
  ``run_repartition_by_size``), an operation after the join or exchange that reads a source
  directly, a reduction a node consumes on either side of the barrier (and an unfolded one at a
  ``join_plan`` output), and ``gak.join(grouped=True)`` with ``how="right"`` or ``"outer"``.
* An operation that fails in a join or repartition plan raises a ``StageError`` at its recording
  line, as in an aggregate plan.
* ``graphed.shuffle.partition_block`` is removed: no stage routes blocks outside the recorded
  graph any more.

Declared output types for external calls
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ``Array.map``, ``graphed.apply`` and ``Session.record_external`` take ``output_type=``, the type
  of each element of the call's value, and record that type instead of the first input's (awkward)
  or ``vector[object]`` (numpy). A boolean mask declared ``"bool"`` indexes as a mask at build
  time, and a declared named record resolves its behavior. awkward takes type strings, type
  objects, forms, numpy dtypes and Python types; numpy takes dtypes of every kind and Python
  types; every numerical dtype works on both, except that awkward 2.14's type grammar refuses
  ``float16`` inside a list or record or with parameters. The declaration is a node
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
* A declared type is checked against the value each time the call runs, never cast. A value of
  another type raises ``graphed.OutputTypeError``, a ``GraphedTypeError``, at the declaring line;
  an aggregate plan raises a ``StageError`` there, from a worker process too, and a bundle's
  ``reproduce`` raises it too. The value is an array when any input is. awkward compares the exact type
  string, letting ``unknown`` (a list with no values) fit anything; numpy compares the dtype, the
  trailing shape a subarray declares and the leading axis. A plugin's ``output_dtype``, and
  ``gak.apply_correction``'s float64, are checked leaf by leaf. Undeclared calls are not checked.

One plan for every output
~~~~~~~~~~~~~~~~~~~~~~~~~

* ``aggregate_plan(writes=[...])`` writes one part per task beside the plan's reductions, from the
  same read and evaluation. ``reduce`` receives the outputs' values as before, then one part path
  per write. A write is a ``graphed.write.PartWrite``: the array, a destination, a ``name`` for
  each task's partition, the backend's codec, and ``metadata`` whose array values are that part's
  own reductions. Colliding part names, a reduction as a written array, and ``store=`` with writes
  are refused when the plan is built.
* ``graphed.collate({name: plan})`` joins plans over different graphs and sources, data and MC
  say, into one plan whose value is ``{name: value}``. Each task runs its own plan's graph, and
  the runner tree-reduces all the tasks together. A part two of its plans would both write, from
  ``writes=`` or ``to_parquet``, is refused when it is built; ``graphed.debug.replay`` refuses a
  plan with writes. Its ``Plan.services`` is the union of its plans' services and
  ``bind_services`` binds every plan's process; a service name two plans declare differently is
  refused.
* ``graphed.awkward.parquet_write`` writes parquet parts through ``ak.to_arrow_table`` and
  ``pyarrow.parquet.write_table`` with options for each, and per-part key-value metadata that
  replaces the schema's.
* ``aggregate_plan(opt_level=0)`` ships the 1:1 cone of the plan's outputs, written arrays and
  metadata arrays (``GraphStore.cone(outputs=)``) instead of the optimized graph, so ``reduce``
  receives one value per distinct output where ``1``, the default, may merge equal ones; any other
  level is refused. Its ``StageError`` reports the level, and ``graphed.debug.replay`` recompiles at
  the plan's level and reduces one value per output of the plan's IR, as the run does.
* ``refuse_chunk_partials(as_outputs=)`` also accepts the compiled output ids to refuse.
* ``gak.num(x, axis=0)`` records a reduction, so a per-chunk count can no longer feed another node
  silently; as a plan output it folds like any other reduction.

Fixes
~~~~~

* ``NetworkMonitor`` no longer drops an event emitted just as its sender wakes with no connection
  open, typically a worker's first event. The sender drains its buffer before deciding to connect, so
  a per-worker monitor in a spawned process ships that event before the process exits.
* A variation-aware ``to_parquet(..., select=)`` no longer corrupts a universe whose leaf type
  differs from nominal's (``float32`` vs ``float64``, ``int`` vs ``float``, option vs not):
  ``read_varied`` gave it nominal's type, with its bits reinterpreted. That universe's value is now
  stored as-is (manifest representation ``"base"``); a primitive or once-jagged leaf that keeps
  nominal's type still stores an XOR delta, and those files are unchanged. A file holding a ``"base"`` column also carries a manifest
  marker that makes ``read_varied`` from graphed 0.0.6 raise ``GraphedError`` instead of returning
  nominal's values for that universe.
* ``read_varied`` no longer crashes on a flat option leaf holding ``None``, and a universe whose
  ``None`` rows differ from nominal's reads back with its own ``None`` rows (a jagged leaf also took
  nominal's). The XOR delta is masked where the label is ``None``.
* A varied regular-list or optional-list leaf keeps its type through ``read_varied`` (it came back
  ``var``, or without its option), and a leaf that ``zip`` broadcasts into a collection, or that a
  universe turns from jagged to flat, no longer crashes the write. Such leaves, and ``datetime64[s]``
  leaves (which parquet stores as ``[ms]``), are stored as-is.
* ``read_varied`` restores each universe's parameters (``__doc__`` and the like) on leaves stored as
  an XOR delta, and the top record's own parameters (``with_name``, ``__doc__``) for every universe,
  nominal included; both came back without them. The manifest records a universe's form when its
  type, parameters included, differs from nominal's, and nominal's when its top record has
  parameters; other files are unchanged.
* A variation-aware ``to_parquet(..., select=)`` no longer raises ``IndexError`` on a record whose
  fields are reached through a nested collection, such as ``gak.with_field(events, varied, "m")``.
* Under a ``with_name`` or ``with_parameter`` record, a leaf that is equal in every universe no
  longer gets an all-zero delta column per universe.

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
