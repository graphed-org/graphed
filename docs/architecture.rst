Repository structure & architecture
===================================

``graphed`` is **one pip-installable distribution** that used to be eight separate prototype
repositories. The former packages are now *subpackages* of ``graphed`` (``graphed_core`` →
``graphed.core``, and so on); the compiled Rust core ships inside the same wheel. This page maps the
import surface, the processing pipeline, the on-disk layout, and the packages that remain separate.

One package, several subpackages
---------------------------------

The base install is deliberately light — the deferred-array frontend, the compiled core, and two
small pure-Python dependencies: ``executing`` (provenance line lookup) and ``cloudpickle``
(serializing genuinely-opaque user callables in a durable plan). Everything heavier is an opt-in
extra.

How a missing extra behaves depends on the subpackage. The two array backends *eagerly* import their
array library, so ``import graphed.awkward`` / ``import graphed.numpy`` without ``[awkward]`` /
``[numpy]`` raises a clear ``ImportError``. ``graphed.debug``, ``graphed.checkpoint`` and
``graphed.preserve`` instead import at the base level and pull their heavier dependencies *lazily*,
only when a feature needs them — the live dashboard (``[dashboard]``) or the correctionlib/ONNX
preservation payloads (``[preserve]``). ``graphed.checkpoint`` needs no extra at all: its
``[checkpoint]`` is an empty back-compat marker, and the ``cloudpickle`` it relies on is already a base
dependency.

.. list-table::
   :header-rows: 1
   :widths: 22 58 20

   * - Import path
     - What it is
     - Install extra
   * - ``graphed``
     - deferred-array **frontend**: ``Session``, ``Array``, the five-method ``Backend`` protocol,
       provenance, and the backend-agnostic machinery (projection, compilation, the parquet/write
       I/O bases)
     - (base)
   * - ``graphed.core``
     - Rust+PyO3 **interned IR**, the equality-saturation optimiser, the deterministic durable codec
       (``GIR1``) and ``DurablePlan``, and the executor/monitor protocols
     - (base)
   * - ``graphed.awkward``
     - the **ragged backend** HEP analyses live on: awkward typetracer forms, the ``gak`` namespace
       (``ak.*`` parity), vector behaviours, buffer-level column projection, parquet I/O (with the
       separate ``[parquet]`` extra), and correctionlib/ONNX ``External`` payloads
     - ``[awkward]``
   * - ``graphed.numpy``
     - the **rectilinear backend**: deferred numpy idiom over the graph (ufunc/array-function
       recording, monoidal reductions)
     - ``[numpy]``
   * - ``graphed.debug``
     - opt-level lowering, source-mapped **picklable** tracebacks (a remote-worker error points at
       the user's line), and the live Perspective dashboard
     - ``[dashboard]``
   * - ``graphed.checkpoint``
     - content-addressed ``Store``, deterministic ``run_resumable`` resume, retry policies +
       dead-letter
     - ``[checkpoint]``
   * - ``graphed.preserve``
     - self-contained content-addressed **preservation bundle** (build / reproduce / inspect) plus
       the ML-framework ``External`` plugins (torch, tf, xgboost, jax, ONNX, correctionlib, Triton)
     - ``[preserve]``

Parquet I/O (``to_parquet`` / ``from_parquet``, available in either backend) needs the separate
``[parquet]`` extra (``pyarrow``). ``pip install "graphed[all]"`` pulls every extra — including
``[parquet]`` — except the heavy ML frameworks; ``[ml]`` adds those (torch/tensorflow/xgboost/jax/
tritonclient), and ``[dev]`` the full test/lint/type toolchain. See :doc:`the migration table </index>`
— or ``MIGRATION.md`` in the repository — for the old-name → import-path mapping.

The processing pipeline
-----------------------

A ``graphed`` analysis flows through the subpackages in one direction. Nothing builds a large graph
first: the frontend hands each recorded operation to the core, which interns and reduces it
incrementally, so only the concise fused-stage graph ever exists.

.. code-block:: text

    record              intern + reduce            evaluate                  run
    ┌──────────┐  op    ┌──────────────┐  plan   ┌──────────────────┐ tasks ┌────────────────────┐
    │ graphed  │ ─────▶ │ graphed.core │ ──────▶ │  graphed.awkward │ ────▶ │ graphed-exec-local │
    │(frontend)│        │ (Rust optim.)│         │  graphed.numpy   │       │  (separate package)│
    └──────────┘        └──────────────┘         └──────────────────┘       └─────────┬──────────┘
                                                                                       │ results
          graphed.debug  ── source-mapped errors + live dashboard (cross-cutting) ─────┤
          graphed.checkpoint ── content-addressed resume ──▶ graphed.preserve ── bundle ┘

1. **Record** — ``graphed`` (frontend). Ordinary array expressions become nodes through the
   pluggable ``Backend`` protocol; type errors surface at the recording line (form inference on
   metadata only).
2. **Intern + reduce** — ``graphed.core``. Structurally identical nodes share one ``NodeId``
   (hash-consing gives CSE for free); DCE is reachability from the outputs; equality saturation
   canonicalises and fuses maximal runs of array ops into stages. Reduction is incremental and
   deterministic (identical graphs serialise to identical bytes) and is CI-guarded against
   super-linear scaling.
3. **Evaluate** — a backend. ``graphed.awkward`` is the default HEP backend (ragged data, ``gak``
   ≙ ``ak.*``); ``graphed.numpy`` is the rectilinear one. Backends never leak into the frontend or
   the core.
4. **Run** — an executor. The reference executor, ``graphed-exec-local``, is a **separate package**
   (single-machine thread/process pools, tree reduction, inter-worker comms). ``graphed.core`` only
   defines the plan/executor protocol it consumes.
5. **Debug & preserve** cut across the pipeline. ``graphed.debug`` re-raises any failure — even one
   deep inside a fused stage on a remote worker — pointing at the user's analysis line, and streams
   a live dashboard. ``graphed.checkpoint`` makes a killed run resumable bit-for-bit;
   ``graphed.preserve`` exports a bundle that reproduces the histograms on a clean machine.

On-disk layout
--------------

.. code-block:: text

    graphed/
    ├── Cargo.toml, src/*.rs        # the Rust core crate → compiled to graphed.core.graphed_core
    ├── pyproject.toml              # maturin build backend; [project.optional-dependencies] = the extras
    ├── python/graphed/             # the pure-Python distribution
    │   ├── __init__.py, session.py, array.py, backend.py, provenance.py, …   # the frontend
    │   ├── core/                   # thin Python wrapper re-exporting the compiled extension
    │   ├── awkward/  numpy/        # the two backends
    │   ├── debug/  checkpoint/  preserve/            # debugging, resume, preservation
    │   └── preserve/externals/     # the ML-framework External plugins
    ├── tests/frozen/<pkg>/mX/      # the frozen acceptance suites, one subtree per subpackage
    └── tests/_corpus/              # the graphed-corpus fixtures, vendored for this repo's tests

The compiled extension keeps its leaf name — it lives at ``graphed.core.graphed_core`` and is
re-exported by ``graphed.core``; import from ``graphed.core``, never the extension directly. Because
the frozen suites of the eight former repositories carry duplicate basenames, they are run **one
subtree at a time** (``scripts/run-tests.sh``); see ``CONTRIBUTING.md``.

Packages that stay separate
---------------------------

Not everything folded into the distribution. These remain their own repositories and depend on
``graphed``:

- `graphed-exec-local <https://github.com/graphed-org/graphed-exec-local-mvp>`_ — the reference
  **executor** (single machine): thread/process pools, tree reduction, work-stealing, inter-worker
  comms. Kept separate because executors are pluggable and this is only the local reference one.
- `graphed-histogram <https://github.com/graphed-org/graphed-histogram-mvp>`_ — deferred
  boost-histogram/``hist`` fills on ``graphed`` graphs (the dask-histogram analogue); powers
  ``hist.graphed``.
- `graphed-orchestrator <https://github.com/graphed-org/graphed-orchestrator>`_ — the deterministic
  state machine that drives the gated three-role (test-author / implementer / reviewer)
  development pipeline; not imported by analyses.
- `graphed-corpus <https://github.com/graphed-org/graphed-corpus-mvp>`_ — the M0.5 Required
  Operations Catalog and canonical-analysis fixtures. Vendored under ``tests/_corpus/`` here for
  this repository's own suite, and still published separately for downstream repositories that
  consume the fixtures (see :doc:`corpus/index`).
