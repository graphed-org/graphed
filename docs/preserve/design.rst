How graphed-preserve works
==========================

``graphed-preserve`` answers the question every analysis eventually faces: *can someone — a
collaborator, a reviewer, you in three years — reproduce this result on a clean machine, bit
for bit?* A **preservation bundle** is a self-contained directory capturing the analysis as
durable data: the canonical IR, the input datasets, every external payload (corrections,
models, histogram specs) by content hash, the provenance map, and the environment record. It
can be *inspected* without executing anything and *reproduced* from its own contents alone.

The package invents no formats: corrections are correctionlib JSON, models are ONNX, the IR is
graphed's canonical bytes, payload identity is SHA-256.

.. contents::
   :local:
   :depth: 2


Anatomy of a bundle
-------------------

On disk, a bundle is two things::

    bundle/
      manifest.json     # the content-addressed bill of materials
      store/            # a graphed-checkpoint Store: every blob, keyed by hash

The manifest *names* everything and *contains* nothing heavy: the IR's hash, each dataset's
hash, each external payload's hash with its kind and node binding, the provenance sourcemap's
hash, the config/seed, and a pinned environment description. ``Bundle.fingerprint()`` hashes
the canonical manifest bytes — one identifier for "exactly this analysis on exactly these
inputs", stable across rebuilds (building the same analysis twice yields the same
fingerprint; the frozen suite pins it).

Two representation decisions carry most of the design:

* **The IR is preserved at opt_level=0** — one node per user operation, unfused. The bundle is
  an *audit artifact* first: ``inspect()`` renders each node with its parameters and the user
  source line that recorded it. Optimization is a *consumer's* step (re-running reduces the IR
  first), not something baked into the preserved form.
* **Everything heavy is content-addressed.** Datasets and payloads live in the store by hash;
  the manifest's references are integrity checks by construction — a corrupted or substituted
  blob simply fails to resolve.

Building
--------

``build_bundle`` takes a recorded session plus what cannot be derived from it: the datasets
behind each source and the bytes behind each external payload. Two calling conventions:

* **Histogram-terminal** (the common case): ``build_bundle(root, session=s, value=fill_node,
  datasets=..., payloads={})``. The analysis *ends at* a histogram fill — an External node —
  and ``reproduce()`` returns the histogram itself. Note ``payloads={}``: a fill's payload is
  its canonical axes/storage spec, which is **synthesized at build time from the node's own
  parameters** (see plugins, below).
* **The (value, weight, spec) triple**: ``build_bundle(..., value=v, weight=w,
  histogram={"name": ..., "bins": ..., "lo": ..., "hi": ...})`` for analyses whose histogram
  step lives outside the graph. ``weight`` and ``histogram`` must be given together or not at
  all — the coherence check is explicit.

The build walks the IR's External nodes: each one's descriptor kind selects a **plugin**; the
payload bytes are integrity-checked against the descriptor's recorded content hash before
storage. An External with *no* plugin — the cloudpickled ``map`` is the canonical example — is
not preservable as durable data; it is recorded in ``opaque_nodes`` and surfaces in
``inspect()`` as an explicit **preservation risk**, never silently.

The plugin registry
-------------------

An :class:`~graphed_preserve.ExternalPlugin` defines, for one payload kind: a deterministic
``content_hash`` over payload bytes, ``load`` (materialize once per run — parse the correction
set, load the model), ``evaluate`` (run it on inputs), ``close``, and ``samples`` — at least
two distinct example payloads used to *validate the hash itself*. ``register_plugin`` rejects
a hash that is non-deterministic across processes (catching anything built from ``hash()``,
``id()``, time, or randomness) or vacuous (distinct samples colliding). Eight plugins ship and
double as templates:

* ``correctionlib`` — hash over the correction *contents*, not incidental file formatting;
* ``onnx_model`` — hash over weights + graph structure;
* ``histogram`` — the structurally interesting one: its payload **is** the fill's canonical
  axes/storage spec, whose SHA-256 is *already the node's identity* (the same bytes hash to the
  same value on both sides of the seam — pinned). Because the spec also rides in the node's
  parameters, the plugin implements the ``synthesize`` hook: at build time the payload is
  derived from the node itself, so callers supply nothing. Its evaluator reconstructs the fill
  from those parameters;
* the **ML-framework family** — ``tensorflow_model`` (``.keras`` archives),
  ``pytorch_model`` (TorchScript), ``xgboost_model`` (XGBoost's open JSON format),
  ``jax_export`` (``jax.export`` StableHLO artifacts), and ``triton_model`` (below). The
  recurring lesson, pinned per framework: **content identity is not byte identity.** Every one
  of these archive formats embeds volatile metadata — zip timestamps, auto-generated layer
  names, MLIR source locations — so re-saving an identical model yields different bytes. Each
  hash therefore goes through the framework's own loader and digests what *is* content: weights
  plus architecture (TorchScript code, name-stripped Keras config, location-stripped StableHLO,
  canonicalized model JSON), stable across re-saves and sensitive to either kind of change.

``triton_model`` is the odd one out and earns its own note: a Triton-served model is *remote* —
the payload preserves the **served model's identity** (a canonical JSON descriptor), while the
connection is environment, resolved per worker through an importable transport factory
(``params["transport"] = "module:attr"``; the default builds a ``tritonclient`` HTTP client).
The bundle preserves *what was called*; it cannot bottle the server — and a vanished server
fails loudly at reproduce time, never silently. The same injectable seam is how the frozen
suite runs the full record→build→reproduce path against a fake transport with no frameworks
installed.

A ``ResourceCache`` loads each payload once per run and reuses it across calls and nodes —
``open_once`` for Externals, so a model is not re-loaded per partition.

Inspecting without executing
----------------------------

``inspect(bundle)`` renders the whole preserved analysis from the manifest and sourcemap
alone — no dataset is read, no payload resolved::

    Preservation Bundle  fingerprint=sha256:...
      environment: python 3.12.10; 10 pinned packages
      histogram: None
      graph (IR, opt_level=0):
        n0   source   events  params={...}  <- []   [analysis.py:12]
        n1   op       field   params={...}  <- [0]  [analysis.py:14]
        ...
      external payloads (HEP standards, content-addressed):
        n41 histogram (uhi) sha256:4f58b35e...
      input datasets:
        events: sha256:...
      no opaque nodes (every node is durable IR or a content-addressed payload)

This is the review artifact: every operation, its source line, every external dependency with
its identity, and an explicit risk section — readable before deciding to run anything.

Reproducing
-----------

``reproduce(bundle)`` re-instantiates the analysis from references alone: resolve the IR from
the store, bind each source to its content-addressed dataset, resolve each external payload
through its plugin (``UnresolvedPayload`` if anything is missing — loudly), interpret the
node list through the ragged backend, and return the result — the histogram itself for a
histogram-terminal bundle. Reproduction is pinned bit-for-bit against the build-time result.

Re-running is deliberately more than ``reproduce``: because the preserved IR carries its
output marks, a consumer can *reduce it first* (the full DCE/CSE/fusion pipeline — on a real
analysis, dozens of audit-grade nodes collapse to a few stages with the fill as terminal) and
evaluate the optimized graph per partition of **new** inputs through any executor. Preserved
analyses are not museum pieces; they re-target.


Phase 2 (deliberately not built)
--------------------------------

* **Behavior-carrying recordings.** The reproduction interpreter evaluates through a bare
  backend; analyses preserved today express behavior properties as explicit formulas. Carrying
  a behavior *reference* (importable, like executor backends) through the manifest is the
  designed extension.
* **Export targets** (REANA/CAP/Zenodo/RECAST packaging) — the bundle is the substrate; the
  exporters are Phase 2 by plan.
* **Partial bundles** (datasets-by-reference for inputs too large to embed, with resolution
  recipes instead of bytes).

See :doc:`improvements` for the live tracked list.
