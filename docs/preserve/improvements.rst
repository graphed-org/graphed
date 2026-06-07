Improvements
============

Tracked design improvements and known limitations for ``graphed-preserve`` (plan M0 requires this
file in every package).

External plugins (delivered)
----------------------------

Externals are an **extensible plugin** system, not a hardwired list. An ``ExternalPlugin`` provides,
for one ``kind``: a deterministic, content-based ``content_hash`` (ONNX → hash of *weights* + graph
structure; correctionlib → hash of *contents*; the ``sha256_bytes`` template → raw bytes), an
``evaluate``, and ``samples`` for validation. ``register_plugin`` validates the hash before trusting
it — **deterministic across processes** (run under two ``PYTHONHASHSEED``\\ s, so a hash built from
``hash()``/``id()``/time/randomness is rejected) and **non-vacuous** (distinct payloads must not
collide). Users record their own preservable Externals with ``record_external(session, plugin, ...)``
and the ``onnx`` / ``correctionlib`` plugins serve as templates. ``build_bundle`` verifies each
stored payload hashes to its recorded id (cache-poisoning-safe).

Resolved design choices (the M9 plan flagged these as needing a decision)
-------------------------------------------------------------------------

- **Embed vs. reference threshold.** *Every* payload (IR, datasets, corrections, models, sourcemap)
  is written to the content-addressed Store and referenced from the manifest **by hash**; nothing is
  inlined into the manifest. This makes the manifest small and uniform and guarantees the bundle is
  "runnable from references alone" (plan guardrail). A future size threshold that inlines tiny blobs
  for convenience is possible but deliberately declined for the MVP.
- **Environment capture.** A lockfile-style record (Python version + pinned package versions) is
  captured for audit and is part of the fingerprint; an optional ``container_digest`` may also be
  recorded. The bundle does **not** itself recreate the environment (that is a container's job,
  Phase-2) — ``reproduce`` runs in the ambient interpreter.

External plugins — validated against real ML frameworks
-------------------------------------------------------

The ``ExternalPlugin`` shape — ``content_hash(bytes)`` + ``evaluate(bytes, params, inputs)`` +
``samples()`` — was stress-tested against **PyTorch** (TorchScript; hash of weights; single- and
multi-input), **XGBoost** (booster bytes; the ``sha256_bytes`` template suffices), and the **NVIDIA
Triton** remote-inference pattern (``tests/frozen/m9/test_ml_plugins.py``; optional, ``pip install -e
.[mltest]``). The shape held; three findings worth knowing:

The plugin now carries an optional ``load(payload, params) -> resource`` + ``close(resource)`` and a
:class:`ResourceCache`, so a model/connection is materialized **once per worker** and reused — both
findings below are addressed:

- **Per-worker resource cache (delivered).** ``evaluate(resource, ...)`` runs against a resource the
  cache ``load``-ed once (``torch.jit.load`` / ``Booster.load_model`` / a Triton connection), not
  per call — ``open_once`` for Externals. ``reproduce`` closes all resources at the end of the run.
- **Remote services are first-class but not bottled.** A Triton (or any remote) model is reached
  through a live client that the plugin ``load``-s from the *environment* (a url in ``params``) and
  ``close``-s at run end; the bundle content-addresses the *served model's weights* and ``reproduce``
  fails loudly if the service is absent. Consistent with "environment captured, not enforced":
  remote externals reproduce only where the service exists. A self-hosting bundle (embedding +
  launching a servable model) is a possible Phase-2 extension.
- **Conflicting native runtimes.** torch and xgboost each vendor an OpenMP runtime and clash in one
  process — another reason the bundle records the environment. The test suite sets
  ``KMP_DUPLICATE_LIB_OK`` / ``OMP_NUM_THREADS`` before import to coexist.

Current limitations
-------------------

- **opt_level=0 IR.** The preserved IR is the unfused, 1:1 op graph (auditable; ``inspect`` ==
  ``reproduce``). Interpreting an M4-stage-fused IR is a tracked extension (a stage executor).
- **Single variation per bundle.** Systematics are captured per bundle via config (the chosen JES
  factor + correctionlib systematic). Systematics-as-a-graph-axis (one bundle, all variations) is
  Phase-2 (plan Part F).
- **External evaluators.** correctionlib (weight) and ONNX (inference) are supported; an opaque
  cloudpickled node is honestly surfaced by ``inspect`` as a preservation risk rather than run.

Planned
-------

- A stage-aware interpreter so an M4-optimized IR can be preserved/reproduced directly.
- HS3 capture for any statistical model; UHI serialization of the reproduced histogram to disk.
- Verifying the captured environment (container build) as part of ``reproduce`` on a fresh machine.
