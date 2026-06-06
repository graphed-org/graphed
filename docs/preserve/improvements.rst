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
