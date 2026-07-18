Improvements
============

Tracked design improvements and known limitations for ``graphed.core`` (plan M0 requires this file
in every package).

Current limitations
-------------------

- **Single-mutex interning.** Correct and simple, but it serializes concurrent builders. Sharded
  interning (keyed on the structural hash) is the planned improvement; it is deferred because a
  node's inputs reference ids in the shared arena, so a sharded design needs cross-shard validation.
- **No optimization.** DCE/CSE/canonicalization/stage fusion are M4. CSE already falls out of
  hash-consing here; the ``Stage`` variant exists but is constructed only by M4.
- **NodeId is an arena index.** Generational ids / removal are not needed yet (the graph is
  append-only at M1).

M8 — plan serialization (delivered)
-----------------------------------

- ``GraphStore.serialize`` / ``GraphStore.deserialize`` give the **canonical, versioned,
  byte-identical** durable IR form (magic ``GIR1``); a round trip reproduces identical bytes.
- ``DurablePlan`` wraps that IR with the executor metadata (partitions, read columns,
  reduction/stopping/locality/resource specs) and is content-addressed (``task_id`` is a SHA-256
  over the IR identity + process spec + partition, so it is cache-poisoning-safe). cloudpickle is
  used **only** for genuinely opaque callables, which are flagged ``opaque=True``.

Planned
-------

- Sharded or lock-free interning once contention is measured to matter.
- A backwards-compatible v2 of the ``GIR`` format (the magic byte is the versioning hook) if the
  node schema ever grows; today only ``GIR1`` is accepted.
