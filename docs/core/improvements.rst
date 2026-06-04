Improvements
============

Tracked design improvements and known limitations for ``graphed-core`` (plan M0 requires this file
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

Planned
-------

- Sharded or lock-free interning once contention is measured to matter.
- A serializable plan form (M8) and the optimizer ``RewriteEngine`` boundary (M4) build on this IR.
