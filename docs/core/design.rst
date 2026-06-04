Design
======

Interning
---------

A node *is* its structural key (kind + name + sorted params + input ids +, for ``External``, the
full :class:`~graphed_core.PayloadDescriptor`). The store keeps an intern table keyed on that
structure; building a structurally identical node returns the existing ``NodeId``. CSE therefore
falls out of construction — there is no separate pass.

Total-order float hashing
-------------------------

Float params use a canonical bit key: every ``NaN`` collapses to one key (so ``NaN`` interns to
itself), while ``0.0`` and ``-0.0`` stay distinct. Treating ``0.0`` and ``-0.0`` as equal is a
*canonicalization*, which is M4's job, not M1's.

Locking discipline
------------------

Interning is a read-modify-write on shared state (look up the key; if absent, push a node and
record its id). The entire inner state sits behind a single ``Mutex`` so that step is atomic and
race-free under the GIL and under free-threaded 3.14t. The critical section is short; finer
sharding is a tracked improvement. The discipline is model-checked with ``loom`` (see the
``loom_model`` test) and stress-tested from many threads in both Rust and Python.

Boundary
--------

``Node`` is ``Source | Op | Reduction | External | Stage``. ``Stage`` exists in the enum but is
constructed only by the M4 optimizer. ``External`` carries a ``PayloadDescriptor`` so external
inputs (models, corrections) are reproducibly identified (plan A.3.1).
