# Test Dispute — `tests/frozen/awkward/m71a/test_m71a_awkward_identity.py::test_declared_and_undeclared_are_two_nodes_with_two_forms`

Contradicts: graphed#143, "an undeclared map records the type its function gives a typetracer".

The test pinned an undeclared map to its input's form.

Correction: the undeclared map records the traced `## * var * bool`; the declared one claims `var * int8`, so the two forms still differ.

Ruled by the owner, 2026-11-01: the change stands. Needs a new freeze tag at the merge commit.
