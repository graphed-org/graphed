# Test Dispute — `tests/frozen/frontend/m60/test_callable_identity.py::test_an_explicit_name_is_the_callers_identity_declaration`

Contradicts: graphed#143, "two callables with different code under one `name=` raise".

The test pinned two different lambdas under one name as one node, which silently returns the first one's result.

Correction: the same code under one name is one node, and different code raises `GraphedError`.

Ruled by the owner, 2026-11-01: the change stands. Needs a new freeze tag at the merge commit.
