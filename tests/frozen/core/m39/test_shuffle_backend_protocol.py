"""M39 — the ``ShuffleBackend`` protocol lives in graphed-core, pure and §A.4-clean (plan §3.0).

The protocol NAMES the vectorized primitives the generic engine calls; it is a ``typing.Protocol`` in
``graphed.core.execution`` beside ``WorkerTransport``/``Executor``/``Plan`` — importing nothing heavy
(§A.4: graphed-core MUST NOT import awkward/numpy). Per ADV-r5.3 the ``@runtime_checkable`` check is
**import hygiene only** — it verifies method NAMES, never the routing rule (that is the golden-vector
theme). The M39 protocol carries the exchange/repartition half only; the join half is M40.
"""

from __future__ import annotations

import inspect
import pathlib

from graphed.core.execution import ShuffleBackend

# the six exchange/repartition primitives introduced in M39 (§3.0), plus the identity token.
_M39_METHODS = ("partition", "concat", "slice_rows", "estimated_bytes", "to_wire", "from_wire")


class _ConformingExchangeBackend:
    """Implements exactly the M39 exchange half — the positive conformance shape."""

    identity = "test-backend/0"

    def partition(self, block, key_field, parts, *, salt=0, boundaries=None):
        return (block, *(None for _ in range(parts - 1)))

    def concat(self, blocks):
        return list(blocks)

    def slice_rows(self, block, start, stop):
        return block

    def estimated_bytes(self, block_or_form):
        return 0

    def to_wire(self, block):
        return b""

    def from_wire(self, data):
        return data


class _MissingPartition:
    identity = "broken/0"

    def concat(self, blocks):
        return blocks

    def slice_rows(self, block, start, stop):
        return block

    def estimated_bytes(self, block_or_form):
        return 0

    def to_wire(self, block):
        return b""

    def from_wire(self, data):
        return data


def test_protocol_declares_every_m39_primitive_plus_identity() -> None:
    members = set(dir(ShuffleBackend))
    for name in _M39_METHODS:
        assert name in members, f"ShuffleBackend must declare {name!r} (§3.0 exchange half)"
    # ``identity`` is a data member of the protocol (folded into V2 task ids, §7.2)
    annotations = getattr(ShuffleBackend, "__annotations__", {})
    assert "identity" in annotations or "identity" in members


def test_conforming_backend_passes_runtime_isinstance() -> None:
    assert isinstance(_ConformingExchangeBackend(), ShuffleBackend)


def test_a_backend_missing_a_primitive_is_not_an_instance() -> None:
    # discrimination: the runtime_checkable hygiene must reject a backend missing `partition`.
    assert not isinstance(_MissingPartition(), ShuffleBackend)


def test_protocol_is_generic_over_opaque_block_and_index() -> None:
    # the engine deals only in opaque handles (Block, Index) — the protocol is parameterised.
    assert ShuffleBackend[object, object] is not None


def test_execution_module_imports_no_backend_library() -> None:
    # §A.4 guard: the protocol's home must not import numpy/awkward (kept a stable, minimal seam).
    src = pathlib.Path(inspect.getsourcefile(ShuffleBackend)).read_text(encoding="utf-8")
    assert "import numpy" not in src
    assert "import awkward" not in src
