"""M41 theme (c.1) / Target T4 — the ``ClusterExecutor`` launch seam CONTRACT in ``graphed.core``
(plan §6.1.3; decomposition §2 T4, §3(c.1)).

The Phase-2 cluster runtime is deferred, but its SEAM ships now — three data-only siblings of
``Executor``/``WorkerTransport`` in ``graphed.core.execution`` (plan §6.1.3):

- ``AddressTable`` — a data-only ``node_id -> routable (host, port)`` table a launcher populates, each
  entry tagged with ``registered_by`` provenance (a concrete ``@dataclass``, like ``Partition``/``StageSpec``);
- ``NodeStore`` — the per-node Store handle Protocol carrying the ``fetch()`` endpoint contract
  (``fetch(digest) -> handle``, idempotent + byte-backpressured — the *body* is exec/M41-T1, the
  *signature* is this seam);
- ``ClusterExecutor`` — an ``Executor``-shaped launch seam (address table + per-node Store handles +
  ``fetch`` endpoint).

This is the PRIMARY non-vacuity anchor for theme (c): on HEAD the seam is absent, so every import here
``ImportError``s (right reason — the §6.1.3 contract does not exist). §A.4: the module stays pure — it
must not pull ``awkward`` or ``graphed_executors`` (that concrete impl lives behind these Protocols).

Discrimination (rejects the traps in decomposition §3(c.1)):
- the seam living in exec instead of ``graphed.core`` -> the import + the ``sys.modules`` purity guard fail;
- a concrete class where a Protocol is required -> ``NodeStore``/``ClusterExecutor`` ``_is_protocol`` fails;
- a core symbol that imports a backend/exec -> the §A.4 subprocess guard fails.
"""

from __future__ import annotations

import dataclasses
import subprocess
import sys


def test_seam_types_import_and_are_exported_from_core() -> None:
    import graphed.core as gc  # noqa: PLC0415
    from graphed.core import execution  # noqa: PLC0415
    from graphed.core.execution import (  # noqa: PLC0415  HEAD: ImportError
        AddressTable,
        ClusterExecutor,
        NodeStore,
    )

    assert ClusterExecutor is not None and NodeStore is not None and AddressTable is not None
    for name in ("ClusterExecutor", "NodeStore", "AddressTable"):
        assert name in gc.__all__, f"{name} must be re-exported from graphed.core.__all__"
        assert getattr(gc, name) is getattr(execution, name)  # the re-export is the SAME object


def test_node_store_is_a_protocol_with_a_fetch_endpoint() -> None:
    from graphed.core.execution import NodeStore  # noqa: PLC0415

    assert getattr(NodeStore, "_is_protocol", False), "NodeStore must be a typing.Protocol (not a concrete class)"
    assert hasattr(NodeStore, "fetch"), "NodeStore must declare the fetch() endpoint contract (fetch(digest))"


def test_cluster_executor_is_an_executor_shaped_protocol() -> None:
    from graphed.core.execution import ClusterExecutor  # noqa: PLC0415

    assert getattr(ClusterExecutor, "_is_protocol", False), "ClusterExecutor must be a typing.Protocol"
    assert hasattr(ClusterExecutor, "run"), "ClusterExecutor must be Executor-shaped (carry a run() launch seam)"


def test_address_table_is_a_data_only_dataclass_with_registration_provenance() -> None:
    from graphed.core.execution import AddressTable  # noqa: PLC0415

    assert dataclasses.is_dataclass(AddressTable), "AddressTable must be a data-only @dataclass"
    for method in ("register", "lookup", "registered_by"):
        assert hasattr(AddressTable, method), f"AddressTable must expose {method}() (populate/lookup/provenance)"


def test_importing_the_core_seam_pulls_no_backend_or_executor() -> None:
    # §A.4 purity guard in a FRESH interpreter (clean sys.modules): importing the seam must not pull a
    # backend (awkward) or the concrete executor package. On HEAD the child ImportErrors at the seam
    # import -> non-zero exit -> this test fails for the right reason.
    code = (
        "import sys\n"
        "from graphed.core.execution import ClusterExecutor, NodeStore, AddressTable\n"
        "banned = [m for m in ('awkward', 'graphed_executors', 'graphed_exec_local') if m in sys.modules]\n"
        "assert not banned, banned\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, f"§A.4 core-purity guard failed:\n{result.stderr}"
