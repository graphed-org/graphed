# M41 frozen suite — graphed-core (the `ClusterExecutor` launch seam CONTRACT)

Milestone **M41** (plan §6.1.3; decomposition `m41-decomposition.md` §2 T4 / §3(c.1)). `graphed-core`
owns the **contract** for the Phase-2 cluster runtime — three data-only siblings of `Executor` /
`WorkerTransport` in `graphed.core.execution`. The concrete cross-process Store + launcher live in
`graphed-executors` (exercised by `tests/frozen/m41/` there), **behind** these Protocols; the seam here
is pure. **Frozen — read-only after `freeze-M41-0`.**

## Files → theme / target (§8 M41)

| File | Theme / Target | What it witnesses | HEAD failure (right reason) |
|---|---|---|---|
| `test_cluster_seam_contract.py` | **(c.1) / T4** | `AddressTable`/`NodeStore`/`ClusterExecutor` import from `graphed.core.execution`, are in `graphed.core.__all__` (same object), have the right shape, and importing the seam pulls no backend/exec (§A.4 `sys.modules` guard) | `ImportError: cannot import name 'AddressTable' from 'graphed.core.execution'` |

This is the PRIMARY, **never-skip** non-vacuity anchor for theme (c): the `[exec]` cross-process forms
are env-gated (routable address + spawn), but the CONTRACT fails on HEAD everywhere via `ImportError`.

## Pinned seam (test-author decisions — the implementer adds these to `graphed.core.execution`)

```python
@dataclass                      # data-only, like Partition / StageSpec
class AddressTable:
    # node_id -> routable (host, port), populated BY a launcher; each entry tagged with provenance.
    def register(self, node_id, host: str, port: int, *, registered_by) -> None: ...
    def lookup(self, node_id) -> tuple[str, int]: ...
    def registered_by(self, node_id): ...     # the CHILD that registered it (not the driver)

@runtime_checkable
class NodeStore(Protocol):       # per-node Store handle; the concrete _HttpCluster/_IpcCluster satisfy it
    def fetch(self, digest: str) -> object: ...   # THE fetch() endpoint contract — idempotent,
                                                  # byte-backpressured/spillable (the BODY is exec/T1)
    # ... (has / put_blob / get_blob as apt)

@runtime_checkable
class ClusterExecutor(Protocol): # an Executor-shaped launch seam (address table + Store handles + fetch)
    def run(self, plan) -> ExecResult: ...
```

All three MUST be added to `graphed.core.__all__` (the test asserts the re-export is the SAME object).

## Contract assertions (per test function)

1. `test_seam_types_import_and_are_exported_from_core` — the three names import from
   `graphed.core.execution` and are re-exported (identically) from `graphed.core.__all__`.
2. `test_node_store_is_a_protocol_with_a_fetch_endpoint` — `NodeStore._is_protocol` and it declares `fetch`.
3. `test_cluster_executor_is_an_executor_shaped_protocol` — `ClusterExecutor._is_protocol` and carries `run`.
4. `test_address_table_is_a_data_only_dataclass_with_registration_provenance` — `is_dataclass` +
   `register`/`lookup`/`registered_by`.
5. `test_importing_the_core_seam_pulls_no_backend_or_executor` — §A.4: a fresh interpreter importing the
   seam has no `awkward` / `graphed_executors` / `graphed_exec_local` in `sys.modules`.

## Non-vacuity evidence (HEAD @7c817f7)

All 5 functions FAIL on HEAD: four `ImportError` (the seam is absent), and the §A.4 guard fails because
its subprocess `ImportError`s at the seam import (returncode 1). Verified: importing
`graphed.core.execution` on HEAD already pulls **no** `awkward`/`numpy`/`graphed_executors` — so the
purity guard PASSES once the (pure) seam is added, and fails ONLY if the impl leaks a backend/exec into
core.

## §A.3.1 note for the implementer

Runtime wiring (host/port/PID/announced addresses) is EPHEMERAL — it must NOT be folded into the durable
plan (`DurablePlanV2`) or the content-addressed task ids (ephemeral addresses would change task ids
run-to-run). Launcher addresses live in `AddressTable` (runtime state), out of the canonical IR.
