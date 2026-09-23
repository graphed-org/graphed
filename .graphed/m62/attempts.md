# m62 — implementer iterations

Branch `lane/checkpoint-remote` on top of `4d27592` (= the frozen suite, tags `freeze-m62{,a,b,c}`).
Run: `python -m pytest tests/frozen/checkpoint tests/extra/checkpoint -q -p no:randomly`.

## Iteration 0 — baseline

All seven m62 modules fail collection: `CheckpointStore` and `FsspecStore` are not exported.

## Iteration 1 — unit A (A1): `CheckpointStore`, verified reads, concurrent puts

`CheckpointStore` is a `runtime_checkable` Protocol over the runners' six calls; both runners take
it. `Store.get` returns `None` for bytes that do not hash to the name. `put` writes unless a
verified copy is present, so it heals a tampered blob. The temp is `.{digest}.{uuid4}.tmp` opened
`"xb"`, which keeps the default mode. If `os.replace` raises `OSError`, the put still succeeds when
a verified copy is then present, and re-raises otherwise. The temp is unlinked on every path.
m62/local 11 passed; m8/m39/m49/extra and preserve unchanged.
