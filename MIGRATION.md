# Migration: the separate `graphed-*` packages → the consolidated `graphed`

`graphed` used to be one distribution per package. It is now a single distribution, `graphed`,
with the former packages as subpackages. Import paths changed accordingly:

| Old package (dist / import)     | New import path        | Install extra |
|---------------------------------|------------------------|---------------|
| `graphed-core` / `graphed_core` | `graphed.core`         | (base)        |
| `graphed` (frontend)            | `graphed`              | (base)        |
| `graphed-awkward` / `graphed_awkward` | `graphed.awkward` | `[awkward]`   |
| `graphed-numpy` / `graphed_numpy`     | `graphed.numpy`   | `[numpy]`     |
| `graphed-debug` / `graphed_debug`     | `graphed.debug`   | `[dashboard]` |
| `graphed-checkpoint` / `graphed_checkpoint` | `graphed.checkpoint` | `[checkpoint]` |
| `graphed-preserve` / `graphed_preserve`     | `graphed.preserve`   | `[preserve]` |

```python
# before
from graphed_core import GraphStore, DurablePlan
from graphed_awkward import AwkwardBackend, from_awkward

# after
from graphed.core import GraphStore, DurablePlan
from graphed.awkward import AwkwardBackend, from_awkward
```

Notes:

- **The compiled extension** keeps its leaf name: it lives at `graphed.core.graphed_core` and is
  re-exported by `graphed.core`. You should import from `graphed.core`, not the extension directly.
- **`graphed_corpus`** — the reference analyses `graphed`'s own numbers are checked against — is
  **not** a shipped subpackage and is not published anywhere. It is vendored under `tests/_corpus/`
  for this repository's test suite, which is where a source checkout finds it.
- **Wire and format constants are unchanged**: backend ids (`graphed-awkward/0`,
  `graphed-numpy/0`), the plan format version (`graphed-plan/2`), and the content-hash schemes are
  byte-stable across the rename, so plans and bundles written before it still deserialize.
- The test suites came over with their history; their imports were rewritten to the new paths and
  no assertion changed.
