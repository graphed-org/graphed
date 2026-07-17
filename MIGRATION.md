# Migration: the `graphed-*-mvp` packages → the consolidated `graphed`

The prototype was split across one repository per package. It is now a single distribution,
`graphed`, with the former packages as subpackages. Import paths changed accordingly:

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
- **`graphed_corpus`** (the M0.5 fixtures/reference data) is **not** a shipped subpackage. It is
  vendored under `tests/_corpus/` for this repository's own test suite only; it is still published
  separately as `graphed-corpus-mvp` for downstream repositories that consume the fixtures.
- **Wire/format constants are unchanged**: backend ids (`graphed-awkward/0`, `graphed-numpy/0`),
  the plan format version (`graphed-plan/2`), and content-hash schemes are byte-stable across the
  rename — deserializing plans/bundles written by the prototypes still works.
- The frozen acceptance suites were carried over with their history; their imports were rewritten
  mechanically to the new paths (no assertion was weakened). See the consolidation commit.
