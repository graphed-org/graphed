# graphed

The **deferred-array frontend** for `graphed` (milestone M2): a user writes ordinary array
expressions and `graphed` records them into the Rust-backed `graphed-core` store via a pluggable
**`Backend`**. The recorded graph is **backend-agnostic** — backends supply only form inference and
evaluation. Part of the [`graphed-org`](https://github.com/graphed-org) project; see
[`graphed-project`](https://github.com/graphed-org/graphed-project-mvp) for the root guidance and plan.

```python
from graphed import Session
import graphed_numpy as gn

s = Session(gn.NumpyBackend())
a = gn.from_array(s, "a", [1, 2, 3])
b = gn.from_array(s, "b", [10, 20, 30])
c = (a + b).filter((a + b).map(lambda x: x > 0, name="positive"))  # records nodes, no eval yet
total = (a + b).reduce("sum")
s.materialize(total)            # 66  (reference node-by-node eval; the real executor is M7)
```

## What it does (M2)

- `Backend` + `Form` protocols (`op_form`, `eval_stage`, `boundary_ops`, `project`,
  `external_payload`).
- A deferred `Array` proxy recording one interned node + one form per op; repeated sub-expressions
  intern to zero new nodes.
- Opaque callables (`map`) record `External` nodes whose backend payload descriptor is flagged a
  preservation risk.
- A provenance **stub** that points build-time type errors at the user's exact source line.

Guardrails: no fusion (M4); no awkward (the awkward backend is M3); provenance is a stub (M3).

## Develop

```bash
pip install "graphed-core @ git+https://github.com/graphed-org/graphed-core-mvp@main"   # needs Rust
pip install -e ".[dev,docs]"
ruff check . && ruff format --check . && mypy
pytest tests/frozen/m2
sphinx-build -W -b html docs docs/_build/html
```

Status: see `.graphed/state.json` and `CLAUDE.md`.
