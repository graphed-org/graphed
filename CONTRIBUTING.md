# Contributing to graphed

Part of the `graphed` project, governed by the gated three-role pipeline. The root
[`graphed-project/CLAUDE.md`](https://github.com/graphed-org/graphed-project-mvp) and the project plan
are authoritative; the plan always wins.

## Guardrails (M2)

- The IR is **backend-agnostic**: no numpy/awkward leakage into graphed core types. Backends supply
  only form inference + evaluation.
- **No fusion** (M4). **No awkward** (the awkward backend is M3). **Provenance is a stub** (M3) —
  enough to point a build-time type error at the user's source line.

## Integrity rules — NON-NEGOTIABLE (plan A.7 / B.6)

Never edit/skip/weaken `tests/frozen/**`; never lower a threshold or relax CI; never stub the thing
under test; never flood `# type: ignore` / `except: pass`. Dispute a frozen test via
`.graphed/<Mx>/disputes/<test_id>.md`, do not route around it.

## Local gates

```bash
pip install "graphed-core @ git+https://github.com/graphed-org/graphed-core-mvp@main"   # needs Rust
pip install -e ".[dev,docs]"
ruff check . && ruff format --check . && mypy
pytest tests/frozen/m2 --cov=graphed --cov-branch    # >=90%
sphinx-build -W -b html docs docs/_build/html
```
