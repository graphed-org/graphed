# CLAUDE.md — graphed

Defers to the root **`graphed-project/CLAUDE.md`**; the **project plan
(`graphed-project-plan-gated.md`) always wins.** This file distills **milestone M2**.

## What this repo is

`graphed`: the **deferred-array frontend**. A user writes ordinary array expressions; an `Array`
proxy records them into the Rust-backed `graphed-core` store via a pluggable `Backend`. The recorded
graph is **backend-agnostic** — backends supply only form inference (`op_form`) and evaluation
(`eval_stage`), plus `boundary_ops`, `project` (M5 stub), and `external_payload`.

> Guardrails: IR stays backend-agnostic (no numpy/awkward leakage into core types) · **no fusion**
> (M4). M2 added the `Backend`/`Form` protocols + the basic `Array`; M3 added the awkward op surface
> (field access, indexing, comparisons, numpy-ufunc hooks, modulo) and **real provenance**.

## M3 additions

- `Array` awkward surface (`array.py`): `__getattr__` (field), `__getitem__` (mask/field),
  comparisons (`> < >= <= == !=`, deferred → Array, so Array is unhashable), boolean ops
  (`& | ~`), scalar-aware arithmetic + reflected ops, `__array_ufunc__` so `np.cos(arr)` etc.
  record canonical ops without graphed importing numpy.
- **Real `provenance.capture()`** (`provenance.py`): first non-`graphed*` frame's filename, line,
  function, and **sub-expression source text** (via `executing`); stateless (thread-safe);
  toggleable (`set_enabled`). Survives helper functions and comprehensions.
- The `gak` awkward-function namespace + the typetracer backend live in **graphed-awkward**.

## M2 — implemented

- `Backend` + `Form` protocols (`backend.py`); `ParamValue = int|float|bool|str`.
- `Array` proxy (`array.py`): `+ - * /`, `filter`, `map`, `reduce`. One interned node + one form
  per op; repeated sub-expressions intern to **zero** new nodes.
- `Session` (`session.py`): owns the `graphed_core.GraphStore` + side tables (form, sources, ops,
  externals, provenance); `materialize()` is a reference node-by-node evaluator (the real executor
  is M7).
- `map(fn)` records an `External` node; its descriptor (from `backend.external_payload`) flags the
  opaque callable as a preservation risk.
- `provenance.capture()` **stub**: first non-`graphed*` frame's (filename, lineno) — enough to raise
  a `GraphedTypeError` at the user's exact source line when a backend reports an ill-typed op.

## Layout

```
src/graphed/backend.py     Backend + Form protocols, ParamValue
src/graphed/array.py       the deferred Array proxy
src/graphed/session.py     build session + reference materialize()
src/graphed/provenance.py  capture() stub (M3 does the rich version)
src/graphed/errors.py      GraphedTypeError (carries provenance)
tests/frozen/m2/           frozen suite + two toy backends (backend-independence)
```

## Gates

`ruff` + `ruff format` · `mypy --strict` · `pytest tests/frozen/m2 --cov=graphed` (≥90%) ·
`sphinx -W`. Needs `graphed-core` installed (from git in CI; `maturin develop` locally). Depends on
`graphed-core` only — `graphed-numpy` is a separate downstream repo.

Status: see `.graphed/state.json`.

## M5 additions (column-projection support)

- `Session.walk(array, *, source, op, external)` — the generic graph walk `materialize` and
  projection share. Source helpers: `source_ids`, `source_name`, `form_of`.
- `projection.py`: `Projection` (source → read columns), `ProjectionError`, `OnFail`
  (`pass|warn|raise`), `handle_opaque` (the shared on-fail policy). The real projection lives in the
  backends (graphed-awkward reporting typetracer; graphed-numpy field-touch tracking).
