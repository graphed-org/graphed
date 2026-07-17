# CLAUDE.md — graphed (consolidated package)

Defers to the root **`graphed-project/CLAUDE.md`**; the **project plan
(`graphed-project-plan-gated.md`) always wins.** This repository consolidates the eight
`graphed-*-mvp` prototype packages into one pip-installable distribution, `graphed`.

## What this repo is

One package, `graphed`, with the former repos as subpackages:

| Import            | Was                  | Guardrail (§A.4)                                        |
|-------------------|----------------------|--------------------------------------------------------|
| `graphed`         | `graphed` frontend   | backend-agnostic; no numpy/awkward in core types       |
| `graphed.core`    | `graphed-core`       | Rust+PyO3 IR/optimizer/plan; **MUST NOT import awkward**|
| `graphed.awkward` | `graphed-awkward`    | awkward typetracer backend                             |
| `graphed.numpy`   | `graphed-numpy`      | trivial numpy backend                                  |
| `graphed.debug`   | `graphed-debug`      | source-mapped tracebacks, lowering, dashboard          |
| `graphed.checkpoint` | `graphed-checkpoint` | content-addressed store / resume                    |
| `graphed.preserve`| `graphed-preserve`   | preservation bundle; invents no formats                |

This is a **packaging/presentation** consolidation — **no new functionality** over the prototypes.

## Hard rules (unchanged from the plan)

- `graphed.core` **MUST NOT import awkward**; the base `import graphed` / `import graphed.core` stays
  free of numpy/awkward (verified: a base install pulls only the compiled core + `executing`).
- Reproducibility: the serializable IR — not cloudpickle — is the canonical durable representation.
  cloudpickle is only for genuinely opaque user callables (flagged `opaque=True`).
- Standards, not invented formats: correctionlib JSON, ONNX, UHI, HS3, content-addressed payloads.
- `tests/frozen/**` is read-only (freeze/integrity rules §A.7/§B.6). The one-time consolidation
  rewrote frozen imports mechanically (`graphed_core` → `graphed.core`, …) without weakening any
  assertion; that baseline is now the freeze. Do not route around a frozen test — file a dispute.

## Build & test

maturin mixed layout: Rust at `src/*.rs` + `Cargo.toml` (root), Python at `python/graphed/`,
`module-name = "graphed.core.graphed_core"`. Tests run **per-subtree** (`./scripts/run-tests.sh`) —
see `CONTRIBUTING.md` for why a single `pytest tests/` cannot collect this tree.

## Separate repos (not consolidated)

`graphed-histogram`, `graphed-exec-local` (executor), and `graphed-orchestrator` stay their own
packages; the first two depend on this consolidated `graphed`.
