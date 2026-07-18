# graphed

A schedulable, serializable, debuggable HEP task-graph system. `graphed` reduces a task graph to a
concise set of stage-nodes **incrementally as you build it**, so a large un-reduced graph never
exists. Reduction runs in a Rust extension via **equality saturation over e-graphs**. The goal is to
minimize Python-interpreter touchpoints and maximize time inside array kernels (awkward by default).

One installable distribution now spans the whole pipeline:

- a deferred-array **frontend** with two backends — `graphed.awkward` (ragged, the HEP default, with
  `ak.*`-parity `gak` operations) and `graphed.numpy` (rectilinear), each a broad,
  dask/dask-awkward-level API;
- a Rust+PyO3 **optimizer** — hash-consing, DCE/CSE, equality-saturation stage fusion — that reduces
  incrementally, deterministically, and is CI-guarded against super-linear scaling;
- source-mapped **debugging** (a remote-worker error re-raised at the user's line) with a live dashboard;
- content-addressed **checkpoint/resume** and a self-contained **preservation bundle** that
  reproduces histograms bit-for-bit on a clean machine, with ML-framework `External` plugins.

> **Status: MVP.** This repository consolidates the `graphed-*-mvp` package prototypes into one
> pip-installable distribution. It is a packaging/presentation release — no new functionality over
> the prototypes it was assembled from. See [MIGRATION.md](MIGRATION.md) and the
> [architecture overview](docs/architecture.rst).

## Install

```bash
pip install graphed                 # frontend + Rust core only (light)
pip install "graphed[awkward]"      # + awkward backend (the default HEP backend)
pip install "graphed[numpy]"        # + numpy backend
pip install "graphed[preserve]"     # + preservation bundle (correctionlib/onnx)
pip install "graphed[dashboard]"    # + live Perspective dashboard
pip install "graphed[all]"          # everything except the heavy ML frameworks
pip install "graphed[dev]"          # full test/lint/type toolchain
```

The base install stays light: the compiled core plus two small pure-Python deps (`executing` for
provenance, `cloudpickle` for opaque-callable plans). Extras are opt-in. The `awkward`/`numpy`
backends eagerly need their array library, so `import graphed.awkward`/`graphed.numpy` without
`[awkward]`/`[numpy]` raises a clear `ImportError`; `graphed.debug`, `graphed.checkpoint`, and
`graphed.preserve` import at base and pull their heavy deps lazily (only the live dashboard or the
correctionlib/ONNX preservation payloads need one). Parquet I/O needs the separate `[parquet]`
(pyarrow) extra.

## Package layout

| Import path         | What it is                                                                         | Extra        |
|---------------------|------------------------------------------------------------------------------------|--------------|
| `graphed`           | deferred-array frontend: `Session`, `Array`, `Backend` protocol, provenance        | (base)       |
| `graphed.core`      | Rust+PyO3 interned IR, equality-saturation optimizer, plan serialization, protocols | (base)       |
| `graphed.awkward`   | ragged backend: `gak` (`ak.*` parity), typetracer forms, vector, buffer projection          | `[awkward]`  |
| `graphed.numpy`     | rectilinear backend: deferred numpy idiom, monoidal reductions                     | `[numpy]`    |
| `graphed.debug`     | source-mapped picklable tracebacks, opt-level lowering, live dashboard             | `[dashboard]`|
| `graphed.checkpoint`| content-addressed store, deterministic resume, retry + dead-letter                 | `[checkpoint]`|
| `graphed.preserve`  | self-contained preservation bundle + ML `External` plugins (torch/tf/xgboost/jax/onnx/triton) | `[preserve]` |

Sibling packages that stay **separate** (they depend on `graphed`): [`graphed-exec-local`](https://github.com/graphed-org/graphed-exec-local-mvp)
(the reference executor), [`graphed-histogram`](https://github.com/graphed-org/graphed-histogram-mvp)
(deferred `hist`/boost-histogram fills), and [`graphed-orchestrator`](https://github.com/graphed-org/graphed-orchestrator)
(the gated development pipeline). See the [architecture overview](docs/architecture.rst) for how they fit together.

## Documentation

Full docs (Sphinx) live under [`docs/`](docs/): start with [`docs/architecture.rst`](docs/architecture.rst)
for the repository structure and pipeline, then each subpackage's `docs/<pkg>/design.rst` for its
engineering walkthrough; `docs/api.rst` is the generated API reference for the whole distribution.

```bash
pip install -e ".[docs]" && sphinx-build -W -b html docs docs/_build/html
```

## Develop

```bash
pip install -e ".[dev]"     # builds the Rust extension via maturin
./scripts/run-tests.sh      # runs each package's frozen suite (per-subtree isolation)
```

`graphed.core` is a Rust crate (`src/*.rs`, `Cargo.toml` at the repo root); everything else is pure
Python under `python/graphed/`. The frozen acceptance suites live under `tests/frozen/<pkg>/` and are
run one subtree at a time (see [CONTRIBUTING.md](CONTRIBUTING.md) for why).

## Provenance

The consolidated history was stitched from the eight prototype repositories (`graphed`,
`graphed-core`, `graphed-awkward`, `graphed-numpy`, `graphed-debug`, `graphed-checkpoint`,
`graphed-preserve`, `graphed-corpus`); `git log --follow` reaches back into each.
