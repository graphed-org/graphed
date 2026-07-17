# graphed

A schedulable, serializable, debuggable HEP task-graph system. `graphed` reduces a task graph to a
concise set of stage-nodes **incrementally as you build it**, so a large un-reduced graph never
exists. Reduction runs in a Rust extension via **equality saturation over e-graphs**. The goal is to
minimize Python-interpreter touchpoints and maximize time inside array kernels (awkward by default).

> **Status: MVP.** This repository consolidates the `graphed-*-mvp` package prototypes into one
> pip-installable distribution. It is a packaging/presentation release — no new functionality over
> the prototypes it was assembled from. See [MIGRATION.md](MIGRATION.md).

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

The base install stays light: it pulls only the compiled core and `executing`. Backend and
preservation dependencies are opt-in extras — importing `graphed.awkward` (etc.) without its extra
raises a clear `ImportError`.

## Package layout

| Import path         | What it is                                                              | Extra        |
|---------------------|-------------------------------------------------------------------------|--------------|
| `graphed`           | deferred-array frontend: `Session`, `Array`, `Backend`, provenance      | (base)       |
| `graphed.core`      | Rust+PyO3 interned IR, optimizer, plan serialization, exec protocol     | (base)       |
| `graphed.awkward`   | awkward-typetracer reference backend + column projection                | `[awkward]`  |
| `graphed.numpy`     | trivial numpy backend                                                   | `[numpy]`    |
| `graphed.debug`     | source-mapped tracebacks, opt-level lowering, live dashboard            | `[dashboard]`|
| `graphed.checkpoint`| content-addressed store, manifest, resume                               | `[checkpoint]`|
| `graphed.preserve`  | self-contained preservation bundle                                      | `[preserve]` |

Sibling packages that stay **separate**: [`graphed-histogram`](https://github.com/graphed-org/graphed-histogram-mvp),
[`graphed-exec-local`](https://github.com/graphed-org/graphed-exec-local-mvp) (executor), and the
standalone [`graphed-orchestrator`](https://github.com/graphed-org/graphed-orchestrator).

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
