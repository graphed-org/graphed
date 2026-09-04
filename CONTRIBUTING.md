# Contributing to `graphed`

`graphed` is a mixed Rust/Python package: a compiled Rust core (the graph store and
optimizer) underneath a pure-Python frontend. Building it from source therefore needs a
**Rust toolchain** (rustc ≥ 1.74, via [rustup](https://rustup.rs)) in addition to
Python ≥ 3.11.

## Layout

```
Cargo.toml, Cargo.lock, src/*.rs     the Rust core crate (compiled to graphed.core.graphed_core)
python/graphed/                      the Python package
  __init__.py, *.py                    frontend (Session, Array, Backend, provenance)
  core/  awkward/  numpy/  debug/  checkpoint/  preserve/   the subpackages
tests/frozen/<pkg>/, tests/extra/<pkg>/   the test suites, one tree per subpackage
tests/_corpus/                       reference fixtures + data the suites compare against
docs/                                one Sphinx project (per-package pages under docs/<pkg>/)
```

`tests/frozen/<pkg>/` holds each subpackage's compatibility suite — the name is historical;
what it means is "don't change these to make your branch pass". New tests go in
`tests/extra/<pkg>/`.

The build backend is **maturin**: the compiled extension is nested at
`graphed.core.graphed_core` and re-exported by `graphed.core`.

Inside the crate, if you are changing the optimizer or the on-disk format:

| File | What lives there |
|---|---|
| `src/param.rs` | operation parameters: total-order float hashing, token escaping |
| `src/node.rs` | a node's structural identity, its tokens, and payload descriptors |
| `src/store.rs` | the intern table behind its mutex, the reduce entry points, dot export |
| `src/optimizer/engine.rs` | the rewrite-engine boundary and its shared rule constants |
| `src/optimizer/mod.rs` | dead-code and duplicate elimination, stage fusion (both modes), the pipeline |
| `src/optimizer/incremental.rs` | the delta canonicalizer and its work counters |
| `src/serialize.rs` | the durable codec a saved graph is written with |
| `src/lib.rs` | the PyO3 bindings — thin; everything above is plain Rust |

On the Python side of the same layer, `python/graphed/core/plan.py` holds the saved-plan format
and the content-addressed task id, and `python/graphed/core/execution.py` holds the
`Plan`/`Task`/`ExecResult` contract every runner implements.

## Setup

```bash
pip install -e ".[dev]"     # compiles the Rust extension and installs the full toolchain
```

Check the build worked:

```python
import graphed
from graphed.core import Partition, Plan, Task

print(graphed.__version__)
print(Plan.__name__, Task.__name__, Partition.__name__)
# 0.0.1
# Plan Task Partition
```

If `graphed.core.graphed_core` fails to import, the Rust extension didn't build — rerun
the editable install and read its output.

## Running tests

Run **one subtree at a time**. A single `pytest tests/` will not collect: test basenames
repeat across packages (`test_topologies.py` appears five times), and some suites assert
process-global state (core's tests require that `awkward` was never imported), so each
needs its own process:

```bash
./scripts/run-tests.sh            # every package's suite, one pytest process per subtree
COV=1 ./scripts/run-tests.sh      # + combined branch coverage, gate >=90
pytest tests/frozen/checkpoint    # a single package
```

Three packages — `frontend`, `numpy`, `awkward` — repeat basenames *between their own
subdirectories* too (two `test_projection.py`, two `shuffle_backends.py`), so even
`pytest tests/frozen/awkward` dies with an "import file mismatch" collection error. Run
those one subdirectory per pytest process; `./scripts/run-tests.sh` already does.

Because the suite runs per-subtree, coverage accumulates with `--cov-append` across
subtrees and is gated once at the end (`coverage report --fail-under=90`);
`COV=1 ./scripts/run-tests.sh` is the enforcing command, and CI runs it directly.

## Lint, types, Rust tests, docs

Run these before pushing — CI runs the same commands:

```bash
uvx prek run --all-files            # ruff check + ruff format --check + mypy --strict + cargo fmt + clippy
cargo test                          # Rust unit tests (see the library-path note below)
RUSTFLAGS="--cfg loom" cargo test --lib loom_model    # concurrency model checks
pip install -e ".[docs]" && sphinx-build -W -b html docs docs/_build/html
```

- `mypy --strict` covers `python/`; ruff and its formatter cover `python` and `tests`.
- The Rust `cargo test` binary links libpython — export `DYLD_FALLBACK_LIBRARY_PATH` (macOS) or
  `LD_LIBRARY_PATH` (Linux) to `python -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))'`
  or the test binary fails to start.
- Docs must build with `-W`: any Sphinx warning is an error.

## Proposing a change

1. Branch, make the change, and add tests under `tests/extra/<pkg>/` next to the code you
   touched. Existing tests under `tests/frozen/**` are the package's compatibility
   surface — if one fails, fix your change, don't edit the test; if you believe the test
   itself is wrong, open an issue explaining what it asserts and why that's incorrect.
2. Run the test script, the lint hooks, `cargo test`, and the docs build (above).
3. Open a pull request against this repository. Describe what changed and why; paste the
   output of the commands you ran if any of them is affected by your change.
