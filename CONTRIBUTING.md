# Contributing to `graphed`

## Layout

```
Cargo.toml, Cargo.lock, src/*.rs     the Rust core crate (graphed.core.graphed_core)
python/graphed/                      the Python package
  __init__.py, *.py                    frontend (Session, Array, Backend, provenance)
  core/  awkward/  numpy/  debug/  checkpoint/  preserve/   the subpackages
tests/frozen/<pkg>/mX/               the frozen acceptance suites, per package + milestone
tests/extra/<pkg>/                   implementer-added (non-frozen) tests
tests/_corpus/                       vendored M0.5 fixtures (graphed_corpus) + reference data
docs/                                one Sphinx project (per-package design pages under docs/<pkg>/)
```

The build backend is **maturin** (mixed Rust/Python): the compiled extension is nested at
`graphed.core.graphed_core` and re-exported by `graphed.core`. `python-source = "python"` ships the
whole `graphed/` tree.

## Setup

```bash
pip install -e ".[dev]"     # builds the Rust extension and installs the full toolchain
```

## Running tests

Run **one subtree at a time** — a single `pytest tests/` will not collect (duplicate test basenames
across packages) and would break frozen guards that assert process-global state:

```bash
./scripts/run-tests.sh          # every package's suite, isolated
COV=1 ./scripts/run-tests.sh    # + combined branch coverage, gate >=90
pytest tests/frozen/awkward     # a single package
```

## Gates (run before pushing)

```bash
uvx prek run --all-files                 # ruff check + ruff format --check + mypy + cargo fmt + clippy
./scripts/run-tests.sh                   # Python suites
cargo test                               # Rust unit tests (set DYLD_FALLBACK_LIBRARY_PATH / LD_LIBRARY_PATH to python's libdir)
RUSTFLAGS="--cfg loom" cargo test --lib loom_model
sphinx-build -W -b html docs docs/_build/html
python -m graphed_orchestrator.precommit .   # the project integrity/gate wrapper (see note)
```

- `mypy --strict` runs on `python/` (`files = ["python"]`); ruff/format cover `python` and `tests`.
- The Rust `cargo test` binary links libpython — export `DYLD_FALLBACK_LIBRARY_PATH` (macOS) or
  `LD_LIBRARY_PATH` (Linux) to `python -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))'`.
- **Integrity / freeze rule:** `tests/frozen/**` is read-only. Do not edit, skip, xfail, weaken, or
  stub what a frozen test verifies. If a frozen test seems wrong, file a dispute — do not route
  around it.
- **Coverage gate note:** because the suite runs per-subtree, coverage is accumulated with
  `--cov-append` across subtrees and gated once (`coverage report --fail-under=90`). The
  `graphed_orchestrator.precommit` coverage sub-check expects a single whole-repo `pytest --cov`
  line; here `COV=1 ./scripts/run-tests.sh` is the enforcing command (CI runs it directly).
