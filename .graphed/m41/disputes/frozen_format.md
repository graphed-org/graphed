# Dispute / config note — frozen suites exempt from `ruff format` (extends the M39 I001 precedent)

**Filed by:** team-lead, post-M41-push (CI `ci` red on the `ruff format --check` prek step).
**Severity:** cosmetic (auto-formatting), NOT a behavioural defect. **No gate weakened for maintained code.**

## The conflict

CI runs `ruff format --check --force-exclude` as a prek hook. Several **immutable frozen** test files
(`tests/frozen/awkward/m40/*`, `tests/frozen/frontend/m40/*`, `tests/frozen/core/m41/*`) were authored
not-`ruff format`-clean. They are read-only under the §A.7 frozen-test integrity rule, so they cannot be
reformatted in place — yet the formatter demands it, so the gate stays red. Same class of conflict the M39
dispute resolved for the linter's `I001` rule, now for the **formatter**.

## Resolution

`[tool.ruff.format] exclude = ["tests/frozen/**"]` in `pyproject.toml`. Rationale:
- `ruff format` is a **pure auto-formatting** pass (no correctness check), so exempting the immutable frozen
  tree loses no defect-detection.
- **`ruff check` (real lint) still covers the frozen files** — the exclude is `format`-only, not `lint`.
- **Every `python/**` src file and every non-frozen test stays fully format-enforced.** The non-frozen format
  debt this masked was fixed in the same commit: `python/graphed/awkward/join.py` (`ruff format` — a long
  call wrapped; no logic change).

## Follow-up (process)

Test-authors should run `ruff format` before freezing. Remove this exclude once the frozen files are re-tagged
format-clean.

> NOTE: this commit does NOT address the separate, pre-existing rust CI breakage on this repo (a
> non-exhaustive `NodeKey::Join` match in the lib TEST build, introduced by the M40 rust join work — breaks
> `cargo clippy`/`cargo test`/`cargo llvm-cov`). That is M40-scope and tracked separately.
