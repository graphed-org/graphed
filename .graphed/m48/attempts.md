# m48 (graphed arc) — implementer attempts

Frozen suite: `tests/frozen/frontend/m48/**` (28) + `tests/frozen/awkward/m48/**` (84) = 112 tests,
tag `m48-freeze` @ 37d86cf. Nothing under `tests/frozen/**` was edited, skipped, xfailed or
weakened at any point.

Environment: `.venv`; every pytest/mypy/ruff invocation is prefixed `PATH=$PWD/.venv/bin:$PATH`
(`scripts/run-tests.sh` invokes bare `python`, which otherwise dies on `--cov=graphed`).

## Iteration 1 — G-I1: vocabulary + container

Added `python/graphed/_tags.py` (§1.1 canonicaliser: exact-decimal e-form, the 32-char cap with
the two refusals split by cause), `varied.py` (`Varied`, the §2.4 label-aligned union, the
per-idiom container registry), `vary.py` (the three §2.1 overloads) and `numpy/varied.py`.

Gates: frontend m48 19/28. Rewrote `varied.py`'s method generation around a single
`expand(fn, args, kwargs)` primitive after the first shape (rebuilding args from a flattened
tuple) proved unreadable; added the pass-through when no container is among the operands, without
which an unvaried expression recorded as a one-member container.

## Iteration 2 — G-I2: event context + lineage seams

Added `context.py` (`EventContext`: lineage, the ambient weight registry, memoised pure
derivations and projections), `awkward/gnano.py` and `accessors.py` (§9.1). Rerouted all five
`self._array_cls(self, node_id)` sites in `session.py` through a new `_wrap` so the §2.3e handle
merge and the `_labels` union happen at one chokepoint (call sites before: 5, left: 0).

`compile_ir`'s unreached-label report cannot key on node reachability — F8's two containers intern
to identical node ids — which is why `_labels` is object provenance on the frontend wrapper.

## Iteration 3 — G-I3a: the gak classification table

Appended §2.3c's five-class table to `awkward/functions.py`: 65 classified / 65 discovered / 0
unclassified / 0 extra, plus `GakSlot`/`GakArgFixture` and 58 argument fixtures for the carrying
classes. `functools.wraps` on `_dispatched` keeps each verb's signature, which §2.3d's annotation
filter reads.

## Iteration 4 — G-I3b: making every carrying fixture callable

Emulating the frozen A5 gate over all 58 fixtures left two red:

- `onnx_inference` → `DecodeError: Wire format was corrupt`. `payloads.onnx_weights_descriptor`
  suppresses parse failures but `onnx_weights_hash` (called outside that suppression) does not.
  Fixed in the fixture: `b""` is a valid empty `ModelProto`, so the weights-identity path parses.
- `unflatten` → `ill-typed op 'ak.unflatten': counts must be integers`. The frozen operand
  vocabulary (`OPERAND_KINDS`, five kinds, all float/bool/record) carries no integer array, and
  five alternative slot pairings all failed. **Root cause, not the fixture**: `gak.unflatten`
  accepted only an `Array` counts, while `ak.unflatten`'s own error names the contract — "counts
  must be an integer or a one-dimensional array of integers". `gak.unflatten` now takes
  `Array | int`, an integer riding in params rather than as a graph edge, with the matching read
  in `awkward/_ops.py`. The fixture is then the ordinary `(jagged, 2, axis=1)` spelling.

Gates: awkward m48 73/84, frontend m48 19/28.

## Iteration 5 — G-I3c: the module-verb dispositions

`VERB_DISPOSITIONS` in `graphed`, `graphed.numpy` and `graphed.awkward` (12 / 6 / 2), the two
refusal contracts (`refuse_boundary` on repartition/pack_key/join/shuffle_plan/join_plan;
`refuse_container` naming `graphed.universe` on compile_ir/aggregate_plan), the `broadcasting` and
`expanding` decorators, `graphed.apply`'s expansion, and the `Varied` branches in
`Array._binary` / `__getitem__` / `filter`.

`read_columns` was not `None`-dominant across several graphs — a conservative graph beside one
that narrows returned the narrow set (measured: `read_columns([plain, opaque], nid)` gave
`('MET',)`). Restructured to answer per graph and combine with `None` dominating, which is what
its own docstring already promised.

Gates: 109/112.

## Iteration 6 — G-I4: the compile and aggregate seams

`CompiledGraph.unreached_labels` (additive field, sorted, default `()`), fed by the Session
registry minus the labels the marked outputs carry; `aggregate_plan(on_compiled=...)` firing once
on the compiled artifact with its return value carried onto `_PartitionReduce.variation_labels`.

The registry stores `(labels, weakref)` pairs rather than bare weak references: F8's third anchor
registers a container and immediately discards it, which is precisely the silent-cost case the
diagnostic exists to report, so the labels must outlive the container. The weak reference to the
container itself is §2.5's pinned registration mechanism and is unchanged.

Gates: 110/112.

## Iteration 7 — lint, types, format

`ruff check python tests` clean, `ruff format --check python` clean (78 files),
`mypy` clean (76 source files, `files=["python"]` unchanged).

Three `# type: ignore[no-any-return]` in `array.py`, each one line with its reason: the container
branches of `_binary`/`__getitem__`/`filter` answer with a `Varied`. Widening those three return
types to `Array | Varied` was tried and reverted — it rippled into `numpy/array.py` and three
`awkward/functions.py` callers for a union that only arises when the *key* is a container, so the
dynamic answer stays confined to the dispatch layer. `_is_context` became a `TypeGuard`, which
removed two `Returning Any` errors honestly rather than by ignore.

## Iteration 8 — the accessor shapes the frozen tree cannot reach

Frozen-suite-only coverage put `accessors.py` at 68%: the uncovered lines are §2.2's `{label: hist}`
RESULT MAPPING and bare-histogram input shapes, the error branches of `labels`/`universe`/`weight`/
`reindex_to`, and `broadcast_like`'s body. Those shapes are `graphed-histogram`'s anchors, in a
different distribution, and `tests/frozen/frontend` must stay importable under
`pytest hypothesis numpy` alone — so this repo's frozen suite cannot reach them.

Added `tests/extra/frontend/m48/test_accessor_input_shapes.py` (awkward-free, a duck-typed
histogram on the one attribute §2.2 reads) and, with `AwkwardBackend.broadcast_like` recording
`ak.broadcast_arrays`, `tests/extra/awkward/m48/test_broadcast_like_seam.py`. `run-tests.sh` picks
both up with no change: it already runs `tests/extra/<pkg>` for the split packages.

`accessors.py` 68% -> 99%; `awkward/backend.py` 98% -> 100%.

## Gate results at the dispute STOP (tip fdbc2ae, freeze `m48-freeze`)

| gate | result |
|---|---|
| m48 frozen | 110/112; the 2 red are the disputed pair |
| whole frozen suite | 7 failures = 5 pre-existing `awkward/m16` + the 2 disputed; no regressions |
| coverage (combined, `COV=1 ./scripts/run-tests.sh`) | TOTAL 93%, gate >=90 |
| coverage (frozen suite ALONE) | TOTAL 92%; every m48 file >=91% except `accessors.py` at 68% (above) |
| ruff | `check python tests` clean; `format --check` clean, 80 files |
| mypy | clean, 76 source files, `files=["python"]` unchanged |
| determinism | frozen F2 green |
| precommit `--fast` | toml-valid ok, integrity-scan ok; workflows-valid skipped (pyyaml absent); prek cannot launch here (`rustup could not choose a version of cargo-clippy`), so ruff and mypy were run directly |

## Iteration 9 — final gate pass on the re-freeze

The dispute was upheld. The test-author applied both adjudicated corrections in affd355 — link kind
(1) re-indexes `shifted.Photon.pt` (a collection the shift does not name, so the read is genuinely
unvaried), and the §6.1d(B) identity arm keeps `sel.MET.pt` and asserts the identity over HANDLES:
equal label lists plus per-label member node-id equality, which is strictly stronger than the single
`.node_id` §2.2 refuses on a container. Suite re-frozen at `m48-freeze2`.

**No implementation change was needed or made.** Every gate below ran against the source as it stood
at the STOP; the only commits since are the test-author's re-freeze and this log entry.

| gate | result | decisive output |
|---|---|---|
| m48 frozen | **112/112** | `pytest tests/frozen/awkward/m48 tests/frozen/frontend/m48` exit 0, zero FAILED |
| whole frozen suite | only the known-env reds | 5 FAILED, all `awkward/m16` `ak.var`/`ak.std` typetracer `MaybeNone` |
| coverage (combined) | green | `TOTAL 5905 314 1712 172 93%`; `coverage report --fail-under=90` printed no failure line |
| coverage (frozen suite ALONE) | `TOTAL … 92%` | every m48 file >=91% except `accessors.py` 68% — the cross-repo shapes covered from `tests/extra` |
| ruff | clean | `All checks passed!` / `84 files already formatted` |
| mypy `--strict` | clean | `Success: no issues found in 76 source files`, `files=["python"]` unchanged |
| determinism | green | frozen F2 passes inside the 112 |
| precommit `--fast` | unchanged | `toml-valid ok`, `integrity-scan ok`; `workflows-valid --  pyyaml not installed`; `prek FAIL` on `rustup could not choose a version of cargo-clippy` |

Per-file coverage under the frozen suite alone: `_tags` 98, `varied` 95, `vary` 91, `numpy/varied`
100, `context` 92, `gnano` 100, `session` 96, `projection` 100, `execute` 93, `aggregate` 95,
`array` 95, `awkward/backend` 98, `accessors` 68 (99 with `tests/extra`).

## Decision: `graphed.labels(ctx)` on a ROOT context

**Answer: `("nominal",)`, not `()`.**

§2.2 defines the label list as "ordered: nominal first, then insertion order", and §2.4 makes
`"nominal"` always first in every union — a `Varied` cannot exist without it. A root context is
the union of three empty terms, and the union of nothing under a rule that always seats
`"nominal"` first is `("nominal",)`, not the empty tuple. It also keeps the verb total across its
four input shapes: `labels` of a bare unvaried histogram is `("nominal",)` by the same reasoning,
and a caller iterating `graphed.labels(ctx)` to fill per universe gets the one nominal fill rather
than silently filling nothing. The empty tuple would make "has this context any variations?" and
"which universes does it have?" the same question, which is the §2.5 silent-drop shape.

`graphed.labels` on a plain `Array` raises instead — an array carries no universes, and answering
`("nominal",)` there would let a dropped container read as a legitimate single-universe result.

## Test dispute — STOPPED on two tests (UPHELD; re-frozen at `m48-freeze2`)

`.graphed/m48/disputes/awkward-m48-test_lineage_seams.md`.

`tests/frozen/awkward/m48/test_lineage_seams.py::test_link_kind_1_a_mask_derivation_makes_an_
UNVARIED_value_varied` and `::test_reindex_to_is_the_identity_for_a_value_already_at_the_target_
or_context_free` both require a read through a shift-varied / `Varied`-mask-derived context to be
a plain `Array`. Plan §2.6b ("thereafter `events.<Collection>` is a `Varied`") and §2.6c ("its
collections READ as `Varied`") bind those reads to be containers, and §2.2's reserved-name rule
makes `.node_id` on a container an `AttributeError`. The dispute carries the measurement, with
`Photon` (a collection the shift does not name) as the positive control, and a proposed
correction for each test.

No implementation shortcut was taken to make either pass.

## Known-environment red (pre-existing, not m48)

`tests/frozen/awkward/m16` fails 5 tests in this environment (`ak.var`/`ak.std` typetracer
`MaybeNone`), independent of m48. Neither fixed nor masked.

Verified against a worktree at `m48-freeze` (37d86cf) running the same command: the two `-q --tb=line`
transcripts diff to ZERO lines once absolute paths and the one shifted `session.py` line number are
normalised — same failing ids, same progress bitmap `.........FF.............F.........................FF`,
same `GraphedTypeError: ill-typed op 'ak.std'/'ak.var' … Encountered unknown type MaybeNone`.
Regenerate with:

    git worktree add --detach <dir> m48-freeze && cp python/graphed/core/graphed_core.*.so <dir>/python/graphed/core/
    (cd <dir> && PYTHONPATH=<dir>/python python -m pytest tests/frozen/awkward/m16 -q --tb=line)
