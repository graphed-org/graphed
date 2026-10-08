# m74 — implementer iterations

Plan: `graphed-workdir/lanes/ckpt-resume/plan.md` (r14b) §2, §3 commits 2–3; addenda and constraints beside
it. Frozen suite: `tests/frozen/checkpoint/m74` (freeze-m74 = 5fd8ae5).

Run: `python -m pytest tests/frozen/checkpoint/m74 -p no:cacheprovider -n 8`.

## Iteration 1 (2026-10-07) — green
- `graphed.checkpoint.resume` (new): `_key_bytes` (the P14 `canonical` rule set on one
  `pickle._Pickler` subclass; payload mode dropped; `checkpoint_resolve` added), `_Keyed`, `_Resumable`,
  `_ResumableStage`, `resumable`, `check_resumable`, `StoreUnavailable`, `EnvironmentChanged`, the
  environment record check, and `open_store`/`task_store` lifted from `_PartitionReduce`.
- Stores: `record_environment` on `Store`/`FsspecStore` (own journal `journal.environment.log`);
  `_replay` returns `Completed`, a dict with the set-aside `environments`.
- `run_resumable`: streaming fold, environment check; `run_shuffle_resumable`: environment check after
  `require_bound`; both refuse a store without `record_environment` before any store call.
  `_partition_tag` now lives in `resume.py`.
- Declarations: `checkpointable`/`checkpoint_ignore`/`checkpoint_resolve` on `_PartitionReduce`,
  `_MapWrite`, `_Gather`, `_Fold`, `_WritePart`, `_VariedWritePart`, `CompiledGraph`; `_Collated` as a
  property. `_resolve_behavior` moved above `_WritePart` (its class body names it).
- Extra: `tests/extra/checkpoint/m74/test_m74_extra.py` (9) for paths the frozen suite reaches only in
  subprocesses or not at all; 8 single-site mutations of `resume.py` each fail one of them.
- Addenda: two lines (§2.4 store cache, §2.6 stage records).
- Gates: m74 frozen 75/75; `COV=1 run-tests.sh` rc 0; ruff, mypy --strict (+ win32) clean; sphinx -W ok
  (docs example executed); frozen diff vs freeze-m74 empty.

## Iteration 2 (2026-10-07) — precommit integrity, green
- `integrity-scan` flagged four `# type: ignore` on the pure pickler's hooks (typeshed declares none of
  them); the hooks are now called through one `Any`-typed alias of `pickle`, no ignores left.
- Extra: `_resolve_behavior`'s import-ref branch (moved code) covered in process.
- Gates: m74 frozen 75/75; `COV=1 run-tests.sh` rc 0; touched files 97–100%; diff-cover vs eea2f8d 100%
  (376 lines); precommit static checks ok, sphinx -W ok. The precommit `pytest` fallback collects the
  whole tree in one process and fails collection on helper-basename collisions (`analyses`,
  `shuffle_backends`) unrelated to m74; `run-tests.sh` is this repo's runner.

## Iteration 3 (2026-10-07) — review fixes (impl1-findings: m74-D1, N1, N2, T-1, T-2, D-1)
- m74-D1: every read of cloudpickle's private API is in `_cloudpickle()` (cached, called from
  `_KeyPickler.__init__`); no class attribute or module global binds one. A missing name makes
  `check_resumable`/`resumable` raise `TypeError` naming cloudpickle; `import graphed.checkpoint`,
  the stores and `run_resumable` keep working. Extra test: one subprocess leg per deleted private name
  plus a control; both deletion legs fail on da9c20c.
- N1: the key pass pickles into a sha256 sink (`_key_digest`), so key memory stays flat. The memory
  regression was this iteration's own eager label in the sink version; da9c20c already built the
  process repr only in `except`, so the lazy `_of` label fixes nothing there. Element sort keys keep bytes.
- N2: each `ak.behavior` entry is keyed alone; an unkeyable one is refused as `ak.behavior[<key>]`.
- T-1: the vacuous tracker test is replaced by by-value TypeVars of different names; it kills
  "blank every str", "no blanking" and "`_TRACKED = (type,)`". T-2: two tests renamed. D-1: the two
  m74 headings no longer nest a literal in bold.

## Iteration 4 (2026-10-07) — delta-review follow-ups (m74-r2-N1, N2, N3)
- r2-N1: `_cloudpickle()` reads the private names listed in `_CLOUDPICKLE_PRIVATE`; the subprocess
  test is parametrized over that list, so an import-time read of any of them fails its leg (the
  reviewer's `_module_reduce`, `dynamic_subimport` and `Pickler._dispatch_table` mutants now die).
- r2-N2: `_task_ids` calls `_cloudpickle()` before keying anything, so a missing name refuses with the
  cloudpickle message alone; the in-process test registers a keyable `ak.behavior` entry and asserts
  the exact message.
- r2-N3: Iteration 3's N1 line corrected.

## Iteration 5 (2026-10-08) — CI fixes (PR #69)
- `test_cloudpickled_plan_loaded_twice` reused `[0, 0]` on 3.13+. From 3.13 a frozen dataclass's
  generated `__setattr__`/`__delattr__` share closure cells, and cloudpickle's loader gives each
  function fresh cells, so the memo wrote BINGET in the building interpreter and new cells in the
  loading one. `_KeyPickler.memoize` skips `types.CellType`.
- On 3.14 the same test still failed: cloudpickle's loader sets a class's annotations by `setattr`,
  which 3.14 stores as `__annotations_cache__`. The class-state rule writes that key as
  `__annotations__`.
- `test_an_abc_registry_made_in_another_order_is_reused` was flaky: the ABC registry holds its classes
  weakly and `_registry` dropped them, so a collection emptied it. The instance now holds them, and
  the test collects between fill and resume.
- Extra: functions sharing a cell, and a class whose annotations were set after creation, are each
  reused from a rebuilt copy; each fails with its fix reverted (the cell one on every version, the
  annotations one on 3.14).

## Iteration 6 (2026-10-08) — closed walk of cloudpickle's by-value reducers
- Walk: every `reducer_override`/`_class_reduce` branch, `Pickler._dispatch_table` key and
  `_*_getstate`/`_*_reduce` helper of cloudpickle 3.1.2, plus lazily filled attributes, probed for the key
  of an object against its loaded copy in a fresh interpreter, before and after first access, on 3.12,
  3.13, 3.14 and 3.14t. Member list, per-version results, causes and cuts:
  `graphed-workdir/lanes/ckpt-resume/probes/walk/` (`members.txt`, `results.md`).
- Unstable on every version, now cut in `_KeyPickler`: an empty class annotations dict a first read
  stores; a slotted instance whose copy (cloudpickle rebuilds the class without slots) carries a
  `__dict__`; a dispatch-table object the by-name rule named while cloudpickle ships it by value
  (typing's `_proto_hook`); a memoryview and a file, which load as bytes and StringIO; set elements with
  equal key bytes, whose iteration order decided which one a later reference shared. On 3.14 the copy of
  an unread annotated class is also cut by the empty-annotations rule.
- Kept, as values a worker uses: a `cached_property` value, and on 3.14 a class's annotations, which
  cloudpickle 3.1.2 ships only once read. The stopped iteration's WIP patch is superseded: its
  `__annotate_func__` condition never held, because cloudpickle pops that key before the key pickler
  sees the class state.
- Extra: one subprocess leg per cut member (two hash seeds); with the cut reverted, seven legs fail on
  3.12 and 3.13, and eight on 3.14 and 3.14t.

## Iteration 7 (2026-10-08) — ties and by-reference slots stay keyed by their values
- Review F1: the twin rule wrote an element whose key bytes tied an earlier one's as a reference to it,
  and a set element's sort key names a by-value class already written by module and qualname alone, so
  `{K1(), K2()}` from one factory keyed like `{K2(), K2()}` (stale reuse). The twin rule and its tie
  break are deleted; tied set elements keep set iteration order, a recompute, never a reuse (F2: ties
  inside tuple or frozenset elements were never covered by it).
- Review F3: the slot-shadow rule applied to every default-BUILD instance; an importable class's copy
  keeps both the slot and the `__dict__` entry. It now applies only when cloudpickle writes the class by
  value (`_should_pickle_by_reference`, read through `_cloudpickle()`).
- Review N3: a class `__dict__` mappingproxy loads as a snapshot dict whose `__dict__` entry is
  `getattr(copy, "__dict__")`, the copy's own mappingproxy, not the descriptor, and the original proxy
  shows later class-dict fills; the copy holds different values, so it stays a recompute.
- `_key_digest` opens with one characterization; the tie and slot bullets are corrected. The docs list
  identity-only set elements among the recompute cases. Walk rerun on four versions
  (`probes/walk/results.md`); `enumerate.py` now lists the `_*_getnewargs` helpers.
- Extra: two factory classes' instances key apart from one class's twice; a by-reference slotted
  instance keys its shadowed entry while a by-value one does not; a memoryview keys as its bytes. Each
  fails at c827ebf or with its rule removed.

## Iteration 8 (2026-10-08) — the key equates an object with its loaded copy only where loading loses nothing
- Owner ruling on the iteration-7 review (`stale.py`, `copies.py`, `slotorder.py`): a rule that keys an
  original like its lossy copy is a stale reuse, since in-process runners run the original. Dropped: the
  slot-shadow rule (and the `_should_pickle_by_reference` read it needed), the memoryview/TextIOWrapper
  rewrite, the empty-annotations drop. `__annotations_cache__` is written as `__annotations__` only when
  the class has no `__annotations__`; with both, two classes differing in `__annotations__` keyed alike.
- Cells, option (b): a first key pass collects the cells a function in the graph rebinds (`STORE_DEREF`/
  `DELETE_DEREF` in its code, or in nested code sharing the name, via `dis`); those are memoized, so their
  sharing is keyed, and the second pass runs only when the first found one. Unrebound cells stay
  unmemoized, so a copy of closures sharing one is still reused.
- `_key_digest` opens with one characterization true of every remaining rule; bullets match. The docs'
  recompute list names the objects cloudpickle cannot copy exactly.
- Extra: seven pairs of processes that compute differently (asserted) must key apart; all seven fail at
  5976f6d, and the annotations and both cell pairs fail at c41b5fd. The memoryview/file/slot/annotation
  copy legs are deleted; the protocol leg stays (fails at c41b5fd). Walk rerun on four versions
  (`probes/walk/results.md`): every unstable member is a stated residual.

## Iteration 9 (2026-10-08) — unread class annotations and shared function globals are keyed
- Review P1 (3.14+): cloudpickle pops an unread class's `__annotate_func__` and its state holds no
  `__annotations_cache__`, so editing an annotation reused a stale result. A class with a callable
  `__annotate_func__` and no annotations in its state now keys that function, unevaluated, under a
  key-only state name; its `__classdict__` cell (the class's own dict) is written as a reference to the
  class, since that dict holds an ABC's unpicklable `_abc_impl` and caches the class state drops. A read
  class keeps its form; below 3.14 nothing changes.
- Review P2: `_function_getnewargs` built a fresh filtered globals dict per function; it is now cached per
  namespace on the pickler, so functions sharing `__globals__` (one rebinding a global) key apart from
  ones with separate equal-valued globals, as cloudpickle's copy keeps them.
- Docs: the cell-rebinding residual (review N1), team-lead's in-place input rewrite line beside `salt`,
  and the unread-annotations recompute case; the `_key_digest` opening names shared globals (review N2).
- Extra: two pairs in `test_processes_that_compute_differently_key_apart`; both fail at 1d9c9e0 on 3.14,
  the globals pair on 3.13; the annotations pair fails with the classdict rewrite removed (`_abc_data`
  cannot be pickled) and runs unskipped below 3.14, where `__annotations__` keys it (a version skip trips
  the integrity scan's `skip_or_xfail_added`).

## Iteration 10 (2026-10-08) — names class-scope code reads from globals are keyed
- Review B1: cloudpickle records a by-value function's globals only from `LOAD_GLOBAL`/`STORE_GLOBAL`/
  `DELETE_GLOBAL`, so a global read by class-scope code was unkeyed: through `LOAD_NAME` (a class body,
  every version) or `LOAD_FROM_DICT_OR_GLOBALS` (a class body's `type` alias from 3.12, a class
  `__annotate__` and a generic class's parameters from 3.14). `T = int` vs `T = str` under an unread
  `class K: x: T`, and `class Rec: q = CONST` with CONST 42 vs 43, shared a key. `_dynamic_function_reduce`
  now adds every global either opcode reads, in the function's code or its nested code, to the recorded
  globals in code order; this covers annotate functions without a path of their own. A class-local or
  builtin name absent from `__globals__` adds nothing; a class body's implicit `__name__` read adds
  `__name__`, which cloudpickle's base globals already hold.
- Extra: `test_processes_that_compute_differently_key_apart` replaces its unread-annotations pair with
  read/unread x {through a global, through a closure cell, of an ABC}, a class-body global, and a
  class-scope `type` alias (3.12+). At 0363337 the class-body and alias pairs fail on 3.12, 3.13, 3.14 and
  3.14t, and the unread global pair on 3.14 and 3.14t; the read global pair passes there, since a read
  class keys its evaluated annotations.
