# Test Dispute — `awkward/m40/test_pack_key.py` child env is POSIX-only (fails on Windows CI)

**Filed by:** team-lead, 2026-07-19, during the post-M40/M41 push CI remediation.
**Status:** RESOLVED — owner ruled "Fix the windows tests, make sure they remain non-vacuous." (2026-07-20).
Proposed non-weakening correction applied by team-lead under that authorization (see Resolution below).
**Severity:** portability defect (the test is deterministic and correct on POSIX; it is *un-launchable* on Windows).

## The test

`tests/frozen/awkward/m40/test_pack_key.py::test_pack_key_is_stable_across_processes_and_hash_seeds`
verifies the join key-packing is independent of `PYTHONHASHSEED` (i.e. never Python `hash()` in the
path — the M39 B2 pattern) by packing the same rows in two child processes launched under different
seeds and asserting equal output. Sound intent, non-vacuous.

## The defect

The child is spawned (helper `_pack_in_child`) with a **hardcoded minimal POSIX environment**:

```python
proc = subprocess.run(
    [sys.executable, "-c", _CHILD, *args],
    env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin"},   # POSIX-only: no SystemRoot, no Windows sys dirs
    capture_output=True, text=True,
)
assert proc.returncode == 0, proc.stderr
```

On **Windows** this `env` has no `SystemRoot`/`SYSTEMROOT` and a POSIX `PATH`, so the child Python
cannot initialize Winsock. Importing `awkward` pulls in `fsspec → asyncio → asyncio.windows_events →
import _overlapped`, which then raises:

```
OSError: [WinError 10106] The requested service provider could not be loaded or initialized
```

The child exits non-zero → the `assert proc.returncode == 0` fails. This is **deterministic on
Windows, not a flake** — the hardcoded env omits what the Windows loader/Winsock require. It passes on
Linux/macOS because a bare `PATH=/usr/bin:/bin` env is sufficient there and no Winsock init occurs.

### Why it surfaced only now

The duplicate-basename collision (`m5`/`m40` `test_projection.py`) previously *interrupted* awkward-package
collection before this test ran, so Windows never reached it. Fixing the collision (per-milestone
isolation) let `awkward/m40` actually run on Windows, exposing the pre-existing env defect.

## Proposed correction (non-weakening)

Preserve the `PYTHONHASHSEED` override (the property under test) but inherit the ambient environment
instead of replacing it, so the child is launchable on every OS:

```python
env={**os.environ, "PYTHONHASHSEED": seed}
```

This keeps the seed override and the determinism assertion **exactly as-is** — it only stops stripping
the platform variables the child needs to start. (If the intent of the bare `PATH` was hygiene, the
Windows-portable equivalent additionally carries `SystemRoot`/`SYSTEMROOT` and `PATHEXT`.) The dispute
proposes the correction; per the frozen-test rule the test-author/owner makes the edit and re-freezes.

## Scope

Only this test. The other three M40 remediation items (py3.11 mypy stubs, freethreaded pyarrow,
rust-coverage <90%) are fixed separately in harness/src (no frozen edits).

## Resolution (2026-07-20)

Owner ruling: **"Fix the windows tests, make sure they remain non-vacuous."** Applied the proposed
correction verbatim — `_pack_in_child` now spawns with `env={**os.environ, "PYTHONHASHSEED": seed}`
(plus an in-line comment explaining the Winsock-init failure the stripped env caused). The five test
bodies and every assertion are byte-for-byte unchanged; only the child's environment construction
moved from replace-with-POSIX-minimal to inherit-and-override-one-variable.

**CI evidence of the exact failure** (before the fix): core CI run `29709245352` (SHA `8ee3446`),
every Windows job, had **exactly one** FAILED test —
`awkward/m40/test_pack_key.py::test_pack_key_is_stable_across_processes_and_hash_seeds` — with the
child dying on `import _overlapped → OSError [WinError 10106]`. No other Windows test failed, so the
correction is correctly scoped to this one file.

**Non-vacuity preserved (measured, `scratchpad/nonvacuity_probe.py`, driving the real fixed
`_pack_in_child`):**
- (A) the real `pack_key` is identical across `PYTHONHASHSEED` 0 vs 1 and equals the in-process value
  (the test passes for the right reason);
- (B) a deliberately `PYTHONHASHSEED`-sensitive mutant packer (str-hash of the triple) produces
  **different** output across the two children under the inherited env → the `assert a == b` would
  **fail** → the test still discriminates a seed-sensitive/`hash()`-based implementation;
- (C) a bare `hash("…")` probe differs across seeds under the full inherited env → `PYTHONHASHSEED`
  is demonstrably still active (the inherited env did not neuter the variable under test).

**Why only this file** (measured — import-chain probe + full frozen-suite `env=` sweep): only the
awkward child pulls `asyncio`/`fsspec` (→ Winsock init → WinError 10106 on Windows). The numpy m40
pack_key, the M8 (`checkpoint`/`core`) durable-plan/no-source, M9 preserve, and exec M39
routing-invariance children are all `asyncio=False` and launch fine with a stripped env — CI's
single-FAILED-line confirms they pass on Windows. The M8/M9 tests additionally **must** keep their
stripped env (they verify "runs with no source on `sys.path`, only installed packages"); inheriting
`os.environ` there would restore `PYTHONPATH` and weaken them. So they are left untouched.
