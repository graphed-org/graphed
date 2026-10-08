# Test Dispute — m74 frozen legs that write parquet have no pyarrow guard (Windows ARM64)

**Filed by:** m74 test-author, 2026-10-08, from the `windows-11-arm` jobs of graphed#69 (py3.11–3.14).
**Status:** RESOLVED under the standing owner ruling of 2026-09-22 (precedent:
`.graphed/m60/disputes/awkward-m60-test_parquet_form-pyarrow-import.md`): frozen tests that need
third-party wheels are skipped on ARM64. Correction applied by the test-author and re-frozen as
`freeze-m74-fixup`.

## The tests

The class, enumerated by the `windows-11-arm` CI run (every m74 test ran; these and only these errored
or failed with `ImportError: to use ak.to_parquet, you must install pyarrow`):

| member | how it needs pyarrow | correction |
|---|---|---|
| `checkpoint/m74/test_m74_key_interpreters.py` (module) | every test takes the module fixture `data`, which writes parquet via `m74_helpers.events` | module-level `pytest.importorskip("pyarrow")` |
| `checkpoint/m74/test_m74_refusals.py::test_allowed` (all four params) | the module fixture `paths` writes parquet via `m74_helpers.events`; the module's other tests do not use it | `pytest.importorskip("pyarrow")` as the fixture's first line |
| `checkpoint/m74/test_m74_resume.py::test_shuffle_plan_interrupted_resumes_with_identical_gather_payloads` | writes and reads parquet via `m74_helpers.events` / `from_parquet` | `pytest.importorskip("pyarrow")` as the test's first line |

## The clause

Owner ruling 2026-09-22: Windows ARM64 needs no testing of what requires third-party wheel builds;
such tests are skipped there. pyarrow publishes no `win_arm64` wheel, and the dev extra installs it only
where `sys_platform != 'win32' or platform_machine != 'ARM64'`.

## The correction (non-weakening)

Only the three guards above. Every assertion is unchanged; wherever pyarrow imports, the collected and
passed m74 test ids are identical before and after the guard. Without pyarrow the listed members skip
and every other m74 test still runs and passes.
