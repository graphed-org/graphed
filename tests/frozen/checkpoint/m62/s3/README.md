# m62 unit C frozen suite — `FsspecStore` on `s3://` through moto

Frozen at `freeze-m62` (alias `freeze-m62c`); read-only afterwards. Spec: lane plan `plan-C.md` §C.3–C.4,
with plan review r4/r5's repair: only TB4 stays out of the re-run (moto answers concurrent same-key PUTs
with HTTP 500); TB16's record names are unique, so it runs here as the s3 concurrent-writer witness.

`conftest.py` defines `store_url` and `shared_url` (each a fresh `s3://m62-checkpoint/<uuid>`) and the
session fixture `m62_moto_bucket`: one `ThreadedMotoServer(ip_address="127.0.0.1", port=0)`, the AWS
variables set in `os.environ` (children inherit them) and restored at teardown, the bucket made once.
moto and s3fs are reached only through `pytest.importorskip` inside that fixture, so where they are not
installed every test here is reported skipped and the other checkpoint trees still run.

`test_fsspec_store_on_s3.py` imports each unit B test that takes `store_url` or `shared_url`, except TB4,
under its own name (regenerate the set with
`grep -nE 'def test_\w+\([^)]*\b(store_url|shared_url)\b' tests/frozen/checkpoint/m62/url/test_*.py`).

| Re-run | Pins (plan-B) | s3-only witness |
|---|---|---|
| TB2, TB3, TB6–TB16, TB18, TB19, TB22 | B.3.2 as in `../url/README.md` | — |
| TB9 `test_fresh_store_is_empty` | B.3.2.7 | C.3.1.2: presence listed with `find`, not `ls` |
| TB20 `test_other_process_resumes_from_the_url_alone` | B.3.2.10 | C.3.1.1: the pre-listed parent sees the child's records (listings read fresh) |
| TB21 `test_other_process_finds_nothing_left_to_do` | B.3.2.10 | — |
| TB16 `test_concurrent_record_dead_loses_nothing` | B.3.2.6 (records) | concurrent writers on s3 |

## TEST_SANITY (lane venv: moto 5.2.3, s3fs 2026.9.0, fsspec 2026.9.0)

- On the constructor-raising stub, all 18 re-run tests fail on the stub (none skipped): the moto fixture
  ran for each.
- On a scratch contract-following `FsspecStore` that reads listings fresh, all 18 pass. With its listing
  cache left on, TB20 fails. With presence listed by `ls`, TB9 fails (and six other re-runs).
- The plan's run of these mutants on B's tip is the implementer's (C2), because B's tip does not exist at
  freeze time.
