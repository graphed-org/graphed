# Test Dispute — `awkward/m60/test_parquet_form.py` + `m60_parquet_fixtures.py` import pyarrow unconditionally (uncollectable on Windows ARM64)

**Filed by:** team-lead, 2026-09-22, while adding the `windows-11-arm` row to the §A.5 test matrix.
**Status:** OPEN — awaiting the owner's ruling.
**Severity:** portability defect (the tests are correct everywhere pyarrow installs; they cannot be *collected* where it does not).

## The tests

`tests/frozen/awkward/m60/test_parquet_form.py` (every test) and its helper
`tests/frozen/awkward/m60/m60_parquet_fixtures.py` verify that a parquet source keeps awkward's record
parameters (integ-m60-P). Sound intent, non-vacuous; they need pyarrow to write the fixture files.

## The defect

Both modules import pyarrow at module top:

```python
import pyarrow.parquet as pq          # test_parquet_form.py
import pyarrow as pa                  # m60_parquet_fixtures.py
import pyarrow.parquet as pq
```

pyarrow publishes no `win_arm64` wheel (pyarrow 25.0.1 ships `win_amd64` only), so on the
`windows-11-arm` CI job the dev extra installs without it (`pyarrow>=15; sys_platform != 'win32' or
platform_machine != 'ARM64'`) and `awkward/m60` fails at **collection** with `ModuleNotFoundError:
No module named 'pyarrow'`. Every other frozen parquet module (`awkward/m15/test_parquet_io.py`,
`awkward/m51/test_single_read.py`, `awkward/m61/test_partitioned_drivers.py`, `frontend/m15/test_parquet_base.py`)
already guards the same dependency with `pytest.importorskip("pyarrow")` and skips cleanly there.

## Proposed correction (non-weakening)

Guard the import the way the sibling parquet modules do, in both files:

```python
pq = pytest.importorskip("pyarrow.parquet")   # test_parquet_form.py
pa = pytest.importorskip("pyarrow")           # m60_parquet_fixtures.py
pq = pytest.importorskip("pyarrow.parquet")
```

Where pyarrow installs (every other matrix cell) the modules import exactly as before and every
assertion runs unchanged; only the platform where pyarrow cannot exist reports the module as skipped
instead of erroring at collection. Per the frozen-test rule the test-author/owner makes the edit and
re-freezes; until then the `windows-11-arm` test jobs fail on the `awkward/m60` subtree alone.
