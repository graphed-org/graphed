# awkward/m60 — a parquet source keeps awkward's record parameters (traceability)

Authority: `graphed-workdir/m60-decomposition.md` (contract lines P1–P3, one frozen property each)
under root-prompt rule **R25.3**. One decision moves:

* **integ-m60-P** — `from_parquet`'s recorded form is the one awkward's OWN reader gives a zero-row
  parquet file of the dataset's arrow schema, so a named record, a parameter on a nested node, an
  option and a regular type all survive into the deferred program. A file written without
  awkward's metadata keeps the form it gets today, and no event data is read to find out.

Today the form comes from `ak.from_arrow_schema` of the arrow schema alone, which carries the
arrow types and nothing awkward stored beside them: a deferred read of a file awkward WROTE does
not type like an eager read of it, and a behavior keyed on a record name never resolves.

Run: `python -m pytest tests/frozen/awkward/m60 -q` (its own process, per the awkward
per-milestone split). The backend-neutral half of the milestone (integ-m60-X/O/V/E) is
`tests/frozen/frontend/m60`. The `m60_` prefix is load-bearing under prepend import mode.

## Fixture — `m60_parquet_fixtures.py`

* `DATA` — two events of `{p: var * m60pair[px, py], o: var * ?int64, r: 2 * float64}`, with the
  custom parameter `m60note` on the list node above the record: a named record nested in a list, a
  parameter on a nested node, an option and a regular type are the four things an arrow schema
  alone cannot say. Every value is dyadic, so `TOTALS` is exact.
* `awkward_dataset` / `pyarrow_dataset` — the same shape written by awkward itself, and the
  no-metadata control written through bare pyarrow.
* `PairArray` + `BEHAVIOR` — a behavior class keyed on `m60pair` that the backend's dict alone
  knows and global `ak.behavior` never sees, so `events.p.total` can only resolve through the
  RECORDED form.
* `eager` / `tracer` — eager awkward as the ORACLE: `ak.from_parquet` of the same file is what a
  recorded form and an executed value must equal, on the typetracer and on the real array.
* `read_guard` — bans every EVENT-DATA read of the named paths (`ak.from_parquet`,
  `pyarrow.parquet.read_table`, and `ParquetFile.read`/`read_row_group`/`read_row_groups`/
  `iter_batches` for a file opened at one of them) and returns the live list of opened paths.
  PATH-SCOPED and not blanket, because the contract's own route writes a zero-row file of the
  dataset's schema and reads it back with `ak.from_parquet` — a blanket ban would ban the answer.

## Traceability (contract line → test → what it witnesses)

| Line | Test | Mechanism witness |
|---|---|---|
| P1 | `test_parquet_form.py::test_the_recorded_form_equals_what_eager_awkward_gives_the_same_file` | `session.form(events).tt.layout.form` against `ak.from_parquet(path).layout.form` (a typetracer's `form` already drops the length, so the comparison is well-posed) |
| P1 (behavior) | `test_parquet_form.py::test_a_behavior_keyed_on_the_record_name_resolves_on_the_deferred_array` | the recorded form of `events.p.total` against the typetracer oracle, its executed values, and the eager answer for the same expression |
| P2 (columns) | `test_parquet_form.py::test_columns_still_selects_and_the_selection_keeps_the_record_name` | the selected form's fields, then the record name in its description |
| P2 (foreign) | `test_parquet_form.py::test_a_file_without_awkwards_metadata_gets_the_form_it_gets_today` | today's description, pinned literally |
| P3 | `test_parquet_form.py::test_recording_a_source_reads_no_event_data_and_opens_only_the_first_file` | the recorded fields, the dataset paths actually opened, and a positive control tripping the same guard |
| P3 (route) | `test_parquet_form.py::test_a_zero_row_file_of_the_datasets_schema_stays_readable_under_the_guard` | a zero-row file of the dataset's schema read back under the ban, then the ban tripping on the dataset itself |

## Non-vacuity — what happens on a pre-m60 tree

The suite COLLECTS with zero errors (6 tests; every m60-new outcome is reached inside a test body)
and gives the same verdict on two consecutive runs. Three legs FAIL, each for its own reason:

* P1 — the two `RecordForm`s differ in exactly the parameters: today's inner record has none where
  eager's carries `{"__record__": "m60pair"}` and the list above it carries `{"m60note": "kept"}`.
* P1 (behavior) — `GraphedTypeError: ill-typed op 'field' … no field named 'total'`: without the
  record name in the recorded form the backend's behavior class is never consulted.
* P2 (columns) — `assert 'm60pair' in '## * {p: var * {px: float64, py: float64}}'`. The selection
  half runs first and passes, so the failure is the name, not the column list.

Three legs PASS on a pre-m60 tree — the declared controls, each a live instrument:

* P2's foreign file: today's description, pinned literally. The empty-row route gives a
  pyarrow-written file the SAME form the schema route does (measured), so this pin holds on both
  sides of the implementation and reds if the new route moves a file it must not.
* P3: today's record path already reads no event data; the leg is the regression guard that the
  new route does not start, and it ends by tripping its own guard so an empty result cannot pass
  for a dead instrument.
* P3's route leg: a zero-row file of the dataset's schema reads back with the fields it should,
  which is the route the contract names, under the very ban that guards it.

## What the frozen expectations were measured against

No assertion here waits on the implementation to be executed for the first time: every leg sitting
behind a first failing one was run by simulation on a pre-m60 tree, under a monkeypatching
stand-in of `_schema_form` applied as a pytest plugin outside this tree (all 6 legs green).

* The stand-in wrote `schema.empty_table()` through pyarrow to a scratch file and read it back with
  `ak.from_parquet`, exactly as the decomposition's Decision states. Against it the recorded form
  equals eager's; `events.p.total` records `## * var * float64` and evaluates to `TOTALS`; the
  `columns=["p"]` selection keeps both fields and the record name; the foreign file's description
  is unchanged; and the whole record path still trips no read guard.
* The guard itself was measured before use: recording under it succeeds while `ak.from_parquet`,
  `pyarrow.parquet.read_table` and `ParquetFile.read` on a dataset path each raise, and a zero-row
  file written elsewhere reads back normally.
