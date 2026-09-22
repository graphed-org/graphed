# awkward/m61 — partition-wise drivers are sound (traceability)

Authority: the uproot5#1720 review remediation (`graphed-workdir/pr1720-review-remediation-journal.md`
§3). One operation produces every shape here: a partition-wise driver evaluates the compiled IR
once per chunk.

| Test | Verifies |
|---|---|
| `test_refusal_reads_the_compiled_ir` | `refuse_chunk_partials` walks the reduced IR: an output partial passes with `as_outputs=False` and is refused with `as_outputs=True`; a consumed partial (`sum(x[2:8])`) is refused either way; a row-local graph passes both |
| `test_aggregate_plan_refuses_an_interior_reduction` | `aggregate_plan` refuses `sum(x[2:8])` at plan time (it returned a per-chunk 33 for a dataset-wide 27) |
| `test_aggregate_plan_still_folds_an_output_reduction` | control: an output reduction and a row-local mask still fold across partitions |
| `test_writers_refuse_a_reduction_output` | both parquet writers refuse `x[2:8]` before creating the destination |
| `test_awkward_writer_wires_externals` / `test_numpy_writer_wires_externals` | `graphed.apply` inside a written expression evaluates in the write task (projection passes through the opaque node; evaluators come from `external_evaluators`) |
| `test_writers_refuse_a_duplicated_input` | listing one file twice raises `duplicate input` instead of writing the same part paths twice |

Run: `python -m pytest tests/frozen/awkward/m61 -q`.
