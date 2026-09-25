# awkward/m72 — `parquet_write`, MC+data collate, `num(axis=0)`, writes-only projection (traceability)

Spec: the m72 brief (a)/(b)/(c) and `graphed-workdir/lanes/multiout/plan-A.md` §2 "Frozen-test contract".
The oracle `m72_awkward_fixtures.dump_to_parquet` mirrors HiggsDNA's `dump_ak_array` /
`inclusive_processor.dump_to_parquet` on an eager chunk.

**New API (an `AttributeError`/`TypeError`/`KeyError` on 0.0.6 is the expected pre-implementation
failure):** `graphed.awkward.parquet_write` (= `graphed.awkward.io.parquet_write`,
`VERB_DISPOSITIONS["parquet_write"] == "refusing"`); `graphed.write.PartWrite`;
`graphed.aggregate_plan(..., writes=)`; `graphed.collate`. Changed behaviour: `gak.num(x, axis=0)`
records as a reduction.

| Test | Contract item | Pins (witness) |
|---|---|---|
| `test_parquet_write_reproduces_a_dump_to_parquet_part` | awkward 1 | option-typed record `?{jag, nest, num}` (jagged, nested record, non-option numeric), sorted in the graph, `extensionarray=False`, KV `{sum_w: this chunk's float32 sum, kind: "MC"}`: schema (with KV), values and column order equal the oracle per part; the float32 `str` differs from `str(float())` |
| `test_parquet_write_defaults_are_the_libraries_defaults` | awkward 2 | no options: schema (incl. the table's own `ak:` KV), values and compression equal `pq.write_table(ak.to_arrow_table(chunk))`; non-option record fields `nullable=False`; non-record wrapped under `data` / `column="val"` |
| `test_mc_and_data_graphs_collate_into_one_plan` | awkward 3 | two parquet datasets, two sessions, different graphs (MC: weights + per-part KV sums; data: static `"Data"` KV): collated value == separate values; parts byte-identical to the separate runs and across two collated runs |
| `test_num_axis0_is_a_reduction` | awkward 4 | compiled node kind `reduction`; feeding `* 2` raises "feeds another node"; as an output folds to the dataset count; as a write root raises "a partitioned write has no combine step" |
| `test_a_writes_only_plan_reads_only_the_written_columns` | awkward 5 | `outputs=()`, toy codec write of `2x`, metadata `sum(y)`: the recording source sees exactly `{x, y}` on every read; parts hold `2x` and each chunk's `sum(y)` |
| `test_parquet_write_refuses_a_varied` | awkward 6 | `Varied` array or `Varied` metadata value raises `GraphedError` "does not accept a Varied" before any directory exists; disposition `"refusing"`; a plain array returns a `PartWrite` |

Run: `python -m pytest tests/frozen/awkward/m72 -q`.
