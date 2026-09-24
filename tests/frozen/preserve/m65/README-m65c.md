# M65 frozen suite, unit C, graphed-preserve slice (`freeze-m65c`, plan-C.md)

Frozen and read-only after `freeze-m65c`. The first file in `preserve/m65/`. Bundles come from the m9 `agc`
builder (`tests/frozen/preserve/m9` is put on `sys.path`; no pyarrow needed) and, for the container-digest
leg, a one-source `from_awkward` bundle through `build_bundle(..., container_digest=)`. Reports come from
`SequentialRunner` runs of plain plans over `Partition.blind("u", "t", k, 4)`.

| Test (`test_m65c_bundle_reports.py`) | Contract | Fails |
|---|---|---|
| `test_attach_leaves_the_fingerprint_alone` (7) | C-5 | a report in `manifest.json`; no Store blob; no `run-report:<digest>` journal entry |
| `test_run_reports_round_trip_and_dedupe` (8) | C-5 | reports out of attach order; a duplicate on re-attach; a digest that differs on re-attach |
| `test_inspect_renders_reports_only_when_present` (9) | C-6 | a section on a report-less bundle; missing header counts, `env=same`/`env=differs`, `environment_digest=`, `failed:` or task lines; an `inspect` that reads `failure["op"]` or writes the raw newline; a report that drops `container_digest` |
| `test_inspect_reads_reports_without_executing` (10) | C-6, m9 precedent | an `inspect` that resolves dataset or payload blobs |
| `test_graph_block_stays_contiguous` (11) | C-6, m9 graph block | a report line starting `"    n"` (a multi-line error `"n1\n    n2"` is attached) |

Decisions on the dispatch constraints:
- Report header lines are located by `"    " + digest[:12] + " "`, so each report's `env=` is read from its own line.
- No byte comparison of `inspect` across `Bundle.open` (C r7 N1 (3)); only line content is asserted.
