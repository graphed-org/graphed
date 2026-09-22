# frontend/m61 — the write base refuses duplicated inputs (traceability)

| Test | Verifies |
|---|---|
| `test_file_bases_refuses_duplicate_keys` | `graphed.write.file_bases` raises `duplicate input` for a repeated uri or (uri, tree) key; distinct keys keep their bases |

Run: `python -m pytest tests/frozen/frontend/m61 -q`. The awkward/m61 suite covers the drivers.
