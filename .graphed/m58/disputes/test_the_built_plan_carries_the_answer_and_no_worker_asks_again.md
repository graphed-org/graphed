# Test dispute — H3's pickle leg cannot pass: `DeclaringSource` carries the output `Array`s (and its
# own `seen` log) through `pickle`

Filed by the m58 implementer. Test:
`tests/frozen/awkward/m58/test_plan_boundary.py::test_the_built_plan_carries_the_answer_and_no_worker_asks_again`.
Fixture: `tests/frozen/awkward/m58/m58_declaration_fixtures.py::DeclaringSource`.

The other ten frozen legs pass against the implementation in the working tree (H1 ×3, H2 ×2, H4, H5,
E1, E2, E3). This one fails for a fixture reason, in two independent places, neither reachable before
the hook existed — the suite README records that: *"H3's later `pickle` leg is unreachable until it
does"*, so TEST_SANITY never executed it.

## The conflict

`projected_columns` stores `self.outputs = tuple(outputs)`, and H1 requires those to be the driver's
own `Array` objects (`assert source.outputs[0] is outputs[0]`). H3 then pickles a `Plan` whose
`process.reader` **is** that same source object, so pickle walks `reader.__dict__["outputs"]` →
`Array.session._store`, which is the Rust `GraphStore`:

    TypeError: cannot pickle 'builtins.GraphStore' object

Measured (`python -c`, this tree, implementation applied):

| probe | result |
|---|---|
| `pickle.dumps(plan)` | `TypeError: cannot pickle 'builtins.GraphStore' object` |
| same, after `source.outputs = ()` | OK |
| `pickle.dumps(GraphStore())` | `TypeError: cannot pickle 'builtins.GraphStore' object` |

So the requirement is: an `Array` must survive `pickle`. It cannot. `Plan` is deliberately built the
other way round — `_PartitionReduce` ships `ir=bytes(compiled.ir)`, never a `Session` (§A.3.1: the
serializable IR, not a live session object, is the durable representation). No implementer-side
choice reconciles the two: handing the hook copies breaks H1's `is`; not carrying the source in the
plan breaks H3's own `shipped.process.reader.calls` / `.seen`; making `GraphStore` picklable is a
Rust change, and m58 is Python-only (`m58-decomposition.md`, line 4).

A second, smaller collision sits behind the first: `seen` also travels. The original plan is run
first (`seen == [DECLARED, DECLARED]`), so the unpickled copy starts with two entries and running
the shipped plan appends two more — `shipped.process.reader.seen` is four entries, not the two the
test asserts. Both fields are recording state that belongs to the *local* witness, not to the
shipped closure.

## Proposed correction

One fixture line, using the precedent already in this repo for a field that must not ship
(`graphed/awkward/functions.py::_RecordedPayload.__getstate__` drops its unpicklable `call`):

```python
    def __getstate__(self) -> dict[str, Any]:
        # the local witness state, not the shipped closure's: the Arrays are unpicklable and the
        # recorded reads must start empty in the worker copy
        return {**self.__dict__, "outputs": (), "seen": []}
```

Verified out of tree (the frozen file untouched; the correction monkey-patched onto the class, then
the test body run verbatim): every assertion of
`test_the_built_plan_carries_the_answer_and_no_worker_asks_again` passes, including
`shipped.process.columns == DECLARED`, `shipped.process.reader.calls == 1` and
`shipped.process.reader.seen == [DECLARED, DECLARED]`. With only `outputs` dropped, the `seen` leg
still fails (4 entries) — both fields are needed.

No other frozen test changes: `outputs` and `seen` are read in-process by H1/H2/H4/H5, which never
pickle, and `PlainSource` is unaffected.

## Implementation status (not committed)

The m58 source change sits uncommitted in the working tree and is complete for all three drivers and
the External stand-in (`graphed/write.py::declared_columns`, the three call sites,
`graphed/awkward/projection.py::_replay.on_external`). `tests/frozen/**` is unmodified
(`git status --short tests/frozen/` is empty). Docs and `tests/extra/awkward/m58/` were not written:
the run stopped here per the integrity rule.

## Adjudication

CORRECTED (lead, all three probe legs reproduced). The test-author amended the fixture
(`DeclaringSource.__getstate__`: the answer and call counter ship, `outputs`/`seen` stay local), added
the control `test_a_pickled_declaring_source_keeps_its_counter_and_drops_its_witness_state`, and ran
every assertion that sits behind a first failing one by simulation on the unimplemented tree.
Re-frozen at tag `freeze-m58-2`.
