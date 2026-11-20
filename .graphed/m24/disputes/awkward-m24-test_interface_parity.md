# Test Dispute — `test_interface_parity.py` (module docstring + README)

File: `tests/frozen/awkward/m24/test_interface_parity.py`, `tests/frozen/awkward/m24/README.md` (graphed#256, PR #266)

Status: CLOSED (owner ruling below)

## The text
> Deliberately absent everywhere: highlevel / attrs (eager concerns) and behavior (backend-owned,
> M18 — never a per-call kwarg).

## The contradiction
Third-party code forwards awkward's keywords at their defaults: `vector.zip` calls
`ak.zip(..., behavior=...)`, and `ak.zip(..., behavior=None)` on deferred arrays (the #256 repro)
raised `zip() got an unexpected keyword argument 'behavior'`. Under the frozen rule such code cannot
run deferred.

## Correction
Text only, no assertion changes (the suite passes unmodified): `highlevel` / `attrs` / `behavior`
and every other defaulted ak keyword a gak function does not implement are accepted at awkward's
default and refused by name otherwise; a `behavior` mapping is accepted only when the backend
already applies it.

## Ruling (owner, 2026-11-20)
"#266 refreeze is authorized". Amendment under `--allow-refreeze tests/frozen/awkward/m24`,
re-tagged `freeze-m24-fixup3` on the merge commit.
