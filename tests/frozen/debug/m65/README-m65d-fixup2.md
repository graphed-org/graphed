# M65 frozen suite, unit D fixup 2: a corrupt capture blob (tag `freeze-m65d-fixup2`)

Frozen and read-only after `freeze-m65d-fixup2`. Owner-ruled dispute: `.graphed/m65/disputes/test_m65d_corrupt_capture_blob.md`.

| Test (`test_m65d_replay_capture_blob.py`) | Contract | What it witnesses | Fails |
|---|---|---|---|
| `test_a_corrupt_input_blob_raises_file_not_found` | plan-D D-2 store arm | task 1's `replay-input` blob rewritten in place (still in `completed()`): `input_source == "store"`, `.value` raises `FileNotFoundError` naming the digest; task 2 replays unaffected | a store check stripped by `python -O` (`assert`); an error from the codec instead; a fall-back to re-reading the partition |
| `test_a_corrupt_output_blob_raises_file_not_found_from_diff` | plan-D D-6 `recorded` reference | task 1's `replay-output` blob rewritten in place: `.value` still correct, `.diff()` raises `FileNotFoundError` naming the digest | as above, on the recorded-output read |
