# Test Dispute — `tests/frozen/debug/m65/test_m65d_replay.py::test_diff_reports_a_difference`

Contradicts: graphed#143, inference, which runs an undeclared map's function once at record time.

`_drop_after_first` counts its calls, so the record-time call consumed the first one.

Correction: the map declares `output_type=`, which skips inference.

Ruled by the owner, 2026-11-01: the change stands. Needs a new freeze tag at the merge commit.
