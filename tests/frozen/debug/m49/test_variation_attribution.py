"""§8.1 and §8.2(ii)/(iii): the worker-side wrap around ``evaluate_ir`` and what it attributes to.

The wrap has TWO arms and both are law: with an entry for the failing key a worker failure becomes a
``StageError`` carrying that key's label and the user's line; with NO entry it re-raises the original
exception untouched — not a ``StageError``, and not the ``IndexError`` an unconditional wrap would
produce out of empty frames.

These tests supply their own (β) hook. §5.2a's self-derivation ban is worded over the LABEL
association (whose only bound producer lives in ``graphed-histogram``); what is asserted here is the
wrap, the keying and the frame tie-break, none of which the hook computes — the frames are COPIED
off the ``CompiledGraph`` ``compile_ir`` built.
"""

from __future__ import annotations

import m49_analyses as m49
import pytest

import graphed.debug as gd
from graphed.execute import compile_ir


def _error(**overrides: object) -> gd.StageError:
    fields: dict[str, object] = {
        "op": "mul",
        "frames": (gd.SourceFrame("analysis.py", 7, "run", "x * 2.0"),),
        "input_forms": ("f",),
        "partition": "toy://list:0",
        "cause_type": "PoisonError",
        "cause_message": m49.POISON_MESSAGE,
        "opt_level": 1,
    }
    fields.update(overrides)
    return gd.StageError(**fields)  # type: ignore[arg-type]


def test_variation_participates_in_equality_and_hash() -> None:
    up, down = _error(variation="jes_up"), _error(variation="jes_down")
    assert up != down
    assert hash(up) != hash(down)
    assert up == _error(variation="jes_up")
    assert hash(up) == hash(_error(variation="jes_up"))
    # control: a field already in the hand-written hash tuple separates two errors the same way,
    # so a failure above means `variation` is missing from it rather than that hashing is broken.
    assert _error(op="add") != _error(op="mul")
    assert hash(_error(op="add")) != hash(_error(op="mul"))


def test_the_empty_string_is_the_default_and_a_label_reaches_the_summary() -> None:
    assert _error().variation == ""
    assert "jes_up" in _error(variation="jes_up").summary()


def test_a_worker_failure_with_no_label_channel_reraises_the_original() -> None:
    _session, _out, _poisoned, plan = m49.plan_for(None)
    assert plan.process.variation_labels is None
    with pytest.raises(m49.PoisonError) as excinfo:
        plan.process(plan.tasks[0].partition, m49.Resources())
    assert str(excinfo.value) == m49.POISON_MESSAGE


def test_a_failure_whose_key_has_no_entry_reraises_the_original() -> None:
    _session, _out, _poisoned, plan = m49.plan_for(m49.hook_skipping_the_failing_key)
    assert plan.process.variation_labels, "the payload must still cover the non-failing keys"
    with pytest.raises(m49.PoisonError) as excinfo:
        plan.process(plan.tasks[0].partition, m49.Resources())
    assert str(excinfo.value) == m49.POISON_MESSAGE


def test_an_attributed_failure_carries_its_own_keys_label_and_the_users_line() -> None:
    session, _out, poisoned, plan = m49.plan_for(m49.hook_labelling_every_key)
    with pytest.raises(gd.StageError) as excinfo:
        plan.process(plan.tasks[0].partition, m49.Resources())
    err = excinfo.value
    # OTHER_LABEL sits on every non-failing key, so reporting "an" entry rather than the failing
    # key's is red here.
    assert err.variation == m49.LABEL
    assert err.cause_type == "PoisonError"
    assert err.cause_message == m49.POISON_MESSAGE
    assert err.partition
    assert err.user_frame.filename.endswith("m49_analyses.py")
    assert err.user_frame.lineno == session.sourcemap()[poisoned.node_id]["lineno"]


def test_two_lines_merged_onto_one_key_report_the_lowest_record_ids_frame() -> None:
    session, out, poisoned, plan = m49.plan_for(m49.hook_labelling_every_key)
    lines = session.sourcemap()
    low, high = sorted((poisoned.node_id, out.node_id))
    assert lines[low]["lineno"] != lines[high]["lineno"]
    # the merge itself: three recorded nodes, two reduced ones, so exactly one pair shares a key
    compiled = compile_ir(session, out)
    assert len(lines) == 3
    assert len(m49.artifact_frames(compiled)) == 2

    with pytest.raises(gd.StageError) as excinfo:
        plan.process(plan.tasks[0].partition, m49.Resources())
    assert excinfo.value.user_frame.lineno == lines[low]["lineno"]
    assert excinfo.value.user_frame.lineno != lines[high]["lineno"]
