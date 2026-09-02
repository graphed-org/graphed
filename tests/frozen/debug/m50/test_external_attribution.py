"""§8.2(ii) at the EXTERNAL dispatch point (Brief C, m50 carryover over shipped m49 source).

`evaluate_ir` attributes at three dispatch sites; the External payload's evaluator is the third and
no frozen anchor reaches it. A failure raised INSIDE an External evaluator, at a key the worker's
label channel has an ENTRY for, must become a LABELLED `StageError`; a key with no entry re-raises
the original untouched. Freezes existing behaviour — the mutant is dropping attribution at the
External arm, which makes the labelled anchor go red (raw `Boom` instead of `StageError`).
"""

from __future__ import annotations

import pytest
from backends import ListBackend, ListForm, from_list

import graphed.core
from graphed import Session
from graphed.aggregate import _PartitionReduce
from graphed.core import PayloadDescriptor
from graphed.debug.errors import StageError
from graphed.execute import Key, compile_ir, evaluate_ir

CHASH = "probe-payload-hash"
KIND = "probe"
FRAME = ("analysis.py", 42, "run", "h.fill(ev.x, weight=w)")


class Boom(RuntimeError):
    pass


def _boom(*ins: object) -> object:
    raise Boom("the payload failed")


def _compiled_external() -> tuple[object, int]:
    """One External node marked as the output, over a list source; returns the reduced output id."""
    session = Session(ListBackend())
    x = from_list(session, "x", [1.0, 2.0])
    node = session.record_external(
        KIND,
        _boom,
        [x],
        descriptor=PayloadDescriptor(
            kind=KIND,
            content_hash=CHASH,
            framework="python",
            version="1",
            io_schema="opaque->opaque",
        ),
        form=ListForm("float"),
    )
    compiled = compile_ir(session, node)
    (output,) = graphed.core.GraphStore.deserialize(bytes(compiled.ir)).outputs()
    return compiled, output


def _worker_hook(entries: tuple[object, ...]) -> object:
    """The shipped §8.2(ii) worker wrap. `_attribute` reads only the label channel; the closure's
    other fields are inert here."""
    return _PartitionReduce(
        ir=b"",
        source_name="x",
        backend_factory=ListBackend,
        reader=None,
        columns=None,
        externals=(),
        reduce=list,
        variation_labels=entries,
    )._attribute("toy://list:0")


def _run(on_failure: object) -> list[object]:
    compiled, _ = _compiled_external()
    return evaluate_ir(
        compiled,
        ListBackend(),
        {"x": [1.0, 2.0]},
        externals={CHASH: _boom},
        on_failure=on_failure,
    )


def test_external_failure_at_an_entried_key_becomes_a_labelled_stage_error() -> None:
    _, output = _compiled_external()
    key: Key = (output, None)
    hook = _worker_hook(((key, (("jes_up",), FRAME)),))
    with pytest.raises(StageError) as excinfo:
        _run(hook)
    assert excinfo.value.variation == "jes_up"
    assert excinfo.value.cause_message == "the payload failed"
    assert excinfo.value.user_frame.lineno == 42
    assert excinfo.value.op == f"external:{KIND}"


def test_external_failure_with_no_entry_for_its_key_re_raises_untouched() -> None:
    # a live label channel, but keyed on a DIFFERENT node — the External arm's key is absent, so
    # `_attribute` declines and the raw payload failure propagates unwrapped.
    _, output = _compiled_external()
    other_key: Key = (output + 999, None)
    hook = _worker_hook(((other_key, (("jes_up",), FRAME)),))
    with pytest.raises(Boom):
        _run(hook)
