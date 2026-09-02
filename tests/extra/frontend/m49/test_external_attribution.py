"""§8.2(ii)/(iii) at the EXTERNAL dispatch point, and the GraphedError carve-out.

§4.1 makes an External payload the canonical carrier of a weight variation, and graphed-histogram's
fills and weight guards are themselves `record_external` nodes — so a RAW External failure attributes
like any other node in the top-level dispatch loop. A `GraphedError` does not: §8.2(ii) re-raises it
untouched on EVERY arm regardless of entry, because it is already an attributed error and §6.1d's
blame parity (the plan path re-raises the guard's message verbatim) binds it.
"""

from __future__ import annotations

import pytest
from backends import ListBackend, ListForm, from_list

import graphed.core
from graphed import Session
from graphed.aggregate import _PartitionReduce
from graphed.core import PayloadDescriptor
from graphed.debug.errors import StageError
from graphed.errors import GraphedError
from graphed.execute import Key, compile_ir, evaluate_ir

CHASH = "probe-payload-hash"
KIND = "probe"
REFUSAL = "weight[0] is not at this fill's row space"
FRAME = ("analysis.py", 42, "run", "h.fill(ev.x, weight=w)")


class Boom(RuntimeError):
    pass


def _boom(*ins: object) -> object:
    raise Boom("the payload failed")


def _refuse(*ins: object) -> object:
    raise GraphedError(REFUSAL)


def _compiled() -> tuple[object, int]:
    """One External node, marked as the output, over a list source."""
    session = Session(ListBackend())
    x = from_list(session, "x", [1.0, 2.0])
    node = session.record_external(
        "probe",
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
    """The shipped §8.2(ii) wrap. `_attribute` reads only the label channel, so the rest of the
    closure's fields are inert here."""
    return _PartitionReduce(
        ir=b"",
        source_name="x",
        backend_factory=ListBackend,
        reader=None,  # `_attribute` never reads it
        columns=None,
        externals=(),
        reduce=list,
        variation_labels=entries,
    )._attribute("toy://list:0")


def _run(evaluator: object, on_failure: object) -> list[object]:
    compiled, _ = _compiled()
    return evaluate_ir(
        compiled,
        ListBackend(),
        {"x": [1.0, 2.0]},
        externals={CHASH: evaluator},
        on_failure=on_failure,
    )


def test_a_raw_external_failure_reaches_the_attribution_hook_keyed_by_its_node() -> None:
    _, output = _compiled()
    seen: list[tuple[Key, str]] = []

    def attribute(key: Key, op: str, ins: list[object], exc: BaseException) -> BaseException:
        seen.append((key, op))
        return ValueError(f"jes_up at {key}")

    with pytest.raises(ValueError, match=rf"jes_up at \({output}, None\)"):
        _run(_boom, attribute)
    # an External carries no `name` in the IR — its identity is the descriptor, so the payload
    # kind is what the attributed failure can name
    assert seen == [((output, None), f"external:{KIND}")]


def test_an_external_failure_the_hook_declines_re_raises_untouched() -> None:
    with pytest.raises(Boom):
        _run(_boom, lambda *_: None)


def test_a_raw_external_failure_at_an_entried_key_becomes_a_labelled_stage_error() -> None:
    _, output = _compiled()
    hook = _worker_hook((((output, None), (("jes_up",), FRAME)),))
    with pytest.raises(StageError) as excinfo:
        _run(_boom, hook)
    assert excinfo.value.variation == "jes_up"
    assert excinfo.value.cause_message == "the payload failed"
    assert excinfo.value.user_frame.lineno == 42


def test_a_graphed_error_passes_verbatim_even_at_an_entried_key() -> None:
    """§8.2(ii)'s carve-out — the same key, the same hook, the same entry as the test above; only
    the exception class differs, and it is re-raised untouched."""
    _, output = _compiled()
    hook = _worker_hook((((output, None), (("jes_up",), FRAME)),))
    with pytest.raises(GraphedError) as excinfo:
        _run(_refuse, hook)
    assert str(excinfo.value) == REFUSAL
    assert not isinstance(excinfo.value, StageError)
