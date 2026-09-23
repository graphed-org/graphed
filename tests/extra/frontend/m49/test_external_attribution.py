"""§8.2(ii)/(iii) at the EXTERNAL dispatch point, and the GraphedError carve-out.

§4.1 makes an External payload the canonical carrier of a weight variation, and graphed-histogram's
fills and weight guards are themselves `record_external` nodes — so a RAW External failure attributes
like any other node in the top-level dispatch loop. A `GraphedError` does not: §8.2(ii) re-raises it
untouched on EVERY arm regardless of entry, because it is already an attributed error and §6.1d's
blame parity (the plan path re-raises the guard's message verbatim) binds it.

An UNVARIED program attributes too: a raw failure at any key with a §8.2(i) frame becomes a
`StageError` with an empty variation; the label channel only adds the label, and a frameless key
keeps re-raising the original.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
from backends import ListBackend, ListForm, from_list

import graphed.core
from graphed import Session
from graphed.aggregate import _PartitionReduce
from graphed.core import PayloadDescriptor
from graphed.debug.errors import StageError
from graphed.errors import GraphedError
from graphed.execute import CompiledGraph, Frame, Key, OnFailure, compile_ir, evaluate_ir

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


def _compiled() -> tuple[CompiledGraph, int]:
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


def _worker_hook(
    entries: tuple[object, ...] | None, frames: tuple[tuple[Key, Frame], ...] = ()
) -> OnFailure | None:
    """The shipped §8.2(ii) wrap. `_attribute` reads only the label and frame channels, so the rest
    of the closure's fields are inert here."""
    return _PartitionReduce(
        ir=b"",
        source_name="x",
        backend_factory=ListBackend,
        reader=None,  # type: ignore[arg-type]  # `_attribute` never reads it
        columns=None,
        externals=(),
        reduce=list,
        variation_labels=entries,
        frames=frames,
    )._attribute("toy://list:0")


def _run(evaluator: Callable[..., object], on_failure: OnFailure | None) -> list[object]:
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


def test_an_unvaried_raw_failure_becomes_a_stage_error_at_the_users_line() -> None:
    compiled, output = _compiled()
    frames = dict(compiled.correspondence.frames)
    assert (output, None) in frames  # the witness: the failing key HAS a frame to attribute to
    with pytest.raises(StageError) as excinfo:
        _run(_boom, _worker_hook(None, compiled.correspondence.frames))
    assert excinfo.value.variation == ""
    assert excinfo.value.cause_type == "Boom"
    assert excinfo.value.cause_message == "the payload failed"
    assert excinfo.value.user_frame.lineno == frames[output, None][1]
    assert excinfo.value.partition == "toy://list:0"


def test_a_frameless_key_still_re_raises_the_original() -> None:
    compiled, output = _compiled()
    others = tuple(e for e in compiled.correspondence.frames if e[0] != (output, None))
    assert others  # the channel is live, it just lacks the failing key
    with pytest.raises(Boom):
        _run(_boom, _worker_hook(None, others))


def test_a_labelled_entry_still_wins_over_the_bare_frame() -> None:
    compiled, output = _compiled()
    entries = (((output, None), (("jes_up",), FRAME)),)
    with pytest.raises(StageError) as excinfo:
        _run(_boom, _worker_hook(entries, compiled.correspondence.frames))
    assert excinfo.value.variation == "jes_up"
    assert excinfo.value.user_frame.lineno == FRAME[1]


def test_a_graphed_error_passes_verbatim_at_a_framed_key() -> None:
    compiled, _ = _compiled()
    with pytest.raises(GraphedError) as excinfo:
        _run(_refuse, _worker_hook(None, compiled.correspondence.frames))
    assert not isinstance(excinfo.value, StageError)
