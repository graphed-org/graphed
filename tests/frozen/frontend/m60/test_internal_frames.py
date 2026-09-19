"""m60 integ-m60-V — a wrapping library's frames are not the user's line.

`capture()` keeps the first frame whose module is not `graphed*`; for an analysis that calls a
wrapping library, that frame is the LIBRARY's, so every node it records points at library source.
`register_internal(prefix)` adds the library to the skipped set, matching whole dotted components.

Registration is process-global and today's module offers no inverse, so every leg lives here and
`"m60_lib"` is the only prefix this suite ever registers — by the dotted-component rule under
test it cannot reach `m60_libx`, and the frozen frontend tree runs one process per milestone dir.
"""

from __future__ import annotations

import inspect
import os
import threading
from types import FrameType
from typing import Any

import m60_lib
import m60_lib.sub
import m60_libx
import pytest
from m60_seams import toy_session

import graphed.debug as gd
import graphed.provenance


def _here() -> FrameType:
    frame = inspect.currentframe()
    assert frame is not None and frame.f_back is not None
    return frame.f_back


def _at(prov: Any) -> tuple[str, str]:
    return (os.path.basename(prov.filename), prov.function)


def test_an_unregistered_module_and_the_built_in_graphed_rule_keep_their_own_lines() -> None:
    session, x = toy_session()

    helper = m60_libx.select(x)
    direct = session.record_op("direct", [x])

    assert _at(session.provenance(helper)) == ("m60_libx.py", "select")
    assert _at(session.provenance(direct)) == (os.path.basename(__file__), _here().f_code.co_name)


def test_a_registered_prefix_matches_whole_dotted_components() -> None:
    graphed.provenance.register_internal("m60_lib")
    session, x = toy_session()
    mine = _here().f_code.co_name

    line = _here().f_lineno + 1
    selected = m60_lib.select(x)
    scaled = m60_lib.sub.scale(selected)
    unregistered = m60_libx.select(scaled)

    assert _at(session.provenance(selected)) == (os.path.basename(__file__), mine)
    assert session.provenance(selected).lineno == line
    assert "m60_lib.select" in session.provenance(selected).source
    assert _at(session.provenance(scaled)) == (os.path.basename(__file__), mine)
    assert _at(session.provenance(unregistered)) == ("m60_libx.py", "select")


def test_registration_is_idempotent_and_survives_concurrent_registrars() -> None:
    graphed.provenance.register_internal("m60_lib")
    graphed.provenance.register_internal("m60_lib")
    session, x = toy_session()
    assert _at(session.provenance(m60_lib.select(x))) == (
        os.path.basename(__file__),
        _here().f_code.co_name,
    )

    seen: list[tuple[str, str]] = []
    lock = threading.Lock()

    def worker() -> None:
        graphed.provenance.register_internal("m60_lib")
        own, value = toy_session()
        where = _at(own.provenance(m60_lib.sub.scale(value)))
        with lock:
            seen.append(where)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert seen == [(os.path.basename(__file__), "worker")] * 8


def test_a_registered_librarys_stage_error_points_at_the_same_user_line() -> None:
    graphed.provenance.register_internal("m60_lib")
    session, x = toy_session()
    mine = _here().f_code.co_name

    line = _here().f_lineno + 1
    burst = m60_lib.sub.failing(x)

    with pytest.raises(gd.StageError) as info:
        gd.run(session, burst, opt_level=0)
    frame = info.value.user_frame
    assert (os.path.basename(frame.filename), frame.lineno, frame.function) == (
        os.path.basename(__file__),
        line,
        mine,
    )
    assert info.value.cause_type == "ValueError"
