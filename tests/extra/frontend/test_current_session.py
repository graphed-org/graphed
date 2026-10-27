"""Where array-less reads record: the innermost open scope, else a lazily made default."""

from __future__ import annotations

import asyncio
import contextvars
import gc
import sys
import threading
import weakref
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

import graphed
from graphed import Session, current_session, new_session, set_default_backend, set_session
from graphed.numpy import NumpyBackend, from_array


@pytest.fixture(autouse=True)
def _no_factory() -> Any:
    previous = set_default_backend(None)
    yield
    set_default_backend(previous)


def _in_thread(fn: Callable[[], Any], timeout: float = 5.0) -> list[Any]:
    out: list[Any] = []
    thread = threading.Thread(target=lambda: out.append(fn()), daemon=True)
    thread.start()
    thread.join(timeout)
    return out


def test_a_bare_call_makes_one_awkward_default() -> None:
    pytest.importorskip("awkward")
    from graphed.awkward import AwkwardBackend  # noqa: PLC0415

    first = current_session()
    assert current_session() is first
    assert isinstance(first.backend, AwkwardBackend)


def test_the_first_backend_makes_the_default_and_later_ones_do_not_split_it() -> None:
    pytest.importorskip("awkward")
    from graphed.awkward import AwkwardBackend  # noqa: PLC0415

    made = current_session(NumpyBackend())
    assert isinstance(made.backend, NumpyBackend)
    assert current_session(AwkwardBackend()) is made


def test_every_thread_and_context_sees_one_default() -> None:
    held = current_session(NumpyBackend())
    assert _in_thread(current_session) == [held]
    assert contextvars.Context().run(current_session) is held


def test_scopes_nest_over_the_default() -> None:
    default = current_session(NumpyBackend())
    s, t = Session(NumpyBackend()), Session(NumpyBackend())
    with s as entered:
        assert entered is s
        with t:
            assert current_session() is t
        assert current_session() is s
    assert current_session() is default


def test_set_session_inside_a_scope_replaces_only_the_default() -> None:
    s, u = Session(NumpyBackend()), Session(NumpyBackend())
    default = current_session(NumpyBackend())
    with s:
        assert set_session(u) is default
        assert current_session() is s
    assert current_session() is u


def test_new_session_installs_a_backend_or_clears_the_default() -> None:
    backend = NumpyBackend()
    new_session(backend)
    made = current_session()
    assert made.backend is backend
    new_session()
    numpy_default = current_session(NumpyBackend())
    assert numpy_default is not made
    assert isinstance(numpy_default.backend, NumpyBackend)


def test_new_session_releases_the_old_default() -> None:
    old = current_session(NumpyBackend())
    x = from_array(old, "x", np.arange(3.0)) + 1.0
    try:
        from_array(old, "x", np.arange(2.0))
    except ValueError as exc:  # a caught error's traceback holds the frames that held `old`
        del exc
    ref = weakref.ref(old)
    del old, x
    new_session(NumpyBackend())
    gc.collect()
    assert ref() is None


def test_racing_threads_make_one_default() -> None:
    barrier = threading.Barrier(32)
    seen: list[Session] = []

    def race() -> None:
        barrier.wait()
        seen.append(current_session(NumpyBackend()))

    threads = [threading.Thread(target=race) for _ in range(32)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(seen) == 32
    assert len({id(s) for s in seen}) == 1


def test_a_misnested_exit_raises_and_leaves_the_stack() -> None:
    s, t = Session(NumpyBackend()), Session(NumpyBackend())
    s.__enter__()
    t.__enter__()
    with pytest.raises(RuntimeError, match="exited out of order"):
        s.__exit__(None, None, None)
    assert current_session() is t
    t.__exit__(None, None, None)
    s.__exit__(None, None, None)


def test_a_misnested_exit_keeps_the_propagating_error_as_context() -> None:
    s, t = Session(NumpyBackend()), Session(NumpyBackend())
    t.__enter__()
    with pytest.raises(RuntimeError, match="exited out of order") as info, s:
        s.__exit__(None, None, None)  # closes s early: the with's own exit then misnests
        raise KeyError("original")
    assert isinstance(info.value.__context__, KeyError)
    t.__exit__(None, None, None)


def test_asyncio_scopes_reach_children_not_siblings() -> None:
    default = current_session(NumpyBackend())
    s = Session(NumpyBackend())

    async def main() -> tuple[Session, Session]:
        entered = asyncio.Event()
        release = asyncio.Event()

        async def scoped() -> Session:
            with s:
                inner = await asyncio.create_task(_current())
                entered.set()
                await release.wait()
            return inner

        async def sibling() -> Session:
            await entered.wait()
            seen = current_session()
            release.set()
            return seen

        return tuple(await asyncio.gather(scoped(), sibling()))  # type: ignore[return-value]

    async def _current() -> Session:
        return current_session()

    child_saw, sibling_saw = asyncio.run(main())
    assert child_saw is s
    assert sibling_saw is default


def test_a_thread_inherits_a_scope_exactly_when_the_build_says_so() -> None:
    default = current_session(NumpyBackend())
    s = Session(NumpyBackend())
    with s:
        seen = _in_thread(current_session)
    inherits = bool(getattr(sys.flags, "thread_inherit_context", 0))
    assert seen == [s if inherits else default]


def test_arrays_recorded_in_a_scope_outlive_it() -> None:
    s = Session(NumpyBackend())
    with s:
        y = from_array(s, "x", np.arange(3.0)) * 2.0
        count = s.node_count()
    assert s.node_count() == count
    np.testing.assert_array_equal(graphed.materialize(y), np.arange(3.0) * 2.0)


def test_a_factory_makes_later_defaults_only() -> None:
    held = current_session(NumpyBackend())
    assert set_default_backend(NumpyBackend) is None
    assert current_session() is held
    new_session()
    assert set_default_backend(None) is NumpyBackend
    made = current_session()  # no factory now: only an eager install gives numpy
    assert made is not held
    assert isinstance(made.backend, NumpyBackend)
    set_default_backend(NumpyBackend)
    set_session(None)
    assert isinstance(current_session().backend, NumpyBackend)


def test_a_factory_may_read_the_current_session() -> None:
    def factory() -> NumpyBackend:
        current_session(NumpyBackend())
        return NumpyBackend()

    set_default_backend(factory)
    assert len(_in_thread(current_session)) == 1


def test_a_factory_set_here_is_used_by_a_new_thread_and_an_empty_context() -> None:
    set_default_backend(NumpyBackend)
    (made,) = _in_thread(current_session)
    assert isinstance(made.backend, NumpyBackend)
    set_session(None)
    assert isinstance(contextvars.Context().run(current_session).backend, NumpyBackend)


def test_without_awkward_or_a_factory_a_bare_call_names_both(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "graphed.awkward", None)
    with pytest.raises(ImportError, match=r"graphed\[awkward\].*set_default_backend"):
        current_session()


def _two_sessions() -> tuple[Any, Any]:
    return (
        from_array(Session(NumpyBackend()), "a", np.arange(3.0)),
        from_array(Session(NumpyBackend()), "b", np.arange(3.0)),
    )


@pytest.mark.parametrize(
    ("refused", "prefix", "pointer"),
    [
        (lambda a, b: graphed.compile_ir(a.session, b), "different Session", "with array.session"),
        (lambda a, b: graphed.join(a, b, on=["k"]), "same Session", "with left.session"),
        (lambda a, b: graphed.apply(np.add, a, b), "one Session", "with operand.session"),
        (
            lambda a, b: graphed.numpy.apply_gufunc(np.add, "(),()->()", a, b, output_dtype=float),
            "one Session",
            "with operand.session",
        ),
        (
            lambda a, b: graphed.aggregate_plan(
                a, b, reduce=list, combine=lambda x, y: [*x, *y], empty=list
            ),
            "collate one plan per Session",
            "with output.session",
        ),
    ],
)
def test_a_cross_session_refusal_points_at_a_scope(
    refused: Callable[[Any, Any], object], prefix: str, pointer: str
) -> None:
    a, b = _two_sessions()
    with pytest.raises(Exception, match=prefix) as info:
        refused(a, b)
    assert pointer in str(info.value)


def test_a_grouped_gak_join_refusal_points_at_a_scope() -> None:
    ak = pytest.importorskip("awkward")
    from graphed.awkward import AwkwardBackend, from_awkward, gak  # noqa: PLC0415

    left = from_awkward(Session(AwkwardBackend()), "left", ak.Array({"k": [1]}))
    right = from_awkward(Session(AwkwardBackend()), "right", ak.Array({"k": [1]}))
    with pytest.raises(graphed.GraphedError, match="same Session") as info:
        gak.join(left, right, on=["k"], grouped=True)
    assert "with left.session" in str(info.value)
