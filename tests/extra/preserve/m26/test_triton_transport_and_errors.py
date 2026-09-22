"""Coverage ratchet: ``triton_external.py`` branches the frozen suites route around.

Every frozen Triton test passes ``params["transport"]`` (a fake, since no real server runs
locally), so the DEFAULT ``tritonclient.http`` transport path is never taken; and every frozen
call template is the NAMED-dict form, so the "non-named template" rejection never fires.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from graphed.preserve import PreserveError
from graphed.preserve.externals.triton_external import (
    _transport_module_and_factory,
    eval_triton,
    triton_http_transport,
)


def test_default_transport_resolves_tritonclient_http(monkeypatch: pytest.MonkeyPatch) -> None:
    connect_urls: list[str] = []

    class FakeInferenceServerClient:
        def __init__(self, url: str) -> None:
            connect_urls.append(url)
            self.url = url

    fake_pkg = types.ModuleType("tritonclient")
    fake_http = types.ModuleType("tritonclient.http")
    fake_http.InferenceServerClient = FakeInferenceServerClient  # type: ignore[attr-defined]
    fake_pkg.http = fake_http  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "tritonclient", fake_pkg)
    monkeypatch.setitem(sys.modules, "tritonclient.http", fake_http)

    module, factory = _transport_module_and_factory({})  # no "transport" key -> the default
    assert factory is triton_http_transport
    assert module is fake_http

    client = factory({"url": "http://triton:8000"})
    assert isinstance(client, FakeInferenceServerClient)
    assert connect_urls == ["http://triton:8000"]


def test_eval_triton_rejects_a_non_named_call_template() -> None:
    # triton inputs are always named (InferInput per name); a positional/group template must be
    # rejected before touching the (unused, so None is fine) resource.
    with pytest.raises(PreserveError, match="NAMED"):
        eval_triton(None, {"model": "m", "args": ["$0"]}, [np.array([1.0])])
