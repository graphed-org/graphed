"""A fake Triton transport for the frozen M26 suite (importable by name, like a real one).

Mirrors the ``tritonclient.http`` calling surface the plugin uses — ``client.infer(model_name,
inputs, outputs=...)`` returning a result with ``.as_numpy(name)`` — so the plugin code under
test is EXACTLY the code a live server would exercise. The fake "server" computes what a Triton
instance hosting the descriptor's model would: ``sigmoid(weight * x0 + bias)``.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

SERVERS: dict[str, FakeTritonClient] = {}


class _Result:
    def __init__(self, outputs: dict[str, np.ndarray]) -> None:
        self._outputs = outputs

    def as_numpy(self, name: str) -> np.ndarray:
        return self._outputs[name]


class FakeTritonClient:
    def __init__(self, descriptor: dict[str, Any]) -> None:
        self.descriptor = descriptor
        self.closed = False
        self.infer_calls = 0

    def infer(self, model_name: str, inputs: list[Any], outputs: list[Any] | None = None) -> _Result:
        assert model_name == self.descriptor["model"], "client asked for a model this server does not host"
        self.infer_calls += 1
        x = inputs[0]._raw_data  # the numpy the plugin set via set_data_from_numpy
        w, b = float(self.descriptor["weights"]["w"]), float(self.descriptor["weights"]["b"])
        y = 1.0 / (1.0 + np.exp(-(w * x[:, 0] + b)))
        name = outputs[0].name() if outputs else "y"
        return _Result({name: y.astype("float32")})

    def close(self) -> None:
        self.closed = True


class _FakeInferInput:
    """Stands in for tritonclient.http.InferInput."""

    def __init__(self, name: str, shape: list[int], datatype: str) -> None:
        self._name, self._shape, self._datatype = name, shape, datatype
        self._raw_data: np.ndarray | None = None

    def set_data_from_numpy(self, arr: Any) -> None:
        self._raw_data = np.asarray(arr)


class _FakeRequestedOutput:
    def __init__(self, name: str) -> None:
        self._name = name

    def name(self) -> str:
        return self._name


def serve(url: str, payload: bytes) -> FakeTritonClient:
    """Start a fake server at ``url`` hosting the descriptor's model."""
    client = FakeTritonClient(json.loads(payload.decode()))
    SERVERS[url] = client
    return client


def transport(params: Any) -> FakeTritonClient:
    """The injectable transport factory: resolve the environment's connection for params['url']."""
    return SERVERS[str(params["url"])]


# the plugin asks the transport module for request classes, so fakes and tritonclient interchange
InferInput = _FakeInferInput
InferRequestedOutput = _FakeRequestedOutput
