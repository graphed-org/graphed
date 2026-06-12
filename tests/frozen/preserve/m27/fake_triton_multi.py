"""A fake Triton transport serving a MULTI-INPUT model (M27).

Same calling surface as ``tritonclient.http`` (and the m26 fake): the served model computes
``sigmoid(sum_name dot(weights[name], x_name))`` from the descriptor — every named InferInput
participates, so a plugin that drops or misroutes an input cannot pass.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

SERVERS: dict[str, FakeMultiTritonClient] = {}


class _Result:
    def __init__(self, outputs: dict[str, np.ndarray]) -> None:
        self._outputs = outputs

    def as_numpy(self, name: str) -> np.ndarray:
        return self._outputs[name]


class FakeMultiTritonClient:
    def __init__(self, descriptor: dict[str, Any]) -> None:
        self.descriptor = descriptor
        self.closed = False
        self.seen_input_names: list[list[str]] = []

    def infer(self, model_name: str, inputs: list[Any], outputs: list[Any] | None = None) -> _Result:
        assert model_name == self.descriptor["model"]
        self.seen_input_names.append(sorted(i._name for i in inputs))
        z = None
        for inp in inputs:
            w = np.asarray(self.descriptor["weights"][inp._name], dtype="float64")
            contrib = inp._raw_data.astype("float64") @ w
            z = contrib if z is None else z + contrib
        y = 1.0 / (1.0 + np.exp(-z))
        name = outputs[0].name() if outputs else "y"
        return _Result({name: y.astype("float32")})

    def close(self) -> None:
        self.closed = True


class _FakeInferInput:
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


def serve(url: str, payload: bytes) -> FakeMultiTritonClient:
    client = FakeMultiTritonClient(json.loads(payload.decode()))
    SERVERS[url] = client
    return client


def transport(params: Any) -> FakeMultiTritonClient:
    return SERVERS[str(params["url"])]


InferInput = _FakeInferInput
InferRequestedOutput = _FakeRequestedOutput
