"""The ``triton_model`` plugin (M26): a REMOTE served model — the payload preserves the served
identity (canonical JSON descriptor); the connection is environment, via an injectable transport."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..errors import PreserveError
from ._base import ExternalPlugin
from ._helpers import (
    _as_event_array,
    _canonical_json_hash,
    _stack_feature_columns,
    ml_matrix,
    parse_call_template,
)


# ---- NVIDIA Triton (remote inference) -------------------------------------------------------------
# The payload is the SERVED MODEL'S IDENTITY (a canonical JSON descriptor: model name/version,
# io names, weight digests) — the bundle preserves what was called, but cannot bottle the server.
# The connection is environment, resolved per worker through an importable transport factory
# (params["transport"] = "module:attr"; default: tritonclient.http). The factory's module also
# supplies the InferInput/InferRequestedOutput request classes, so fakes and tritonclient
# interchange without touching plugin code.
def triton_content_hash(payload: bytes) -> str:
    return _canonical_json_hash(b"triton-descriptor-v1", payload)


def triton_http_transport(params: Mapping[str, Any]) -> Any:
    """The default transport: a tritonclient HTTP connection to params['url']."""
    import tritonclient.http as triton_http  # noqa: PLC0415

    return triton_http.InferenceServerClient(url=str(params["url"]))


def _transport_module_and_factory(params: Mapping[str, Any]) -> tuple[Any, Any]:
    import importlib  # noqa: PLC0415

    ref = str(params.get("transport", "")) or ""
    if not ref:
        import tritonclient.http as triton_http  # noqa: PLC0415

        return triton_http, triton_http_transport
    module_name, _, attr = ref.partition(":")
    module = importlib.import_module(module_name)
    return module, getattr(module, attr)


class _TritonResource:
    """A live connection plus the transport module supplying request classes."""

    def __init__(self, client: Any, module: Any) -> None:
        self.client = client
        self.module = module


def load_triton(payload: bytes, params: Mapping[str, Any]) -> Any:
    module, factory = _transport_module_and_factory(params)
    return _TritonResource(factory(params), module)


def eval_triton(resource: Any, params: Mapping[str, Any], inputs: list[Any]) -> Any:
    output_name = str(params.get("output_name", "y"))
    template = parse_call_template(params, len(inputs), allow_kwargs=False)
    if template is None:  # legacy: one named input
        named = {str(params.get("input_name", "x")): _stack_feature_columns(inputs)}
    else:
        args, _ = template
        if not (len(args) == 1 and args[0][0] == "named"):
            raise PreserveError("triton inputs are NAMED: use the dict form of params['args']")
        named = {name: ml_matrix(entry, inputs) for name, entry in args[0][1].items()}
    requests = []
    for name, x in named.items():
        request = resource.module.InferInput(name, list(x.shape), "FP32")
        request.set_data_from_numpy(x)
        requests.append(request)
    wanted = resource.module.InferRequestedOutput(output_name)
    result = resource.client.infer(str(params["model"]), requests, outputs=[wanted])
    return _as_event_array(result.as_numpy(output_name))


def close_triton(resource: Any) -> None:
    resource.client.close()  # release the connection at end of run


def _triton_samples() -> list[bytes]:
    return [
        b'{"model": "scorer", "version": "1", "weights": {"w": 0.5, "b": 0.0}}',
        b'{"model": "scorer", "version": "2", "weights": {"w": 0.8, "b": 0.2}}',
    ]


TRITON_PLUGIN = ExternalPlugin(
    kind="triton_model",
    content_hash=triton_content_hash,
    evaluate=eval_triton,
    samples=_triton_samples,
    load=load_triton,
    close=close_triton,
    framework="tritonclient",
)
