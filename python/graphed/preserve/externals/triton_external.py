"""The ``triton_model`` plugin (M26): a REMOTE served model — the payload preserves the served
identity (canonical JSON descriptor) and the node names the service; the connection is environment."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

from ...services import split_endpoint
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
# The connection is environment: a node names a declared service (params["service"], its endpoint
# bound per run by graphed.services.bind_services, never recorded) or a literal params["url"].
# An importable params["transport"] ("module:attr") always wins; otherwise the endpoint's scheme
# picks tritonclient.http or tritonclient.grpc, and a url without "://" stays tritonclient.http.
# The transport's module also supplies the InferInput/InferRequestedOutput request classes, so
# fakes and tritonclient interchange without touching plugin code.
_WIRES = {"http": "http", "https": "http", "grpc": "grpc", "grpcs": "grpc"}


def triton_content_hash(payload: bytes) -> str:
    return _canonical_json_hash(b"triton-descriptor-v1", payload)


def _check_triton_params(params: Mapping[str, Any]) -> None:
    if ("url" in params) == ("service" in params):
        raise PreserveError("a Triton node names exactly one of params['url'] or params['service']")


def triton_http_transport(params: Mapping[str, Any]) -> Any:
    """A tritonclient HTTP connection to params['url'], passed as given."""
    import tritonclient.http as triton_http  # noqa: PLC0415

    return triton_http.InferenceServerClient(url=str(params["url"]))


def _wire(url: str) -> tuple[str, str, bool]:
    """``(tritonclient module, "host:port", ssl)`` for a ``scheme://host:port`` url."""
    try:
        scheme, address = split_endpoint(url)
    except ValueError as err:
        raise PreserveError(f"Triton cannot reach {url!r}: {err}") from err
    if scheme not in _WIRES:
        raise PreserveError(f"Triton speaks {', '.join(_WIRES)}, not {scheme!r} ({url!r})")
    return f"tritonclient.{_WIRES[scheme]}", address, scheme.endswith("s")


def triton_transport(params: Mapping[str, Any]) -> Any:
    """A tritonclient connection chosen by params['url']'s scheme, given the bare host:port."""
    module, address, ssl = _wire(str(params["url"]))
    return importlib.import_module(module).InferenceServerClient(url=address, ssl=ssl)


def _transport_module_and_factory(params: Mapping[str, Any]) -> tuple[Any, Any]:
    ref = str(params.get("transport", ""))
    if ref:
        module_name, _, attr = ref.partition(":")
        module = importlib.import_module(module_name)
        return module, getattr(module, attr)
    url = str(params.get("url", ""))
    if "://" not in url:
        import tritonclient.http as triton_http  # noqa: PLC0415

        return triton_http, triton_http_transport
    return importlib.import_module(_wire(url)[0]), triton_transport


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
    check_params=_check_triton_params,
    load_params=("url", "transport"),
)
