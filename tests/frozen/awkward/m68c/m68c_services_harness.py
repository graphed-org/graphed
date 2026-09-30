"""Shared builders for the frozen m68c awkward suite: in-memory partitioned sources, a local HTTP
stand-in for a scale-factor service, and the External that calls it on every evaluation.

Stage processes reach a runner as cloudpickle copies, so the spies are module state (classes and
functions here pickle by reference) rather than instance state."""

from __future__ import annotations

import json
import socketserver
import threading
import urllib.request
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import awkward as ak
import numpy as np

from graphed import Session
from graphed.awkward import AwkwardBackend, AwkwardForm, gak
from graphed.core import Partition, WorkerResources
from graphed.preserve import ExternalPlugin, record_external, sha256_bytes
from graphed.services import ServiceSpec

SF = 1.25
SPEC = ServiceSpec("sf", "http")
EXTRA = ServiceSpec("extra", "http")
#: every ``read_partition`` call, as ``(source name, partition uri)``
READS: list[tuple[str, str]] = []
#: every value a ``ResolvingReduce.resolve_services`` was handed
RESOLVED: list[Any] = []


class Chunks:
    """A ``PartitionedSource`` over in-memory "files"; each file splits into ``steps_per_file``
    blind partitions. ``columns`` is honoured, so a map stage that reads too few columns fails."""

    def __init__(self, name: str, *files: ak.Array) -> None:
        self.name = name
        self.files = files

    def partitions(self, steps_per_file: int) -> tuple[Partition, ...]:
        return tuple(
            Partition.blind(f"mem://{self.name}/{i}", "", s, steps_per_file)
            for i in range(len(self.files))
            for s in range(steps_per_file)
        )

    def read_partition(self, partition: Partition, columns: Sequence[str] | None, resources: WorkerResources) -> Any:
        READS.append((self.name, partition.uri))
        data = self.files[int(partition.uri.rsplit("/", 1)[1])]
        part = partition.resolve(len(data))
        block = data[part.entry_start : part.entry_stop]
        return block if columns is None else block[list(columns)]

    def whole(self) -> ak.Array:
        return ak.concatenate(self.files)

    def __call__(self) -> ak.Array:
        """The whole dataset, for ``Session.materialize`` (a lazy loader, as a parquet dataset is)."""
        return self.whole()


def source(session: Session, name: str, *files: ak.Array) -> Any:
    data = Chunks(name, *files)
    form = AwkwardForm(ak.Array(data.whole().layout.to_typetracer(forget_length=True)))
    return session.source(name, form=form, data=data)


def events_files() -> tuple[ak.Array, ...]:
    """One file; its first half has x < 25 only, so ``x > 25`` empties that partition. Every run
    routes to dest 0 of 2, so dest 1 of a two-dest join gets rows from the right side only."""
    return (
        ak.Array(
            {
                "run": np.array([1, 1, 3, 1, 1, 3, 3, 3], dtype=np.int64),
                "x": np.array([10.0, 20.0, 5.0, 15.0, 30.0, 40.0, 50.0, 60.0]),
            }
        ),
    )


def lumi_files() -> tuple[ak.Array, ...]:
    """Two files (so a join's build side has fewer map tasks than its probe side)."""
    return (
        ak.Array({"run": np.array([1, 2, 5], dtype=np.int64), "w": np.array([0.5, 1.5, 2.5])}),
        ak.Array({"run": np.array([2, 6, 7], dtype=np.int64), "w": np.array([3.5, 4.5, 5.5])}),
    )


def new_session() -> Session:
    s = Session(AwkwardBackend())
    s.declare_service(SPEC)
    s.declare_service(EXTRA)
    return s


def two_sources() -> tuple[Any, Any]:
    s = new_session()
    return source(s, "events", *events_files()), source(s, "lumi", *lumi_files())


def one_source() -> Any:
    return source(new_session(), "events", *events_files())


# ---- the scale-factor service and the External that calls it -------------------------------------
class _LookupFreeHTTPServer(ThreadingHTTPServer):
    """Binds without ``HTTPServer.server_bind``'s ``socket.getfqdn``, which stalls for tens of seconds on
    macOS CI runners."""

    def server_bind(self) -> None:
        socketserver.TCPServer.server_bind(self)
        host, port = self.server_address[:2]
        self.server_name = str(host)
        self.server_port = int(port)


class SFServer:
    """A local HTTP server answering ``{"sf": SF}``; ``requests`` counts the calls it served."""

    def __init__(self) -> None:
        self.requests = 0
        self._lock = threading.Lock()
        self._http = _LookupFreeHTTPServer(("127.0.0.1", 0), partial(_Handler, self))
        self._thread = threading.Thread(target=self._http.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._http.server_port}"

    def count(self) -> None:
        with self._lock:
            self.requests += 1


class _Handler(BaseHTTPRequestHandler):
    def __init__(self, server: SFServer, *args: Any) -> None:
        self._sf_server = server
        super().__init__(*args)

    def do_GET(self) -> None:
        self._sf_server.count()
        body = json.dumps({"sf": SF}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        pass


@contextmanager
def serve() -> Iterator[SFServer]:
    server = SFServer()
    server._thread.start()
    try:
        yield server
    finally:
        server._http.shutdown()
        server._http.server_close()
        server._thread.join()


def _load(payload: bytes, params: Any) -> str:
    return str(params["url"])


def _scale(url: str, params: Any, inputs: list[Any]) -> Any:
    with urllib.request.urlopen(f"{params['url']}/sf") as reply:
        sf = json.load(reply)["sf"]
    return inputs[0] * sf


def _samples() -> list[bytes]:
    return [b"sf-v1"]


SF_PLUGIN = ExternalPlugin(kind="m68c_sf", content_hash=sha256_bytes, load=_load, evaluate=_scale, samples=_samples)


def scaled(array: Any) -> Any:
    """``array * SF``, fetched from the ``sf`` service at run time."""
    return record_external(array.session, SF_PLUGIN, b"sf-v1", [array], params={"service": "sf"})


def corrected(events: Any) -> Any:
    """``events`` with ``x`` scaled by the service."""
    return gak.zip({"run": events.run, "x": scaled(events.x)}, depth_limit=1)


def by_hand(events: Any) -> Any:
    """What :func:`corrected` computes, with no service."""
    return gak.zip({"run": events.run, "x": events.x * SF}, depth_limit=1)


# ---- reduce/combine/empty that keep every gathered block ------------------------------------------
def keep(values: list[Any]) -> list[Any]:
    return [values[0]]


def concat(a: list[Any], b: list[Any]) -> list[Any]:
    return [*a, *b]


def nothing() -> list[Any]:
    return []


FOLD: dict[str, Any] = {"reduce": keep, "combine": concat, "empty": nothing}


class ResolvingReduce:
    """A ``reduce`` with a ``resolve_services`` hook that records what it is handed."""

    def __call__(self, values: list[Any]) -> list[Any]:
        return [values[0]]

    def resolve_services(self, value: Any) -> Any:
        RESOLVED.append(value)
        return ("resolved", value)


# ---- comparisons ---------------------------------------------------------------------------------
def rows(array: Any) -> list[str]:
    """The row multiset of ``array``: each row canonicalized, sorted."""
    return sorted(json.dumps(r, sort_keys=True) for r in ak.to_list(array))


def sublists(array: Any) -> list[str]:
    """The multiset of a grouped result's sublists, each sublist's rows sorted."""
    return sorted(json.dumps(sorted(json.dumps(r, sort_keys=True) for r in sub)) for sub in ak.to_list(array))


def joined(blocks: Sequence[Any]) -> ak.Array:
    return ak.concatenate(list(blocks))
