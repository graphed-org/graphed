"""Shared helpers for the M65 A1 dashboard-control tests: POST, polling and a TCP relay."""

from __future__ import annotations

import contextlib
import json
import socket
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any


def post(url: str, body: bytes, content_type: str = "application/json") -> tuple[int, Any]:
    req = urllib.request.Request(url, data=body, method="POST", headers={"Content-Type": content_type})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read() or b"null")
    except urllib.error.HTTPError as err:
        return err.code, None


def post_cmd(base_url: str, cmd: str) -> tuple[int, Any]:
    return post(base_url + "api/control", json.dumps({"cmd": cmd}).encode())


def get_json(url: str) -> Any:
    with urllib.request.urlopen(url, timeout=10) as resp:
        return json.loads(resp.read())


def until(pred: Callable[[], bool], timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return
        time.sleep(0.02)
    raise AssertionError(f"condition not met within {timeout}s")


class Relay:
    """A fixed local port that forwards each accepted connection to ``target`` (set by the test),
    so a server can be replaced behind one address without rebinding a port."""

    def __init__(self) -> None:
        self.target: int | None = None
        self._lsock = socket.create_server(("127.0.0.1", 0))
        self.port: int = self._lsock.getsockname()[1]
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self) -> None:
        while True:
            try:
                client, _ = self._lsock.accept()
            except OSError:
                return
            try:
                upstream = socket.create_connection(("127.0.0.1", self.target or 0), timeout=5)
                upstream.settimeout(None)
            except OSError:
                client.close()
                continue
            for src, dst in ((client, upstream), (upstream, client)):
                threading.Thread(target=self._pump, args=(src, dst), daemon=True).start()

    @staticmethod
    def _pump(src: socket.socket, dst: socket.socket) -> None:
        with contextlib.suppress(OSError):
            while data := src.recv(65536):
                dst.sendall(data)
        for s in (src, dst):
            with contextlib.suppress(OSError):
                s.shutdown(socket.SHUT_RDWR)
            s.close()

    def close(self) -> None:
        self._lsock.close()
