"""A loopback moto S3 bucket and a per-operation S3 request counter."""

from __future__ import annotations

import collections
import os
import uuid
from collections.abc import Callable, Iterator

import pytest

_BUCKET = "store-scaling"


@pytest.fixture(scope="session")
def s3_bucket() -> Iterator[str]:
    moto_server = pytest.importorskip("moto.server")
    s3fs = pytest.importorskip("s3fs")
    server = moto_server.ThreadedMotoServer(ip_address="127.0.0.1", port=0)
    server.start()
    host, port = server.get_host_and_port()
    env = {
        "AWS_ACCESS_KEY_ID": "testing",
        "AWS_SECRET_ACCESS_KEY": "testing",
        "AWS_DEFAULT_REGION": "us-east-1",
        "AWS_ENDPOINT_URL": f"http://{host}:{port}",
    }
    saved = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        s3fs.S3FileSystem(skip_instance_cache=True).mkdir(_BUCKET)
        yield _BUCKET
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        server.stop()


@pytest.fixture
def s3_url(s3_bucket: str) -> str:
    return f"s3://{s3_bucket}/{uuid.uuid4().hex}"


@pytest.fixture
def s3_counter() -> Callable[..., collections.Counter[str]]:
    """``count(store, fn)`` runs ``fn()`` and returns the S3 API calls ``store.fs`` made, by name."""

    def count(store: object, fn: Callable[[], object]) -> collections.Counter[str]:
        calls: collections.Counter[str] = collections.Counter()

        def before(model: object, **_kw: object) -> None:
            calls[model.name] += 1  # type: ignore[attr-defined]

        client = store.fs.s3  # type: ignore[attr-defined]
        client.meta.events.register("before-call.s3", before)
        try:
            fn()
        finally:
            client.meta.events.unregister("before-call.s3", before)
        return calls

    return count
