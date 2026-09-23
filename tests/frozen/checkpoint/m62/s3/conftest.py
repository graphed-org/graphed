"""m62 unit C: unit B's two URL fixtures, rooted on s3 behind one loopback moto server."""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator

import pytest

_BUCKET = "m62-checkpoint"


@pytest.fixture(scope="session")
def m62_moto_bucket() -> Iterator[str]:
    moto_server = pytest.importorskip("moto.server")
    s3fs = pytest.importorskip("s3fs")
    # Windows cannot connect to the default 0.0.0.0 bind that get_host_and_port() reports.
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
def store_url(m62_moto_bucket) -> str:
    return f"s3://{m62_moto_bucket}/{uuid.uuid4().hex}"


@pytest.fixture
def shared_url(m62_moto_bucket) -> str:
    return f"s3://{m62_moto_bucket}/{uuid.uuid4().hex}"
