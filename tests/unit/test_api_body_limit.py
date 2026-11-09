import asyncio

import pytest

pytest.importorskip("fastapi")

from api.app import MAX_UPLOAD_BYTES, MULTIPART_OVERHEAD_BYTES, BodyLimit  # noqa: E402


def _call(app, headers, receive):
    sent = []

    async def send(message):
        sent.append(message)

    scope = {"type": "http", "method": "POST", "path": "/", "headers": headers}
    asyncio.run(BodyLimit(app)(scope, receive, send))
    return sent


async def _never_called(*args):
    raise AssertionError("must not run")


def test_oversize_content_length_is_413_before_the_body_is_read():
    size = str(MAX_UPLOAD_BYTES + MULTIPART_OVERHEAD_BYTES + 1).encode()
    sent = _call(_never_called, [(b"content-length", size)], _never_called)
    assert sent[0]["status"] == 413


def test_small_body_passes_through():
    async def app(scope, receive, send):
        message = await receive()
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": message["body"]})

    async def receive():
        return {"type": "http.request", "body": b"ok", "more_body": False}

    sent = _call(app, [(b"content-length", b"2")], receive)
    assert sent[0]["status"] == 200 and sent[1]["body"] == b"ok"


def test_non_http_scope_is_not_touched():
    seen = []

    async def app(scope, receive, send):
        seen.append(scope["type"])

    asyncio.run(BodyLimit(app)({"type": "lifespan"}, _never_called, _never_called))
    assert seen == ["lifespan"]
