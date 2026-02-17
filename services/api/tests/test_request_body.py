import pytest
from fastapi import HTTPException
from starlette.requests import Request

from app.utils.request_body import read_json_body


def _make_request(messages: list[dict], content_type: str = "application/json") -> Request:
    queue = list(messages)

    async def receive() -> dict:
        if queue:
            return queue.pop(0)
        return {"type": "http.disconnect"}

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/analyze",
        "raw_path": b"/v1/analyze",
        "query_string": b"",
        "root_path": "",
        "headers": [(b"content-type", content_type.encode("ascii"))],
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
    }
    return Request(scope, receive)


@pytest.mark.asyncio
async def test_read_json_body_valid_object():
    request = _make_request([{"type": "http.request", "body": b'{"text":"hello"}', "more_body": False}])

    payload = await read_json_body(request)

    assert payload == {"text": "hello"}


@pytest.mark.asyncio
async def test_read_json_body_invalid_json():
    request = _make_request([{"type": "http.request", "body": b"{", "more_body": False}])

    with pytest.raises(HTTPException) as exc:
        await read_json_body(request)

    assert exc.value.status_code == 400
    assert exc.value.detail == "Invalid JSON body"


@pytest.mark.asyncio
async def test_read_json_body_client_disconnect():
    request = _make_request([{"type": "http.disconnect"}])

    with pytest.raises(HTTPException) as exc:
        await read_json_body(request)

    assert exc.value.status_code == 499
    assert exc.value.detail == "Client disconnected"


@pytest.mark.asyncio
async def test_read_json_body_rejects_non_object():
    request = _make_request([{"type": "http.request", "body": b"[]", "more_body": False}])

    with pytest.raises(HTTPException) as exc:
        await read_json_body(request)

    assert exc.value.status_code == 400
    assert exc.value.detail == "JSON body must be an object"
