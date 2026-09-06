"""Tests for shared inbound media download guards (#5223).

Covers destination validation (scheme + host allowlist) and the in-flight
byte cap that aborts a download before the full body is buffered.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest

from app.channels.inbound_media import (
    MAX_INBOUND_HTTP_FILE_BYTES,
    download_inbound_media,
    validate_inbound_media_url,
)


class _ChunkedBody:
    """Async-iterable body that records how many chunks were actually consumed."""

    def __init__(self, chunk_sizes: list[int]):
        self.chunk_sizes = list(chunk_sizes)
        self.total_chunks = len(chunk_sizes)
        self.read_chunks = 0

    def __aiter__(self):
        self._sizes = iter(self.chunk_sizes)
        return self

    async def __anext__(self) -> bytes:
        for size in self._sizes:
            self.read_chunks += 1
            return b"a" * size
        raise StopAsyncIteration


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _chunked_handler(body: _ChunkedBody, *, content_length: str | None = None):
    headers = {"content-length": content_length} if content_length is not None else {}

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers=headers, content=body)

    return handler


# ---------------------------------------------------------------------------
# validate_inbound_media_url
# ---------------------------------------------------------------------------


def test_validate_accepts_http_urls_and_trims():
    assert validate_inbound_media_url("  https://cdn.example.com/a.bin  ") == "https://cdn.example.com/a.bin"
    assert validate_inbound_media_url("http://cdn.example.com/a.bin") == "http://cdn.example.com/a.bin"


@pytest.mark.parametrize(
    "url",
    [
        None,
        "",
        "   ",
        "ftp://cdn.example.com/a.bin",
        "file:///etc/passwd",
        "data:text/plain,hello",
        "https:///no-host.bin",
        "cdn.example.com/no-scheme.bin",
    ],
)
def test_validate_rejects_unsafe_urls(url):
    assert validate_inbound_media_url(url) is None


def test_validate_enforces_host_allowlist_when_non_empty():
    hosts = ["Novac2c.cdn.weixin.qq.com"]
    assert validate_inbound_media_url("https://novac2c.cdn.weixin.qq.com/c2c/a", allowed_hosts=hosts) is not None
    assert validate_inbound_media_url("https://evil.example.com/a", allowed_hosts=hosts) is None
    # An empty allowlist skips host matching but still applies scheme checks.
    assert validate_inbound_media_url("https://evil.example.com/a", allowed_hosts=[]) is not None
    assert validate_inbound_media_url("file:///etc/passwd", allowed_hosts=[]) is None


# ---------------------------------------------------------------------------
# download_inbound_media
# ---------------------------------------------------------------------------


def _run(coro):
    import asyncio

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def test_download_returns_body_when_within_cap_sync():
    body = _ChunkedBody([8, 8])

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body)

    async def go():
        async with _client(handler) as client:
            return await download_inbound_media(client, "https://cdn.example.com/a.bin", max_bytes=100)

    assert _run(go()) == b"a" * 16
    assert body.read_chunks == body.total_chunks


def test_download_aborts_when_stream_exceeds_cap():
    body = _ChunkedBody([8, 8, 8])

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body)

    async def go():
        async with _client(handler) as client:
            return await download_inbound_media(client, "https://cdn.example.com/a.bin", max_bytes=10)

    assert _run(go()) is None
    # The cap must abort before the tail of the body is buffered.
    assert body.read_chunks == 2


def test_download_rejects_content_length_over_cap_without_reading():
    body = _ChunkedBody([1024])
    read = {"requested": False}

    def handler(request: httpx.Request) -> httpx.Response:
        read["requested"] = True
        return httpx.Response(200, headers={"content-length": str(MAX_INBOUND_HTTP_FILE_BYTES + 1)}, content=b"")

    async def go():
        async with _client(handler) as client:
            return await download_inbound_media(
                client,
                "https://cdn.example.com/a.bin",
                max_bytes=MAX_INBOUND_HTTP_FILE_BYTES,
            )

    assert _run(go()) is None
    assert read["requested"] is True
    assert body.read_chunks == 0


def test_download_rejects_invalid_url_without_network_call():
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("network must not be reached for an invalid URL")

    async def go():
        async with _client(handler) as client:
            return await download_inbound_media(client, "file:///etc/passwd")

    assert _run(go()) is None


def test_download_rejects_host_outside_allowlist_without_network_call():
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("network must not be reached for a host outside the allowlist")

    async def go():
        async with _client(handler) as client:
            return await download_inbound_media(
                client,
                "https://evil.example.com/a.bin",
                allowed_hosts={"cdn.example.com"},
            )

    assert _run(go()) is None


def test_download_uncapped_when_max_bytes_disabled():
    body = _ChunkedBody([2048, 2048])

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body)

    async def go():
        async with _client(handler) as client:
            return await download_inbound_media(client, "https://cdn.example.com/a.bin", max_bytes=0)

    assert _run(go()) == b"a" * 4096


def test_download_propagates_http_errors():
    async def go():
        async with _client(lambda request: httpx.Response(404)) as client:
            return await download_inbound_media(client, "https://cdn.example.com/missing.bin")

    with pytest.raises(httpx.HTTPStatusError):
        _run(go())


# ---------------------------------------------------------------------------
# _read_http_inbound_file (generic channel reader)
# ---------------------------------------------------------------------------


def test_read_http_inbound_file_returns_small_payload():
    from app.channels.manager import _read_http_inbound_file

    async def go():
        async with _client(lambda request: httpx.Response(200, content=b"file-bytes")) as client:
            return await _read_http_inbound_file({"url": "https://media.example.com/f.bin"}, client)

    assert _run(go()) == b"file-bytes"


def test_read_http_inbound_file_rejects_non_http_url():
    from app.channels.manager import _read_http_inbound_file

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("network must not be reached for a non-http URL")

    async def go():
        async with _client(handler) as client:
            return await _read_http_inbound_file({"url": "file:///etc/passwd"}, client)

    assert _run(go()) is None


def test_read_http_inbound_file_aborts_when_over_default_cap():
    from app.channels.manager import _read_http_inbound_file

    requested = {"value": False}

    def handler(request: httpx.Request) -> httpx.Response:
        requested["value"] = True
        return httpx.Response(200, headers={"content-length": str(MAX_INBOUND_HTTP_FILE_BYTES + 1)}, content=b"")

    async def go():
        async with _client(handler) as client:
            return await _read_http_inbound_file({"url": "https://media.example.com/huge.bin"}, client)

    assert _run(go()) is None
    assert requested["value"] is True


# ---------------------------------------------------------------------------
# WeChat channel integration
# ---------------------------------------------------------------------------


def _wechat_channel(tmp_path: Path, config: dict[str, Any] | None = None):
    from app.channels.message_bus import MessageBus
    from app.channels.wechat import WechatChannel

    return WechatChannel(
        bus=MessageBus(),
        config={"bot_token": "test-token", "state_dir": str(tmp_path), **(config or {})},
    )


def test_wechat_default_media_hosts_cover_platform_domains(tmp_path: Path):
    channel = _wechat_channel(tmp_path)
    assert channel._allowed_media_hosts == {"novac2c.cdn.weixin.qq.com", "ilinkai.weixin.qq.com"}


def test_wechat_allowed_media_hosts_config_overrides_default(tmp_path: Path):
    channel = _wechat_channel(tmp_path, {"allowed_media_hosts": ["Cdn.Custom.Example"]})
    assert channel._allowed_media_hosts == {"cdn.custom.example"}


def test_wechat_cdn_download_rejects_foreign_host(tmp_path: Path):
    channel = _wechat_channel(tmp_path)

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("network must not be reached for a host outside the allowlist")

    async def go():
        channel._client = _client(handler)
        return await channel._download_cdn_bytes("https://evil.example.com/media.bin")

    assert _run(go()) is None


def test_wechat_cdn_download_allows_configured_cdn_host(tmp_path: Path):
    channel = _wechat_channel(tmp_path)

    async def go():
        channel._client = _client(lambda request: httpx.Response(200, content=b"cdn-bytes"))
        return await channel._download_cdn_bytes("https://novac2c.cdn.weixin.qq.com/c2c/media.bin", max_bytes=100)

    assert _run(go()) == b"cdn-bytes"


def test_wechat_cdn_download_aborts_when_over_cap(tmp_path: Path):
    channel = _wechat_channel(tmp_path)

    async def go():
        channel._client = _client(lambda request: httpx.Response(200, content=b"1234567890"))
        return await channel._download_cdn_bytes("https://novac2c.cdn.weixin.qq.com/c2c/media.bin", max_bytes=5)

    assert _run(go()) is None


def test_wechat_image_skipped_when_download_rejected(tmp_path: Path):
    channel = _wechat_channel(tmp_path)

    async def fake_download(url: str, *, timeout: float | None = None, **_kwargs):
        return None

    channel._download_cdn_bytes = fake_download  # type: ignore[method-assign]

    async def go():
        return await channel._extract_image_file(
            {
                "type": 2,
                "image_item": {
                    "aeskey": b"1234567890abcdef".hex(),
                    "media": {"full_url": "https://novac2c.cdn.weixin.qq.com/c2c/a.bin"},
                },
            },
            message_id="m-1",
            index=0,
        )

    assert _run(go()) is None


def test_wechat_file_skipped_when_download_rejected(tmp_path: Path):
    channel = _wechat_channel(tmp_path)

    async def fake_download(url: str, *, timeout: float | None = None, **_kwargs):
        return None

    channel._download_cdn_bytes = fake_download  # type: ignore[method-assign]

    async def go():
        return await channel._extract_file_item(
            {
                "type": 4,
                "file_item": {
                    "file_name": "report.pdf",
                    "aeskey": b"1234567890abcdef".hex(),
                    "media": {"full_url": "https://novac2c.cdn.weixin.qq.com/c2c/a.bin"},
                },
            },
            message_id="m-2",
            index=0,
        )

    assert _run(go()) is None
