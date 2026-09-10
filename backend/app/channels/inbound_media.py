"""Shared inbound media download helpers for channel integrations.

Inbound media URLs arrive inside relayed platform message payloads, so they
are treated as untrusted input (the same posture DingTalk documents for its
``download_code`` handling): a fetch must validate the destination before
connecting, and must stream the response with an in-flight byte cap so an
oversized attachment is refused *before* it is fully buffered in memory.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)

#: Default in-flight cap for inbound media fetched over plain HTTP by the
#: generic channel reader. Matches DingTalk's ``_MAX_INBOUND_FILE_SIZE_BYTES``
#: and the WeChat channel's default inbound file limit.
MAX_INBOUND_HTTP_FILE_BYTES = 50 * 1024 * 1024

_ALLOWED_URL_SCHEMES = frozenset({"https", "http"})


def validate_inbound_media_url(url: str | None, *, allowed_hosts: Iterable[str] | None = None) -> str | None:
    """Return the trimmed URL when it is safe to fetch, else ``None``.

    Only ``http``/``https`` URLs with an explicit hostname are accepted. When
    *allowed_hosts* is a non-empty iterable, the URL host must match one of
    the entries (case-insensitive, exact host match); an empty/``None``
    allowlist skips the host check but still enforces scheme validation.
    """
    candidate = url.strip() if isinstance(url, str) else ""
    if not candidate:
        return None
    parsed = urlparse(candidate)
    hostname = parsed.hostname
    if parsed.scheme not in _ALLOWED_URL_SCHEMES or not hostname:
        return None
    if allowed_hosts:
        hosts = {str(host).strip().lower() for host in allowed_hosts if str(host).strip()}
        if hosts and hostname.lower() not in hosts:
            return None
    return candidate


async def download_inbound_media(
    client: httpx.AsyncClient,
    url: str | None,
    *,
    max_bytes: int = MAX_INBOUND_HTTP_FILE_BYTES,
    timeout: float | None = None,
    allowed_hosts: Iterable[str] | None = None,
) -> bytes | None:
    """Stream an inbound media attachment with an in-flight byte cap.

    Returns ``None`` — dropping the attachment — when the destination fails
    validation or the response exceeds *max_bytes*; the cap aborts the stream
    before the body is fully buffered, mirroring DingTalk's
    ``_download_by_code``. ``max_bytes <= 0`` disables the size cap (scheme
    and host validation still apply).
    """
    validated = validate_inbound_media_url(url, allowed_hosts=allowed_hosts)
    if validated is None:
        logger.warning("[Channels] inbound media URL rejected: %s", str(url)[:200])
        return None

    request_kwargs: dict[str, Any] = {}
    if timeout is not None:
        request_kwargs["timeout"] = timeout

    chunks: list[bytes] = []
    total = 0
    async with client.stream("GET", validated, **request_kwargs) as response:
        response.raise_for_status()
        if max_bytes > 0:
            content_length = response.headers.get("content-length")
            if content_length and content_length.isdigit() and int(content_length) > max_bytes:
                logger.warning(
                    "[Channels] inbound media content-length %s exceeds %d bytes, dropping",
                    content_length,
                    max_bytes,
                )
                return None
        async for chunk in response.aiter_bytes():
            total += len(chunk)
            if max_bytes > 0 and total > max_bytes:
                logger.warning("[Channels] inbound media exceeds %d bytes, dropping", max_bytes)
                return None
            chunks.append(chunk)
    return b"".join(chunks)
