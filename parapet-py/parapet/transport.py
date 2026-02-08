"""httpx transport patch for routing LLM API requests through the parapet engine.

Monkey-patches ``httpx.Client`` and ``httpx.AsyncClient`` so that requests
to known LLM API hosts are transparently routed through ``localhost:<port>``.

Failopen behavior:
    Connection refused -> fall back to original URL (engine down is not fatal).
Failclosed behavior:
    Timeout -> raise error (do not silently bypass).
"""
from __future__ import annotations

import logging
from typing import Optional

import httpx

__all__ = [
    "LLM_HOSTS",
    "ParapetTransport",
    "AsyncParapetTransport",
    "patch_httpx",
    "should_intercept",
]

logger = logging.getLogger("parapet.transport")

# Known LLM API hosts that should be routed through the engine.
# Strict equality match -- subdomains are NOT intercepted.
LLM_HOSTS: frozenset[str] = frozenset(
    {
        "api.openai.com",
        "api.anthropic.com",
    }
)


def should_intercept(url: httpx.URL) -> bool:
    """Return True if the request URL targets a known LLM API host.

    Uses strict host equality -- subdomains like ``evil.api.openai.com``
    are not matched.
    """
    return url.host in LLM_HOSTS


def _rewrite_url(original_url: httpx.URL, port: int) -> httpx.URL:
    """Rewrite a URL to target localhost on the given port.

    Preserves the path, query, and fragment from the original URL.
    Forces http scheme since the engine listens locally.
    """
    return original_url.copy_with(
        scheme="http",
        host="127.0.0.1",
        port=port,
    )


class ParapetTransport(httpx.BaseTransport):
    """Synchronous httpx transport that routes LLM requests through the engine.

    Non-LLM requests are passed through to the wrapped transport unchanged.
    Connection errors to the engine trigger failopen -- the original URL is
    retried directly.
    """

    def __init__(self, wrapped: httpx.BaseTransport, port: int) -> None:
        self._wrapped = wrapped
        self._port = port

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        if not should_intercept(request.url):
            return self._wrapped.handle_request(request)

        original_url = request.url
        proxy_url = _rewrite_url(original_url, self._port)

        # Build a new request targeting the engine.
        proxy_request = httpx.Request(
            method=request.method,
            url=proxy_url,
            headers=request.headers,
            content=request.content,
            extensions=request.extensions,
        )

        try:
            return self._wrapped.handle_request(proxy_request)
        except (ConnectionError, OSError) as exc:
            # Failopen: engine is unreachable, fall back to original URL.
            logger.warning(
                "Engine unreachable (%s), failing open to %s",
                exc,
                original_url,
            )
            fallback_request = httpx.Request(
                method=request.method,
                url=original_url,
                headers=request.headers,
                content=request.content,
                extensions=request.extensions,
            )
            return self._wrapped.handle_request(fallback_request)

    def close(self) -> None:
        self._wrapped.close()


class AsyncParapetTransport(httpx.AsyncBaseTransport):
    """Async httpx transport that routes LLM requests through the engine.

    Same failopen/failclosed semantics as ``ParapetTransport``.
    """

    def __init__(self, wrapped: httpx.AsyncBaseTransport, port: int) -> None:
        self._wrapped = wrapped
        self._port = port

    async def handle_async_request(
        self, request: httpx.Request
    ) -> httpx.Response:
        if not should_intercept(request.url):
            return await self._wrapped.handle_async_request(request)

        original_url = request.url
        proxy_url = _rewrite_url(original_url, self._port)

        proxy_request = httpx.Request(
            method=request.method,
            url=proxy_url,
            headers=request.headers,
            content=request.content,
            extensions=request.extensions,
        )

        try:
            return await self._wrapped.handle_async_request(proxy_request)
        except (ConnectionError, OSError) as exc:
            logger.warning(
                "Engine unreachable (%s), failing open to %s",
                exc,
                original_url,
            )
            fallback_request = httpx.Request(
                method=request.method,
                url=original_url,
                headers=request.headers,
                content=request.content,
                extensions=request.extensions,
            )
            return await self._wrapped.handle_async_request(fallback_request)

    async def aclose(self) -> None:
        await self._wrapped.aclose()


def patch_httpx(port: int) -> None:
    """Monkey-patch httpx.Client and httpx.AsyncClient constructors.

    After this call, all newly created ``httpx.Client`` and
    ``httpx.AsyncClient`` instances will route LLM API requests through
    the parapet engine at ``localhost:<port>``.

    Args:
        port: The port the engine is listening on.
    """
    _original_client_init = httpx.Client.__init__
    _original_async_init = httpx.AsyncClient.__init__

    def _patched_client_init(self: httpx.Client, *args, **kwargs) -> None:
        _original_client_init(self, *args, **kwargs)
        original_transport = self._transport
        self._transport = ParapetTransport(
            wrapped=original_transport,
            port=port,
        )

    def _patched_async_init(
        self: httpx.AsyncClient, *args, **kwargs
    ) -> None:
        _original_async_init(self, *args, **kwargs)
        original_transport = self._transport
        self._transport = AsyncParapetTransport(
            wrapped=original_transport,
            port=port,
        )

    httpx.Client.__init__ = _patched_client_init  # type: ignore[assignment]
    httpx.AsyncClient.__init__ = _patched_async_init  # type: ignore[assignment]

    logger.info("httpx patched to route LLM requests through port %d", port)
