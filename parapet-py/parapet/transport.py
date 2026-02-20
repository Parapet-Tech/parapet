# Copyright 2026 The Parapet Project
# SPDX-License-Identifier: Apache-2.0

"""httpx transport patch for routing LLM API requests through the parapet engine.

Monkey-patches ``httpx.Client`` and ``httpx.AsyncClient`` so that requests
to known LLM API hosts are transparently routed through ``localhost:<port>``.

Failopen behavior:
    Connection refused / connect timeout -> fall back to original URL
    (engine down is not fatal).
Failclosed behavior:
    Read timeout -> raise error (a slow engine is not silently bypassed).
"""
from __future__ import annotations

import logging
from typing import Optional

import httpx

from parapet.header import build_trust_header, get_active_baggage
from parapet.trust import get_registry

__all__ = [
    "DEFAULT_LLM_HOSTS",
    "LLM_HOSTS",
    "ParapetTransport",
    "AsyncParapetTransport",
    "patch_httpx",
    "should_intercept",
]

logger = logging.getLogger("parapet.transport")

# Default LLM API hosts that should be routed through the engine.
# Strict equality match -- subdomains are NOT intercepted.
DEFAULT_LLM_HOSTS: frozenset[str] = frozenset(
    {
        "api.openai.com",
        "api.anthropic.com",
        # OpenAI-compatible providers
        "api.cerebras.ai",
        "api.groq.com",
        "generativelanguage.googleapis.com",
    }
)

# Active host set — starts as defaults, extended by patch_httpx(extra_hosts=...).
LLM_HOSTS: frozenset[str] = DEFAULT_LLM_HOSTS


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

    def __init__(
        self,
        wrapped: httpx.BaseTransport,
        port: int,
        hosts: Optional[frozenset[str]] = None,
    ) -> None:
        self._wrapped = wrapped
        self._port = port
        self._hosts = hosts or LLM_HOSTS

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        if request.url.host not in self._hosts:
            return self._wrapped.handle_request(request)

        original_url = request.url
        proxy_url = _rewrite_url(original_url, self._port)

        # Build a new request targeting the engine.
        # Carry the original host so the engine can forward to the right upstream.
        # Remove the original Host header so httpx sets it from proxy_url.
        headers = {k: v for k, v in request.headers.items() if k.lower() != "host"}
        headers["x-parapet-original-host"] = original_url.host

        # Inject session baggage if an active session exists.
        # Merge with any existing caller-provided baggage rather than replacing it.
        baggage = get_active_baggage()
        if baggage:
            existing = headers.get("baggage")
            headers["baggage"] = f"{existing},{baggage}" if existing else baggage

        # Inject trust spans if registry has entries.
        # Guard against RequestNotRead for streaming request bodies —
        # degrade gracefully (skip trust header) rather than crash.
        body: bytes | None = None
        try:
            body = request.content
        except httpx.RequestNotRead:
            logger.debug("Request body is a stream; skipping trust span injection")

        registry = get_registry()
        if registry is not None and body:
            spans = registry.find_spans(body)
            trust_header = build_trust_header(spans)
            if trust_header is not None:
                headers["x-guard-trust"] = trust_header

        proxy_request = httpx.Request(
            method=request.method,
            url=proxy_url,
            headers=headers,
            content=body if body is not None else request.stream,
            extensions=request.extensions,
        )

        try:
            return self._wrapped.handle_request(proxy_request)
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            # Failopen: engine is unreachable, fall back to original URL.
            # ConnectError = refused/DNS failure. ConnectTimeout = engine not responding.
            # ReadTimeout is NOT caught — a slow engine is failclosed.
            logger.warning(
                "Engine unreachable (%s), failing open to %s",
                exc,
                original_url,
            )
            fallback_request = httpx.Request(
                method=request.method,
                url=original_url,
                headers=request.headers,
                content=body if body is not None else request.stream,
                extensions=request.extensions,
            )
            return self._wrapped.handle_request(fallback_request)

    def close(self) -> None:
        self._wrapped.close()


class AsyncParapetTransport(httpx.AsyncBaseTransport):
    """Async httpx transport that routes LLM requests through the engine.

    Same failopen/failclosed semantics as ``ParapetTransport``.
    """

    def __init__(
        self,
        wrapped: httpx.AsyncBaseTransport,
        port: int,
        hosts: Optional[frozenset[str]] = None,
    ) -> None:
        self._wrapped = wrapped
        self._port = port
        self._hosts = hosts or LLM_HOSTS

    async def handle_async_request(
        self, request: httpx.Request
    ) -> httpx.Response:
        if request.url.host not in self._hosts:
            return await self._wrapped.handle_async_request(request)

        original_url = request.url
        proxy_url = _rewrite_url(original_url, self._port)

        # Carry the original host so the engine can forward to the right upstream.
        # Remove the original Host header so httpx sets it from proxy_url.
        headers = {k: v for k, v in request.headers.items() if k.lower() != "host"}
        headers["x-parapet-original-host"] = original_url.host

        # Inject session baggage if an active session exists.
        # Merge with any existing caller-provided baggage rather than replacing it.
        baggage = get_active_baggage()
        if baggage:
            existing = headers.get("baggage")
            headers["baggage"] = f"{existing},{baggage}" if existing else baggage

        # Inject trust spans if registry has entries.
        # Guard against RequestNotRead for streaming request bodies.
        body: bytes | None = None
        try:
            body = request.content
        except httpx.RequestNotRead:
            logger.debug("Request body is a stream; skipping trust span injection")

        registry = get_registry()
        if registry is not None and body:
            spans = registry.find_spans(body)
            trust_header = build_trust_header(spans)
            if trust_header is not None:
                headers["x-guard-trust"] = trust_header

        proxy_request = httpx.Request(
            method=request.method,
            url=proxy_url,
            headers=headers,
            content=body if body is not None else request.stream,
            extensions=request.extensions,
        )

        try:
            return await self._wrapped.handle_async_request(proxy_request)
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
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
                content=body if body is not None else request.stream,
                extensions=request.extensions,
            )
            return await self._wrapped.handle_async_request(fallback_request)

    async def aclose(self) -> None:
        await self._wrapped.aclose()


def patch_httpx(
    port: int,
    extra_hosts: Optional[frozenset[str]] = None,
) -> None:
    """Monkey-patch httpx.Client and httpx.AsyncClient constructors.

    After this call, all newly created ``httpx.Client`` and
    ``httpx.AsyncClient`` instances will route LLM API requests through
    the parapet engine at ``localhost:<port>``.

    Args:
        port: The port the engine is listening on.
        extra_hosts: Additional hosts to intercept beyond the defaults.
    """
    global LLM_HOSTS
    if extra_hosts:
        LLM_HOSTS = DEFAULT_LLM_HOSTS | extra_hosts
    else:
        LLM_HOSTS = DEFAULT_LLM_HOSTS

    hosts = LLM_HOSTS
    _original_client_init = httpx.Client.__init__
    _original_async_init = httpx.AsyncClient.__init__

    def _patched_client_init(self: httpx.Client, *args, **kwargs) -> None:
        _original_client_init(self, *args, **kwargs)
        original_transport = self._transport
        self._transport = ParapetTransport(
            wrapped=original_transport,
            port=port,
            hosts=hosts,
        )

    def _patched_async_init(
        self: httpx.AsyncClient, *args, **kwargs
    ) -> None:
        _original_async_init(self, *args, **kwargs)
        original_transport = self._transport
        self._transport = AsyncParapetTransport(
            wrapped=original_transport,
            port=port,
            hosts=hosts,
        )

    httpx.Client.__init__ = _patched_client_init  # type: ignore[assignment]
    httpx.AsyncClient.__init__ = _patched_async_init  # type: ignore[assignment]

    logger.info("httpx patched to route LLM requests through port %d", port)
