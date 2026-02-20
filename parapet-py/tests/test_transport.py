# Copyright 2026 The Parapet Project
# SPDX-License-Identifier: Apache-2.0

"""Tests for parapet.transport — httpx monkey-patch."""
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from urllib.parse import urlparse

import httpx
import pytest

from parapet.transport import (
    AsyncParapetTransport,
    DEFAULT_LLM_HOSTS,
    LLM_HOSTS,
    ParapetTransport,
    should_intercept,
)


class TestShouldIntercept:
    """Determine which requests are routed to the parapet engine."""

    def test_openai_intercepted(self):
        """Requests to api.openai.com are intercepted."""
        url = httpx.URL("https://api.openai.com/v1/chat/completions")
        assert should_intercept(url) is True

    def test_anthropic_intercepted(self):
        """Requests to api.anthropic.com are intercepted."""
        url = httpx.URL("https://api.anthropic.com/v1/messages")
        assert should_intercept(url) is True

    def test_non_llm_not_intercepted(self):
        """Requests to non-LLM hosts are NOT intercepted."""
        url = httpx.URL("https://example.com/api/data")
        assert should_intercept(url) is False

    def test_subdomain_not_intercepted(self):
        """Requests to subdomains of LLM hosts are NOT intercepted (strict match)."""
        url = httpx.URL("https://evil.api.openai.com/v1/chat/completions")
        assert should_intercept(url) is False


class TestParapetTransport:
    """Transport layer that routes LLM requests through parapet engine."""

    def test_openai_routed_to_localhost(self):
        """OpenAI requests have their URL rewritten to localhost:port."""
        transport = ParapetTransport(
            wrapped=MagicMock(),
            port=9800,
        )
        request = httpx.Request(
            "POST",
            "https://api.openai.com/v1/chat/completions",
            json={"model": "gpt-4"},
        )

        # Mock the wrapped transport to capture the request
        mock_response = httpx.Response(200, json={"choices": []})
        transport._wrapped.handle_request.return_value = mock_response

        response = transport.handle_request(request)

        # The request sent to wrapped transport should target localhost
        called_request = transport._wrapped.handle_request.call_args[0][0]
        assert called_request.url.host == "127.0.0.1"
        assert called_request.url.port == 9800

    def test_anthropic_routed_to_localhost(self):
        """Anthropic requests have their URL rewritten to localhost:port."""
        transport = ParapetTransport(
            wrapped=MagicMock(),
            port=9800,
        )
        request = httpx.Request(
            "POST",
            "https://api.anthropic.com/v1/messages",
            json={"model": "claude-3"},
        )

        mock_response = httpx.Response(200, json={"content": []})
        transport._wrapped.handle_request.return_value = mock_response

        response = transport.handle_request(request)

        called_request = transport._wrapped.handle_request.call_args[0][0]
        assert called_request.url.host == "127.0.0.1"
        assert called_request.url.port == 9800

    def test_non_llm_not_intercepted(self):
        """Non-LLM requests pass through to original URL unchanged."""
        transport = ParapetTransport(
            wrapped=MagicMock(),
            port=9800,
        )
        request = httpx.Request(
            "GET",
            "https://example.com/api/data",
        )

        mock_response = httpx.Response(200, json={"ok": True})
        transport._wrapped.handle_request.return_value = mock_response

        response = transport.handle_request(request)

        called_request = transport._wrapped.handle_request.call_args[0][0]
        assert called_request.url.host == "example.com"

    def test_connect_error_falls_back(self):
        """httpx.ConnectError triggers failopen to original URL."""
        mock_wrapped = MagicMock()
        transport = ParapetTransport(wrapped=mock_wrapped, port=9800)
        request = httpx.Request(
            "POST",
            "https://api.openai.com/v1/chat/completions",
            json={"model": "gpt-4"},
        )

        fallback_response = httpx.Response(200, json={"choices": []})
        mock_wrapped.handle_request.side_effect = [
            httpx.ConnectError("Connection refused"),
            fallback_response,
        ]

        response = transport.handle_request(request)

        assert response.status_code == 200
        assert mock_wrapped.handle_request.call_count == 2
        fallback_request = mock_wrapped.handle_request.call_args_list[1][0][0]
        assert fallback_request.url.host == "api.openai.com"

    def test_connect_timeout_falls_back(self):
        """httpx.ConnectTimeout triggers failopen to original URL."""
        mock_wrapped = MagicMock()
        transport = ParapetTransport(wrapped=mock_wrapped, port=9800)
        request = httpx.Request(
            "POST",
            "https://api.openai.com/v1/chat/completions",
            json={"model": "gpt-4"},
        )

        fallback_response = httpx.Response(200, json={"choices": []})
        mock_wrapped.handle_request.side_effect = [
            httpx.ConnectTimeout("Timed out connecting"),
            fallback_response,
        ]

        response = transport.handle_request(request)

        assert response.status_code == 200
        assert mock_wrapped.handle_request.call_count == 2

    def test_read_timeout_does_not_failopen(self):
        """httpx.ReadTimeout is NOT caught — a slow engine is failclosed."""
        mock_wrapped = MagicMock()
        transport = ParapetTransport(wrapped=mock_wrapped, port=9800)
        request = httpx.Request(
            "POST",
            "https://api.openai.com/v1/chat/completions",
            json={"model": "gpt-4"},
        )

        mock_wrapped.handle_request.side_effect = httpx.ReadTimeout(
            "Timed out reading"
        )

        with pytest.raises(httpx.ReadTimeout):
            transport.handle_request(request)

        # Only called once — no fallback attempted.
        assert mock_wrapped.handle_request.call_count == 1


class TestStreamingTrustScan:
    """Streaming request bodies must not crash the trust scan path."""

    def _make_stream_request(self) -> httpx.Request:
        """Create a request then remove _content to simulate a stream-only body."""
        request = httpx.Request(
            "POST",
            "https://api.openai.com/v1/chat/completions",
            json={"model": "gpt-4", "stream": True},
        )
        del request._content
        return request

    def test_stream_body_does_not_crash_transport(self):
        """When request.content raises RequestNotRead, transport skips trust header."""
        transport = ParapetTransport(wrapped=MagicMock(), port=9800)
        request = self._make_stream_request()

        transport._wrapped.handle_request.return_value = httpx.Response(
            200, json={"choices": []}
        )

        response = transport.handle_request(request)
        assert response.status_code == 200

        called_request = transport._wrapped.handle_request.call_args[0][0]
        assert called_request.url.host == "127.0.0.1"
        assert "x-guard-trust" not in called_request.headers

    def test_stream_body_with_connect_error_failopen(self):
        """Streaming body + ConnectError must failopen without raising RequestNotRead."""
        mock_wrapped = MagicMock()
        transport = ParapetTransport(wrapped=mock_wrapped, port=9800)
        request = self._make_stream_request()

        fallback_response = httpx.Response(200, json={"choices": []})
        mock_wrapped.handle_request.side_effect = [
            httpx.ConnectError("Connection refused"),
            fallback_response,
        ]

        response = transport.handle_request(request)

        assert response.status_code == 200
        assert mock_wrapped.handle_request.call_count == 2
        fallback_request = mock_wrapped.handle_request.call_args_list[1][0][0]
        assert fallback_request.url.host == "api.openai.com"

    def test_stream_body_with_connect_timeout_failopen(self):
        """Streaming body + ConnectTimeout must failopen without raising RequestNotRead."""
        mock_wrapped = MagicMock()
        transport = ParapetTransport(wrapped=mock_wrapped, port=9800)
        request = self._make_stream_request()

        fallback_response = httpx.Response(200, json={"choices": []})
        mock_wrapped.handle_request.side_effect = [
            httpx.ConnectTimeout("Timed out connecting"),
            fallback_response,
        ]

        response = transport.handle_request(request)

        assert response.status_code == 200
        assert mock_wrapped.handle_request.call_count == 2


class TestBaggagePropagation:
    """Session baggage is injected into outbound LLM requests."""

    def test_baggage_header_injected_when_session_active(self):
        """Active session baggage is forwarded as a `baggage` header."""
        from parapet.header import _active_baggage

        transport = ParapetTransport(wrapped=MagicMock(), port=9800)
        request = httpx.Request(
            "POST",
            "https://api.openai.com/v1/chat/completions",
            json={"model": "gpt-4"},
        )
        transport._wrapped.handle_request.return_value = httpx.Response(
            200, json={"choices": []}
        )

        token = _active_baggage.set("user_id=u_1,role=admin")
        try:
            transport.handle_request(request)
        finally:
            _active_baggage.reset(token)

        called_request = transport._wrapped.handle_request.call_args[0][0]
        assert called_request.headers["baggage"] == "user_id=u_1,role=admin"

    def test_no_baggage_header_when_no_session(self):
        """Without an active session, no `baggage` header is added."""
        from parapet.header import _active_baggage

        _active_baggage.set(None)
        transport = ParapetTransport(wrapped=MagicMock(), port=9800)
        request = httpx.Request(
            "POST",
            "https://api.openai.com/v1/chat/completions",
            json={"model": "gpt-4"},
        )
        transport._wrapped.handle_request.return_value = httpx.Response(
            200, json={"choices": []}
        )

        transport.handle_request(request)

        called_request = transport._wrapped.handle_request.call_args[0][0]
        assert "baggage" not in called_request.headers

    def test_baggage_not_injected_for_non_llm_hosts(self):
        """Baggage header is not injected for non-LLM requests (pass-through)."""
        from parapet.header import _active_baggage

        transport = ParapetTransport(wrapped=MagicMock(), port=9800)
        request = httpx.Request("GET", "https://example.com/api/data")
        transport._wrapped.handle_request.return_value = httpx.Response(
            200, json={"ok": True}
        )

        token = _active_baggage.set("user_id=u_1")
        try:
            transport.handle_request(request)
        finally:
            _active_baggage.reset(token)

        # Non-LLM request is passed through unchanged — no baggage injection.
        called_request = transport._wrapped.handle_request.call_args[0][0]
        assert "baggage" not in called_request.headers


class TestLLMHosts:
    """The set of known LLM hosts is correct."""

    def test_openai_in_hosts(self):
        assert "api.openai.com" in DEFAULT_LLM_HOSTS

    def test_anthropic_in_hosts(self):
        assert "api.anthropic.com" in DEFAULT_LLM_HOSTS


class TestCustomHosts:
    """Custom hosts can be added via the hosts parameter."""

    def test_should_intercept_with_custom_hosts(self):
        """Transport with custom hosts intercepts those hosts."""
        custom_hosts = DEFAULT_LLM_HOSTS | frozenset({"api.together.xyz"})
        transport = ParapetTransport(
            wrapped=MagicMock(),
            port=9800,
            hosts=custom_hosts,
        )

        # Custom host should be intercepted
        request = httpx.Request(
            "POST",
            "https://api.together.xyz/v1/chat/completions",
            json={"model": "meta-llama/Llama-3-70b"},
        )
        mock_response = httpx.Response(200, json={"choices": []})
        transport._wrapped.handle_request.return_value = mock_response
        transport.handle_request(request)

        called_request = transport._wrapped.handle_request.call_args[0][0]
        assert called_request.url.host == "127.0.0.1"
        assert called_request.url.port == 9800

    def test_default_hosts_still_work_with_custom(self):
        """Adding custom hosts doesn't break default host interception."""
        custom_hosts = DEFAULT_LLM_HOSTS | frozenset({"api.together.xyz"})
        transport = ParapetTransport(
            wrapped=MagicMock(),
            port=9800,
            hosts=custom_hosts,
        )

        request = httpx.Request(
            "POST",
            "https://api.openai.com/v1/chat/completions",
            json={"model": "gpt-4"},
        )
        mock_response = httpx.Response(200, json={"choices": []})
        transport._wrapped.handle_request.return_value = mock_response
        transport.handle_request(request)

        called_request = transport._wrapped.handle_request.call_args[0][0]
        assert called_request.url.host == "127.0.0.1"

    def test_unknown_host_not_intercepted_with_custom(self):
        """Hosts not in defaults or custom set are still passed through."""
        custom_hosts = DEFAULT_LLM_HOSTS | frozenset({"api.together.xyz"})
        transport = ParapetTransport(
            wrapped=MagicMock(),
            port=9800,
            hosts=custom_hosts,
        )

        request = httpx.Request("GET", "https://example.com/api/data")
        mock_response = httpx.Response(200, json={"ok": True})
        transport._wrapped.handle_request.return_value = mock_response
        transport.handle_request(request)

        called_request = transport._wrapped.handle_request.call_args[0][0]
        assert called_request.url.host == "example.com"


class TestBaggageMerge:
    """Session baggage merges with existing caller-provided baggage."""

    def test_merges_with_existing_baggage(self):
        """Caller-provided baggage keys are preserved alongside session baggage."""
        from parapet.header import _active_baggage

        transport = ParapetTransport(wrapped=MagicMock(), port=9800)
        request = httpx.Request(
            "POST",
            "https://api.openai.com/v1/chat/completions",
            headers={"baggage": "trace_id=abc123"},
            json={"model": "gpt-4"},
        )
        transport._wrapped.handle_request.return_value = httpx.Response(
            200, json={"choices": []}
        )

        token = _active_baggage.set("user_id=u_1")
        try:
            transport.handle_request(request)
        finally:
            _active_baggage.reset(token)

        called_request = transport._wrapped.handle_request.call_args[0][0]
        baggage = called_request.headers["baggage"]
        assert "trace_id=abc123" in baggage
        assert "user_id=u_1" in baggage

    def test_no_existing_baggage_sets_cleanly(self):
        """Without existing baggage, session baggage is set without leading comma."""
        from parapet.header import _active_baggage

        transport = ParapetTransport(wrapped=MagicMock(), port=9800)
        request = httpx.Request(
            "POST",
            "https://api.openai.com/v1/chat/completions",
            json={"model": "gpt-4"},
        )
        transport._wrapped.handle_request.return_value = httpx.Response(
            200, json={"choices": []}
        )

        token = _active_baggage.set("user_id=u_1")
        try:
            transport.handle_request(request)
        finally:
            _active_baggage.reset(token)

        called_request = transport._wrapped.handle_request.call_args[0][0]
        assert called_request.headers["baggage"] == "user_id=u_1"


class TestAsyncParapetTransport:
    """Async transport has the same failopen/streaming/baggage behavior."""

    @pytest.mark.asyncio
    async def test_connect_error_falls_back(self):
        """httpx.ConnectError triggers failopen in async transport."""
        mock_wrapped = MagicMock()
        transport = AsyncParapetTransport(wrapped=mock_wrapped, port=9800)
        request = httpx.Request(
            "POST",
            "https://api.openai.com/v1/chat/completions",
            json={"model": "gpt-4"},
        )

        fallback_response = httpx.Response(200, json={"choices": []})
        mock_wrapped.handle_async_request = AsyncMock(
            side_effect=[httpx.ConnectError("refused"), fallback_response]
        )

        response = await transport.handle_async_request(request)

        assert response.status_code == 200
        assert mock_wrapped.handle_async_request.call_count == 2
        fallback_request = mock_wrapped.handle_async_request.call_args_list[1][0][0]
        assert fallback_request.url.host == "api.openai.com"

    @pytest.mark.asyncio
    async def test_connect_timeout_falls_back(self):
        """httpx.ConnectTimeout triggers failopen in async transport."""
        mock_wrapped = MagicMock()
        transport = AsyncParapetTransport(wrapped=mock_wrapped, port=9800)
        request = httpx.Request(
            "POST",
            "https://api.openai.com/v1/chat/completions",
            json={"model": "gpt-4"},
        )

        fallback_response = httpx.Response(200, json={"choices": []})
        mock_wrapped.handle_async_request = AsyncMock(
            side_effect=[httpx.ConnectTimeout("timed out connecting"), fallback_response]
        )

        response = await transport.handle_async_request(request)

        assert response.status_code == 200
        assert mock_wrapped.handle_async_request.call_count == 2
        fallback_request = mock_wrapped.handle_async_request.call_args_list[1][0][0]
        assert fallback_request.url.host == "api.openai.com"

    @pytest.mark.asyncio
    async def test_read_timeout_does_not_failopen(self):
        """httpx.ReadTimeout propagates in async transport (failclosed)."""
        mock_wrapped = MagicMock()
        transport = AsyncParapetTransport(wrapped=mock_wrapped, port=9800)
        request = httpx.Request(
            "POST",
            "https://api.openai.com/v1/chat/completions",
            json={"model": "gpt-4"},
        )

        mock_wrapped.handle_async_request = AsyncMock(
            side_effect=httpx.ReadTimeout("slow")
        )

        with pytest.raises(httpx.ReadTimeout):
            await transport.handle_async_request(request)

    @pytest.mark.asyncio
    async def test_stream_body_with_connect_error_failopen(self):
        """Streaming body + ConnectError must failopen in async transport."""
        mock_wrapped = MagicMock()
        transport = AsyncParapetTransport(wrapped=mock_wrapped, port=9800)
        request = httpx.Request(
            "POST",
            "https://api.openai.com/v1/chat/completions",
            json={"model": "gpt-4", "stream": True},
        )
        del request._content

        fallback_response = httpx.Response(200, json={"choices": []})
        mock_wrapped.handle_async_request = AsyncMock(
            side_effect=[httpx.ConnectError("refused"), fallback_response]
        )

        response = await transport.handle_async_request(request)

        assert response.status_code == 200
        assert mock_wrapped.handle_async_request.call_count == 2
