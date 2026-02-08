"""Tests for parapet.transport — httpx monkey-patch."""
from unittest.mock import MagicMock, patch, PropertyMock
from urllib.parse import urlparse

import httpx
import pytest

from parapet.transport import (
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

    def test_connection_refused_falls_back(self):
        """Connection refused to engine falls back to original URL (failopen)."""
        mock_wrapped = MagicMock()
        transport = ParapetTransport(
            wrapped=mock_wrapped,
            port=9800,
        )
        request = httpx.Request(
            "POST",
            "https://api.openai.com/v1/chat/completions",
            json={"model": "gpt-4"},
        )

        # First call (to localhost) raises ConnectionError
        # Second call (fallback to original) succeeds
        fallback_response = httpx.Response(200, json={"choices": []})
        mock_wrapped.handle_request.side_effect = [
            ConnectionError("Connection refused"),
            fallback_response,
        ]

        response = transport.handle_request(request)

        assert response.status_code == 200
        # Should have been called twice: once for proxy, once for fallback
        assert mock_wrapped.handle_request.call_count == 2
        # Second call should use original URL
        fallback_request = mock_wrapped.handle_request.call_args_list[1][0][0]
        assert fallback_request.url.host == "api.openai.com"


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
