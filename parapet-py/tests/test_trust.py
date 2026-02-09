"""Tests for parapet.trust -- TrustRegistry and span finding."""
from __future__ import annotations

import json

import pytest

from parapet.trust import TrustRegistry, TrustSpan, _clear_registry, _ensure_registry, get_registry


class TestTrustSpan:
    def test_construction(self):
        span = TrustSpan(start=10, end=20, source="rag")
        assert span.start == 10
        assert span.end == 20
        assert span.source == "rag"

    def test_default_source(self):
        span = TrustSpan(start=0, end=5)
        assert span.source == "unknown"

    def test_frozen(self):
        span = TrustSpan(start=0, end=5)
        with pytest.raises(AttributeError):
            span.start = 10  # type: ignore


class TestTrustRegistry:
    def test_register_returns_content_unchanged(self):
        reg = TrustRegistry()
        content = "some RAG content"
        result = reg.register(content, source="rag")
        assert result == content
        assert result is content

    def test_register_increments_count(self):
        reg = TrustRegistry()
        assert reg.entry_count == 0
        reg.register("a", source="rag")
        assert reg.entry_count == 1
        reg.register("b", source="user")
        assert reg.entry_count == 2

    def test_register_empty_string_is_noop(self):
        reg = TrustRegistry()
        result = reg.register("", source="rag")
        assert result == ""
        assert reg.entry_count == 0

    def test_register_max_entries_raises(self):
        reg = TrustRegistry(max_entries=3)
        reg.register("a")
        reg.register("b")
        reg.register("c")
        with pytest.raises(ValueError, match="TrustRegistry full"):
            reg.register("d")

    def test_clear(self):
        reg = TrustRegistry()
        reg.register("a")
        reg.register("b")
        assert reg.entry_count == 2
        reg.clear()
        assert reg.entry_count == 0

    def test_find_spans_empty_registry(self):
        reg = TrustRegistry()
        body = json.dumps({"content": "hello"}).encode()
        assert reg.find_spans(body) == []

    def test_find_spans_simple_match(self):
        reg = TrustRegistry()
        reg.register("untrusted data", source="rag")
        body = json.dumps({"content": "some untrusted data here"}).encode()
        spans = reg.find_spans(body)
        assert len(spans) >= 1
        # Verify the span points to the right content
        for span in spans:
            found = body[span.start:span.end].decode()
            assert "untrusted data" in found or found == "untrusted data"

    def test_find_spans_json_escaped_content(self):
        """Content with special JSON chars (quotes, backslash) is found correctly."""
        reg = TrustRegistry()
        content_with_quotes = 'He said "hello"'
        reg.register(content_with_quotes, source="user")
        body = json.dumps({"content": f"Prefix: {content_with_quotes}"}).encode()
        spans = reg.find_spans(body)
        assert len(spans) >= 1
        assert all(s.source == "user" for s in spans)

    def test_find_spans_unicode_content(self):
        reg = TrustRegistry()
        reg.register("こんにちは世界", source="rag")
        body = json.dumps({"content": "Greeting: こんにちは世界"}).encode()
        spans = reg.find_spans(body)
        assert len(spans) >= 1

    def test_find_spans_multiple_occurrences(self):
        reg = TrustRegistry()
        reg.register("needle", source="rag")
        body = json.dumps({"a": "needle in a", "b": "needle stack"}).encode()
        spans = reg.find_spans(body)
        assert len(spans) >= 2

    def test_find_spans_multiple_entries(self):
        reg = TrustRegistry()
        reg.register("alpha", source="src_a")
        reg.register("beta", source="src_b")
        body = json.dumps({"content": "alpha and beta together"}).encode()
        spans = reg.find_spans(body)
        sources = {s.source for s in spans}
        assert "src_a" in sources
        assert "src_b" in sources

    def test_find_spans_sorted_by_offset(self):
        reg = TrustRegistry()
        reg.register("bbb", source="b")
        reg.register("aaa", source="a")
        body = json.dumps({"content": "aaa then bbb"}).encode()
        spans = reg.find_spans(body)
        assert len(spans) >= 2
        offsets = [s.start for s in spans]
        assert offsets == sorted(offsets)

    def test_find_spans_no_match(self):
        reg = TrustRegistry()
        reg.register("not present", source="rag")
        body = json.dumps({"content": "something else entirely"}).encode()
        spans = reg.find_spans(body)
        assert spans == []


class TestContextVarIntegration:
    def test_get_registry_default_none(self):
        _clear_registry()
        assert get_registry() is None

    def test_ensure_registry_creates(self):
        _clear_registry()
        reg = _ensure_registry()
        assert reg is not None
        assert isinstance(reg, TrustRegistry)

    def test_ensure_registry_returns_same(self):
        _clear_registry()
        reg1 = _ensure_registry()
        reg2 = _ensure_registry()
        assert reg1 is reg2

    def test_clear_registry(self):
        _clear_registry()
        reg = _ensure_registry()
        reg.register("test")
        _clear_registry()
        assert get_registry() is None


class TestPublicApi:
    """Test the parapet.untrusted() public function."""

    def test_untrusted_registers_and_returns(self):
        _clear_registry()
        from parapet import untrusted
        result = untrusted("rag content", source="rag")
        assert result == "rag content"
        reg = get_registry()
        assert reg is not None
        assert reg.entry_count == 1

    def test_untrusted_default_source(self):
        _clear_registry()
        from parapet import untrusted
        untrusted("data")
        reg = get_registry()
        assert reg is not None
        # Verify it was registered (source defaults to "unknown")
        spans = reg.find_spans(json.dumps({"x": "data"}).encode())
        assert len(spans) >= 1
        assert spans[0].source == "unknown"
