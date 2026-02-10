# Copyright 2026 The Parapet Project
# SPDX-License-Identifier: Apache-2.0

"""Tests for parapet.header — W3C Baggage and X-Guard-Trust serialization."""
import base64
import json

import pytest

from parapet.header import build_baggage_header, build_trust_header
from parapet.trust import TrustSpan


class TestBuildBaggageHeader:
    """Tests for baggage header construction."""

    def test_user_id_and_role(self):
        """Baggage header with both user_id and role produces correct format."""
        result = build_baggage_header(user_id="u_1", role="admin")
        assert result == "user_id=u_1,role=admin"

    def test_user_id_only(self):
        """Baggage header with only user_id omits role."""
        result = build_baggage_header(user_id="u_42")
        assert result == "user_id=u_42"

    def test_role_only(self):
        """Baggage header with only role omits user_id."""
        result = build_baggage_header(role="viewer")
        assert result == "role=viewer"

    def test_no_fields_returns_empty(self):
        """Baggage header with no fields returns empty string."""
        result = build_baggage_header()
        assert result == ""

    def test_special_characters_percent_encoded(self):
        """Values with special characters are percent-encoded per RFC 8941."""
        result = build_baggage_header(user_id="user with spaces")
        assert "user_id=user%20with%20spaces" in result

    def test_comma_in_value_encoded(self):
        """Commas in values are percent-encoded to avoid ambiguity."""
        result = build_baggage_header(user_id="a,b")
        assert "user_id=a%2Cb" in result

    def test_equals_in_value_encoded(self):
        """Equals signs in values are percent-encoded."""
        result = build_baggage_header(role="a=b")
        assert "role=a%3Db" in result

    def test_semicolon_in_value_encoded(self):
        """Semicolons in values are percent-encoded."""
        result = build_baggage_header(user_id="a;b")
        assert "user_id=a%3Bb" in result

    def test_unicode_in_value_encoded(self):
        """Unicode characters in values are percent-encoded."""
        result = build_baggage_header(user_id="usr_\u00e9")
        # Should be percent-encoded UTF-8 bytes
        assert "user_id=" in result
        assert "\u00e9" not in result  # Raw unicode should not appear


class TestBuildTrustHeader:
    """Tests for X-Guard-Trust header serialization."""

    def test_empty_spans_returns_none(self):
        """Empty span list returns None (no header needed)."""
        assert build_trust_header([]) is None

    def test_single_span_round_trip(self):
        """Single span serializes and can be decoded back correctly."""
        spans = [TrustSpan(start=10, end=20, source="rag")]
        header = build_trust_header(spans)
        assert header is not None
        assert header.startswith("inline:")

        # Decode and verify
        encoded = header[len("inline:"):]
        decoded = json.loads(base64.b64decode(encoded))
        assert decoded == [{"s": 10, "e": 20, "src": "rag"}]

    def test_multiple_spans(self):
        """Multiple spans are all included in the header."""
        spans = [
            TrustSpan(start=0, end=10, source="rag"),
            TrustSpan(start=20, end=30, source="user"),
        ]
        header = build_trust_header(spans)
        assert header is not None
        decoded = json.loads(base64.b64decode(header[len("inline:"):]))
        assert len(decoded) == 2
        assert decoded[0] == {"s": 0, "e": 10, "src": "rag"}
        assert decoded[1] == {"s": 20, "e": 30, "src": "user"}

    def test_oversized_header_returns_none(self):
        """Header exceeding 4KB limit returns None (graceful degradation)."""
        # Create enough spans to exceed 4KB
        spans = [TrustSpan(start=i, end=i + 100, source=f"source_{i}") for i in range(200)]
        header = build_trust_header(spans)
        assert header is None

    def test_compact_json_format(self):
        """JSON payload uses compact separators (no extra spaces)."""
        spans = [TrustSpan(start=5, end=15, source="web")]
        header = build_trust_header(spans)
        encoded = header[len("inline:"):]
        raw_json = base64.b64decode(encoded).decode("utf-8")
        # Should have no spaces in the JSON (compact separators)
        assert " " not in raw_json
