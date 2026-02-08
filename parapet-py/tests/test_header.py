"""Tests for parapet.header — W3C Baggage serialization."""
import pytest

from parapet.header import build_baggage_header


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
