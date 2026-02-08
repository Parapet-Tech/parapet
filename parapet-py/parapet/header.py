"""W3C Baggage header serialization.

Produces baggage headers per the W3C Baggage specification.
Format: ``baggage: key1=value1,key2=value2``

Values are percent-encoded to handle special characters safely.
"""
from __future__ import annotations

from urllib.parse import quote

__all__ = ["build_baggage_header"]

# Characters that must be percent-encoded in baggage values.
# W3C Baggage uses a subset of RFC 3986 — commas, semicolons, equals,
# and spaces are delimiters and must not appear unescaped in values.
_SAFE_CHARS = ""  # encode everything except unreserved chars


def _encode_value(value: str) -> str:
    """Percent-encode a baggage value per W3C Baggage spec."""
    return quote(value, safe=_SAFE_CHARS)


def build_baggage_header(
    *,
    user_id: str | None = None,
    role: str | None = None,
) -> str:
    """Build a W3C Baggage header value from optional fields.

    Args:
        user_id: Optional user identifier.
        role: Optional role string.

    Returns:
        Baggage header value string (e.g. ``user_id=u_1,role=admin``).
        Returns empty string when no fields are provided.
    """
    parts: list[str] = []

    if user_id is not None:
        parts.append(f"user_id={_encode_value(user_id)}")

    if role is not None:
        parts.append(f"role={_encode_value(role)}")

    return ",".join(parts)
