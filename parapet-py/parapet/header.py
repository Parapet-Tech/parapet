# Copyright 2026 The Parapet Project
# SPDX-License-Identifier: Apache-2.0

"""Header serialization for Parapet SDK.

Includes:
- W3C Baggage header (``baggage: key1=value1,key2=value2``)
- X-Guard-Trust header (``inline:<base64-encoded JSON>``)

Values are percent-encoded to handle special characters safely.
"""
from __future__ import annotations

import base64
import json
import logging
from urllib.parse import quote

__all__ = ["build_baggage_header", "build_trust_header"]

MAX_TRUST_HEADER_BYTES = 4096  # 4KB inline limit

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


def build_trust_header(spans: list) -> str | None:
    """Serialize trust spans as X-Guard-Trust header value.

    Format: ``inline:<base64-encoded JSON>``
    JSON schema: ``[{"s": start, "e": end, "src": source}, ...]``

    Returns None if no spans or if encoded value exceeds 4KB limit.
    Compact keys (s/e/src) to minimize header size.
    """
    if not spans:
        return None

    compact = [{"s": s.start, "e": s.end, "src": s.source} for s in spans]
    payload = json.dumps(compact, separators=(",", ":")).encode("utf-8")

    if len(payload) > MAX_TRUST_HEADER_BYTES:
        logging.getLogger("parapet.header").warning(
            "X-Guard-Trust header exceeds %d byte limit (%d spans, %d bytes); dropping",
            MAX_TRUST_HEADER_BYTES,
            len(spans),
            len(payload),
        )
        return None

    encoded = base64.b64encode(payload).decode("ascii")
    return f"inline:{encoded}"
