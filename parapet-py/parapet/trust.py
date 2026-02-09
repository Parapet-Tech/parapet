"""Byte-range trust registry for marking untrusted content.

SDK users call ``parapet.untrusted(content, source)`` to register strings
that originate from untrusted sources (RAG, user input, web search).

On request intercept, the transport layer builds an Aho-Corasick automaton
from registered strings, searches the serialized request body, and emits
byte ranges as ``X-Guard-Trust`` header spans.
"""
from __future__ import annotations

import json
from contextvars import ContextVar
from dataclasses import dataclass

__all__ = ["TrustRegistry", "TrustSpan", "get_registry"]

# Per-context registry (thread/async-safe via contextvars)
_registry: ContextVar[TrustRegistry | None] = ContextVar(
    "parapet_trust_registry", default=None
)


@dataclass(frozen=True)
class TrustSpan:
    """A byte range in the request body marked as untrusted."""
    start: int
    end: int
    source: str = "unknown"


@dataclass
class _Entry:
    """A registered untrusted string."""
    content: str
    source: str


class TrustRegistry:
    """Registry of untrusted content strings for the current request scope.

    Thread-safe via contextvars -- each async context gets its own registry.

    Usage:
        registry = TrustRegistry()
        registry.register("some RAG content", source="rag")
        spans = registry.find_spans(serialized_body_bytes)
    """

    MAX_ENTRIES = 1000  # Configurable cap per request

    def __init__(self, max_entries: int = MAX_ENTRIES) -> None:
        self._entries: list[_Entry] = []
        self._max_entries = max_entries

    def register(self, content: str, source: str = "unknown") -> str:
        """Register a string as untrusted. Returns the string unchanged.

        Args:
            content: The untrusted content string.
            source: Provenance label (e.g., "rag", "user_input", "web_search").

        Returns:
            The content string, unchanged (for inline use).

        Raises:
            ValueError: If the registry is full (> max_entries).
        """
        if not content:
            return content

        if len(self._entries) >= self._max_entries:
            raise ValueError(
                f"TrustRegistry full: {self._max_entries} entries. "
                "Increase max_entries or reduce untrusted content registrations."
            )

        self._entries.append(_Entry(content=content, source=source))
        return content

    def find_spans(self, body: bytes) -> list[TrustSpan]:
        """Find all registered untrusted strings in a serialized request body.

        Uses a simple multi-pattern search (str.find-based for correctness in v2;
        Aho-Corasick library can be swapped in for performance if needed).

        Handles JSON string escaping: searches for the JSON-escaped form of each
        registered string, since the body is JSON-serialized.

        Args:
            body: The serialized request body bytes.

        Returns:
            List of TrustSpan for each occurrence found, sorted by start offset.
        """
        if not self._entries:
            return []

        text = body.decode("utf-8", errors="replace")
        spans: list[TrustSpan] = []

        for entry in self._entries:
            # Search for JSON-escaped form of the string since the body is JSON.
            # json.dumps adds quotes, so strip them to get the escaped interior.
            escaped = json.dumps(entry.content)[1:-1]

            # Find all occurrences of the escaped string in the body
            start = 0
            while True:
                idx = text.find(escaped, start)
                if idx == -1:
                    break
                # Convert char offset to byte offset
                byte_start = len(text[:idx].encode("utf-8"))
                byte_end = byte_start + len(escaped.encode("utf-8"))
                spans.append(TrustSpan(
                    start=byte_start, end=byte_end, source=entry.source,
                ))
                start = idx + 1

        # Sort by start offset, then by end offset (longer spans first for same start)
        spans.sort(key=lambda s: (s.start, -s.end))
        return spans

    @property
    def entry_count(self) -> int:
        """Number of registered entries."""
        return len(self._entries)

    def clear(self) -> None:
        """Remove all registered entries."""
        self._entries.clear()


def get_registry() -> TrustRegistry | None:
    """Get the trust registry for the current context, if any."""
    return _registry.get()


def _ensure_registry() -> TrustRegistry:
    """Get or create the trust registry for the current context."""
    reg = _registry.get()
    if reg is None:
        reg = TrustRegistry()
        _registry.set(reg)
    return reg


def _clear_registry() -> None:
    """Clear the trust registry for the current context."""
    reg = _registry.get()
    if reg is not None:
        reg.clear()
    _registry.set(None)
