# Parapet Python SDK — defined in M1.11
#
# Public API:
#   parapet.init(config_path)  — validate config, start engine sidecar, patch httpx
#   parapet.session(...)       — context manager that sets W3C Baggage for the scope
#   parapet.untrusted(content) — mark a string as untrusted content
from __future__ import annotations

import logging
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Generator

from parapet.header import build_baggage_header
from parapet.sidecar import EngineState, start_engine
from parapet.transport import patch_httpx
from parapet.trust import _ensure_registry
from typing import Iterable

__all__ = ["init", "session", "untrusted"]

logger = logging.getLogger("parapet")

_DEFAULT_PORT = 9800

# Module-level engine state — single instance shared across init/session.
_engine_state = EngineState()

# Active baggage for the current context (set inside session(), cleared on exit).
_active_baggage: ContextVar[str | None] = ContextVar(
    "parapet_active_baggage", default=None
)


class SessionContext:
    """Holds the baggage string for the duration of a session scope."""

    __slots__ = ("baggage",)

    def __init__(self, baggage: str) -> None:
        self.baggage = baggage


def init(
    config_path: str,
    *,
    port: int = _DEFAULT_PORT,
    extra_hosts: Iterable[str] | None = None,
) -> None:
    """Initialize the parapet SDK.

    Validates that *config_path* exists, starts the engine sidecar (if not
    already running), and patches httpx to route LLM requests through the
    engine.

    Args:
        config_path: Path to the parapet.yaml configuration file.
        port: Port for the engine to listen on (default 9800).
        extra_hosts: Additional LLM API hosts to intercept beyond the
            built-in defaults (e.g., ``["api.together.xyz"]``).

    Raises:
        FileNotFoundError: If *config_path* does not exist.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}"
        )

    hosts = frozenset(extra_hosts) if extra_hosts else None
    start_engine(config_path=config_path, port=port, state=_engine_state)
    patch_httpx(port=port, extra_hosts=hosts)


@contextmanager
def session(
    *,
    user_id: str | None = None,
    role: str | None = None,
) -> Generator[SessionContext, None, None]:
    """Set W3C Baggage headers for the duration of this scope.

    Must be called after ``init()``.

    Args:
        user_id: Optional user identifier to include in baggage.
        role: Optional role string to include in baggage.

    Yields:
        A ``SessionContext`` with the computed baggage string.

    Raises:
        RuntimeError: If ``init()`` has not been called.
    """
    if not _engine_state.initialized:
        raise RuntimeError(
            "parapet.session() called before init(). Call parapet.init() first."
        )

    baggage = build_baggage_header(user_id=user_id, role=role)
    token = _active_baggage.set(baggage)
    try:
        yield SessionContext(baggage=baggage)
    finally:
        _active_baggage.reset(token)


def untrusted(content: str, source: str = "unknown") -> str:
    """Mark a string as untrusted content.

    Registers the string in the per-context trust registry so the SDK
    can locate it in serialized requests and emit byte-range trust spans.

    The string is returned unchanged for inline use::

        prompt = f"Context: {parapet.untrusted(rag_snippet, source='rag')}"

    Args:
        content: The untrusted content string.
        source: Provenance label (e.g., "rag", "user_input", "web_search").

    Returns:
        The content string, unchanged.
    """
    registry = _ensure_registry()
    return registry.register(content, source=source)
