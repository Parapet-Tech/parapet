# Parapet Python SDK — defined in M1.11
#
# Public API:
#   parapet.init(config_path)  — validate config, start engine sidecar, patch httpx
#   parapet.session(...)       — context manager that sets W3C Baggage for the scope
from __future__ import annotations

import logging
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Generator

from parapet.header import build_baggage_header
from parapet.sidecar import EngineState, start_engine
from parapet.transport import patch_httpx

__all__ = ["init", "session"]

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
) -> None:
    """Initialize the parapet SDK.

    Validates that *config_path* exists, starts the engine sidecar (if not
    already running), and patches httpx to route LLM requests through the
    engine.

    Args:
        config_path: Path to the parapet.yaml configuration file.
        port: Port for the engine to listen on (default 9800).

    Raises:
        FileNotFoundError: If *config_path* does not exist.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}"
        )

    start_engine(config_path=config_path, port=port, state=_engine_state)
    patch_httpx(port=port)


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
