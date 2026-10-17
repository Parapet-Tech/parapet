# Copyright 2026 The Parapet Project
# SPDX-License-Identifier: Apache-2.0

# Parapet Python SDK — defined in M1.11
#
# Public API:
#   parapet.init(config_path)  — validate config, start engine sidecar, patch httpx
#   parapet.session(...)       — context manager that sets W3C Baggage for the scope
#   parapet.shutdown()         — stop owned engine if permitted by file-cleanup gate
#   parapet.untrusted(content) — mark a string as untrusted content
from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from parapet.header import _active_baggage, build_baggage_header
from parapet.sidecar import EngineState, start_engine, stop_engine
from parapet.transport import patch_httpx
from parapet.trust import _ensure_registry, _restore_registry, _swap_registry
from typing import Iterable

__all__ = ["init", "session", "shutdown", "untrusted"]

logger = logging.getLogger("parapet")

_DEFAULT_PORT = 9800

# Module-level engine state — single instance shared across init/session.
_engine_state = EngineState()


class SessionContext:
    """Holds the baggage string for the duration of a session scope."""

    __slots__ = ("baggage",)

    def __init__(self, baggage: str) -> None:
        self.baggage = baggage


def init(
    config_path: str,
    *,
    port: int = _DEFAULT_PORT,
    engine_bin: str | None = None,
    extra_hosts: Iterable[str] | None = None,
    state_dir: str | Path | None = None,
    engine_args: Iterable[str] | None = None,
) -> None:
    """Initialize the parapet SDK.

    Validates that *config_path* exists, starts the engine sidecar (if not
    already running), and patches httpx to route LLM requests through the
    engine.

    Args:
        config_path: Path to the parapet.yaml configuration file.
        port: Port for the engine to listen on (default 9800).
        engine_bin: Explicit path to the ``parapet-engine`` binary.
            Falls back to ``PARAPET_ENGINE_PATH`` env var, then ``PATH``.
        extra_hosts: Additional LLM API hosts to intercept beyond the
            built-in defaults (e.g., ``["api.together.xyz"]``).
        state_dir: Optional private state directory for receipt evidence
            and the ``parapet-control/v0`` channel.
        engine_args: Optional additional CLI flags passed to ``parapet-engine``.

    Raises:
        FileNotFoundError: If *config_path* does not exist or the engine
            binary cannot be found.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}"
        )

    hosts = frozenset(extra_hosts) if extra_hosts else None
    resolved_state_dir = Path(state_dir) if state_dir is not None else None
    extra_engine_args = list(engine_args) if engine_args is not None else None
    start_engine(
        config_path=config_path,
        port=port,
        engine_bin=engine_bin,
        state=_engine_state,
        state_dir=resolved_state_dir,
        engine_args=extra_engine_args,
    )
    patch_httpx(port=port, extra_hosts=hosts)


def shutdown() -> bool:
    """Stop the owned engine sidecar if permitted by the file-cleanup gate.

    Returns:
        ``True`` if an owned engine was stopped, ``False`` if the engine is
        not owned or shutdown was refused by the cleanup gate.
    """
    return stop_engine(state=_engine_state)


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
    baggage_token = _active_baggage.set(baggage)
    trust_token = _swap_registry()
    try:
        yield SessionContext(baggage=baggage)
    finally:
        _restore_registry(trust_token)
        _active_baggage.reset(baggage_token)


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
