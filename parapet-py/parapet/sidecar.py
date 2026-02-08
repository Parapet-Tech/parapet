"""Engine process lifecycle management.

Manages spawning the Rust engine as a subprocess, PID file tracking,
stale PID cleanup, heartbeat watchdog, and clean shutdown via atexit.
"""
from __future__ import annotations

import atexit
import logging
import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO

import httpx

__all__ = ["EngineState", "start_engine", "stop_engine"]

logger = logging.getLogger("parapet.sidecar")

_HEARTBEAT_INTERVAL_SECONDS = 10
_PARAPET_DIR = Path.home() / ".parapet"
_DEFAULT_PID_PATH = _PARAPET_DIR / "engine.pid"
_DEFAULT_LOG_PATH = _PARAPET_DIR / "engine.log"


@dataclass
class EngineState:
    """Holds the state of the running engine sidecar.

    This is not global mutable state scattered across the module --
    it is a single, explicit object passed to functions that need it.
    """

    process: subprocess.Popen | None = None
    initialized: bool = False
    port: int | None = None
    _log_file: IO[bytes] | None = field(default=None, repr=False)
    _heartbeat_thread: threading.Thread | None = field(
        default=None, repr=False
    )
    _heartbeat_stop: threading.Event = field(
        default_factory=threading.Event, repr=False
    )


def _is_process_alive(pid: int) -> bool:
    """Check whether a process with the given PID is still running."""
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def _start_heartbeat(state: EngineState, port: int) -> None:
    """Start a daemon thread that pings the engine heartbeat endpoint."""

    def _heartbeat_loop() -> None:
        while not state._heartbeat_stop.is_set():
            try:
                httpx.get(
                    f"http://127.0.0.1:{port}/v1/heartbeat",
                    timeout=5.0,
                )
            except Exception:
                logger.debug("Heartbeat send failed (engine may be starting)")
            state._heartbeat_stop.wait(_HEARTBEAT_INTERVAL_SECONDS)

    thread = threading.Thread(
        target=_heartbeat_loop,
        name="parapet-heartbeat",
        daemon=True,
    )
    thread.start()
    state._heartbeat_thread = thread


def start_engine(
    config_path: str,
    port: int,
    *,
    state: EngineState,
    pid_path: Path = _DEFAULT_PID_PATH,
    log_path: Path = _DEFAULT_LOG_PATH,
) -> None:
    """Spawn the Rust engine as a subprocess.

    Idempotent: if the engine is already running (tracked via *state*),
    this is a no-op. If a PID file exists from a previous run but the
    process is dead, the stale PID is cleaned up and a fresh engine
    is started.

    Args:
        config_path: Path to the parapet.yaml configuration file.
        port: Port for the engine to listen on.
        state: Mutable engine state object.
        pid_path: Path to write the engine PID file.
        log_path: Path to write the engine log file.
    """
    # Already running in this process -- reuse.
    if state.initialized and state.process is not None:
        if state.process.poll() is None:
            logger.debug("Engine already running (pid=%d), reusing", state.process.pid)
            return
        # Process died unexpectedly; fall through to restart.
        logger.warning("Engine process died unexpectedly, restarting")
        state.process = None
        state.initialized = False

    # Check for stale PID file from a previous run.
    if pid_path.exists():
        try:
            old_pid = int(pid_path.read_text().strip())
        except (ValueError, OSError):
            old_pid = None

        if old_pid is not None and _is_process_alive(old_pid):
            # Another engine is still running -- reuse it.
            logger.info(
                "Reusing existing engine from PID file (pid=%d)", old_pid
            )
            state.initialized = True
            state.port = port
            _start_heartbeat(state, port)
            return

        # Stale PID file -- clean up.
        logger.info("Cleaning up stale PID file (pid=%s)", old_pid)
        pid_path.unlink(missing_ok=True)

    # Ensure the parapet directory exists.
    pid_path.parent.mkdir(parents=True, exist_ok=True)

    # Open log file for engine stdout/stderr.
    log_file: IO[bytes] = open(log_path, "ab")

    # Spawn the engine process.
    proc = subprocess.Popen(
        [
            "parapet-engine",
            "--config",
            config_path,
            "--port",
            str(port),
        ],
        stdout=log_file,
        stderr=log_file,
    )

    state.process = proc
    state.initialized = True
    state.port = port
    state._log_file = log_file

    # Write PID file.
    pid_path.write_text(str(proc.pid))
    logger.info("Engine started (pid=%d, port=%d)", proc.pid, port)

    # Start heartbeat watchdog.
    _start_heartbeat(state, port)

    # Register atexit handler for clean shutdown.
    atexit.register(stop_engine, state=state, pid_path=pid_path)


def stop_engine(
    *,
    state: EngineState,
    pid_path: Path = _DEFAULT_PID_PATH,
) -> None:
    """Terminate the engine subprocess and clean up.

    Args:
        state: The engine state to clean up.
        pid_path: Path to the PID file to remove.
    """
    # Signal heartbeat thread to stop.
    state._heartbeat_stop.set()

    if state.process is not None:
        logger.info("Stopping engine (pid=%d)", state.process.pid)
        try:
            state.process.terminate()
            state.process.wait(timeout=5)
        except Exception:
            logger.warning("Engine did not stop gracefully, killing")
            try:
                state.process.kill()
            except Exception:
                pass

    # Close the log file handle.
    if state._log_file is not None:
        try:
            state._log_file.close()
        except Exception:
            pass
        state._log_file = None

    state.process = None
    state.initialized = False

    if pid_path.exists():
        pid_path.unlink(missing_ok=True)
        logger.debug("Removed PID file %s", pid_path)
