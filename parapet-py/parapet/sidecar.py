# Copyright 2026 The Parapet Project
# SPDX-License-Identifier: Apache-2.0

"""Engine process lifecycle management.

Manages spawning the Rust engine as a subprocess, PID file tracking,
stale PID cleanup, heartbeat watchdog, and clean shutdown via atexit.
"""
from __future__ import annotations

import atexit
from collections.abc import Iterable
import json
import logging
import os
import socket
import subprocess
import sys
import sysconfig
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO

import httpx

__all__ = ["EngineState", "start_engine", "stop_engine"]

logger = logging.getLogger("parapet.sidecar")

_HEARTBEAT_INTERVAL_SECONDS = 10
_CONTROL_TIMEOUT_SECONDS = 5.0
_CONTROL_SCHEMA = "parapet-control/v0"
_MAX_CONTROL_LINE_BYTES = 4096
_PARAPET_DIR = Path.home() / ".parapet"
_DEFAULT_PID_PATH = _PARAPET_DIR / "engine.pid"
_DEFAULT_LOG_PATH = _PARAPET_DIR / "engine.log"
_OWNED_TRANSIENT_FILES = (
    "control.sock",
    "control.token",
    "engine.owner.json",
    "engine.pid",
    "engine.meta",
)


@dataclass
class EngineState:
    """Holds the state of the running engine sidecar.

    This is not global mutable state scattered across the module --
    it is a single, explicit object passed to functions that need it.
    """

    process: subprocess.Popen | None = None
    initialized: bool = False
    port: int | None = None
    owns_process: bool = False
    state_dir: Path | None = None
    engine_args: list[str] = field(default_factory=list)
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


def _has_flag(args: list[str], flag: str) -> bool:
    """Return True if *flag* or ``flag=...`` is present in *args*."""
    return any(arg == flag or arg.startswith(f"{flag}=") for arg in args)


def _extract_state_dir_from_args(engine_args: list[str]) -> Path | None:
    """Extract a ``--state-dir`` path from *engine_args* if present."""
    for idx, arg in enumerate(engine_args):
        if arg == "--state-dir" and idx + 1 < len(engine_args):
            return Path(engine_args[idx + 1])
        if arg.startswith("--state-dir="):
            return Path(arg.split("=", 1)[1])
    return None


def _read_owner_pid(state_dir: Path) -> int | None:
    """Read the owning engine PID from ``<state_dir>/engine.owner.json`` if present."""
    owner_path = state_dir / "engine.owner.json"
    try:
        if not owner_path.exists():
            return None
        data = json.loads(owner_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    pid = data.get("pid")
    return pid if isinstance(pid, int) and pid > 0 else None


def _requires_control_gate(state: EngineState, control_dir: Path | None) -> bool:
    """Return True when stopping an owned engine must pass the control-socket gate."""
    if _has_flag(state.engine_args, "--effect-root"):
        return True
    if control_dir is None:
        return False
    return (
        (control_dir / "control.sock").exists()
        or (control_dir / "control.token").exists()
        or (control_dir / "engine.owner.json").exists()
    )


def _request_control_disable(
    state_dir: Path,
    timeout: float = _CONTROL_TIMEOUT_SECONDS,
) -> bool:
    """Send an authenticated ``disable`` request over ``<state_dir>/control.sock``.

    Returns ``True`` only when the control channel responds with
    ``schema == "parapet-control/v0"``, ``ok is True``, ``stop_approved is True``,
    and ``file_cleanup`` in ``("clear", "not_applicable")``.
    """
    if not hasattr(socket, "AF_UNIX"):
        return False

    sock_path = state_dir / "control.sock"
    token_path = state_dir / "control.token"
    try:
        if not sock_path.exists() or not token_path.exists():
            return False
        token = token_path.read_text(encoding="utf-8").strip()
        if not token:
            return False
    except OSError:
        return False

    payload = (
        json.dumps(
            {
                "schema": _CONTROL_SCHEMA,
                "token": token,
                "action": "disable",
            },
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )

    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout)
            sock.connect(str(sock_path))
            sock.sendall(payload)
            chunks: list[bytes] = []
            received = 0
            while received <= _MAX_CONTROL_LINE_BYTES:
                chunk = sock.recv(1024)
                if not chunk:
                    break
                chunks.append(chunk)
                received += len(chunk)
                if b"\n" in chunk:
                    break
    except OSError:
        return False

    if not chunks or received > _MAX_CONTROL_LINE_BYTES:
        return False

    raw_line = b"".join(chunks).split(b"\n", 1)[0].strip()
    if not raw_line:
        return False

    try:
        resp = json.loads(raw_line.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return False

    if not isinstance(resp, dict):
        return False

    return (
        resp.get("schema") == _CONTROL_SCHEMA
        and resp.get("ok") is True
        and resp.get("stop_approved") is True
        and resp.get("file_cleanup") in ("clear", "not_applicable")
    )


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


def _resolve_engine_binary(engine_bin: str | None) -> str:
    """Resolve the engine binary path.

    Resolution order:
      1. Explicit *engine_bin* argument.
      2. ``PARAPET_ENGINE_PATH`` environment variable.
      3. Installed wheel script in the current environment.
      4. Bare ``parapet-engine`` name (must be on ``PATH``).
    """
    if engine_bin:
        return engine_bin
    from_env = os.environ.get("PARAPET_ENGINE_PATH")
    if from_env:
        return from_env
    executable = "parapet-engine.exe" if sys.platform == "win32" else "parapet-engine"
    scripts_dir = sysconfig.get_path("scripts")
    if scripts_dir:
        installed_script = Path(scripts_dir) / executable
        if installed_script.exists():
            return str(installed_script)
    return "parapet-engine"


def start_engine(
    config_path: str,
    port: int,
    *,
    state: EngineState,
    engine_bin: str | None = None,
    pid_path: Path = _DEFAULT_PID_PATH,
    log_path: Path = _DEFAULT_LOG_PATH,
    state_dir: str | Path | None = None,
    engine_args: Iterable[str] | None = None,
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
        engine_bin: Explicit path to the ``parapet-engine`` binary.
            Falls back to ``PARAPET_ENGINE_PATH`` env var, then ``PATH``.
        pid_path: Path to write the engine PID file.
        log_path: Path to write the engine log file.
        state_dir: Optional private state directory for receipt evidence
            and the ``parapet-control/v0`` channel.
        engine_args: Optional additional CLI flags passed to ``parapet-engine``.
    """
    extra_args = list(engine_args) if engine_args is not None else []
    resolved_state_dir = (
        Path(state_dir)
        if state_dir is not None
        else _extract_state_dir_from_args(extra_args)
    )
    if resolved_state_dir is not None:
        if pid_path == _DEFAULT_PID_PATH:
            pid_path = resolved_state_dir / "engine.pid"
        if log_path == _DEFAULT_LOG_PATH:
            log_path = resolved_state_dir / "engine.log"

    # Already running in this process -- reuse.
    if state.initialized and state.process is not None:
        if state.process.poll() is None:
            logger.debug("Engine already running (pid=%d), reusing", state.process.pid)
            return
        # Process died unexpectedly; fall through to restart.
        logger.warning("Engine process died unexpectedly, restarting")
        state.process = None
        state.initialized = False
        state.owns_process = False

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
            state.process = None
            state.owns_process = False
            state.initialized = True
            state.port = port
            state.state_dir = resolved_state_dir
            state.engine_args = extra_args
            state._heartbeat_stop.clear()
            _start_heartbeat(state, port)
            atexit.register(stop_engine, state=state, pid_path=pid_path)
            return

        # Stale PID file -- clean up.
        logger.info("Cleaning up stale PID file (pid=%s)", old_pid)
        pid_path.unlink(missing_ok=True)
    elif resolved_state_dir is not None:
        owner_pid = _read_owner_pid(resolved_state_dir)
        if owner_pid is not None and _is_process_alive(owner_pid):
            logger.info(
                "Reusing existing engine from engine.owner.json (pid=%d)",
                owner_pid,
            )
            state.process = None
            state.owns_process = False
            state.initialized = True
            state.port = port
            state.state_dir = resolved_state_dir
            state.engine_args = extra_args
            state._heartbeat_stop.clear()
            _start_heartbeat(state, port)
            atexit.register(stop_engine, state=state, pid_path=pid_path)
            return

    # Ensure the state / parapet directory exists.
    if resolved_state_dir is not None:
        resolved_state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        if os.name != "nt":
            try:
                os.chmod(resolved_state_dir, 0o700)
            except OSError:
                pass
    pid_path.parent.mkdir(parents=True, exist_ok=True)

    # Open log file for engine stdout/stderr.
    log_file: IO[bytes] = open(log_path, "ab")

    # Resolve engine binary.
    binary = _resolve_engine_binary(engine_bin)

    cmd = [
        binary,
        "--config",
        config_path,
        "--port",
        str(port),
    ]
    if resolved_state_dir is not None and not _has_flag(extra_args, "--state-dir"):
        cmd.extend(["--state-dir", str(resolved_state_dir)])
    if extra_args:
        cmd.extend(extra_args)

    # Spawn the engine process.
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=log_file,
        )
    except FileNotFoundError:
        log_file.close()
        raise FileNotFoundError(
            f"Engine binary not found: {binary!r}. "
            "Install parapet-engine and add it to PATH, or set "
            "PARAPET_ENGINE_PATH to the binary location, or pass "
            "engine_bin= to init()."
        ) from None

    state.process = proc
    state.owns_process = True
    state.initialized = True
    state.port = port
    state.state_dir = resolved_state_dir
    state.engine_args = extra_args
    state._log_file = log_file
    state._heartbeat_stop.clear()

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
) -> bool:
    """Stop the owned engine subprocess if permitted by the cleanup gate.

    When ``state.owns_process`` is ``False``, this is a strict no-op on the
    remote process and its PID/control files. When ``state.owns_process`` is
    ``True`` and the control gate is active, sends the authenticated
    ``parapet-control/v0`` ``disable`` handshake over ``control.sock`` and
    refuses to terminate, kill, or unlink files unless ``stop_approved`` is
    ``True``.

    Args:
        state: The engine state to clean up.
        pid_path: Path to the PID file to remove.

    Returns:
        ``True`` if an owned engine was stopped, ``False`` if the engine is
        not owned or shutdown was refused by the cleanup gate.
    """
    if state.state_dir is not None and pid_path == _DEFAULT_PID_PATH:
        pid_path = state.state_dir / "engine.pid"

    # Never signal or unlink files for an unowned / reused engine.
    if not state.owns_process:
        state._heartbeat_stop.set()
        if state._log_file is not None:
            try:
                state._log_file.close()
            except Exception:
                pass
            state._log_file = None
        state.process = None
        state.initialized = False
        return False

    control_dir = state.state_dir
    if control_dir is None and pid_path != _DEFAULT_PID_PATH:
        candidate_dir = pid_path.parent
        if (
            (candidate_dir / "control.sock").exists()
            or (candidate_dir / "control.token").exists()
            or (candidate_dir / "engine.owner.json").exists()
        ):
            control_dir = candidate_dir

    if _requires_control_gate(state, control_dir):
        if control_dir is None or not _request_control_disable(control_dir):
            logger.warning(
                "Engine stop refused by control cleanup gate; leaving engine and control state intact"
            )
            return False

        state._heartbeat_stop.set()
        if state.process is not None:
            logger.info("Stopping engine after approved control disable (pid=%d)", state.process.pid)
            try:
                state.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning(
                    "Engine did not exit after approved control disable; leaving control state intact"
                )
                return False

        if state._log_file is not None:
            try:
                state._log_file.close()
            except Exception:
                pass
            state._log_file = None

        state.process = None
        state.initialized = False
        state.owns_process = False

        if pid_path.exists():
            pid_path.unlink(missing_ok=True)
        for name in _OWNED_TRANSIENT_FILES:
            (control_dir / name).unlink(missing_ok=True)
        return True

    # Non-control-gated owned engine shutdown.
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
    state.owns_process = False

    if pid_path.exists():
        pid_path.unlink(missing_ok=True)
        logger.debug("Removed PID file %s", pid_path)

    return True
