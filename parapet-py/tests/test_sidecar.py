# Copyright 2026 The Parapet Project
# SPDX-License-Identifier: Apache-2.0

"""Tests for parapet.sidecar — Engine process lifecycle."""
import os
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import pytest

from parapet.sidecar import EngineState, start_engine, stop_engine, _resolve_engine_binary


class TestStartEngineIdempotent:
    """start_engine called twice reuses the running engine."""

    @patch("parapet.sidecar.subprocess.Popen")
    @patch("parapet.sidecar._start_heartbeat")
    def test_second_call_reuses_engine(self, mock_heartbeat, mock_popen):
        """Calling start_engine twice does not spawn a second process."""
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_proc.poll.return_value = None  # Process is alive
        mock_popen.return_value = mock_proc

        state = EngineState()

        with tempfile.TemporaryDirectory() as tmpdir:
            pid_path = Path(tmpdir) / "engine.pid"
            log_path = Path(tmpdir) / "engine.log"

            start_engine(
                config_path="parapet.yaml",
                port=9800,
                state=state,
                pid_path=pid_path,
                log_path=log_path,
            )
            start_engine(
                config_path="parapet.yaml",
                port=9800,
                state=state,
                pid_path=pid_path,
                log_path=log_path,
            )

            stop_engine(state=state, pid_path=pid_path)

        # Popen should only be called once
        assert mock_popen.call_count == 1


class TestStalePidFile:
    """Stale PID file (dead process) is cleaned up."""

    @patch("parapet.sidecar.subprocess.Popen")
    @patch("parapet.sidecar._start_heartbeat")
    @patch("parapet.sidecar._is_process_alive", return_value=False)
    def test_stale_pid_cleaned_and_new_engine_started(
        self, mock_alive, mock_heartbeat, mock_popen
    ):
        """If PID file exists but process is dead, clean up and start fresh."""
        mock_proc = MagicMock()
        mock_proc.pid = 99999
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc

        state = EngineState()

        with tempfile.TemporaryDirectory() as tmpdir:
            pid_path = Path(tmpdir) / "engine.pid"
            log_path = Path(tmpdir) / "engine.log"

            # Write a stale PID
            pid_path.write_text("11111")

            start_engine(
                config_path="parapet.yaml",
                port=9800,
                state=state,
                pid_path=pid_path,
                log_path=log_path,
            )

            # Clean up before temp dir removal (Windows file locks).
            stop_engine(state=state, pid_path=pid_path)

        # Should have spawned a new process
        assert mock_popen.call_count == 1


class TestPidFileWritten:
    """PID file is written correctly after engine start."""

    @patch("parapet.sidecar.subprocess.Popen")
    @patch("parapet.sidecar._start_heartbeat")
    def test_pid_file_contains_correct_pid(self, mock_heartbeat, mock_popen):
        """PID file should contain the PID of the spawned process."""
        mock_proc = MagicMock()
        mock_proc.pid = 54321
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc

        state = EngineState()

        with tempfile.TemporaryDirectory() as tmpdir:
            pid_path = Path(tmpdir) / "engine.pid"
            log_path = Path(tmpdir) / "engine.log"

            start_engine(
                config_path="parapet.yaml",
                port=9800,
                state=state,
                pid_path=pid_path,
                log_path=log_path,
            )

            assert pid_path.read_text().strip() == "54321"

            stop_engine(state=state, pid_path=pid_path)


class TestStopEngine:
    """stop_engine terminates the process and cleans up."""

    def test_stop_terminates_and_cleans_pid(self):
        """stop_engine terminates the subprocess and removes the PID file."""
        mock_proc = MagicMock()
        mock_proc.pid = 11111
        mock_proc.poll.return_value = None

        state = EngineState()
        state.process = mock_proc
        state.owns_process = True
        state.initialized = True

        with tempfile.TemporaryDirectory() as tmpdir:
            pid_path = Path(tmpdir) / "engine.pid"
            pid_path.write_text("11111")

            assert stop_engine(state=state, pid_path=pid_path) is True

        mock_proc.terminate.assert_called_once()
        assert not pid_path.exists()
        assert state.process is None
        assert state.initialized is False
        assert state.owns_process is False


class TestResolveEngineBinary:
    """_resolve_engine_binary uses the locked 4-step fallback."""

    def test_explicit_arg_wins(self, monkeypatch):
        monkeypatch.setenv("PARAPET_ENGINE_PATH", "/env/parapet-engine")
        monkeypatch.setattr("parapet.sidecar.sysconfig.get_path", lambda key: "/installed/bin")
        assert _resolve_engine_binary("/explicit/bin") == "/explicit/bin"

    def test_env_var_wins_over_installed_script(self, monkeypatch):
        monkeypatch.setenv("PARAPET_ENGINE_PATH", "/env/parapet-engine")
        monkeypatch.setattr("parapet.sidecar.sysconfig.get_path", lambda key: "/installed/bin")
        monkeypatch.setattr("parapet.sidecar.Path.exists", lambda self: True)
        assert _resolve_engine_binary(None) == "/env/parapet-engine"

    def test_installed_script_fallback(self, monkeypatch, tmp_path):
        monkeypatch.delenv("PARAPET_ENGINE_PATH", raising=False)
        scripts_dir = tmp_path / "bin"
        scripts_dir.mkdir()
        engine = scripts_dir / "parapet-engine"
        engine.write_text("")
        monkeypatch.setattr("parapet.sidecar.sysconfig.get_path", lambda key: str(scripts_dir))

        assert _resolve_engine_binary(None) == str(engine)

    def test_installed_script_missing_falls_through(self, monkeypatch, tmp_path):
        monkeypatch.delenv("PARAPET_ENGINE_PATH", raising=False)
        scripts_dir = tmp_path / "bin"
        scripts_dir.mkdir()
        monkeypatch.setattr("parapet.sidecar.sysconfig.get_path", lambda key: str(scripts_dir))

        assert _resolve_engine_binary(None) == "parapet-engine"

    def test_bare_name_fallback(self, monkeypatch):
        monkeypatch.delenv("PARAPET_ENGINE_PATH", raising=False)
        monkeypatch.setattr("parapet.sidecar.sysconfig.get_path", lambda key: None)
        assert _resolve_engine_binary(None) == "parapet-engine"

    def test_windows_installed_script_suffix(self, monkeypatch, tmp_path):
        monkeypatch.delenv("PARAPET_ENGINE_PATH", raising=False)
        scripts_dir = tmp_path / "Scripts"
        scripts_dir.mkdir()
        engine = scripts_dir / "parapet-engine.exe"
        engine.write_text("")
        monkeypatch.setattr("parapet.sidecar.sys.platform", "win32")
        monkeypatch.setattr("parapet.sidecar.sysconfig.get_path", lambda key: str(scripts_dir))

        assert _resolve_engine_binary(None) == str(engine)


class TestEngineBinaryNotFound:
    """start_engine raises actionable error when binary is missing."""

    @patch("parapet.sidecar.subprocess.Popen", side_effect=FileNotFoundError)
    @patch("parapet.sidecar._start_heartbeat")
    def test_missing_binary_actionable_error(self, mock_hb, mock_popen):
        state = EngineState()

        with tempfile.TemporaryDirectory() as tmpdir:
            pid_path = Path(tmpdir) / "engine.pid"
            log_path = Path(tmpdir) / "engine.log"

            with pytest.raises(FileNotFoundError, match="PARAPET_ENGINE_PATH"):
                start_engine(
                    config_path="parapet.yaml",
                    port=9800,
                    state=state,
                    pid_path=pid_path,
                    log_path=log_path,
                )

    @patch("parapet.sidecar.subprocess.Popen")
    @patch("parapet.sidecar._start_heartbeat")
    def test_engine_bin_kwarg_passed_to_popen(self, mock_hb, mock_popen):
        mock_proc = MagicMock()
        mock_proc.pid = 42
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc

        state = EngineState()

        with tempfile.TemporaryDirectory() as tmpdir:
            pid_path = Path(tmpdir) / "engine.pid"
            log_path = Path(tmpdir) / "engine.log"

            start_engine(
                config_path="parapet.yaml",
                port=9800,
                state=state,
                engine_bin="/custom/parapet-engine",
                pid_path=pid_path,
                log_path=log_path,
            )

            cmd = mock_popen.call_args[0][0]
            assert cmd[0] == "/custom/parapet-engine"

            stop_engine(state=state, pid_path=pid_path)


class TestStateDirAndEngineArgs:
    """start_engine wires explicit state_dir and engine_args without touching ~/.parapet."""

    @patch("parapet.sidecar.atexit.register")
    @patch("parapet.sidecar.subprocess.Popen")
    @patch("parapet.sidecar._start_heartbeat")
    def test_state_dir_and_engine_args_passed_to_popen(
        self, mock_hb, mock_popen, mock_atexit, tmp_path
    ):
        mock_proc = MagicMock()
        mock_proc.pid = 7777
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc

        state_dir = tmp_path / "custom_state"
        state = EngineState()

        start_engine(
            config_path="parapet.yaml",
            port=9899,
            state=state,
            engine_bin="/bin/parapet-engine",
            state_dir=state_dir,
            engine_args=["--offline-fixture", "v0", "--ready-file", "ready.port"],
        )

        cmd = mock_popen.call_args[0][0]
        assert cmd == [
            "/bin/parapet-engine",
            "--config",
            "parapet.yaml",
            "--port",
            "9899",
            "--state-dir",
            str(state_dir),
            "--offline-fixture",
            "v0",
            "--ready-file",
            "ready.port",
        ]
        assert state.owns_process is True
        assert state.state_dir == state_dir
        assert (state_dir / "engine.pid").read_text().strip() == "7777"
        assert (state_dir / "engine.log").exists()
        mock_atexit.assert_called_once_with(
            stop_engine, state=state, pid_path=state_dir / "engine.pid"
        )

        stop_engine(state=state)
        assert not (state_dir / "engine.pid").exists()


class TestOwnedVsReusedEngine:
    """Reused engines set owns_process=False and are never stopped or unlinked."""

    @patch("parapet.sidecar.atexit.register")
    @patch("parapet.sidecar.socket.socket")
    @patch("parapet.sidecar.subprocess.Popen")
    @patch("parapet.sidecar._start_heartbeat")
    @patch("parapet.sidecar._is_process_alive", return_value=True)
    def test_reused_engine_from_pid_is_never_stopped_or_unlinked(
        self, mock_alive, mock_hb, mock_popen, mock_socket, mock_atexit, tmp_path
    ):
        state_dir = tmp_path / "reused_state"
        state_dir.mkdir(mode=0o700)
        pid_file = state_dir / "engine.pid"
        sock_file = state_dir / "control.sock"
        token_file = state_dir / "control.token"
        owner_file = state_dir / "engine.owner.json"
        pid_file.write_text("43210")
        sock_file.write_text("")
        token_file.write_text("a" * 64)
        owner_file.write_text('{"schema":"parapet-engine-owner/v0","pid":43210}')

        state = EngineState()
        start_engine(
            config_path="parapet.yaml",
            port=9800,
            state=state,
            state_dir=state_dir,
        )

        assert mock_popen.call_count == 0
        assert state.initialized is True
        assert state.owns_process is False
        assert state.process is None
        assert state.state_dir == state_dir

        # Explicit stop_engine must be a strict no-op on remote process and files.
        assert stop_engine(state=state) is False
        mock_socket.assert_not_called()
        assert pid_file.exists()
        assert sock_file.exists()
        assert token_file.exists()
        assert owner_file.exists()

        # Registered atexit callback must also be a strict no-op.
        atexit_fn = mock_atexit.call_args[0][0]
        atexit_kwargs = mock_atexit.call_args[1]
        assert atexit_fn(**atexit_kwargs) is False
        mock_socket.assert_not_called()
        assert pid_file.exists()
        assert sock_file.exists()
        assert token_file.exists()
        assert owner_file.exists()

    @patch("parapet.sidecar.atexit.register")
    @patch("parapet.sidecar.socket.socket")
    @patch("parapet.sidecar.subprocess.Popen")
    @patch("parapet.sidecar._start_heartbeat")
    @patch("parapet.sidecar._is_process_alive", return_value=True)
    def test_reused_engine_from_owner_json_is_never_stopped(
        self, mock_alive, mock_hb, mock_popen, mock_socket, mock_atexit, tmp_path
    ):
        state_dir = tmp_path / "owner_reused_state"
        state_dir.mkdir(mode=0o700)
        sock_file = state_dir / "control.sock"
        token_file = state_dir / "control.token"
        owner_file = state_dir / "engine.owner.json"
        sock_file.write_text("")
        token_file.write_text("d" * 64)
        owner_file.write_text('{"schema":"parapet-engine-owner/v0","pid":54321}')

        state = EngineState()
        start_engine(
            config_path="parapet.yaml",
            port=9800,
            state=state,
            state_dir=state_dir,
        )

        assert mock_popen.call_count == 0
        assert state.initialized is True
        assert state.owns_process is False
        assert state.process is None
        assert stop_engine(state=state) is False
        mock_socket.assert_not_called()
        assert sock_file.exists()
        assert token_file.exists()
        assert owner_file.exists()


def _make_mock_unix_socket(response_line: bytes | Exception):
    """Build a context-manager mock for socket.socket(AF_UNIX, SOCK_STREAM)."""
    mock_sock = MagicMock()
    mock_sock.__enter__.return_value = mock_sock
    mock_sock.__exit__.return_value = False
    if isinstance(response_line, Exception):
        mock_sock.recv.side_effect = response_line
    else:
        mock_sock.recv.side_effect = [response_line, b""]
    return mock_sock


class TestAuthenticatedShutdownGate:
    """Owned engines with control channel / effect root obey the disable cleanup gate."""

    @pytest.mark.parametrize(
        "cleanup_code",
        ["not_run", "fail", "indeterminate"],
    )
    @patch("parapet.sidecar.atexit.register")
    @patch("parapet.sidecar.socket.socket")
    @patch("parapet.sidecar.subprocess.Popen")
    @patch("parapet.sidecar._start_heartbeat")
    def test_refused_disable_gate_leaves_engine_and_control_files_intact(
        self,
        mock_hb,
        mock_popen,
        mock_socket_cls,
        mock_atexit,
        cleanup_code,
        tmp_path,
    ):
        import json

        mock_proc = MagicMock()
        mock_proc.pid = 55555
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc

        state_dir = tmp_path / "state"
        state = EngineState()
        start_engine(
            config_path="parapet.yaml",
            port=9800,
            state=state,
            state_dir=state_dir,
            engine_args=["--effect-root", str(tmp_path / "effect")],
        )

        token_hex = "b" * 64
        sock_file = state_dir / "control.sock"
        token_file = state_dir / "control.token"
        owner_file = state_dir / "engine.owner.json"
        control_json = state_dir / "control.json"
        sock_file.write_text("")
        token_file.write_text(token_hex)
        owner_file.write_text('{"schema":"parapet-engine-owner/v0","pid":55555}')
        control_json.write_text('{"schema":"parapet-control-state/v0","effect_root_must_be_new":true}')

        refusal_bytes = (
            json.dumps(
                {
                    "schema": "parapet-control/v0",
                    "ok": False,
                    "stop_approved": False,
                    "file_cleanup": cleanup_code,
                    "evidence_retained": True,
                }
            ).encode("utf-8")
            + b"\n"
        )

        # 1. Explicit stop_engine must refuse to terminate, kill, or unlink.
        mock_sock_1 = _make_mock_unix_socket(refusal_bytes)
        mock_socket_cls.return_value = mock_sock_1
        assert stop_engine(state=state) is False

        mock_sock_1.settimeout.assert_called_once_with(5.0)
        mock_sock_1.connect.assert_called_once_with(str(sock_file))
        sent_req = json.loads(mock_sock_1.sendall.call_args[0][0].decode("utf-8").strip())
        assert sent_req == {
            "schema": "parapet-control/v0",
            "token": token_hex,
            "action": "disable",
        }
        mock_proc.terminate.assert_not_called()
        mock_proc.kill.assert_not_called()
        assert state.process is mock_proc
        assert state.owns_process is True
        assert (state_dir / "engine.pid").exists()
        assert sock_file.exists()
        assert token_file.exists()
        assert owner_file.exists()

        # 2. Registered atexit handler must also refuse when gate returns stop_approved=False.
        mock_sock_2 = _make_mock_unix_socket(refusal_bytes)
        mock_socket_cls.return_value = mock_sock_2
        atexit_fn = mock_atexit.call_args[0][0]
        atexit_kwargs = mock_atexit.call_args[1]
        assert atexit_fn(**atexit_kwargs) is False

        mock_proc.terminate.assert_not_called()
        mock_proc.kill.assert_not_called()
        assert (state_dir / "engine.pid").exists()
        assert sock_file.exists()
        assert token_file.exists()
        assert owner_file.exists()

        # 3. Subsequent restore -> clear -> stop_approved=True allows clean stop & unlink.
        approved_bytes = (
            json.dumps(
                {
                    "schema": "parapet-control/v0",
                    "ok": True,
                    "stop_approved": True,
                    "file_cleanup": "clear",
                    "evidence_retained": True,
                }
            ).encode("utf-8")
            + b"\n"
        )
        mock_sock_3 = _make_mock_unix_socket(approved_bytes)
        mock_socket_cls.return_value = mock_sock_3
        assert stop_engine(state=state) is True

        mock_proc.wait.assert_called_once_with(timeout=5)
        mock_proc.kill.assert_not_called()
        assert not (state_dir / "engine.pid").exists()
        assert not sock_file.exists()
        assert not token_file.exists()
        assert not owner_file.exists()
        assert control_json.exists()
        assert state.process is None
        assert state.owns_process is False
        assert state.initialized is False

    @patch("parapet.sidecar.atexit.register")
    @patch("parapet.sidecar.socket.socket")
    @patch("parapet.sidecar.subprocess.Popen")
    @patch("parapet.sidecar._start_heartbeat")
    def test_timeout_or_missing_control_files_never_kills_or_unlinks(
        self, mock_hb, mock_popen, mock_socket_cls, mock_atexit, tmp_path
    ):
        mock_proc = MagicMock()
        mock_proc.pid = 66666
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc

        state_dir = tmp_path / "state_timeout"
        state = EngineState()
        start_engine(
            config_path="parapet.yaml",
            port=9800,
            state=state,
            state_dir=state_dir,
            engine_args=["--effect-root", str(tmp_path / "effect")],
        )

        sock_file = state_dir / "control.sock"
        token_file = state_dir / "control.token"
        sock_file.write_text("")
        token_file.write_text("c" * 64)

        mock_sock = _make_mock_unix_socket(TimeoutError("control socket timed out"))
        mock_socket_cls.return_value = mock_sock

        assert stop_engine(state=state) is False
        mock_proc.terminate.assert_not_called()
        mock_proc.kill.assert_not_called()
        assert (state_dir / "engine.pid").exists()
        assert sock_file.exists()
        assert token_file.exists()

        # Even if control.sock is missing while --effect-root is active, atexit refuses to kill.
        sock_file.unlink()
        atexit_fn = mock_atexit.call_args[0][0]
        atexit_kwargs = mock_atexit.call_args[1]
        assert atexit_fn(**atexit_kwargs) is False
        mock_proc.terminate.assert_not_called()
        mock_proc.kill.assert_not_called()
        assert (state_dir / "engine.pid").exists()
        assert token_file.exists()

    @patch("parapet.sidecar._request_control_disable", return_value=True)
    def test_approved_gate_wait_timeout_preserves_live_state(
        self, mock_disable, tmp_path
    ):
        state_dir = tmp_path / "state_slow_exit"
        state_dir.mkdir(mode=0o700)
        for name in (
            "control.sock",
            "control.token",
            "engine.owner.json",
            "engine.pid",
            "engine.meta",
        ):
            (state_dir / name).write_text("owned", encoding="utf-8")

        mock_proc = MagicMock()
        mock_proc.pid = 77777
        mock_proc.wait.side_effect = subprocess.TimeoutExpired(
            cmd="parapet-engine", timeout=5
        )
        state = EngineState(
            process=mock_proc,
            initialized=True,
            owns_process=True,
            state_dir=state_dir,
            engine_args=["--effect-root", str(tmp_path / "effect")],
        )

        assert stop_engine(state=state) is False

        mock_disable.assert_called_once_with(state_dir)
        mock_proc.wait.assert_called_once_with(timeout=5)
        mock_proc.terminate.assert_not_called()
        mock_proc.kill.assert_not_called()
        assert state.process is mock_proc
        assert state.initialized is True
        assert state.owns_process is True
        for name in (
            "control.sock",
            "control.token",
            "engine.owner.json",
            "engine.pid",
            "engine.meta",
        ):
            assert (state_dir / name).exists()
