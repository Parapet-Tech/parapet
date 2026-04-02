# Copyright 2026 The Parapet Project
# SPDX-License-Identifier: Apache-2.0

"""Tests for parapet.sidecar — Engine process lifecycle."""
import os
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
        state.initialized = True

        with tempfile.TemporaryDirectory() as tmpdir:
            pid_path = Path(tmpdir) / "engine.pid"
            pid_path.write_text("11111")

            stop_engine(state=state, pid_path=pid_path)

        mock_proc.terminate.assert_called_once()
        assert not pid_path.exists()
        assert state.process is None
        assert state.initialized is False


class TestResolveEngineBinary:
    """_resolve_engine_binary uses 3-tier fallback."""

    def test_explicit_arg_wins(self, monkeypatch):
        monkeypatch.setenv("PARAPET_ENGINE_PATH", "/env/parapet-engine")
        assert _resolve_engine_binary("/explicit/bin") == "/explicit/bin"

    def test_env_var_fallback(self, monkeypatch):
        monkeypatch.setenv("PARAPET_ENGINE_PATH", "/env/parapet-engine")
        assert _resolve_engine_binary(None) == "/env/parapet-engine"

    def test_bare_name_fallback(self, monkeypatch):
        monkeypatch.delenv("PARAPET_ENGINE_PATH", raising=False)
        assert _resolve_engine_binary(None) == "parapet-engine"


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
