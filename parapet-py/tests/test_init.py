# Copyright 2026 The Parapet Project
# SPDX-License-Identifier: Apache-2.0

"""Tests for parapet.__init__ — main API (init, session)."""
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

import parapet
from parapet import _engine_state


class TestInitWithNonexistentYaml:
    """init() with nonexistent YAML file produces a clear error."""

    def setup_method(self):
        """Reset module-level state before each test."""
        _engine_state.initialized = False
        _engine_state.process = None
        _engine_state.port = None

    def test_nonexistent_config_raises_file_not_found(self):
        """init() with a missing config file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="parapet.yaml"):
            parapet.init(config_path="/nonexistent/parapet.yaml")

    def test_no_engine_started_on_config_error(self):
        """When config file is missing, no engine should be started."""
        with pytest.raises(FileNotFoundError):
            parapet.init(config_path="/nonexistent/parapet.yaml")
        assert _engine_state.initialized is False


class TestSessionWithoutInit:
    """session() without init() produces a clear error."""

    def setup_method(self):
        """Reset module-level state before each test."""
        _engine_state.initialized = False
        _engine_state.process = None
        _engine_state.port = None

    def test_session_without_init_raises_runtime_error(self):
        """session() before init() raises RuntimeError."""
        with pytest.raises(RuntimeError, match="init\\(\\)"):
            with parapet.session(user_id="u_1"):
                pass


class TestSessionContextManager:
    """session() context manager sets and clears baggage."""

    def setup_method(self):
        """Reset module-level state before each test."""
        _engine_state.initialized = True
        _engine_state.process = MagicMock()
        _engine_state.port = 9800

    def teardown_method(self):
        """Clean up state after each test."""
        _engine_state.initialized = False
        _engine_state.process = None
        _engine_state.port = None

    def test_session_sets_baggage(self):
        """Inside session(), baggage is set with user_id and role."""
        with parapet.session(user_id="u_1", role="admin") as ctx:
            assert ctx.baggage == "user_id=u_1,role=admin"

    def test_session_clears_baggage_on_exit(self):
        """After exiting session(), baggage is cleared."""
        with parapet.session(user_id="u_1", role="admin") as ctx:
            pass
        # After context exit, the context object's baggage should still be set
        # but the module-level active baggage should be cleared
        from parapet import _active_baggage
        assert _active_baggage.get() is None

    def test_session_with_user_id_only(self):
        """session() with only user_id sets partial baggage."""
        with parapet.session(user_id="u_42") as ctx:
            assert ctx.baggage == "user_id=u_42"

    def test_session_with_role_only(self):
        """session() with only role sets partial baggage."""
        with parapet.session(role="viewer") as ctx:
            assert ctx.baggage == "role=viewer"
