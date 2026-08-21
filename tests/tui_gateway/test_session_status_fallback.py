"""Regression tests for #20: session.status fallback semantics.

The old code carried a dead `status_params` stub whose comment promised an
unimplemented profile_home fallback. These tests pin the ACTUAL contract so a
future edit cannot misread that stub and break the real behavior:

1. A session bound to a profile home resolves status from THAT profile's
   state.db (via _session_db) - even when params.profile names another db.
2. A session with no bound db falls back to params.profile's db
   (via _profile_db).
3. With neither, the response is still a well-formed ok payload.
"""
from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def server():
    with patch.dict("sys.modules", {
        "hermes_constants": MagicMock(get_hermes_home=MagicMock(return_value="/tmp/hermes_test")),
        "hermes_cli.env_loader": MagicMock(),
        "hermes_cli.banner": MagicMock(),
        "hermes_state": MagicMock(),
    }):
        import importlib

        mod = importlib.import_module("tui_gateway.server")
    methods = dict(mod._methods)
    yield mod
    mod._methods.clear()
    mod._methods.update(methods)
    for sid in list(mod._sessions):
        mod._close_session_by_id(sid, end_reason="test_cleanup")


def _call_status(server, session, params):
    server._sess_nowait = lambda _params, _rid: (session, None)
    return server._methods["session.status"]("status", params)


def _session_without_home(key="sess-key-1"):
    return {
        "agent": None,
        "history_lock": threading.Lock(),
        "running": False,
        "session_key": key,
        "_run_thread": None,
    }


def _titled_db(profile_home, sid, title):
    from hermes_state import SessionDB

    profile_home.mkdir(parents=True, exist_ok=True)
    db = SessionDB(db_path=profile_home / "state.db")
    db.ensure_session(sid, source="cli")
    with db._lock:
        db._conn.execute("UPDATE sessions SET title = ? WHERE id = ?", (title, sid))
        db._conn.commit()
    db.close()


def test_status_bound_profile_home_db_wins(server, tmp_path, monkeypatch):
    """A session bound to profile_home reads status from that home's state.db."""
    home_a = tmp_path / "profileA"
    _titled_db(home_a, "sess-key-1", "BOUND-HOME-TITLE")

    session = _session_without_home()
    session["profile_home"] = str(home_a)

    # params.profile points somewhere else entirely - the bound home must win.
    import contextlib

    @contextlib.contextmanager
    def _none_profile_db(params):
        yield None

    monkeypatch.setattr(server, "_profile_db", _none_profile_db)
    resp = _call_status(server, session, {"session_id": "sess-key-1", "profile": "other"})
    out = resp["result"]["output"]
    assert "Session ID: sess-key-1" in out
    assert "Title: BOUND-HOME-TITLE" in out


def test_status_falls_back_to_params_profile_db(server, tmp_path, monkeypatch):
    """No bound db: params.profile's db is consulted for the session key."""
    home_b = tmp_path / "profileB"
    _titled_db(home_b, "sess-key-2", "FROM-PROFILE-DB")
    from hermes_state import SessionDB

    reopened = SessionDB(db_path=home_b / "state.db", read_only=True)

    session = _session_without_home("sess-key-2")

    yielded = []

    import contextlib

    @contextlib.contextmanager
    def fake_profile_db(params):
        yielded.append(dict(params or {}))
        yield reopened

    # Simulate the bound db being unavailable (the only case that reaches the
    # _profile_db fallback for a session without profile_home).
    import contextlib as _cl

    @_cl.contextmanager
    def _none_session_db(session):
        yield None

    monkeypatch.setattr(server, "_session_db", _none_session_db)
    monkeypatch.setattr(server, "_profile_db", fake_profile_db)
    resp = _call_status(server, session, {"session_id": "sess-key-2", "profile": "profileB"})

    out = resp["result"]["output"]
    assert "Title: FROM-PROFILE-DB" in out
    assert yielded and yielded[0].get("profile") == "profileB"


def test_status_ok_when_no_db_anywhere(server, tmp_path, monkeypatch):
    """Neither bound db nor profile db: still a well-formed status payload."""
    session = _session_without_home("missing-key")
    import contextlib

    @contextlib.contextmanager
    def _none_profile_db(params):
        yield None

    monkeypatch.setattr(server, "_profile_db", _none_profile_db)
    resp = _call_status(server, session, {"session_id": "missing-key"})
    out = resp["result"]["output"]
    assert "Session ID: missing-key" in out
    assert "Hermes TUI Status" in out
