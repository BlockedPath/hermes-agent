"""Regression tests for #36: background plugin discovery race + silent menu
callback fallbacks."""
from __future__ import annotations

import sys
import threading
import types


def test_start_background_discovery_noop_after_sync_discovery(tmp_path, monkeypatch):
    """Once discovery completed synchronously, a later start call must not
    spawn another discovery thread — the check-and-set window that used to sit
    outside the lock (#36)."""
    from hermes_cli import plugins as plugins_mod

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    manager = plugins_mod.get_plugin_manager()
    spawned = []
    real_thread = threading.Thread

    class _TrackingThread(real_thread):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            spawned.append(self)

        def start(self):
            pass

    monkeypatch.setattr(manager, "_discovered", True)
    monkeypatch.setattr(plugins_mod, "_background_discovery_thread", None)
    monkeypatch.setattr(plugins_mod.threading, "Thread", _TrackingThread)

    plugins_mod.start_background_plugin_discovery()
    assert spawned == [], "redundant discovery thread after sync discovery"


def test_start_background_discovery_spawns_when_undiscovered(tmp_path, monkeypatch):
    from hermes_cli import plugins as plugins_mod

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    manager = plugins_mod.get_plugin_manager()
    spawned = []
    real_thread = threading.Thread

    class _TrackingThread(real_thread):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            spawned.append(self)

        def start(self):
            pass

    monkeypatch.setattr(manager, "_discovered", False)
    monkeypatch.setattr(plugins_mod, "_background_discovery_thread", None)
    monkeypatch.setattr(plugins_mod.threading, "Thread", _TrackingThread)

    plugins_mod.start_background_plugin_discovery()
    assert len(spawned) == 1


def test_concurrent_starts_spawn_at_most_one_thread(tmp_path, monkeypatch):
    from hermes_cli import plugins as plugins_mod

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    manager = plugins_mod.get_plugin_manager()
    spawned = []
    real_thread = threading.Thread
    lock = threading.Lock()

    class _TrackingThread(real_thread):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            with lock:
                spawned.append(self)

        def start(self):
            pass

    monkeypatch.setattr(manager, "_discovered", False)
    monkeypatch.setattr(plugins_mod, "_background_discovery_thread", None)
    monkeypatch.setattr(plugins_mod.threading, "Thread", _TrackingThread)

    gate = threading.Event()
    threads = [
        real_thread(
            target=lambda: (gate.wait(), plugins_mod.start_background_plugin_discovery())
        )
        for _ in range(8)
    ]
    for t in threads:
        t.start()
    gate.set()
    for t in threads:
        t.join(timeout=5)

    assert len(spawned) <= 1


def test_menu_callback_exception_is_logged_not_swallowed(caplog, monkeypatch):
    """A raising menu loop must LOG before degrading to fallback (#36)."""
    import logging

    import hermes_cli.curses_ui as cui

    class _RaisingWrapper:
        def __call__(self, fn):
            raise RuntimeError("menu handler bug")

    fake_curses = types.ModuleType("curses")
    fake_curses.wrapper = _RaisingWrapper()
    monkeypatch.setitem(sys.modules, "curses", fake_curses)
    fallback_called = []

    with caplog.at_level(logging.ERROR):
        result = cui._run_curses_menu(
            initial_cursor=0,
            item_count=1,
            draw_header=lambda *_: None,
            draw_row=lambda *_: None,
            on_action=lambda *a: None,
            fallback=lambda: fallback_called.append(1) or "fb",
            cancel_value="cancelled",
        )

    assert result == "fb" and fallback_called == [1]
    assert any(
        r.levelno >= logging.ERROR and "menu callback failed" in r.getMessage()
        for r in caplog.records
    ), "exception swallowed without logging"
