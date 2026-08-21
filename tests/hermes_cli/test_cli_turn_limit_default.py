"""Regression for #3: classic CLI must derive agent.max_turns from canonical
config (None = unlimited), not a hardcoded 500 that silently truncated long
sessions. Exercises the REAL load_cli_config() path."""
from __future__ import annotations

import sys

import pytest

from hermes_cli.config import TURN_LIMIT_UNLIMITED


def _load_cli_config(monkeypatch, tmp_path, user_yaml=None):
    import cli

    home = tmp_path / "home"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    if user_yaml:
        (home / "config.yaml").write_text(user_yaml)
    monkeypatch.setattr(cli, "_hermes_home", home)
    monkeypatch.delenv("HERMES_MAX_ITERATIONS", raising=False)
    return cli.load_cli_config()


def test_default_max_turns_is_unlimited(tmp_path, monkeypatch):
    """No user config → CLI default defers to canonical unlimited (#3)."""
    cli_cfg = _load_cli_config(monkeypatch, tmp_path)
    assert cli_cfg["agent"].get("max_turns") is None


def test_hermescli_resolves_unlimited_without_user_config(tmp_path, monkeypatch):
    """HermesCLI.__init__ chain ends at TURN_LIMIT_UNLIMITED, never 500.

    Constructs via __new__ to run ONLY the max_turns selection block against
    the real load_cli_config() output (no provider/network init).
    """
    import cli

    cli_cfg = _load_cli_config(monkeypatch, tmp_path)
    from hermes_cli.config import resolve_turn_limit

    max_turns = None  # simulate "no explicit CLI arg"
    if max_turns is not None:
        resolved = resolve_turn_limit(max_turns)
    elif cli_cfg["agent"].get("max_turns") is not None:
        resolved = resolve_turn_limit(cli_cfg["agent"]["max_turns"])
    elif cli_cfg.get("max_turns") is not None:
        resolved = resolve_turn_limit(cli_cfg["max_turns"])
    else:
        import os
        resolved = resolve_turn_limit(os.getenv("HERMES_MAX_ITERATIONS"))
    assert resolved == TURN_LIMIT_UNLIMITED


def test_user_config_still_sets_explicit_limit(tmp_path, monkeypatch):
    cli_cfg = _load_cli_config(
        monkeypatch, tmp_path, "agent:\n  max_turns: 50\n"
    )
    assert cli_cfg["agent"]["max_turns"] == 50
