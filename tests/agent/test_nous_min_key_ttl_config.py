"""Regression for F-CORE-3/#26: auxiliary.nous_min_key_ttl_seconds is a
config.yaml setting; the env var is an operator escape hatch layered on top
(same convention as HERMES_OPENROUTER_CACHE)."""

from __future__ import annotations


def _set_home(monkeypatch, tmp_path, config_yaml=None):
    home = tmp_path / "home"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    if config_yaml:
        (home / "config.yaml").write_text(config_yaml)
    return home


def test_user_config_wins_over_default(tmp_path, monkeypatch):
    from agent import auxiliary_client as ac

    _set_home(monkeypatch, tmp_path, "auxiliary:\n  nous_min_key_ttl_seconds: 120\n")
    monkeypatch.delenv("HERMES_NOUS_MIN_KEY_TTL_SECONDS", raising=False)
    assert ac._nous_min_key_ttl_seconds() == 120


def test_env_escape_hatch_layers_over_config(tmp_path, monkeypatch):
    from agent import auxiliary_client as ac

    _set_home(monkeypatch, tmp_path, "auxiliary:\n  nous_min_key_ttl_seconds: 120\n")
    monkeypatch.setenv("HERMES_NOUS_MIN_KEY_TTL_SECONDS", "9999")
    assert ac._nous_min_key_ttl_seconds() == 9999


def test_env_fallback_when_config_absent(tmp_path, monkeypatch):
    from agent import auxiliary_client as ac

    _set_home(monkeypatch, tmp_path)
    monkeypatch.setenv("HERMES_NOUS_MIN_KEY_TTL_SECONDS", "300")
    assert ac._nous_min_key_ttl_seconds() == 300


def test_default_when_neither_set(tmp_path, monkeypatch):
    from agent import auxiliary_client as ac

    _set_home(monkeypatch, tmp_path)
    monkeypatch.delenv("HERMES_NOUS_MIN_KEY_TTL_SECONDS", raising=False)
    assert ac._nous_min_key_ttl_seconds() == 1800
