"""Regression for #4: gateway config gates must resolve from the MERGED,
managed-overlay-aware config view — not raw user YAML.

Before: _resolve_config_gates() used read_raw_config(), so admin-managed
values (skills.write_approval, display.tool_progress_command) enabled
commands at runtime while /help and Telegram menus omitted them.
"""
from __future__ import annotations

import textwrap

import pytest


@pytest.fixture
def isolated_homes(tmp_path, monkeypatch):
    """Point HERMES_HOME + HERMES_MANAGED_DIR at tmp dirs; clear loader caches."""
    home = tmp_path / "home"
    home.mkdir(parents=True)
    managed = tmp_path / "managed"
    managed.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    monkeypatch.delenv("HERMES_IGNORE_USER_CONFIG", raising=False)

    import hermes_cli.config as cfg_mod
    from hermes_cli import managed_scope

    monkeypatch.setattr(cfg_mod, "_LAST_EXPANDED_CONFIG_BY_PATH", {})
    monkeypatch.setattr(cfg_mod, "_RAW_CONFIG_CACHE", {})
    monkeypatch.setattr(cfg_mod, "_LOAD_CONFIG_CACHE", {})
    managed_scope.invalidate_managed_cache()
    return home, managed


def _write(path, content):
    path.write_text(textwrap.dedent(content))


def test_managed_gate_visible_in_resolve_config_gates(isolated_homes):
    """Admin sets skills.write_approval in MANAGED scope only — the registry
    surface (/help, menus) must now see what runtime dispatch sees."""
    from hermes_cli.commands import _resolve_config_gates

    home, managed = isolated_homes
    # User file does NOT set the gate.
    _write(home / "config.yaml", "model:\n  default: test-model\n")
    # Admin-managed overlay DOES.
    _write(managed / "config.yaml", "skills:\n  write_approval: true\n")

    gated = _resolve_config_gates()
    assert "skills" in gated


def test_raw_read_misses_what_merged_view_sees(isolated_homes):
    """Pin the previously-contradictory pair: raw read misses the managed
    value while the merged view (what dispatch uses) contains it."""
    from hermes_cli.commands import _resolve_config_gates
    from hermes_cli.config import load_config_readonly, read_raw_config

    home, managed = isolated_homes
    _write(home / "config.yaml", "display:\n  skin: mono\n")
    _write(managed / "config.yaml", "skills:\n  write_approval: true\n")

    assert read_raw_config().get("skills") is None  # old behavior missed it
    assert (
        load_config_readonly()
        .get("skills", {})
        .get("write_approval") is True
    )  # merged view (dispatch) has it
    assert "skills" in _resolve_config_gates()  # surfaces now agree


def test_user_gate_still_resolves_without_managed_scope(isolated_homes, monkeypatch):
    from hermes_cli.commands import _resolve_config_gates

    home, _ = isolated_homes
    monkeypatch.delenv("HERMES_MANAGED_DIR", raising=False)
    import hermes_cli.config as cfg_mod
    from hermes_cli import managed_scope

    monkeypatch.setattr(cfg_mod, "_LAST_EXPANDED_CONFIG_BY_PATH", {})
    monkeypatch.setattr(cfg_mod, "_RAW_CONFIG_CACHE", {})
    monkeypatch.setattr(cfg_mod, "_LOAD_CONFIG_CACHE", {})
    managed_scope.invalidate_managed_cache()

    _write(
        home / "config.yaml",
        "display:\n  tool_progress_command: true\n",
    )
    assert "verbose" in _resolve_config_gates()


def test_falsy_gate_not_advertised(isolated_homes):
    from hermes_cli.commands import _resolve_config_gates

    home, managed = isolated_homes
    _write(home / "config.yaml", "model:\n  default: test-model\n")
    _write(managed / "config.yaml", "skills:\n  write_approval: false\n")
    assert "skills" not in _resolve_config_gates()
