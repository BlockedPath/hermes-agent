"""Regression for #34: built-in skins must not be shadowable by user files,
and the active-skin name must report the EFFECTIVE skin after fallback."""
from __future__ import annotations

import textwrap

import pytest


@pytest.fixture()
def fresh_engine(monkeypatch, tmp_path):
    import importlib

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import hermes_cli.skin_engine as se

    monkeypatch.setattr(se, "_active_skin", None)
    monkeypatch.setattr(se, "_active_skin_name", "default")
    return se


def _write_user_skin(home, name):
    d = home / "skins"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.yaml").write_text(
        textwrap.dedent(
            f"""\
            name: {name}
            description: user shadow copy
            colors:
              banner_border: "#123456"
            """
        )
    )


def test_builtin_not_shadowed_by_user_file(fresh_engine, tmp_path):
    """list_skins shows only the built-in 'ares'; load must agree (#34)."""
    se = fresh_engine
    _write_user_skin(tmp_path, "ares")

    listed = [s["name"] for s in se.list_skins()]
    assert listed.count("ares") == 1  # no duplicate/shadow entry

    cfg = se.load_skin("ares")
    assert cfg.name == "ares"
    # Built-in palette won: the shadow copy's banner_border must NOT apply.
    assert cfg.colors.get("banner_border") != "#123456"


def test_unknown_skin_fallback_reports_effective_name(fresh_engine):
    se = fresh_engine
    cfg = se.set_active_skin("no-such-skin-anywhere")
    assert getattr(cfg, "name", "default") == "default"
    assert se.get_active_skin_name() == "default"


def test_real_skin_reports_its_own_name(fresh_engine):
    se = fresh_engine
    se.set_active_skin("mono")
    assert se.get_active_skin_name() == "mono"
