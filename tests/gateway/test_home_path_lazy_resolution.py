"""Regression for #8: gateway home-scoped paths must resolve per call so the
multiplex per-turn ContextVar override (set_hermes_home_override) is honored —
import-time binding pinned every profile to the boot profile's home."""
from __future__ import annotations

from hermes_constants import reset_hermes_home_override, set_hermes_home_override


def test_sticker_cache_resolves_through_override(tmp_path):
    from gateway import sticker_cache

    token = set_hermes_home_override(str(tmp_path))
    try:
        assert sticker_cache.cache_path() == tmp_path / "sticker_cache.json"
        assert sticker_cache.CACHE_PATH == tmp_path / "sticker_cache.json"
    finally:
        sticker_cache.__dict__.pop("CACHE_PATH", None)
        reset_hermes_home_override(token)


def test_mirror_sessions_index_resolves_through_override(tmp_path):
    from gateway import mirror

    token = set_hermes_home_override(str(tmp_path))
    try:
        assert mirror.sessions_index_path() == tmp_path / "sessions" / "sessions.json"
    finally:
        reset_hermes_home_override(token)


def test_hooks_dir_resolves_through_override(tmp_path):
    from gateway import hooks

    token = set_hermes_home_override(str(tmp_path))
    try:
        assert hooks.hooks_dir() == tmp_path / "hooks"
    finally:
        reset_hermes_home_override(token)


def test_channel_directory_paths_resolve_through_override(tmp_path):
    from gateway import channel_directory as cd

    token = set_hermes_home_override(str(tmp_path))
    try:
        assert cd.directory_path() == tmp_path / "channel_directory.json"
        assert cd.channel_aliases_path() == tmp_path / "channel_aliases.json"
    finally:
        reset_hermes_home_override(token)


def test_monkeypatched_attribute_still_wins(tmp_path, monkeypatch):
    """Back-compat: tests that patch the module attribute keep working."""
    import gateway.hooks as hooks_mod

    monkeypatch.setattr(hooks_mod, "HOOKS_DIR", tmp_path / "custom")
    assert hooks_mod.hooks_dir() == tmp_path / "custom"
