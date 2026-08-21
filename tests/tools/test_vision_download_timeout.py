"""Regression for #18: auxiliary.vision.download_timeout is read lazily per
download (config edits apply without restart) and takes precedence over the
HERMES_VISION_DOWNLOAD_TIMEOUT env mirror."""
from __future__ import annotations


def _set_home(monkeypatch, tmp_path, config_yaml=None):
    home = tmp_path / "home"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    if config_yaml:
        (home / "config.yaml").write_text(config_yaml)
    return home


def test_config_value_wins_over_env_and_default(tmp_path, monkeypatch):
    from tools import vision_tools

    _set_home(monkeypatch, tmp_path,
              "auxiliary:\n  vision:\n    download_timeout: 77\n")
    monkeypatch.setenv("HERMES_VISION_DOWNLOAD_TIMEOUT", "999")
    assert vision_tools._resolve_download_timeout() == 77.0


def test_env_fallback_when_config_absent(tmp_path, monkeypatch):
    from tools import vision_tools

    _set_home(monkeypatch, tmp_path)
    monkeypatch.setenv("HERMES_VISION_DOWNLOAD_TIMEOUT", "42")
    assert vision_tools._resolve_download_timeout() == 42.0


def test_lazy_resolution_reflects_config_edit_without_restart(tmp_path, monkeypatch):
    """The symptom: an import-time freeze meant config edits needed a restart."""
    from tools import vision_tools

    home = _set_home(monkeypatch, tmp_path)
    monkeypatch.delenv("HERMES_VISION_DOWNLOAD_TIMEOUT", raising=False)
    assert vision_tools._resolve_download_timeout() == 30.0
    # Operator edits config.yaml mid-process:
    (home / "config.yaml").write_text(
        "auxiliary:\n  vision:\n    download_timeout: 55\n"
    )
    assert vision_tools._resolve_download_timeout() == 55.0


def test_no_import_time_freeze():
    import inspect

    from tools import vision_tools

    src = inspect.getmodule(vision_tools)
    # The module must not bind a frozen timeout constant at import time.
    assert not hasattr(vision_tools, "_VISION_DOWNLOAD_TIMEOUT")
    assert callable(src._resolve_download_timeout)
