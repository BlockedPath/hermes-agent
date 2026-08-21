"""Regression tests for F-CORE-2 / issue #7: Nous non-pool auth fallback
must resolve auth.json per call (profile-aware), not from the import-time
module constant pinned to the boot profile.

Old behavior: ``_AUTH_JSON_PATH = get_hermes_home() / "auth.json"`` was bound
at import time. Under gateway.multiplex_profiles, a secondary profile's
``_read_nous_auth()`` fallback silently authenticated against whichever
profile was active at process boot — cross-profile credential use/billing.
"""

import json

import hermes_constants
from hermes_constants import (
    reset_hermes_home_override,
    set_hermes_home_override,
)


def _write_auth(home, agent_key="boot-profile-key"):
    home.mkdir(parents=True, exist_ok=True)
    (home / "auth.json").write_text(
        json.dumps({
            "active_provider": "nous",
            "providers": {"nous": {"agent_key": agent_key, "access_token": "tok"}},
        }),
        encoding="utf-8",
    )


def _read(monkeypatch):
    from agent import auxiliary_client

    monkeypatch.setattr(
        auxiliary_client, "_select_pool_entry", lambda provider: (False, None)
    )
    return auxiliary_client._read_nous_auth()


def test_secondary_profile_without_own_auth_json_gets_no_fallback(
    monkeypatch, tmp_path
):
    """Boot profile has auth.json; the active (secondary) profile does not.

    The fallback must return None — never the boot profile's grant.
    """
    boot_home = tmp_path / "boot-home"
    _write_auth(boot_home, agent_key="BOOT-PROFILE-KEY")
    secondary_home = tmp_path / "secondary-home"
    secondary_home.mkdir(parents=True)

    # Simulate the boot process having imported the module while HERMES_HOME
    # pointed at the boot profile (what the old constant captured).
    monkeypatch.setenv("HERMES_HOME", str(boot_home))
    import importlib

    from agent import auxiliary_client

    importlib.reload(auxiliary_client)

    # Now a secondary-profile turn runs under the ContextVar override.
    token = set_hermes_home_override(str(secondary_home))
    try:
        result = _read(monkeypatch)
    finally:
        reset_hermes_home_override(token)

    assert result is None


def test_secondary_profile_with_own_auth_json_uses_it(monkeypatch, tmp_path):
    """The active profile's own auth.json wins over any other profile's."""
    boot_home = tmp_path / "boot-home"
    _write_auth(boot_home, agent_key="BOOT-PROFILE-KEY")
    secondary_home = tmp_path / "secondary-home"
    _write_auth(secondary_home, agent_key="SECONDARY-PROFILE-KEY")

    monkeypatch.setenv("HERMES_HOME", str(boot_home))
    import importlib

    from agent import auxiliary_client

    importlib.reload(auxiliary_client)

    token = set_hermes_home_override(str(secondary_home))
    try:
        result = _read(monkeypatch)
    finally:
        reset_hermes_home_override(token)

    assert result is not None
    assert result.get("agent_key") == "SECONDARY-PROFILE-KEY"


def test_no_override_reads_default_home(monkeypatch, tmp_path):
    """Without an override (single-profile deployment), legacy behavior holds."""
    home = tmp_path / "only-home"
    _write_auth(home, agent_key="ONLY-HOME-KEY")

    monkeypatch.setenv("HERMES_HOME", str(home))
    import importlib

    from agent import auxiliary_client

    importlib.reload(auxiliary_client)

    result = _read(monkeypatch)
    assert result is not None
    assert result.get("agent_key") == "ONLY-HOME-KEY"
