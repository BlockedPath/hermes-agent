"""Regression for #25: no credential-class env name may bypass the fail-closed
scope via a global-env PREFIX wildcard. Every name starting with a formerly
wildcarded prefix must be either an explicit knob or treated as scoped."""
from __future__ import annotations

import re
from pathlib import Path

from agent import secret_scope as ss


def test_telegram_knobs_are_explicit_not_prefix():
    # The known tuning knobs resolve globally...
    assert ss._is_global_env("HERMES_TELEGRAM_HTTP_POOL_TIMEOUT") is True
    # ...but an UNKNOWN HERMES_TELEGRAM_* name (potential future credential,
    # e.g. HERMES_TELEGRAM_BOT_TOKEN alias) must NOT bypass the scope.
    assert ss._is_global_env("HERMES_TELEGRAM_BOT_TOKEN") is False
    assert ss._is_global_env("HERMES_TELEGRAM_ANYTHING_NEW") is False


def test_no_optional_env_var_starts_with_global_prefix():
    from hermes_cli.config_defaults import OPTIONAL_ENV_VARS

    knobs = ss._GLOBAL_ENV_TELEGRAM_KNOBS
    leaked = [
        name for name in OPTIONAL_ENV_VARS
        if name not in knobs and any(
            name.startswith(p) for p in ss._GLOBAL_ENV_PREFIXES
        )
    ]
    assert leaked == [], f"credential names collide with global prefixes: {leaked}"


def test_knob_list_covers_all_repo_usages():
    """Every HERMES_TELEGRAM_* literal referenced in the repo is a declared knob."""
    repo = Path(__file__).resolve().parents[2]
    pat = re.compile(r"HERMES_TELEGRAM_[A-Z0-9_]+")
    found = set()
    for base in ("gateway", "plugins/platforms/telegram"):
        for f in (repo / base).rglob("*.py"):
            found |= set(pat.findall(f.read_text(encoding="utf-8", errors="ignore")))
    missing = sorted(found - set(ss._GLOBAL_ENV_TELEGRAM_KNOBS))
    assert missing == [], f"undeclared telegram knobs (add to _GLOBAL_ENV_TELEGRAM_KNOBS): {missing}"
