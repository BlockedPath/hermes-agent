"""Regression tests for terminal config -> env-var bridging.

terminal_tool._get_env_config() reads ALL terminal settings from os.environ
(TERMINAL_*).  config.yaml values therefore have to be bridged into env vars
at startup, by THREE separate code paths:

  1. cli.py            -> ``CLI_TERMINAL_ENV_MAPPINGS`` (CLI / TUI startup)
  2. gateway/run.py    -> ``GATEWAY_TERMINAL_ENV_MAP`` (gateway / messaging
                          platforms)
  3. hermes_cli/config.py:set_config_value
                       -> bridges via the canonical ``TERMINAL_CONFIG_ENV_MAP``
                          (one-shot when the user runs ``hermes config set …``)

If any one of these is missing a key, the corresponding config.yaml setting
silently does nothing for that entry-point.  This bug already shipped once
for ``docker_run_as_host_user`` (gateway and CLI maps) and once for
``docker_mount_cwd_to_workspace`` (gateway map).

The tests below import the LIVE dicts (hoisted to module level for exactly
this purpose) instead of parsing source text, so they exercise what actually
ships.
"""

from hermes_cli import config as hc_config


def _cli_env_map_keys() -> set[str]:
    """terminal config keys bridged by cli.load_cli_config()."""
    import cli

    return set(cli.CLI_TERMINAL_ENV_MAPPINGS)


def _gateway_env_map_keys() -> set[str]:
    """terminal config keys bridged by gateway/run.py at startup."""
    from gateway import run as gr

    return set(gr.GATEWAY_TERMINAL_ENV_MAP)


def _save_config_env_sync_keys() -> set[str]:
    """terminal config keys bridged by ``hermes config set foo bar``.

    ``set_config_value`` bridges through the canonical
    ``TERMINAL_CONFIG_ENV_MAP`` via ``terminal_config_env_var_for_key()``,
    excluding ``cwd`` (handled separately); mirror that exclusion here.
    """
    return {k for k in hc_config.TERMINAL_CONFIG_ENV_MAP if k != "cwd"}


# Keys present in cli.py env_mappings but intentionally absent from
# gateway/run.py or set_config_value.  Each entry must be justified.
_CLI_ONLY_OK = frozenset({
    # `env_type` is a legacy YAML key alias for `backend` that cli.py
    # accepts for backwards-compat with older cli-config.yaml.  The
    # gateway path normalizes on the canonical `backend` key, which is
    # also in the map and handles the same bridging.
    "env_type",
    # sudo_password is not a terminal-backend option — it's a credential
    # used across backends, bridged to $SUDO_PASSWORD (not TERMINAL_*).
    # Treating it as terminal-only would be misleading.
    "sudo_password",
})


def test_cli_and_gateway_env_maps_agree():
    """cli.py and gateway/run.py must bridge the same set of terminal keys.

    Both feed the same downstream consumer (terminal_tool).  Drift between
    them means a config.yaml setting that "works in CLI mode but not gateway
    mode" (or vice-versa) — the bug class that shipped twice already.
    """
    cli_keys = _cli_env_map_keys() - _CLI_ONLY_OK
    gw_keys = _gateway_env_map_keys()

    # Normalize the legacy `env_type` alias: cli.py accepts both `env_type`
    # and `backend` as source keys for TERMINAL_ENV; gateway only accepts
    # `backend`.  Remove `backend` from the gateway side to avoid a spurious
    # "backend missing from cli" failure.
    gw_keys = gw_keys - {"backend"}

    missing_in_gateway = cli_keys - gw_keys
    missing_in_cli = gw_keys - cli_keys

    assert not missing_in_gateway, (
        f"Keys in cli.py CLI_TERMINAL_ENV_MAPPINGS but missing from "
        f"gateway GATEWAY_TERMINAL_ENV_MAP: {sorted(missing_in_gateway)}. "
        f"Add them to both maps (same bug class as docker_run_as_host_user "
        f"shipping wired in cli but not gateway in April 2026)."
    )
    assert not missing_in_cli, (
        f"Keys in gateway GATEWAY_TERMINAL_ENV_MAP but missing from cli.py "
        f"CLI_TERMINAL_ENV_MAPPINGS: {sorted(missing_in_cli)}. Add them to "
        f"both maps."
    )


def test_save_config_set_supports_critical_bridged_keys():
    """``hermes config set terminal.X true`` must propagate to .env for
    known-critical keys.  SSH terminal keys are handled via the separate
    api_keys TERMINAL_SSH_* fallback path or user-edits-yaml-directly.
    """
    save_keys = _save_config_env_sync_keys()
    required = {
        "docker_run_as_host_user",
        "docker_mount_cwd_to_workspace",
        "backend",
        "docker_image",
        "container_cpu",
        "container_memory",
        "container_disk",
        "container_persistent",
    }
    missing = required - save_keys
    assert not missing, (
        f"`hermes config set terminal.X` doesn't sync these load-bearing "
        f"keys to .env: {sorted(missing)}.  Add them to TERMINAL_CONFIG_ENV_MAP "
        f"in hermes_cli/config.py (set_config_value bridges through it)."
    )
