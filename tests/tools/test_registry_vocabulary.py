"""Regression for #17: registry vocabulary agreement.

1. browser_cdp / browser_exec must be registered under the canonical
   'browser' toolset (the static toolsets.py list owns them), NOT under
   phantom 'browser-cdp' / 'browser-use' sets that config cannot target.
2. banner/doctor must read requirement metadata from the LIVE registry
   accessor (registry.get_toolset_requirements()), not the import-time
   snapshot in model_tools.TOOLSET_REQUIREMENTS which predates MCP/plugin
   discovery.
"""
from __future__ import annotations

import pytest

from tools.registry import registry


PHANTOM = ("browser-cdp", "browser-use")


def _import_browser_tool_modules():
    import tools.browser_cdp_tool  # noqa: F401
    import tools.browser_dialog_tool  # noqa: F401
    import tools.browser_use_cli  # noqa: F401


def test_browser_cdp_and_exec_belong_to_browser_toolset():
    _import_browser_tool_modules()
    members = set(registry.get_tool_names_for_toolset("browser"))
    assert {"browser_cdp", "browser_exec"} <= members
    for name in ("browser_cdp", "browser_exec"):
        tool = registry._tools.get(name)
        assert tool is not None, name
        assert tool.toolset == "browser", name


def test_no_phantom_browser_toolsets():
    names = set(registry.get_registered_toolset_names())
    for phantom in PHANTOM:
        assert phantom not in names, phantom


def test_late_registered_requirements_visible_via_live_accessor():
    """The accessor banner/doctor use sees post-import registrations; the
    import-time snapshot in model_tools does not — which is exactly why
    they must read live (#17)."""
    from model_tools import TOOLSET_REQUIREMENTS

    toolset = "audit-probe-vocabulary"
    registry.register(
        name="audit_vocab_probe",
        toolset=toolset,
        schema={"name": "audit_vocab_probe", "description": "probe",
                "parameters": {"type": "object", "properties": {}}},
        handler=lambda args, **kw: "{}",
        requires_env=["AUDIT_VOCAB_PROBE_KEY"],
        override=True,
    )

    live = registry.get_toolset_requirements()
    assert toolset in live
    assert live[toolset].get("env_vars") == ["AUDIT_VOCAB_PROBE_KEY"]
    # The frozen import-time snapshot missed the late registration.
    assert toolset not in TOOLSET_REQUIREMENTS
