"""Regression for #16: ``_last_resolved_tool_names`` must reflect the
POST-assembly tool universe on BOTH compute and cache-hit paths.

Pre-fix, the compute path assigned the global from pre-sanitization /
pre-assembly ``filtered_tools`` while the cache-hit path assigned from the
cached post-assembly result — so consumers like execute_code's sandbox
enumeration saw a different tool universe depending on memo warmth.
"""
from __future__ import annotations

import pytest

from tools.tool_search import ToolSearchConfig


BRIDGE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": name,
            "description": "bridge",
            "parameters": {"type": "object", "properties": {}},
        },
    }
    for name in ("tool_search", "tool_describe", "tool_call")
]


class _FakeAssembly:
    activated = True
    tier = 2
    deferred_count = 3
    deferred_tokens = 1234
    listing_form = "none"
    tool_defs = BRIDGE_TOOLS


@pytest.fixture()
def _clean_cache():
    import model_tools as mt

    saved = dict(mt._tool_defs_cache)
    mt._tool_defs_cache.clear()
    yield
    mt._tool_defs_cache.clear()
    mt._tool_defs_cache.update(saved)


def _returned_names(defs):
    return [t["function"]["name"] for t in defs]


def test_compute_path_global_matches_returned_defs(monkeypatch, _clean_cache):
    import model_tools as mt
    import tools.tool_search as ts

    monkeypatch.setattr(
        ts, "load_config",
        lambda: ToolSearchConfig(
            enabled="auto", threshold_pct=10.0,
            search_default_limit=5, max_search_limit=20,
        ),
    )
    monkeypatch.setattr(ts, "assemble_tool_defs", lambda *a, **k: _FakeAssembly())

    defs = mt.get_tool_definitions(enabled_toolsets=["terminal"], quiet_mode=True)

    # The global must equal the RETURNED (post-assembly) universe.
    assert mt._last_resolved_tool_names == _returned_names(defs)
    # And it must be the bridged set, not pre-assembly names.
    assert set(mt._last_resolved_tool_names) == {"tool_search", "tool_describe", "tool_call"}


def test_global_is_set_even_when_assembly_skipped(monkeypatch, _clean_cache):
    """No tool-search bridging: global still matches the returned universe."""
    import model_tools as mt
    import tools.tool_search as ts

    monkeypatch.setattr(
        ts, "load_config",
        lambda: ToolSearchConfig(
            enabled="off", threshold_pct=10.0,
            search_default_limit=5, max_search_limit=20,
        ),
    )
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    defs = mt.get_tool_definitions(enabled_toolsets=["terminal"], quiet_mode=True)
    assert mt._last_resolved_tool_names == _returned_names(defs)
    assert len(mt._last_resolved_tool_names) > 0


def test_cache_hit_path_agrees_with_compute_path(monkeypatch, _clean_cache):
    import model_tools as mt
    import tools.tool_search as ts

    monkeypatch.setattr(
        ts, "load_config",
        lambda: ToolSearchConfig(
            enabled="auto", threshold_pct=10.0,
            search_default_limit=5, max_search_limit=20,
        ),
    )
    monkeypatch.setattr(ts, "assemble_tool_defs", lambda *a, **k: _FakeAssembly())

    first = mt.get_tool_definitions(enabled_toolsets=["terminal"], quiet_mode=True)
    first_global = list(mt._last_resolved_tool_names)
    second = mt.get_tool_definitions(enabled_toolsets=["terminal"], quiet_mode=True)

    assert _returned_names(first) == _returned_names(second)
    assert first_global == _returned_names(second)
    assert mt._last_resolved_tool_names == _returned_names(second)
