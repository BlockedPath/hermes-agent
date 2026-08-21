"""Regression for #15: cross-tool references in schema descriptions must be
stripped/narrowed PER-TOOL when the referenced tools are unavailable —
partial toolset availability must not leave hallucinatable mentions behind."""
from __future__ import annotations

from model_tools import _strip_web_tool_refs

NAVIGATE_DESC = (
    "Navigate to a URL in the browser. Must be called before other browser tools. "
    "For simple information retrieval, prefer web_search or web_extract (faster, cheaper). "
    "For plain-text endpoints — URLs ending in .md, .txt, .json, .yaml, .yml, .csv, .xml, "
    "raw.githubusercontent.com, or any documented API endpoint — prefer curl via the "
    "terminal tool or web_extract; the browser stack is overkill and much slower for these."
)


def test_both_web_tools_available_leaves_description_untouched():
    out = _strip_web_tool_refs(NAVIGATE_DESC, {"browser_navigate", "web_search", "web_extract", "terminal"})
    assert out == NAVIGATE_DESC


def test_both_web_tools_absent_strips_first_sentence_and_curl_sentence():
    out = _strip_web_tool_refs(NAVIGATE_DESC, {"browser_navigate"})
    assert "web_search" not in out
    assert "web_extract" not in out
    assert "prefer curl via the terminal tool" not in out


def test_partial_availability_narrowed_to_present_tool():
    # web_extract present, web_search absent -> sentence narrows to web_extract only
    out = _strip_web_tool_refs(
        NAVIGATE_DESC, {"browser_navigate", "web_extract", "terminal"}
    )
    assert "web_search" not in out
    assert "web_extract" in out
    assert "terminal" in out  # terminal available: curl guidance stays


def test_terminal_absent_removes_curl_mention():
    out = _strip_web_tool_refs(
        NAVIGATE_DESC, {"browser_navigate", "web_search", "web_extract"}
    )
    assert "curl via the terminal tool" not in out


def test_no_absent_tool_named_in_any_combination():
    combos = [
        {"browser_navigate"},
        {"browser_navigate", "web_search"},
        {"browser_navigate", "web_extract"},
        {"browser_navigate", "web_search", "web_extract"},
        {"browser_navigate", "terminal"},
        {"browser_navigate", "web_extract", "terminal"},
        {"browser_navigate", "web_search", "terminal"},
    ]
    for avail in combos:
        out = _strip_web_tool_refs(NAVIGATE_DESC, avail)
        for absent in ({"web_search", "web_extract", "terminal"} - avail):
            assert absent not in out, f"{absent} named while absent from {avail}"


def test_browser_cdp_static_schema_names_web_extract_for_dynamic_strip():
    """The cdp schema names web_extract; the dynamic post-processing in
    model_tools rewords it when web_extract is absent. Pin that the static
    text matches the replacement target so the strip cannot silently rot."""
    from tools.browser_cdp_tool import BROWSER_CDP_SCHEMA

    desc = BROWSER_CDP_SCHEMA["description"]
    assert "use web_extract" in desc


def test_yuanbao_sticker_schema_capability_worded():
    """#15: yuanbao sticker description must not name execute_code."""
    import tools.yuanbao_tools as yt

    found = False
    for schema in getattr(yt, "YUANBAO_TOOL_SCHEMAS", []):
        if "execute_code" in schema.get("description", ""):
            found = True
    assert not found, "yuanbao schema still names execute_code"
