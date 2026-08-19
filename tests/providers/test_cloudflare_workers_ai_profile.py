"""Tests for the bundled Cloudflare Workers AI provider plugin.

The reasoning-effort expectations below are not guesses: each accepted set
was probed against the live API, where an unsupported level is a hard HTTP
400 rather than an ignored field.  ``test_never_emits_a_rejected_level``
pins the whole mapping against those measurements.
"""

import json
import sys
from unittest.mock import patch

import pytest

from providers import get_provider_profile, list_providers

# ``_import_plugin_dir`` gives bundled plugins a stable synthetic module name
# with dashes swapped for underscores; the directory itself is never importable
# under that path, so the module has to be picked out of sys.modules after
# discovery rather than imported directly.
PLUGIN = "plugins.model_providers.cloudflare_workers_ai"

# Levels hermes can ask for (hermes_constants.VALID_REASONING_EFFORTS + "none").
LEVELS = ("none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra")

# Measured live, 2026-08. Empty-string entries mean "field omitted".
ACCEPTED_BY_MODEL = {
    # Harmony + friends: reject none/minimal/xhigh/max.
    "@cf/openai/gpt-oss-120b": {"low", "medium", "high"},
    "@cf/openai/gpt-oss-20b": {"low", "medium", "high"},
    "@cf/nvidia/nemotron-3-120b-a12b": {"low", "medium", "high"},
    "@cf/zai-org/glm-4.7-flash": {"low", "medium", "high"},
    # "Supported types are xhigh (default), medium, and low" — note `high` is
    # rejected while `xhigh` is accepted.
    "@cf/qwen/qwen3.8-27b": {"low", "medium", "xhigh"},
    # Accepts a real off switch and `max`, but rejects `minimal`/`xhigh`.
    "@cf/moonshotai/kimi-k2.6": {"none", "low", "medium", "high", "max"},
    "@cf/moonshotai/kimi-k2.7-code": {"none", "low", "medium", "high", "max"},
    # Took every level tested.
    "@cf/deepseek-ai/deepseek-v4-flash-0731": {
        "none", "minimal", "low", "medium", "high", "xhigh", "max",
    },
    "@cf/zai-org/glm-5.2": {
        "none", "minimal", "low", "medium", "high", "xhigh", "max",
    },
    "@cf/google/gemma-4-26b-a4b-it": {
        "none", "minimal", "low", "medium", "high", "xhigh", "max",
    },
    "@cf/qwen/qwen3-30b-a3b-fp8": {
        "none", "minimal", "low", "medium", "high", "xhigh", "max",
    },
}


def _profile():
    p = get_provider_profile("cloudflare-workers-ai")
    assert p is not None
    return p


def _plugin_module():
    """Return the exact module object the registered profile was defined in."""
    list_providers()  # force plugin discovery
    mod = sys.modules.get(PLUGIN)
    assert mod is not None, f"{PLUGIN} not in sys.modules after discovery"
    assert mod is sys.modules[type(_profile()).__module__]
    return mod


def _search_url(base):
    return _profile()._models_search_url(base)  # type: ignore[attr-defined]


def _effort(model, cfg, supports_reasoning=True):
    _extra, top = _profile().build_api_kwargs_extras(
        reasoning_config=cfg, supports_reasoning=supports_reasoning, model=model
    )
    return top.get("reasoning_effort")


class TestProfileRegistration:
    def test_profile_registered(self):
        p = _profile()
        assert p.name == "cloudflare-workers-ai"
        assert p.auth_type == "api_key"
        assert p.api_mode == "chat_completions"
        assert p.supports_vision is True
        assert p.get_hostname() == "api.cloudflare.com"
        assert p.default_aux_model == "@cf/ibm-granite/granite-4.0-h-micro"

    def test_account_id_is_not_offered_as_a_credential(self):
        """CLOUDFLARE_ACCOUNT_ID must not look like an API key to auth.py.

        auth.py treats every env var not ending in _BASE_URL/_URL as an
        api-key candidate; an account id sent as a bearer token would
        authenticate nothing while masking the real misconfiguration.
        """
        p = _profile()
        assert "CLOUDFLARE_ACCOUNT_ID" not in p.env_vars
        assert p.env_vars == ("CLOUDFLARE_API_KEY", "CLOUDFLARE_BASE_URL")

    @pytest.mark.parametrize("alias", ["cloudflare", "workers-ai", "cloudflare-ai"])
    def test_aliases_resolve(self, alias):
        assert get_provider_profile(alias) is _profile()

    def test_auto_wired_into_auth_registry(self):
        from hermes_cli.auth import PROVIDER_REGISTRY

        cfg = PROVIDER_REGISTRY["cloudflare-workers-ai"]
        assert cfg.auth_type == "api_key"
        assert cfg.api_key_env_vars == ("CLOUDFLARE_API_KEY",)
        assert cfg.base_url_env_var == "CLOUDFLARE_BASE_URL"

    def test_listed_in_canonical_providers(self):
        from hermes_cli.models import CANONICAL_PROVIDERS

        assert "cloudflare-workers-ai" in {p.slug for p in CANONICAL_PROVIDERS}

    def test_fallback_models_are_all_cf_scoped(self):
        p = _profile()
        assert len(p.fallback_models) == 16
        assert all(m.startswith("@cf/") for m in p.fallback_models)


class TestBaseUrl:
    def test_account_id_is_interpolated(self):
        mod = _plugin_module()
        with patch.object(mod, "_read_env", return_value="abc123"):
            assert mod._resolve_base_url() == (
                "https://api.cloudflare.com/client/v4/accounts/abc123/ai/v1"
            )

    def test_missing_account_id_yields_empty_not_a_placeholder(self):
        """No account -> "" so setup/doctor report it as unconfigured."""
        mod = _plugin_module()
        with patch.object(mod, "_read_env", return_value=""):
            assert mod._resolve_base_url() == ""


class TestModelsSearchUrl:
    """GET /v1/models answers 405; the catalog lives at /ai/models/search."""

    def test_maps_inference_base_onto_catalog_endpoint(self):
        url = _search_url("https://api.cloudflare.com/client/v4/accounts/acct/ai/v1")
        assert url == (
            "https://api.cloudflare.com/client/v4/accounts/acct/ai/models/search"
        )

    def test_trailing_slash_tolerated(self):
        url = _search_url("https://api.cloudflare.com/client/v4/accounts/acct/ai/v1/")
        assert url.endswith("/ai/models/search")

    def test_non_workers_ai_base_url_declines(self):
        """A proxy/gateway base URL has no Cloudflare-shaped catalog."""
        assert _search_url("https://proxy.example.com/v1") == ""


class _FakeResponse:
    def __init__(self, payload):
        self._payload = json.dumps(payload).encode()

    def read(self):
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


_SEARCH_PAYLOAD = {
    "success": True,
    "result": [
        {
            "name": "@cf/openai/gpt-oss-120b",
            "properties": [
                {"property_id": "function_calling", "value": "true"},
                {"property_id": "context_window", "value": "128000"},
            ],
        },
        {
            "name": "@cf/deepseek-ai/deepseek-v4-flash-0731",
            "properties": [{"property_id": "function_calling", "value": "true"}],
        },
        {   # no function_calling property -> dropped
            "name": "@cf/meta/llama-guard-3-8b",
            "properties": [{"property_id": "context_window", "value": "131072"}],
        },
        {   # explicitly false -> dropped
            "name": "@cf/qwen/qwq-32b",
            "properties": [{"property_id": "function_calling", "value": "false"}],
        },
        {"properties": []},  # nameless -> skipped, must not raise
    ],
}


class TestFetchModels:
    def test_keeps_only_tool_calling_models_sorted(self):
        p = _profile()
        base = "https://api.cloudflare.com/client/v4/accounts/acct/ai/v1"
        with patch(
            "hermes_cli.urllib_security.open_credentialed_url",
            return_value=_FakeResponse(_SEARCH_PAYLOAD),
        ):
            models = p.fetch_models(api_key="k", base_url=base)
        assert models == [
            "@cf/deepseek-ai/deepseek-v4-flash-0731",
            "@cf/openai/gpt-oss-120b",
        ]

    def test_network_failure_returns_none_so_caller_uses_fallbacks(self):
        p = _profile()
        base = "https://api.cloudflare.com/client/v4/accounts/acct/ai/v1"
        with patch(
            "hermes_cli.urllib_security.open_credentialed_url",
            side_effect=OSError("boom"),
        ):
            assert p.fetch_models(api_key="k", base_url=base) is None

    def test_empty_result_returns_none_not_empty_list(self):
        p = _profile()
        base = "https://api.cloudflare.com/client/v4/accounts/acct/ai/v1"
        with patch(
            "hermes_cli.urllib_security.open_credentialed_url",
            return_value=_FakeResponse({"success": True, "result": []}),
        ):
            assert p.fetch_models(api_key="k", base_url=base) is None

    def test_unconfigured_account_declines_without_network(self):
        mod = _plugin_module()
        p = _profile()
        # base_url is an ordinary instance attribute, so patch the instance.
        with patch.object(mod, "_resolve_base_url", return_value=""), \
             patch.object(p, "base_url", ""):
            assert p.fetch_models(api_key="k") is None


class TestReasoningEffort:
    @pytest.mark.parametrize("model,accepted", sorted(ACCEPTED_BY_MODEL.items()))
    def test_never_emits_a_rejected_level(self, model, accepted):
        """Every hermes level must map into the model's accepted set or be omitted."""
        for level in LEVELS:
            sent = _effort(model, {"effort": level})
            assert sent is None or sent in accepted, (
                f"{model} would receive reasoning_effort={sent!r} for {level!r}, "
                f"which the API rejects (accepted: {sorted(accepted)})"
            )

    def test_qwen38_remaps_high_to_xhigh(self):
        """The one case where forwarding hermes's default level 400s."""
        assert _effort("@cf/qwen/qwen3.8-27b", {"effort": "high"}) == "xhigh"

    def test_harmony_clamps_xhigh_down_to_high(self):
        assert _effort("@cf/openai/gpt-oss-120b", {"effort": "xhigh"}) == "high"

    def test_kimi_maps_xhigh_to_max(self):
        assert _effort("@cf/moonshotai/kimi-k2.6", {"effort": "xhigh"}) == "max"

    def test_unknown_model_passes_through(self):
        """A model Cloudflare adds later still gets the requested level."""
        assert _effort("@cf/brand/new-model", {"effort": "high"}) == "high"

    def test_ultra_folds_onto_max(self):
        """`ultra` is hermes-only and has no upstream meaning."""
        assert _effort("@cf/zai-org/glm-5.2", {"effort": "ultra"}) == "max"

    def test_explicit_off_uses_none_where_supported(self):
        assert _effort("@cf/moonshotai/kimi-k2.6", {"enabled": False}) == "none"

    def test_explicit_off_omitted_where_unsupported(self):
        """Harmony 400s on reasoning_effort='none' — omit instead."""
        assert _effort("@cf/openai/gpt-oss-120b", {"enabled": False}) is None

    def test_non_reasoning_model_gets_no_effort(self):
        assert _effort(
            "@cf/ibm-granite/granite-4.0-h-micro",
            {"effort": "high"},
            supports_reasoning=False,
        ) is None

    def test_absent_effort_defers_to_model_default(self):
        assert _effort("@cf/zai-org/glm-5.2", {"effort": ""}) is None

    def test_unrecognised_effort_is_omitted(self):
        assert _effort("@cf/zai-org/glm-5.2", {"effort": "bogus"}) is None
