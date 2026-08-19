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

    def test_appears_in_the_model_picker(self):
        from hermes_cli.models import list_available_providers

        row = next(
            r for r in list_available_providers()
            if r["id"] == "cloudflare-workers-ai"
        )
        assert row["label"] == "Cloudflare Workers AI"
        # Surfaced from _PROVIDER_ALIASES, not from the profile — see
        # TestProviderAliasRouting for why that distinction bites.
        assert set(row["aliases"]) == {"cloudflare", "workers-ai", "cloudflare-ai"}

    def test_fallback_models_are_all_cf_scoped(self):
        p = _profile()
        assert len(p.fallback_models) == 16
        assert all(m.startswith("@cf/") for m in p.fallback_models)


class TestProviderAliasRouting:
    """Profile aliases must also be mirrored into models._PROVIDER_ALIASES.

    A profile's ``aliases`` auto-extend ``PROVIDER_REGISTRY`` (auth.py) and
    ``CANONICAL_PROVIDERS``, but NOT ``_PROVIDER_ALIASES``.  An alias missing
    from that map does not raise — ``parse_model_input`` silently falls
    through to openrouter carrying the whole ``"cloudflare:@cf/..."`` string
    as a model name, which reaches the user as a bogus OpenRouter model error
    that never mentions Cloudflare.
    """

    @pytest.mark.parametrize("alias", ["cloudflare", "workers-ai", "cloudflare-ai"])
    def test_alias_normalizes_to_canonical_slug(self, alias):
        from hermes_cli.models import normalize_provider, provider_label

        assert normalize_provider(alias) == "cloudflare-workers-ai"
        assert provider_label(alias) == "Cloudflare Workers AI"

    @pytest.mark.parametrize(
        "typed,expected_model",
        [
            ("cloudflare:@cf/zai-org/glm-5.2", "@cf/zai-org/glm-5.2"),
            ("workers-ai:@cf/qwen/qwen3.8-27b", "@cf/qwen/qwen3.8-27b"),
            ("cloudflare-workers-ai:@cf/openai/gpt-oss-120b", "@cf/openai/gpt-oss-120b"),
        ],
    )
    def test_provider_model_shorthand_routes_to_cloudflare(self, typed, expected_model):
        from hermes_cli.models import parse_model_input

        assert parse_model_input(typed, "openrouter") == (
            "cloudflare-workers-ai",
            expected_model,
        )

    def test_does_not_hijack_an_existing_alias(self):
        """``qwen`` is claimed by alibaba; adding ours must not steal it."""
        from hermes_cli.models import normalize_provider

        assert normalize_provider("qwen") == "alibaba"


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


class TestProviderModelIds:
    """The picker must not require a non-empty import-time profile base URL."""

    _LATE_MODEL = "@cf/example/new-tool-model"
    _LIVE_PAYLOAD = {
        "success": True,
        "result": [
            {
                "name": _LATE_MODEL,
                "properties": [
                    {"property_id": "function_calling", "value": "true"}
                ],
            }
        ],
    }

    def test_account_id_loaded_after_profile_import_still_fetches_live_models(self):
        from hermes_cli.models import provider_model_ids

        mod = _plugin_module()
        p = _profile()
        late_base = "https://api.cloudflare.com/client/v4/accounts/late/ai/v1"
        with patch.object(p, "base_url", ""), \
             patch.object(mod, "_resolve_base_url", return_value=late_base), \
             patch(
                 "hermes_cli.auth.resolve_api_key_provider_credentials",
                 return_value={"api_key": "k", "base_url": ""},
             ), \
             patch(
                 "hermes_cli.urllib_security.open_credentialed_url",
                 return_value=_FakeResponse(self._LIVE_PAYLOAD),
             ) as opened:
            models = provider_model_ids("cloudflare-workers-ai")

        assert self._LATE_MODEL in models
        request = opened.call_args.args[0]
        assert request.full_url.startswith(
            "https://api.cloudflare.com/client/v4/accounts/late/ai/models/search?"
        )

    def test_base_url_only_configuration_still_fetches_live_models(self):
        from hermes_cli.models import provider_model_ids

        mod = _plugin_module()
        p = _profile()
        configured_base = (
            "https://api.cloudflare.com/client/v4/accounts/from-base-url/ai/v1"
        )
        with patch.object(p, "base_url", ""), \
             patch.object(mod, "_resolve_base_url", return_value=""), \
             patch(
                 "hermes_cli.auth.resolve_api_key_provider_credentials",
                 return_value={"api_key": "k", "base_url": configured_base},
             ), \
             patch(
                 "hermes_cli.urllib_security.open_credentialed_url",
                 return_value=_FakeResponse(self._LIVE_PAYLOAD),
             ) as opened:
            models = provider_model_ids("cloudflare-workers-ai")

        assert self._LATE_MODEL in models
        request = opened.call_args.args[0]
        assert request.full_url.startswith(
            "https://api.cloudflare.com/client/v4/accounts/from-base-url/ai/models/search?"
        )

    def test_missing_endpoint_still_returns_profile_fallbacks(self):
        from hermes_cli.models import provider_model_ids

        mod = _plugin_module()
        p = _profile()
        with patch.object(p, "base_url", ""), \
             patch.object(mod, "_resolve_base_url", return_value=""), \
             patch(
                 "hermes_cli.auth.resolve_api_key_provider_credentials",
                 return_value={"api_key": "k", "base_url": ""},
             ):
            models = provider_model_ids("cloudflare-workers-ai")

        assert models == list(p.fallback_models)


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
