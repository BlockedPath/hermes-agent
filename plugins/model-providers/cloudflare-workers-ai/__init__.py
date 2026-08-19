"""Cloudflare Workers AI provider profile.

Workers AI serves an OpenAI-compatible Chat Completions surface at::

    https://api.cloudflare.com/client/v4/accounts/<ACCOUNT_ID>/ai/v1

Three things stop this from being a plain ``ProviderProfile(...)`` literal,
and all three are handled here rather than leaking into shared code:

1. **The base URL is account-scoped.**  ``CLOUDFLARE_ACCOUNT_ID`` is
   interpolated into the *path*, so the endpoint is assembled from the
   environment instead of being a constant.  When the account id is absent
   the profile registers with an empty ``base_url`` — the same convention
   ``azure-foundry`` uses for a per-resource endpoint — and the standard
   ``CLOUDFLARE_BASE_URL`` override supplies the URL at runtime.

2. **``GET /v1/models`` returns HTTP 405** (``GET not supported for
   requested URI``).  The catalog lives at ``/ai/models/search`` behind a
   Cloudflare envelope (``{"result": [...]}``) rather than the OpenAI
   ``{"data": [...]}`` shape, so ``fetch_models`` is overridden.  Only
   models advertising ``function_calling=true`` are surfaced, matching the
   "picker shows agentic models only" rule in the plugin README.

3. **Reasoning effort is a per-family dialect, and a wrong level is a hard
   HTTP 400** — not a silently-ignored field.  The accepted sets below were
   probed live against the API rather than inferred:

   ==========================  ====================================
   family                      accepted ``reasoning_effort``
   ==========================  ====================================
   gpt-oss (Harmony),          ``low`` ``medium`` ``high``
   nemotron-3, glm-4.7-flash
   qwen3.8-27b                 ``low`` ``medium`` ``xhigh``
   kimi-k2.x                   ``none`` ``low`` ``medium`` ``high``
                               ``max``
   deepseek-v4, glm-5.2,       everything
   gemma-4, qwen3-30b
   ==========================  ====================================

   Hermes's eight-level dial (``none|minimal|low|medium|high|xhigh|max|
   ultra``) is folded onto whichever set the target model accepts.  A level
   that maps to ``None`` is omitted entirely so the model applies its own
   default, which is the only correct behaviour for families that cannot
   disable thinking at all.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request
from typing import Any
from urllib.parse import urlencode

from providers import register_provider
from providers.base import ProviderProfile

logger = logging.getLogger(__name__)

_BASE_URL_TEMPLATE = "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1"


def _read_env(name: str) -> str:
    """Return credential *name*, honouring the active secret scope.

    Imported lazily: provider plugins are discovered while
    ``hermes_cli.auth`` is still executing its module body, and a top-level
    ``agent.secret_scope`` import there would close an import cycle.

    Never raises.  Under an active multiplexed turn ``get_secret`` fails
    closed for an unscoped read; that is reported as "absent" rather than
    falling back to ``os.environ``, which under multiplexing may hold a
    different profile's account.
    """
    get_secret = None
    try:
        from agent.secret_scope import get_secret as _get_secret
    except ImportError:
        get_secret = None
    else:
        get_secret = _get_secret

    if get_secret is None:
        raw = os.environ.get(name)
        return raw.strip() if raw else ""
    try:
        value = get_secret(name)
    except Exception:
        return ""
    return value.strip() if value else ""


def _resolve_base_url() -> str:
    """Return the account-scoped inference URL, or "" when unconfigured.

    Returning "" (rather than a URL carrying a literal placeholder) makes an
    account-less install read as *unconfigured* to ``hermes doctor`` and the
    setup flow, instead of emitting requests against a nonexistent account.
    """
    account_id = _read_env("CLOUDFLARE_ACCOUNT_ID")
    if not account_id:
        return ""
    return _BASE_URL_TEMPLATE.format(account_id=account_id)


# ── Reasoning-effort dialects ────────────────────────────────────────────
# Maps hermes's level -> the level to send, or None to omit the field.

_HARMONY_EFFORT: dict[str, str | None] = {
    # gpt-oss-120b/20b, nemotron-3, glm-4.7-flash. Rejects none/minimal/
    # xhigh/max: "reasoning_effort='xhigh' is not supported by Harmony".
    # There is no off switch, so `none` omits the field.
    "none": None,
    "minimal": "low",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "high",
    "max": "high",
    "ultra": "high",
}

_QWEN38_EFFORT: dict[str, str | None] = {
    # qwen3.8-27b: "Supported types are xhigh (default), medium, and low."
    # Note `high` is NOT accepted while `xhigh` is — the one case where a
    # verbatim passthrough of hermes's default level 400s.
    "none": None,
    "minimal": "low",
    "low": "low",
    "medium": "medium",
    "high": "xhigh",
    "xhigh": "xhigh",
    "max": "xhigh",
    "ultra": "xhigh",
}

_KIMI_EFFORT: dict[str, str | None] = {
    # kimi-k2.6 / kimi-k2.7-code: accepts `none` (a real off switch) and
    # `max`, but rejects `minimal` and `xhigh`.
    "none": "none",
    "minimal": "low",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "max",
    "max": "max",
    "ultra": "max",
}

_PERMISSIVE_EFFORT: dict[str, str | None] = {
    # deepseek-v4-*, glm-5.2, gemma-4-26b, qwen3-30b took every level tested.
    # `ultra` is a hermes-only name with no upstream meaning, so it is folded
    # onto `max` rather than sent verbatim into an untested code path.
    "none": "none",
    "minimal": "minimal",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "xhigh",
    "max": "max",
    "ultra": "max",
}

# Longest-prefix-wins; anything unmatched (including models Cloudflare adds
# later) gets the permissive table, mirroring the upstream default of
# forwarding the requested level.
_EFFORT_TABLES: tuple[tuple[str, dict[str, str | None]], ...] = (
    ("@cf/qwen/qwen3.8", _QWEN38_EFFORT),
    ("@cf/openai/gpt-oss", _HARMONY_EFFORT),
    ("@cf/nvidia/nemotron", _HARMONY_EFFORT),
    ("@cf/zai-org/glm-4.7-flash", _HARMONY_EFFORT),
    ("@cf/moonshotai/kimi", _KIMI_EFFORT),
)


def _effort_table(model: str | None) -> dict[str, str | None]:
    """Return the accepted-effort map for *model*."""
    name = (model or "").strip().lower()
    if not name:
        return _PERMISSIVE_EFFORT
    best: dict[str, str | None] = _PERMISSIVE_EFFORT
    best_len = -1
    for prefix, table in _EFFORT_TABLES:
        if name.startswith(prefix) and len(prefix) > best_len:
            best, best_len = table, len(prefix)
    return best


class CloudflareWorkersAIProfile(ProviderProfile):
    """Workers AI — account-scoped base URL, bespoke catalog, effort dialects."""

    # ── Catalog ──────────────────────────────────────────────────────────

    def _models_search_url(self, base_url: str | None) -> str:
        """Map an ``.../ai/v1`` inference base onto ``.../ai/models/search``.

        The account id is re-resolved here rather than reused from
        ``self.base_url`` so a ``CLOUDFLARE_ACCOUNT_ID`` that only becomes
        visible after import (a late-loaded ``~/.hermes/.env``, a profile
        switch) still yields a live catalog.

        Returns "" when no effective base URL exists, or when it points at
        something that is not a Workers AI path (a proxy or AI Gateway),
        since the Cloudflare-shaped catalog endpoint would not exist there.
        """
        base = (base_url or "").strip() or _resolve_base_url() or self.base_url
        base = base.rstrip("/")
        if not base.endswith("/ai/v1"):
            return ""
        return base[: -len("/v1")] + "/models/search"

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        """Return tool-calling Workers AI text models, or None to fall back.

        ``GET {base_url}/models`` — what the base implementation would call —
        answers 405 here, so this queries Cloudflare's own catalog and reads
        the ``function_calling`` property off each entry.  Models without it
        are dropped: hermes drives every turn through tool calls, so a
        non-agentic model in the picker is a guaranteed failure.
        """
        url = self._models_search_url(base_url)
        if not url:
            return None

        from hermes_cli.urllib_security import open_credentialed_url

        # per_page=100 comfortably covers the ~30 text-generation models
        # Cloudflare publishes; the endpoint paginates if that ever changes.
        query = urlencode({"task": "Text Generation", "per_page": "100"})
        req = urllib.request.Request(f"{url}?{query}")
        if api_key:
            req.add_header("Authorization", f"Bearer {api_key}")
        req.add_header("Accept", "application/json")
        for key, value in self.default_headers.items():
            req.add_header(key, value)

        try:
            with open_credentialed_url(req, timeout=timeout) as resp:
                payload = json.loads(resp.read().decode())
        except Exception as exc:
            logger.debug("fetch_models(%s): %s", self.name, exc)
            return None

        if not isinstance(payload, dict):
            return None
        models: list[str] = []
        for item in payload.get("result") or []:
            if not isinstance(item, dict):
                continue
            model_id = item.get("name")
            if not model_id:
                continue
            props = {
                p.get("property_id"): p.get("value")
                for p in (item.get("properties") or [])
                if isinstance(p, dict)
            }
            if str(props.get("function_calling", "")).strip().lower() != "true":
                continue
            models.append(model_id)
        # "" -> caller falls back to fallback_models, which is the right
        # answer for an empty/garbled response as well as a failed one.
        return sorted(models) or None

    # ── Request quirks ───────────────────────────────────────────────────

    def build_api_kwargs_extras(
        self,
        *,
        reasoning_config: dict | None = None,
        supports_reasoning: bool = False,
        model: str | None = None,
        **ctx: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Emit a top-level ``reasoning_effort`` the target model accepts.

        Gated on ``supports_reasoning`` so non-thinking models (granite,
        llama, mistral-small) never receive the field.  Any level the family
        rejects is folded onto its nearest accepted neighbour, and a family
        with no off switch omits the field rather than 400-ing on ``none``.
        """
        if not supports_reasoning or not isinstance(reasoning_config, dict):
            return {}, {}

        table = _effort_table(model)

        if not reasoning_config.get("enabled", True):
            off = table.get("none")
            return ({}, {"reasoning_effort": off}) if off else ({}, {})

        effort = (reasoning_config.get("effort") or "").strip().lower()
        if not effort:
            # No explicit dial — let Cloudflare apply the model default.
            return {}, {}
        mapped = table.get(effort)
        if not mapped:
            return {}, {}
        return {}, {"reasoning_effort": mapped}


cloudflare_workers_ai = CloudflareWorkersAIProfile(
    name="cloudflare-workers-ai",
    aliases=("cloudflare", "workers-ai", "cloudflare-ai"),
    display_name="Cloudflare Workers AI",
    description="Cloudflare Workers AI — serverless GPU inference, OpenAI-compatible",
    signup_url="https://dash.cloudflare.com/profile/api-tokens",
    # CLOUDFLARE_ACCOUNT_ID is deliberately NOT listed: auth.py treats every
    # non-URL env var as an API-key candidate, and an account id offered as a
    # bearer token would authenticate nothing while masking the real error.
    # It is consumed by the account-scoped URL resolver instead.
    env_vars=("CLOUDFLARE_API_KEY", "CLOUDFLARE_BASE_URL"),
    # Assembled from CLOUDFLARE_ACCOUNT_ID; "" when that is unset, in which
    # case CLOUDFLARE_BASE_URL (wired as this provider's base_url_env_var)
    # supplies the endpoint at request time.
    base_url=_resolve_base_url(),
    auth_type="api_key",
    hostname="api.cloudflare.com",
    # gemma-4, llama-4-scout, kimi-k2.x and qwen3.8 all take image input.
    supports_vision=True,
    # Output caps span 16K (gpt-oss) to 1M (deepseek-v4); no provider-wide
    # default, so each model's own limit applies unless the user sets one.
    default_max_tokens=None,
    # Cheapest tool-calling model in the catalog ($0.017/$0.112 per M).
    default_aux_model="@cf/ibm-granite/granite-4.0-h-micro",
    # Shown when the live catalog is unreachable (no key, no network, or a
    # user-supplied base URL that isn't a Workers AI path). Tool-calling
    # models only, matching the fetch_models filter.
    fallback_models=(
        "@cf/deepseek-ai/deepseek-v4-flash-0731",
        "@cf/deepseek-ai/deepseek-v4-pro-0813",
        "@cf/google/gemma-4-26b-a4b-it",
        "@cf/ibm-granite/granite-4.0-h-micro",
        "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
        "@cf/meta/llama-4-scout-17b-16e-instruct",
        "@cf/mistralai/mistral-small-3.1-24b-instruct",
        "@cf/moonshotai/kimi-k2.6",
        "@cf/moonshotai/kimi-k2.7-code",
        "@cf/nvidia/nemotron-3-120b-a12b",
        "@cf/openai/gpt-oss-120b",
        "@cf/openai/gpt-oss-20b",
        "@cf/qwen/qwen3-30b-a3b-fp8",
        "@cf/qwen/qwen3.8-27b",
        "@cf/zai-org/glm-4.7-flash",
        "@cf/zai-org/glm-5.2",
    ),
)

register_provider(cloudflare_workers_ai)
