#!/usr/bin/env python3
"""model-compare: pick the best value-for-money LLM on OpenRouter.

Run it before launching a coding agent to auto-select the model that gives
the most capability per dollar today. Data sources:

  * OpenRouter public API (/api/v1/models) - catalog, pricing (input and
    output), context window, release date and capabilities.
  * Artificial Analysis intelligence index - model quality. Primary
    source: OpenRouter's republished AA benchmarks (data.benchmarks in
    the frontend models API, exact per-model keys). When a model is not
    covered there: the AA API v2 (--aa-api-key or the AA_API_KEY
    environment variable; free key at artificialanalysis.ai), matched by
    exact slug/name only. Without a key, AA data comes only from
    OpenRouter. Unmatched models rank on price/context/age.

Ranking: every criterion is normalized to [0, 1] and combined with
priority-dependent weights:

    balanced:  quality 0.40 / price 0.40 / context 0.10 / age 0.10
    price:     quality 0.20 / price 0.60 / context 0.10 / age 0.10
    quality:   quality 0.60 / price 0.20 / context 0.10 / age 0.10

  price_score    log-scaled blended price (USD per 1M tokens,
                 --input-share input / rest output) within the candidate
                 pool; free listings score 1.0.
  quality_score  AA intelligence index divided by --quality-ref (default
                 70), measured at the model's best selectable depth and
                 clamped to [0, 1]; unmatched models score 0.
  context_score  logarithmic bonus from --min-context up to 4x that
                 threshold (extra headroom helps, but with diminishing
                 returns).
  age_score      exponential decay with --recency-half-life (default 120
                 days): the fresher the listing, the better.

Examples:
  model_compare.py                          top 5, balanced
  model_compare.py --best                   print only "#1 opencode id" (openrouter/<model>)
  model_compare.py --priority price --top 3 three cheapest-sensible picks
  model_compare.py --discount               only currently-discounted models
  model_compare.py --off-peak               also list scheduled off-peak deals
  model_compare.py --detail simple          fewer columns, friendlier labels
  model_compare.py --bare                   just the table(s): no summary or legend
  model_compare.py --catalog                full catalog (ranked + filtered) as one JSON document
  MODEL=$(model_compare.py --best)          shell integration (opencode --model $MODEL)
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import re
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
OPENROUTER_DISCOUNTS_URL = (
    "https://openrouter.ai/api/frontend/v1/models/find?output_modalities=text"
)
# Public per-endpoint list of zero-data-retention endpoints; DESIGN.md
# ("Zero data retention") describes how its entries are interpreted.
OPENROUTER_ZDR_URL = "https://openrouter.ai/api/v1/endpoints/zdr"
# Supported V2 free-tier replacement for the retired /api/v2/data/llms/models
# (legacy endpoints 410 after 2026-11-04; see
# https://artificialanalysis.ai/data-api/migrate-v2-data).
AA_API_URL = "https://artificialanalysis.ai/api/v2/language/models/free"
# Cache key for the AA API entries envelope. v4 adds the per-effort fields
# (id, slug, base, effort, creator, cost_per_task) parsed from each row's
# name/slug; v3 payloads predate them and must never be read.
AA_ENTRIES_CACHE = "aa-intelligence-v4"
# Reasoning-effort labels AA encodes in a row's name parenthetical or slug
# suffix. AA_EFFORT_ORDER is the display order (deepest first);
# REASONING_GATEWAY_EFFORTS is what OpenRouter accepts as a discrete effort.
AA_EFFORT_ORDER = (
    "max",
    "xhigh",
    "high",
    "medium",
    "low",
    "minimal",
    "none",
    "reasoning",
)
REASONING_GATEWAY_EFFORTS = ("max", "xhigh", "high", "medium", "low", "minimal", "none")
# --reasoning-effort values: "default" selects each model's best measured
# selectable effort and does not filter; the rest select and filter (spec §7).
REASONING_EFFORT_CHOICES = ("default",) + tuple(reversed(REASONING_GATEWAY_EFFORTS))
# The repo's own published catalog.json -- built by keyed CI runs, hydrated
# by unkeyed local runs (design: quality source reconciliation, source 2).
# The github.io host 301s to plain http://model-compare.dev/catalog.json, an
# unauthenticated downgrade on ranking-critical data; the https direct URL is
# verified 200 application/json. Do not point this back at github.io.
QUALITY_CATALOG_URL = "https://model-compare.dev/catalog.json"
# Cache key for the hydration envelope {"by_id", "generated_at",
# "schema_version"}. Only usable (non-empty by_id) results are ever written.
QUALITY_CATALOG_CACHE = "published-catalog"
# A published catalog generated more than this many days ago is stale:
# warn but use -- the records' as_of makes the staleness visible downstream.
QUALITY_CATALOG_MAX_AGE_DAYS = 14
VERSION = "0.7.10"
USER_AGENT = f"model-compare/{VERSION} (https://github.com/canonical/model-compare)"

# opencode expects provider-qualified model ids: openrouter/<provider/model>.
# Single source of truth -- other provider namespaces can be supported later
# without touching consumers of --best / --json.
OPENCODE_PROVIDER = "openrouter"


def opencode_model_id(model_id: str) -> str:
    return f"{OPENCODE_PROVIDER}/{model_id}"


PRIORITY_WEIGHTS = {
    "balanced": {"quality": 0.40, "price": 0.40, "context": 0.10, "age": 0.10},
    "price": {"quality": 0.20, "price": 0.60, "context": 0.10, "age": 0.10},
    "quality": {"quality": 0.60, "price": 0.20, "context": 0.10, "age": 0.10},
}

CATALOG_SCHEMA_VERSION = 6

# The drop-reason keys build_candidates counts, verbatim -- keep in sync with
# its drop() call sites. pool.dropped lists all of them zero-filled so the
# contract is self-describing. The last two come from the endpoint stage
# (apply_endpoint_pricing's drop sites), not build_candidates: snake_case
# while the older reasons are human phrases. The mixed naming convention is
# a deliberate, documented extension -- machine-readable reasons for the
# endpoint-stage drops the publish drop-rate gate keys on.
CATALOG_DROP_REASONS = (
    "malformed id",
    "context",
    "pricing",
    "schedule",
    "free",
    "batch",
    "no discount",
    "not ZDR",
    "modality",
    "tool calling",
    "expired",
    "age",
    "reasoning effort",
    "no_baseline_eligible_endpoint",
    "endpoint_fetch_failed",
)

# The only reasons the endpoint stage itself produces (the last two
# CATALOG_DROP_REASONS entries) -- the empty-pool warning keys on them;
# web/publish.py keeps its own copy of this tuple by design.
ENDPOINT_DROP_REASONS = (
    "no_baseline_eligible_endpoint",
    "endpoint_fetch_failed",
)

PAREN_RE = re.compile(r"\([^)]*\)")
PROVIDER_PREFIX_RE = re.compile(r"^[^:\s]{1,30}:\s*")

# Quantization precision ladder: label -> rank, highest fidelity first. The
# ladder ordering in the design doc is authoritative: fp6 > int8 is strict,
# and formats within one fidelity tier share the same rank. "unknown" never
# enters the ladder -- it only routes through the first-party eligibility
# carve-out, never through rank comparison.
PRECISION_LADDER: dict[str, int] = {
    "fp32": 4,
    "bf16": 3,
    "fp16": 3,
    "fp8": 2,
    "mxfp8": 2,
    "fp6": 1,
    "int8": 0,
    "fp4": -1,
    "mxfp4": -1,
    "nvfp4": -1,
    "int4": -2,
}

# Every quantization value an endpoint may report: the ladder keys plus the
# "unknown" sentinel. Whether a recorded value is recognized at all is
# membership here, not ladder rank.
QUANTIZATION_ENUM: frozenset[str] = frozenset(PRECISION_LADDER) | {"unknown"}

# Official partner hosts that serve closed labs' models: an endpoint on one
# of these providers counts as first-party for the eligibility carve-out
# even though it is not the maker itself. Entries are compared after
# normalize_provider (casefold + strip non-alphanumerics), so the slugs here
# are display forms -- "aws-bedrock" and "AWS Bedrock" both hit. Extend only
# with live-data evidence (e.g. an endpoint tag matching <lab>-official).
FIRST_PARTY_HOSTS = frozenset(
    {"bedrock", "azure", "vertex", "azure-ai", "google-vertex", "aws-bedrock"}
)

# Explicit first-party alias pairs that normalization alone cannot bridge
# (design §4.2): makers that self-name differently than their OpenRouter
# author slug. Key = the exact live provider_name string, value = the
# author slug the pair is first-party FOR; both sides are compared
# normalized (case/punctuation variants match), and the alias only fires
# when the model's author equals the value. Verified against live data
# 2026-10-20 (task-13-fix-report.md section 8b; endpoints fetched fresh,
# every endpoint quantization "unknown", status 0):
#   ("Amazon Bedrock", "amazon")   -- amazon/nova-2-lite-v1, 2 endpoints
#   ("Google AI Studio", "google") -- google/gemini-2.5-pro-preview, 3 endpoints
#   ("Mistral", "mistralai")       -- mistralai/mistral-large-4-0, 3 endpoints
# Add entries only after the same live verification -- never pin an
# unverified string.
FIRST_PARTY_ALIASES: dict[str, str] = {
    "Amazon Bedrock": "amazon",
    "Google AI Studio": "google",
    "Mistral": "mistralai",
}


def warn(msg: str) -> None:
    print(f"warning: {msg}", file=sys.stderr)


def norm_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


# ---------------------------------------------------------------------------
# Provider normalization + first-party detection
# ---------------------------------------------------------------------------


def normalize_provider(name: str) -> str:
    """Casefold a provider name and strip ALL non-alphanumeric characters.

    "Z.AI", "z-ai" and "z ai" all normalize to "zai", so a live-recorded
    maker name matches its author slug regardless of punctuation or case.
    None or empty input normalizes to "".
    """
    return re.sub(r"[^a-z0-9]+", "", (name or "").casefold())


def is_first_party(provider_name: str, author: str) -> bool:
    """True iff the provider is the model's own maker or a partner host.

    Criterion 1: the normalized provider name equals the normalized author
    slug. Criterion 2: it matches a normalized FIRST_PARTY_HOSTS entry, or
    a FIRST_PARTY_ALIASES key whose value equals the normalized author slug
    (the §4.2 alias pairs: a maker self-naming differently than its author
    slug is first-party only for that author). Everything else is
    third-party.
    """
    provider = normalize_provider(provider_name)
    if provider == normalize_provider(author):
        return True
    if provider in {normalize_provider(host) for host in FIRST_PARTY_HOSTS}:
        return True
    author_slug = normalize_provider(author)
    return any(
        provider == normalize_provider(alias_provider)
        and author_slug == normalize_provider(alias_author)
        for alias_provider, alias_author in FIRST_PARTY_ALIASES.items()
    )


# ---------------------------------------------------------------------------
# HTTP + cache
# ---------------------------------------------------------------------------


_DEFAULT_PORTS = {"https": 443, "http": 80}


class StripApiKeyRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Drop API credentials when a redirect leaves the original origin.

    urllib copies custom request headers onto the redirected request even
    across hosts. The AA API key ("X-api-key") and the session-cost bearer
    token ("Authorization") are kept only when scheme, host and effective
    port all match -- an explicit default port counts as its scheme's
    default -- so a cross-host redirect or an https->http downgrade never
    carries them.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None:
            old, dest = (
                urllib.parse.urlsplit(req.full_url),
                urllib.parse.urlsplit(new.full_url),
            )
            # An explicit default port is the scheme's default, not a
            # different origin: https://host:443 == https://host.
            same_origin = (
                old.scheme,
                old.hostname,
                old.port or _DEFAULT_PORTS.get(old.scheme),
            ) == (
                dest.scheme,
                dest.hostname,
                dest.port or _DEFAULT_PORTS.get(dest.scheme),
            )
            if not same_origin:
                # Request stores header names capitalize()d: "X-api-key".
                new.remove_header("X-api-key")
                # fetch_session_cost authenticates with a bearer token.
                new.remove_header("Authorization")
        return new


def http_get(url: str, headers: dict | None = None, timeout: int = 30) -> bytes:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept-Encoding": "gzip",
            **(headers or {}),
        },
    )
    opener = urllib.request.build_opener(StripApiKeyRedirectHandler)
    with opener.open(req, timeout=timeout) as resp:
        data = resp.read()
    if data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    return data


def fetch_json(
    url: str, headers: dict | None = None, timeout: int = 30, retries: int = 1
):
    last_exc: Exception = RuntimeError("request failed")
    for attempt in range(retries + 1):
        try:
            return json.loads(http_get(url, headers, timeout))
        except urllib.error.HTTPError as exc:
            # HTTPError is a URLError, so this clause must come first. A
            # 4xx (bad key, forbidden, rate limited) will not change on an
            # immediate retry; raise it at once. 5xx falls through to retry.
            if 400 <= exc.code < 500:
                raise
            last_exc = exc
            if attempt < retries:
                time.sleep(1)
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            last_exc = exc
            if attempt < retries:
                time.sleep(1)
    raise last_exc


def cache_path(name: str):
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(
        os.path.expanduser("~"), ".cache"
    )
    return base and os.path.join(base, "model-compare", f"{name}.json")


def load_cache(name: str, ttl_seconds: float):
    path = cache_path(name)
    if not path:
        return None
    try:
        with open(path) as fh:
            blob = json.load(fh)
        if time.time() - blob["fetched_at"] <= ttl_seconds:
            return blob["payload"]
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


def save_cache(name: str, payload) -> None:
    path = cache_path(name)
    if not path:
        return
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w") as fh:
            json.dump({"fetched_at": time.time(), "payload": payload}, fh)
        os.replace(tmp_path, path)
    except OSError as exc:
        warn(f"could not write cache: {exc}")


# ---------------------------------------------------------------------------
# OpenRouter endpoints (per-model provider list)
# ---------------------------------------------------------------------------


OPENROUTER_ENDPOINTS_URL = "https://openrouter.ai/api/v1/models/{id}/endpoints"


def load_cache_expired(name: str) -> dict | None:
    """Return the cached payload regardless of age; None when absent/corrupt.

    load_cache drops payloads past their TTL; this variant serves them anyway
    so a dropped model's last known endpoint annotations can still be shown
    (marked stale by the caller). Never used for ranking.
    """
    path = cache_path(name)
    if not path:
        return None
    try:
        with open(path) as fh:
            blob = json.load(fh)
        return blob["payload"]
    except (OSError, ValueError, KeyError, TypeError):
        return None


def parse_endpoints(payload) -> list[dict]:
    """Keep the verified per-endpoint fields of an endpoints response.

    The live GET /api/v1/models/{id}/endpoints response nests the list one
    level below the usual envelope: `data` is the model object (id, name,
    architecture, ...) and the per-provider entries live in
    `data["endpoints"]`. `pricing` keeps prompt/completion (parsed through
    parse_price) and the raw `overrides` list -- the tiered-pricing
    machinery parses the entries itself. Fields beyond the verified
    per-entry shape are dropped, and the parse keeps `status` as-is (the
    caller filters).
    """
    # Task 13 lesson: the first unit fixtures fed the models/find shape
    # ({"data": [entry, ...]}) -- a list directly under "data" -- so the
    # suite stayed green while every live parse returned [] and the whole
    # pool died as no_baseline_eligible_endpoint. Pin the live envelope:
    # the list is read ONLY from data["endpoints"], and every other shape
    # parses to [] (no compat branch for the wrong fixture shape).
    data = payload.get("data") if isinstance(payload, dict) else None
    entries = data.get("endpoints") if isinstance(data, dict) else None
    endpoints = []
    for entry in entries or []:
        if not isinstance(entry, dict):
            continue
        raw = entry.get("pricing")
        raw_pricing = raw if isinstance(raw, dict) else {}
        max_completion = entry.get("max_completion_tokens")
        if type(max_completion) is not int or max_completion <= 0:
            max_completion = None
        endpoints.append(
            {
                "provider_name": entry.get("provider_name"),
                "tag": entry.get("tag"),
                "quantization": entry.get("quantization"),
                "pricing": {
                    "prompt": parse_price(raw_pricing.get("prompt")),
                    "completion": parse_price(raw_pricing.get("completion")),
                    "input_cache_read": parse_non_negative_price(
                        raw_pricing.get("input_cache_read")
                    ),
                    "input_cache_write": parse_non_negative_price(
                        raw_pricing.get("input_cache_write")
                    ),
                    "input_cache_write_1h": parse_non_negative_price(
                        raw_pricing.get("input_cache_write_1h")
                    ),
                    "overrides": raw_pricing.get("overrides"),
                },
                "context_length": entry.get("context_length"),
                "max_completion_tokens": max_completion,
                "status": entry.get("status"),
                "service_tier": entry.get("service_tier"),
            }
        )
    return endpoints


def endpoint_cache_name(model_id: str) -> str:
    """The one cache key both the fetch and the stale-annotation read use."""
    return f"endpoints-v2:{model_id.replace('/', '_')}"


def fetch_model_endpoints(model_id: str, args) -> list[dict]:
    """Fetch + parse one model's endpoint list, cached per model.

    The cache name flattens the model id ("/" -> "_") to keep the cache
    files flat. Fetches retry 3 times (4 attempts total) via fetch_json,
    which raises at once on 4xx and after the retries on 5xx/network
    errors -- the caller owns the fail-closed policy.
    """
    cache_name = endpoint_cache_name(model_id)
    if not args.no_cache:
        cached = load_cache(cache_name, args.cache_ttl)
        if cached:
            return cached
    payload = fetch_json(
        OPENROUTER_ENDPOINTS_URL.format(id=model_id), timeout=30, retries=3
    )
    endpoints = parse_endpoints(payload)
    save_cache(cache_name, endpoints)
    return endpoints


def is_variant_endpoint(endpoint) -> bool:
    """True iff the endpoint is a :free/:batch variant or entirely free.

    Variant iff its slug or tag carries a `:free`/`:batch` marker OR both
    input and output prices are zero. `:batch` is cheaper-but-restricted,
    not zero-priced -- only the marker test catches it. Missing or
    unparseable prices are not zero.
    """
    for field in ("slug", "tag"):
        marker = endpoint.get(field)
        if isinstance(marker, str) and (":free" in marker or ":batch" in marker):
            return True
    pricing = endpoint.get("pricing")
    prices = pricing if isinstance(pricing, dict) else {}
    return (
        parse_price(prices.get("prompt")) == 0.0
        and parse_price(prices.get("completion")) == 0.0
    )


def endpoint_schedule_ok(endpoint, args) -> bool:
    """False iff the endpoint's pricing.overrides fails the v2 schedule rule.

    Reuses effective_pricing in full: the gate fires on malformed windows,
    exactly-one-of utc_start/utc_end pairs, bad utc_days, combined
    token+time entries and missing inheritance -- not just non-tiling
    windows. An endpoint without overrides has no schedule to fail.
    """
    result = effective_pricing(
        endpoint.get("pricing"), args.min_context, args.input_share
    )
    return not result["schedule_error"]


# Schedule gate for args-less callers: effective_pricing's schedule_error
# depends only on the SHAPE of pricing.overrides (malformed windows, bad
# utc_days, ...), never on min_context/input_share -- those only affect the
# resolved prices. derive_default_precision has no args parameter (pinned
# interface), so it gates through this fixed stand-in; any args would give
# the same schedule_error verdict.
_SCHEDULE_GATE_ARGS = argparse.Namespace(min_context=0, input_share=0.75)


def _baseline_universe(endpoints: list[dict], args=None) -> list[dict]:
    """The shared derivation universe (design section 4.1).

    Live (status == 0), not a variant endpoint (:free/:batch marker or
    zero-priced), and passing the schedule gate. Schedule-broken endpoints
    must not pin the default that governs healthy endpoints, same rationale
    as the variant exclusion. args defaults to the structural stand-in
    (see _SCHEDULE_GATE_ARGS); callers with real args pass them.
    """
    if args is None:
        args = _SCHEDULE_GATE_ARGS
    return [
        endpoint
        for endpoint in endpoints
        if endpoint.get("status") == 0
        and not is_variant_endpoint(endpoint)
        and endpoint_schedule_ok(endpoint, args)
    ]


def derive_default_precision(endpoints: list[dict], author: str) -> str | None:
    """Derive a model's default precision from its endpoint list (section 4.1).

    Universe: live non-variant schedule-passing endpoints (_baseline_universe).
    Primary rule: the highest-ranked KNOWN quantization among the maker's
    endpoints -- matched by normalized author-slug equality only, so partner
    hosts never pin the maker's default (section 4.2 criterion 2 does not
    apply here). Fallback: when the maker has no known quantization, the
    highest-ranked known quantization across the FULL universe (partners
    included -- best-available). None when no endpoint in the universe has a
    known quantization (the all-unknown closed-model case): eligibility then
    relies solely on the first-party carve-out.

    Known means a PRECISION_LADDER key: "unknown" and unparseable/absent
    values never enter the ladder. An endpoint whose provider_name is
    absent/empty (or normalizes to empty) is skipped for maker matching
    (never first-party -- fail-closed), but still feeds the fallback
    universe.
    """
    universe = _baseline_universe(endpoints, _SCHEDULE_GATE_ARGS)
    maker = normalize_provider(author)
    maker_best = None
    fallback_best = None
    for endpoint in universe:
        quantization = endpoint.get("quantization")
        if not isinstance(quantization, str) or quantization not in PRECISION_LADDER:
            continue  # "unknown", null, unrecognized: no rank, no pinning
        rank = PRECISION_LADDER[quantization]
        if fallback_best is None or rank > fallback_best[0]:
            fallback_best = (rank, quantization)
        # Maker matching only: a provider_name that normalizes to nothing
        # must never match an (equally empty) author slug.
        provider_slug = (
            normalize_provider(endpoint.get("provider_name"))
            if isinstance(endpoint.get("provider_name"), str)
            else ""
        )
        if (
            provider_slug
            and provider_slug == maker
            and (maker_best is None or rank > maker_best[0])
        ):
            maker_best = (rank, quantization)
    return (maker_best or fallback_best or (None, None))[1]


def baseline_eligible(endpoint, default_precision, author, args) -> bool:
    """Baseline-eligibility rule (design section 4.3, fail-closed).

    True iff the endpoint is live (status == 0), not a variant, passes the
    schedule gate, and then either:

    * its quantization is a known ladder label with rank >=
      rank(default_precision) -- provider-blind (default_routing considers
      every endpoint OpenRouter may send traffic to), or
    * its quantization is "unknown" or null AND the endpoint is first-party
      (the D3 carve-out: the maker or a partner host).

    When default_precision is None (all-unknown model) the rank comparison is
    meaningless -- only the carve-out applies. A provider_name that is
    absent/empty (or normalizes to empty) is never first-party, so such an
    endpoint can only qualify through the rank branch. An unrecognized
    quantization string (neither a ladder label nor "unknown"/null) is a
    schema anomaly: no rank comparison and no carve-out -- fail-closed
    (fleet-wide new-label drift is what the publish drop-rate gate exists
    to catch).
    """
    if (
        endpoint.get("status") != 0
        or is_variant_endpoint(endpoint)
        or not endpoint_schedule_ok(endpoint, args)
    ):
        return False
    quantization = endpoint.get("quantization")
    if isinstance(quantization, str) and quantization in PRECISION_LADDER:
        # Known quantization: the rank branch, provider-blind.
        if default_precision is None:
            return False
        return PRECISION_LADDER[quantization] >= PRECISION_LADDER[default_precision]
    # Unverifiable: the carve-out applies to "unknown" and null only --
    # an unrecognized string fails closed. A provider_name that normalizes
    # to nothing is never first-party (fail-closed; the literal
    # is_first_party("", "") iff is guarded out here).
    provider_name = endpoint.get("provider_name")
    provider_slug = (
        normalize_provider(provider_name) if isinstance(provider_name, str) else ""
    )
    return (
        (quantization is None or quantization == "unknown")
        and bool(provider_slug)
        and is_first_party(provider_name, author)
    )


def pick_baseline(
    endpoints: list[dict], eligible_flags: list[bool], args
) -> dict | None:
    """Pick the baseline endpoint among the eligible ones (design section 4.5).

    Cheapest blended per-token price wins, computed from effective_pricing
    with the args' min_context and input_share (the shared --input-share
    convention). Ties broken by (normalized provider name, tag), then --
    still tied -- first in list order (deterministic; Python's sort is
    stable and the endpoints list order is the API's). Returns None when no
    eligible endpoint has a usable price (the caller owns the drop).
    """
    candidates = []
    for endpoint, eligible in zip(endpoints, eligible_flags):
        if not eligible:
            continue
        eff = effective_pricing(
            endpoint.get("pricing"), args.min_context, args.input_share
        )
        price_in, price_out = eff["price_in"], eff["price_out"]
        if (
            price_in is None
            or price_out is None
            or not math.isfinite(price_in)
            or not math.isfinite(price_out)
        ):
            continue  # unpriceable: no executable blended price
        blended = args.input_share * price_in + (1 - args.input_share) * price_out
        provider_name = endpoint.get("provider_name")
        candidates.append(
            (
                blended,
                normalize_provider(provider_name)
                if isinstance(provider_name, str)
                else "",
                endpoint.get("tag") or "",
                endpoint,
            )
        )
    if not candidates:
        return None
    return min(candidates, key=lambda candidate: candidate[:3])[3]


def _blended_price(endpoint, args) -> float | None:
    """Blended per-token price of one endpoint (pick_baseline's price rule).

    effective_pricing with the args' min_context and input_share, then
    input_share * price_in + (1 - input_share) * price_out. None when either
    resolved price is missing or non-finite -- no executable blended price,
    the same skip pick_baseline applies. A zero result is returned as-is:
    zero-priced endpoints are variants (already excluded upstream), so the
    callers apply their own zero guards defensively.
    """
    eff = effective_pricing(endpoint.get("pricing"), args.min_context, args.input_share)
    price_in, price_out = eff["price_in"], eff["price_out"]
    if (
        price_in is None
        or price_out is None
        or not math.isfinite(price_in)
        or not math.isfinite(price_out)
    ):
        return None
    blended = args.input_share * price_in + (1 - args.input_share) * price_out
    return blended if math.isfinite(blended) else None


def _flag_list(endpoints: list[dict], eligible_flags: list[bool]):
    """Zip endpoints with their parallel eligibility flags, hardened.

    eligible_flags is a parallel list to endpoints (the baseline_eligible
    verdict per endpoint, same order -- Task 4 contract). A length mismatch
    must raise, never truncate silently (Task 4 review ruling).
    """
    if len(endpoints) != len(eligible_flags):
        raise ValueError(
            "eligible_flags must be a parallel list to endpoints: "
            f"{len(endpoints)} endpoints, {len(eligible_flags)} flags"
        )
    return zip(endpoints, eligible_flags, strict=True)


def NON_ELIGIBLE_UNIVERSE(endpoints: list[dict], eligible_flags: list[bool]) -> list:
    """The shared annotations universe (design section 4.6).

    Live (status == 0), non-variant endpoints that are NOT baseline-eligible:
    sub-default known quantization, third-party unknown/null quantization, or
    schedule-gate-failed. All annotations derive from this one universe.
    Eligibility comes from the parallel eligible_flags list (the
    baseline_eligible verdicts); variants and non-live endpoints are excluded
    even though they too are not eligible -- unlike the steering `ignore`
    list, which includes live variants (section 4.7).

    Raises ValueError when eligible_flags does not match endpoints in length.
    """
    return [
        endpoint
        for endpoint, eligible in _flag_list(endpoints, eligible_flags)
        if endpoint.get("status") == 0
        and not is_variant_endpoint(endpoint)
        and not eligible
    ]


def build_annotations(
    endpoints: list[dict],
    eligible_flags: list[bool],
    baseline,
    catalog_context,
    args,
) -> dict:
    """Per-model endpoint annotations (design section 4.6).

    All four annotations derive from one shared universe -- the live
    non-variant endpoints that are not baseline-eligible
    (NON_ELIGIBLE_UNIVERSE):

    * price_spread: cheapest non-eligible blended price divided by the
      cheapest eligible blended price, both blended with the args'
      min_context and input_share (the shared --input-share convention).
      None when the universe is empty -- generalized to None when either
      side has no priceable member at all -- and None when either cheapest
      price is 0 (zero-priced endpoints are variants by definition, so the
      zero half of the guard is defensive only). Unpriceable endpoints are
      skipped when looking for a cheapest price.
    * cheapest_quantized: {provider, quantization, price} of the cheapest
      member of the quantization-caused subset of the universe -- endpoints
      non-eligible due to quantization only (sub-default known quantization
      or third-party unknown/null), NOT schedule-gate-failed ones, which are
      excluded even when cheaper. Unpriceable subset members are skipped
      (defensively also non-positive ones, matching the repo-wide treatment
      of negative prices as malformed); None when the subset has no
      priceable member. Price ties keep the first in list order
      (deterministic).
    * risk_flag: True iff the sum of 1/blended**2 over the universe strictly
      exceeds the sum over the baseline-eligible endpoints (exact tie ->
      False). Any endpoint whose blended price is 0/None (defensively also
      negative) is skipped from BOTH sums -- variants are already excluded,
      so this only guards malformed data, and the flag stays deterministic.
    * baseline_context_below_catalog: True iff the baseline endpoint's
      context_length is smaller than the catalog context (the models-API
      value). A None baseline -- or missing context on either side -- yields
      False: the caveat marks a positive finding, never missing data.

    Raises ValueError when eligible_flags does not match endpoints in length.
    """
    universe = NON_ELIGIBLE_UNIVERSE(endpoints, eligible_flags)

    universe_prices = []
    for endpoint in universe:
        price = _blended_price(endpoint, args)
        if price is not None:
            universe_prices.append(price)
    eligible_prices = []
    for endpoint, eligible in _flag_list(endpoints, eligible_flags):
        if not eligible:
            continue
        price = _blended_price(endpoint, args)
        if price is not None:
            eligible_prices.append(price)
    cheapest_non_eligible = min(universe_prices) if universe_prices else None
    cheapest_eligible = min(eligible_prices) if eligible_prices else None
    price_spread = (
        cheapest_non_eligible / cheapest_eligible
        if cheapest_non_eligible is not None
        and cheapest_eligible is not None
        and cheapest_non_eligible > 0
        and cheapest_eligible > 0
        else None
    )

    quantized_best = None
    for endpoint in universe:
        if not endpoint_schedule_ok(endpoint, args):
            continue  # schedule-gate-failed: not a quantization-caused exclusion
        price = _blended_price(endpoint, args)
        if price is None or price <= 0:
            continue  # unpriceable/non-positive: no comparable price
        if quantized_best is None or price < quantized_best[0]:
            quantized_best = (price, endpoint)  # ties keep the first in list order
    if quantized_best is None:
        cheapest_quantized = None
    else:
        price, endpoint = quantized_best
        cheapest_quantized = {
            "provider": endpoint.get("provider_name"),
            "quantization": endpoint.get("quantization"),
            "price": price,
        }

    non_eligible_weight = 0.0
    for endpoint in universe:
        price = _blended_price(endpoint, args)
        if price is None or price <= 0:
            continue  # skipped from BOTH sums, deterministically
        non_eligible_weight += 1.0 / (price * price)
    eligible_weight = 0.0
    for endpoint, eligible in _flag_list(endpoints, eligible_flags):
        if not eligible:
            continue
        price = _blended_price(endpoint, args)
        if price is None or price <= 0:
            continue
        eligible_weight += 1.0 / (price * price)
    risk_flag = non_eligible_weight > eligible_weight

    baseline_context = baseline.get("context_length") if baseline is not None else None
    baseline_context_below_catalog = (
        baseline_context is not None
        and catalog_context is not None
        and baseline_context < catalog_context
    )

    return {
        "price_spread": price_spread,
        "cheapest_quantized": cheapest_quantized,
        "risk_flag": risk_flag,
        "baseline_context_below_catalog": baseline_context_below_catalog,
    }


def build_steering(
    model_id: str,
    endpoints: list[dict],
    eligible_flags: list[bool],
    default_precision: str | None,
    args,
) -> dict:
    """OpenRouter provider steering object (design section 4.7).

    {"model": model_id, "provider": {"order": [...], "ignore": [...],
    "quantizations": [...], "allow_fallbacks": True}}.

    * order: tags of the baseline-eligible endpoints, cheapest blended first
      (blended price asc, ties broken by (normalized provider name, tag) --
      the same comparator as pick_baseline, stable so the final tie is list
      order). Eligible endpoints whose blended price is unpriceable cannot
      be price-ranked; they are appended after the ranked ones (still
      (provider, tag)-ordered) so every eligible endpoint keeps a place --
      ranked last, never silently left to default priority.
    * ignore: tags of EVERY live endpoint that is not baseline-eligible --
      sub-default quantization, third-party unknown/null, schedule-gate-
      failed AND variants (variants are live but not eligible; the universal
      quantifier includes them). Non-live (status != 0) endpoints are never
      listed. List order (the API's).
    * quantizations: sorted KNOWN quantizations (PRECISION_LADDER labels)
      with rank >= rank(default_precision) among live non-variant endpoints,
      deduplicated -- no schedule or eligibility filter. OMITTED from the
      provider dict entirely when default_precision is None -- never an
      empty list (OpenRouter semantics would filter out every endpoint).
    * allow_fallbacks: True.

    Signature deviation from the plan (controller ruling at Task 5 spawn):
    args is a required parameter -- the order ranking is real pricing math
    (effective_pricing via min_context/input_share), and reusing the
    shape-only _SCHEDULE_GATE_ARGS stand-in for pricing was explicitly
    ruled out in the Task 4 review.

    Raises ValueError when eligible_flags does not match endpoints in length.
    """
    ignore = []
    ranked = []  # (blended, normalized provider name, tag) -- pick_baseline order
    unranked = []  # (normalized provider name, tag) -- eligible but unpriceable
    for endpoint, eligible in _flag_list(endpoints, eligible_flags):
        if endpoint.get("status") == 0 and not eligible:
            ignore.append(endpoint.get("tag") or "")
        if not eligible:
            continue
        provider_name = endpoint.get("provider_name")
        provider_key = (
            normalize_provider(provider_name) if isinstance(provider_name, str) else ""
        )
        tag = endpoint.get("tag") or ""
        price = _blended_price(endpoint, args)
        if price is None:
            unranked.append((provider_key, tag))
        else:
            ranked.append((price, provider_key, tag))
    ranked.sort()  # stable: the final tie is list order, as in pick_baseline
    order = [tag for _, _, tag in ranked] + [tag for _, tag in sorted(unranked)]

    provider = {"order": order, "ignore": ignore}
    if default_precision is not None:
        default_rank = PRECISION_LADDER[default_precision]
        quants = set()
        for endpoint in endpoints:
            if endpoint.get("status") != 0 or is_variant_endpoint(endpoint):
                continue
            quantization = endpoint.get("quantization")
            if (
                not isinstance(quantization, str)
                or quantization not in PRECISION_LADDER
            ):
                continue  # "unknown"/null/unrecognized: not a known quantization
            if PRECISION_LADDER[quantization] >= default_rank:
                quants.add(quantization)
        provider["quantizations"] = sorted(quants)
    provider["allow_fallbacks"] = True
    return {"model": model_id, "provider": provider}


def _stale_filtered_annotations(model_id, catalog_context, args):
    """Annotations for a dropped model's filtered entry, or None.

    Built from the model's last cached endpoint list (load_cache_expired --
    TTL ignored), never from a fresh fetch: the model is out of the ranking,
    so the block is marked "stale": True for display in the filtered section
    only (design section 4.9). Same builders as the live stage: eligibility
    and baseline are re-derived from the cached list, so the shape matches a
    ranked candidate's annotations. A model whose cache holds no endpoint
    list at all yields None (the entry carries the drop reason alone).
    """
    cached = load_cache_expired(endpoint_cache_name(model_id))
    if not isinstance(cached, list):
        return None
    author = model_id.partition("/")[0]
    try:
        default_precision = derive_default_precision(cached, author)
        eligible_flags = [
            baseline_eligible(endpoint, default_precision, author, args)
            for endpoint in cached
        ]
        baseline = pick_baseline(cached, eligible_flags, args)
        annotations = build_annotations(
            cached, eligible_flags, baseline, catalog_context, args
        )
    except Exception:
        # Display-only block for an already-dropped model: never fail the run.
        return None
    annotations["stale"] = True
    return annotations


def _endpoint_stage_drop(extra_dropped, filtered, cand, reason, args):
    """Count one endpoint-stage drop and append its filtered entry.

    Same validated entry shape as build_candidates' drop() plus the optional
    stale annotations block (absent cache -> reasons only).
    """
    extra_dropped[reason] = extra_dropped.get(reason, 0) + 1
    entry = {"id": cand["id"], "name": cand["name"] or cand["id"], "reasons": [reason]}
    annotations = _stale_filtered_annotations(cand["id"], cand.get("context"), args)
    if annotations is not None:
        entry["annotations"] = annotations
    filtered.append(entry)


def apply_endpoint_pricing(candidates, args, filtered):
    """Reprice every candidate at its baseline endpoint (design section 4.9).

    The CLI stage between build_candidates and resolve_quality. Endpoint
    fetches run concurrently (ThreadPoolExecutor, 8 workers) through
    fetch_model_endpoints' per-model cache, but the results are applied PER
    CANDIDATE ID in the original candidates order -- never in completion
    order -- so the stage is deterministic.

    Per candidate: endpoints = fetch. On fetch failure after fetch_json's
    internal retries the model is dropped "endpoint_fetch_failed" and removed
    from the returned list. With endpoints, default_precision is derived,
    every endpoint scored for baseline eligibility and the baseline picked;
    no usable baseline drops the model "no_baseline_eligible_endpoint"
    (the stale-annotations filtered-entry rule covers both drop reasons).
    Otherwise the candidate is mutated in place: the pricing block
    (price_in/price_out/blended/base_*/tier/schedule/notes) is replaced with
    the baseline endpoint's effective_pricing outputs scaled per-1M exactly
    like build_candidates, and the additive keys "endpoints", "default_precision",
    "annotations" (catalog context = the candidate's existing "context"),
    "advertised_pricing" (the models-API scalars the candidate carried before
    the mutation, source-tagged) and "steering" are attached.

    A baseline whose prices are missing/non-finite counts as unusable
    (pick_baseline's blended-price skip is authoritative for the top-level
    effective prices). The same bar applies to the base prices the mutation
    would publish: parse_price returns float | None, so a None base price
    (tier-over-None-base endpoint) is reachable -- the isinstance(..., float)
    half of the guard is load-bearing for None, not a redundant type check.
    Returns (surviving candidates in the original order, dropped counts for
    the two new reasons).
    """
    extra_dropped: dict[str, int] = {}
    if not candidates:
        return candidates, extra_dropped
    endpoints_by_id: dict[str, list] = {}
    failed_ids: set[str] = set()
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {}
        for cand in candidates:
            # Duplicate ids fetch once and share the result.
            futures.setdefault(
                cand["id"], pool.submit(fetch_model_endpoints, cand["id"], args)
            )
        for cand_id, future in futures.items():
            try:
                endpoints_by_id[cand_id] = future.result()
            except Exception:
                # fetch_json already retried 3 times; the caller owns the
                # fail-closed policy.
                failed_ids.add(cand_id)
    survivors = []
    for cand in candidates:
        cand_id = cand["id"]
        if cand_id in failed_ids:
            _endpoint_stage_drop(
                extra_dropped, filtered, cand, "endpoint_fetch_failed", args
            )
            continue
        endpoints = endpoints_by_id[cand_id]
        author = cand_id.partition("/")[0]
        default_precision = derive_default_precision(endpoints, author)
        eligible_flags = [
            baseline_eligible(endpoint, default_precision, author, args)
            for endpoint in endpoints
        ]
        baseline = pick_baseline(endpoints, eligible_flags, args)
        eff = (
            effective_pricing(
                baseline.get("pricing"), args.min_context, args.input_share
            )
            if baseline is not None
            else None
        )
        if baseline is None or not all(
            isinstance(eff[key], float) and math.isfinite(eff[key])
            for key in ("price_in", "price_out", "base_price_in", "base_price_out")
        ):
            _endpoint_stage_drop(
                extra_dropped, filtered, cand, "no_baseline_eligible_endpoint", args
            )
            continue
        # The advertised models-API scalars, captured before the mutation.
        advertised_pricing = {
            "input_per_1m": cand["price_in"],
            "output_per_1m": cand["price_out"],
            "blended_per_1m": cand["blended"],
            "source": "models-api",
        }
        # Replace the pricing block with the baseline endpoint's outputs,
        # scaled per-1M exactly like build_candidates.
        cand["price_in"] = eff["price_in"] * 1_000_000.0
        cand["price_out"] = eff["price_out"] * 1_000_000.0
        cand["blended"] = (
            args.input_share * cand["price_in"]
            + (1.0 - args.input_share) * cand["price_out"]
        )
        cand["base_price_in"] = eff["base_price_in"] * 1_000_000.0
        cand["base_price_out"] = eff["base_price_out"] * 1_000_000.0
        cand["cache_read_per_1m"] = (
            _per_1m(eff["price_cache_read"])
            if eff["price_cache_read"] is not None
            else None
        )
        cand["cache_write_per_1m"] = (
            _per_1m(eff["price_cache_write"])
            if eff["price_cache_write"] is not None
            else None
        )
        cand["base_cache_read_per_1m"] = (
            _per_1m(eff["base_cache_read"])
            if eff["base_cache_read"] is not None
            else None
        )
        cand["base_cache_write_per_1m"] = (
            _per_1m(eff["base_cache_write"])
            if eff["base_cache_write"] is not None
            else None
        )
        baseline_max_output = baseline.get("max_completion_tokens")
        if type(baseline_max_output) is int and baseline_max_output > 0:
            cand["max_output_tokens"] = baseline_max_output
        cand["tier_prompt_tokens"] = eff["tier_prompt_tokens"]
        cand["tiers"] = eff["tiers"]
        cand["schedule"] = eff["schedule"]
        cand["max_discount"] = eff["max_discount"]
        cand["tier_note"] = fmt_tier_note(eff["tier_prompt_tokens"])
        cand["sched_note"] = fmt_sched_note(eff["max_discount"])
        cand["sched_detail"] = fmt_sched_detail(eff["schedule"], eff["peak_blended"])
        cand["endpoints"] = endpoints
        cand["default_precision"] = default_precision
        cand["annotations"] = build_annotations(
            endpoints, eligible_flags, baseline, cand["context"], args
        )
        cand["advertised_pricing"] = advertised_pricing
        cand["steering"] = build_steering(
            cand_id, endpoints, eligible_flags, default_precision, args
        )
        survivors.append(cand)
    return survivors, extra_dropped


# ---------------------------------------------------------------------------
# OpenRouter session-cost dataset (per-task cost, outside the score)
# ---------------------------------------------------------------------------


SESSION_COST_URL = "https://openrouter.ai/api/v1/datasets/session-cost"
SESSION_COST_HARNESS_SLUG = "hermes-agent"
# The turn range the catalog/CLI surfaces (Task 8).
SESSION_COST_DISPLAY_TURN_RANGE = "10-49-turns"
# The dataset's four turn ranges (brainstorm section 2.6) -- the catalog
# contract's closed set (web/build_site_data.py pins the same tuple).
# Cells naming anything else are dropped: an unknown range is dataset
# drift, and the block must never fail the publish (design section 4.8).
SESSION_COST_TURN_RANGES = ("1-turn", "2-9-turns", "10-49-turns", "50-plus-turns")
SESSION_COST_ATTRIBUTION = "Data © OpenRouter, CC BY 4.0"


def strip_permaslug_date(permaslug: str) -> str:
    """Permaslug without its trailing -YYYYMMDD (undated ids map to themselves)."""
    match = re.match(r"^(.*)-(\d{8})$", permaslug)
    return match.group(1) if match else permaslug


def catalog_pairing_entries(models):
    """Catalog listings projected to the slug/permaslug pairing
    _pair_permaslug_payloads consumes: models-API entries key on id and
    canonical_slug, which the pairing reads as slug/permaslug. Entries
    without an id cannot pair and are dropped."""
    return [
        {"slug": model["id"], "permaslug": model.get("canonical_slug") or ""}
        for model in models or []
        if isinstance(model, dict) and model.get("id")
    ]


def fetch_session_cost(args, models=None) -> dict | None:
    """Per-task session-cost medians from OpenRouter's dataset, or None.

    A separate published dimension (design section 4.8): the median USD
    cost of real sessions through the pinned published harness -- never
    part of the weighted score. One bulk paginated fetch per invocation
    (?app_slug=<harness>&limit=500&offset=k) regardless of the model list;
    pagination stops on a page shorter than the limit or once the next
    offset would exceed the API's 5000 cap. Cells key on dated permaslugs
    (z-ai/glm-5.3-flash-20260826), resolved to exact catalog slugs with
    the same three-pass pairing as build_aa_benchmarks (#22): a blind
    strip with latest-date-wins merges snapshots across listings of one
    model family, the exact conflation #17 removed for the benchmarks.
    ``models`` is the raw catalog listing (id/canonical_slug entries);
    catalog_pairing_entries translates it to the slug/permaslug map the
    pairing consumes. When several dated permaslugs pair with one slug,
    the payloads are turn-range maps, not whole records, so the merge
    unions the maps and each range resolves to the latest date's cell
    (first wins on an equal date); an ambiguous strip with no catalog
    pairing to disambiguate drops rather than guesses.

    Returns None (graceful degradation, design section 4.9) when no key is
    configured (the --openrouter-api-key flag, else OPENROUTER_API_KEY),
    when the fetch fails after fetch_json's retries, when the response
    shape is unexpected, or when no cell survives the pairing -- a
    malformed payload must never crash the run. The merged result is
    cached under "session-cost-v3" with the standard TTL.
    """
    api_key = args.openrouter_api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return None
    if not args.no_cache:
        cached = load_cache("session-cost-v3", args.cache_ttl)
        if isinstance(cached, dict) and isinstance(cached.get("medians"), dict):
            return cached
    cells = []
    # The API caps limit at 500 and offset at 5000 (11 full pages max).
    offset = 0
    while offset <= 5000:
        try:
            payload = fetch_json(
                f"{SESSION_COST_URL}?app_slug={SESSION_COST_HARNESS_SLUG}"
                f"&limit=500&offset={offset}",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=30,
                retries=3,
            )
        except Exception as exc:
            warn(f"session-cost dataset request failed ({exc})")
            return None
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list):
            warn("session-cost dataset returned an unexpected response shape")
            return None
        cells.extend(cell for cell in data if isinstance(cell, dict))
        if len(data) < 500:
            break
        offset += 500
    # Cells group one payload per permaslug before the pairing passes. A
    # cell naming a new turn range is dropped (the catalog validator
    # rejects unknown ranges, and session cost must never fail the
    # publish), as is any non-finite median.
    valid: dict[str, tuple[dict[str, float], str]] = {}
    seen_date: dict[tuple[str, str], str] = {}
    for cell in cells:
        permaslug = cell.get("model_permaslug")
        turn_range = cell.get("turn_range")
        raw = cell.get("median_session_cost_usd")
        if not (
            isinstance(permaslug, str)
            and permaslug
            and isinstance(turn_range, str)
            and turn_range
        ):
            continue
        # Closed contract set: a cell naming a new turn range is dropped,
        # never published (the catalog validator rejects unknown ranges,
        # and session cost must never fail the publish).
        if turn_range not in SESSION_COST_TURN_RANGES:
            continue
        # Non-finite medians would serialize as NaN and poison the catalog.
        if not isinstance(raw, (int, float)) or isinstance(raw, bool):
            continue
        try:
            median = float(raw)  # huge ints raise OverflowError
        except OverflowError:
            continue
        if not math.isfinite(median):
            continue
        bare = strip_permaslug_date(permaslug)
        # strip_permaslug_date removed a trailing -YYYYMMDD exactly when
        # the key was dated; the date is then the final 8 characters.
        date = "" if bare == permaslug else permaslug[-8:]
        # The dataset can republish a (permaslug, range) cell; the latest
        # date wins, first on an equal date.
        key = (permaslug, turn_range)
        if key in seen_date and seen_date[key] >= date:
            continue
        seen_date[key] = date
        if permaslug not in valid:
            valid[permaslug] = ({}, date)
        valid[permaslug][0][turn_range] = median
    # Resolve keys to exact catalog slugs with the same three-pass pairing
    # as build_aa_benchmarks (#22). The payloads are turn-range maps, so
    # same-slug snapshots merge per range instead of replacing wholesale.
    medians = {
        slug: payload
        for slug, (payload, _date) in _pair_permaslug_payloads(
            valid, catalog_pairing_entries(models), merge=True
        ).items()
    }
    if not medians:
        warn("session-cost dataset returned no usable cells")
        return None
    meta = payload.get("meta") if isinstance(payload, dict) else None
    if not isinstance(meta, dict):
        warn("session-cost dataset returned an unexpected response shape")
        return None
    as_of = meta.get("as_of")
    window_days = meta.get("window_days")
    window_end_date = meta.get("window_end_date")
    # The catalog validator demands non-empty as_of/window_end_date strings
    # and a positive integer window_days; anything else is dataset drift.
    # Degrade to None (block omitted, "not available this refresh") rather
    # than emit a block the validator would reject -- session cost must
    # never fail the publish (design section 4.8).
    if not (
        isinstance(as_of, str)
        and as_of
        and isinstance(window_end_date, str)
        and window_end_date
        and isinstance(window_days, int)
        and not isinstance(window_days, bool)
        and window_days > 0
    ):
        warn("session-cost dataset returned malformed meta; omitting the block")
        return None
    result = {
        "attribution": SESSION_COST_ATTRIBUTION,
        "app_slug": SESSION_COST_HARNESS_SLUG,
        "as_of": as_of,
        "window_days": window_days,
        "window_end_date": window_end_date,
        "medians": medians,
    }
    save_cache("session-cost-v3", result)
    return result


# ---------------------------------------------------------------------------
# Published catalog quality (hydration for unkeyed runs)
# ---------------------------------------------------------------------------


# Published catalogs before v6 are retired: v6 is the only shape whose
# `quality` means the best-selectable-depth index.
CATALOG_QUALITY_SCHEMA_VERSION = CATALOG_SCHEMA_VERSION

# One display-only unlabeled AA reference: the intelligence index and AA's
# benchmark cost, both measured without a reasoning-effort label, plus the
# source label and its as_of. The record deliberately has no `effort` key.
CATALOG_UNLABELED_KEYS = ("index", "cost_per_task_usd", "source", "as_of")


def _catalog_index(value) -> float | None:
    """A finite in-range catalog index, or None for poisoned/absent input."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    try:
        index = float(value)  # huge ints raise OverflowError
    except OverflowError:
        return None
    if not math.isfinite(index) or not 0.0 <= index <= 100.0:
        return None
    return index


def _parse_catalog_v6(doc) -> tuple[dict, dict]:
    """Pure parser: a published catalog v6 document -> (by_id, depths_by_id).

    by_id maps id -> {"index", "source", "as_of", "effort"} for the model's
    main quality (the index at its best selectable depth). The
    depth map maps id -> {"default_effort", "depths": {effort: record},
    "unlabeled": record-or-None} for the model's full measured depth list
    and, when AA measured it without an effort label, the display-only
    reference; resolve_quality re-selects from it using the
    best-selectable-depth rules and the unkeyed run's support data.

    Gates (design: failure modes):
    - schema_version must be an int == 6. Earlier catalogs predate the
      best-selectable-depth contract, so they warn
      "published catalog schema vX, expected 6; skipping hydration" and
      yield empty maps. Mixing semantics is never allowed.
    - Any shape surprise -- a non-dict document, a non-int schema_version,
      a non-list models, a row without a usable id, a main quality that is
      not finite/in-range, or malformed depth data -- warns and yields
      empty maps for the WHOLE artifact. Never a partial hydration, never
      a raise: the catalog is machine-generated, so one malformed row
      means this is not the artifact we think it is.
    - Rows with quality None and quality_match None are legitimate
      unmatched models; they are skipped in by_id but their depth block is
      still parsed (a model can have measured depths without a main score).

    Per-depth records use the published shape {effort, index,
    cost_per_task_usd, source, as_of} -- the quality record's own shape
    (`index`, never `quality_index`).
    """
    if not isinstance(doc, dict):
        warn("published catalog returned an unexpected response shape")
        return {}, {}
    schema_version = doc.get("schema_version")
    if type(schema_version) is not int:
        warn("published catalog returned an unexpected response shape")
        return {}, {}
    if schema_version != CATALOG_QUALITY_SCHEMA_VERSION:
        warn(
            f"published catalog schema v{schema_version}, expected "
            f"{CATALOG_QUALITY_SCHEMA_VERSION}; skipping hydration"
        )
        return {}, {}
    models = doc.get("models")
    if not isinstance(models, list):
        warn("published catalog returned an unexpected response shape")
        return {}, {}
    by_id: dict = {}
    depths_by_id: dict = {}
    for entry in models:
        if not isinstance(entry, dict):
            warn("published catalog returned an unexpected response shape")
            return {}, {}
        model_id = entry.get("id")
        if not (isinstance(model_id, str) and model_id):
            warn("published catalog returned an unexpected response shape")
            return {}, {}
        quality = entry.get("quality")
        match = entry.get("quality_match")
        if not (match is None and quality is None):
            if not (isinstance(match, str) and match):
                warn("published catalog returned an unexpected response shape")
                return {}, {}
            index = _catalog_index(quality)
            if index is None:
                warn("published catalog returned an unexpected response shape")
                return {}, {}
            effort = entry.get("quality_effort")
            if effort is not None and not (isinstance(effort, str) and effort):
                warn("published catalog returned an unexpected response shape")
                return {}, {}
            recorded = entry.get("quality_as_of")
            by_id[model_id] = {
                "index": index,
                "source": match,
                "as_of": recorded if isinstance(recorded, str) and recorded else None,
                "effort": effort,
            }
        reasoning = entry.get("reasoning")
        if reasoning is None:
            continue
        if not isinstance(reasoning, dict):
            warn("published catalog returned an unexpected response shape")
            return {}, {}
        default_effort = reasoning.get("default_effort")
        if default_effort is not None and not isinstance(default_effort, str):
            warn("published catalog returned an unexpected response shape")
            return {}, {}
        raw_depths = reasoning.get("depths")
        if raw_depths is None:
            raw_depths = []
        if not isinstance(raw_depths, list):
            warn("published catalog returned an unexpected response shape")
            return {}, {}
        depths: dict = {}
        for depth in raw_depths:
            if not isinstance(depth, dict):
                warn("published catalog returned an unexpected response shape")
                return {}, {}
            effort = depth.get("effort")
            if not (isinstance(effort, str) and effort):
                warn("published catalog returned an unexpected response shape")
                return {}, {}
            index = _catalog_index(depth.get("index"))
            if index is None:
                warn("published catalog returned an unexpected response shape")
                return {}, {}
            cost = depth.get("cost_per_task_usd")
            if cost is not None:
                cost = _finite_cost(cost)
                if cost is None:
                    warn("published catalog returned an unexpected response shape")
                    return {}, {}
            source = depth.get("source")
            if not (isinstance(source, str) and source):
                warn("published catalog returned an unexpected response shape")
                return {}, {}
            as_of = depth.get("as_of")
            if as_of is not None and not isinstance(as_of, str):
                warn("published catalog returned an unexpected response shape")
                return {}, {}
            if effort in depths:
                # Duplicate efforts are ambiguous: fail closed, never guess.
                warn("published catalog returned an unexpected response shape")
                return {}, {}
            depths[effort] = {
                "effort": effort,
                "index": index,
                "cost_per_task_usd": cost,
                "source": source,
                "as_of": as_of if as_of else None,
            }
        unlabeled = reasoning.get("aa_unlabeled")
        if unlabeled is not None:
            if not isinstance(unlabeled, dict):
                warn("published catalog returned an unexpected response shape")
                return {}, {}
            if any(key not in CATALOG_UNLABELED_KEYS for key in unlabeled):
                warn("published catalog returned an unexpected response shape")
                return {}, {}
            if any(key not in unlabeled for key in CATALOG_UNLABELED_KEYS):
                warn("published catalog returned an unexpected response shape")
                return {}, {}
            index = _catalog_index(unlabeled.get("index"))
            if index is None:
                warn("published catalog returned an unexpected response shape")
                return {}, {}
            cost = unlabeled.get("cost_per_task_usd")
            if cost is not None:
                cost = _finite_cost(cost)
                if cost is None:
                    warn("published catalog returned an unexpected response shape")
                    return {}, {}
            source = unlabeled.get("source")
            if not (isinstance(source, str) and source):
                warn("published catalog returned an unexpected response shape")
                return {}, {}
            as_of = unlabeled.get("as_of")
            if as_of is not None and not isinstance(as_of, str):
                warn("published catalog returned an unexpected response shape")
                return {}, {}
            unlabeled = {
                "index": index,
                "cost_per_task_usd": cost,
                "source": source,
                "as_of": as_of if as_of else None,
            }
        depths_by_id[model_id] = {
            "default_effort": default_effort if default_effort else None,
            "depths": depths,
            "unlabeled": unlabeled,
        }
    return by_id, depths_by_id


def catalog_quality_by_id(doc) -> dict:
    """The v6 main-quality map {id: {"index", "source", "as_of", "effort"}}.

    Pure parser; see _parse_catalog_v6 for the gates and the depth map.
    """
    return _parse_catalog_v6(doc)[0]


def _catalog_schedule_entries(doc) -> dict:
    """Pure parser: published catalog v6 schedules -> id -> recorded provenance.

    Spec 6.5: unkeyed runs rebuild the Deals table from the published
    catalog's schedules array, because schedule-only models (sub-1M context,
    non-ZDR) never appear in the models array. Each value is
    {"quality", "quality_effort", "quality_as_of", "quality_source"},
    copied verbatim so the rebuild publishes the keyed run's recorded
    provenance unchanged.

    A non-list schedules block yields an empty map. Individual malformed
    rows (not an object, no usable id, a non-finite/out-of-range quality, or
    a wrong-typed provenance field) are skipped rather than guessed; the
    model then falls through the existing tiers.

    Schema gate: only a v6 catalog contributes schedule entries. Earlier
    schedules predate ``quality_effort``, so hydrating them would mix old
    quality semantics into a best-selectable-depth run.
    """
    if not isinstance(doc, dict):
        return {}
    schema_version = doc.get("schema_version")
    if (
        type(schema_version) is not int
        or schema_version != CATALOG_QUALITY_SCHEMA_VERSION
    ):
        return {}
    schedules = doc.get("schedules")
    if not isinstance(schedules, list):
        return {}
    entries = {}
    for entry in schedules:
        if not isinstance(entry, dict):
            continue
        model_id = entry.get("id")
        if not (isinstance(model_id, str) and model_id):
            continue
        quality = entry.get("quality")
        if quality is not None:
            quality = _catalog_index(quality)
            if quality is None:
                continue
        quality_effort = entry.get("quality_effort")
        if quality_effort is not None and not (
            isinstance(quality_effort, str) and quality_effort
        ):
            continue
        quality_as_of = entry.get("quality_as_of")
        if quality_as_of is not None and not isinstance(quality_as_of, str):
            continue
        quality_source = entry.get("quality_source")
        if quality_source is not None and not isinstance(quality_source, str):
            continue
        entries[model_id] = {
            "quality": quality,
            "quality_effort": quality_effort,
            "quality_as_of": quality_as_of,
            "quality_source": quality_source,
        }
    return entries


def fetch_quality_catalog(args) -> dict | None:
    """Quality records from the published catalog.json, or None.

    The third quality source (design: source 2): the repo's own published
    catalog artifact, built by keyed CI runs and hydrated by unkeyed local
    runs. Never called on keyed runs -- a CI/site build hydrating from the
    published catalog would consume its own output (self-reference loop);
    run() owns that wiring.

    Returns None only on TRANSPORT failure (after fetch_json's retries);
    otherwise the envelope {"by_id": {id: record}, "depths_by_id":
    {id: {"default_effort", "depths": {effort: record}}}, "generated_at":
    str|None, "schedules_by_id": {id: recorded schedule provenance}}. An
    empty by_id AND empty depths_by_id AND empty schedules_by_id means
    fetched-but-unusable (schema gate fail, malformed shape) -- Task 4's
    footer treats None and empty maps identically ("catalog unavailable").

    The envelope is cached under QUALITY_CATALOG_CACHE with the standard
    TTL. A usable result (non-empty by_id, depths_by_id or schedules_by_id)
    is written; so is a v5 catalog's deliberately empty envelope -- that is
    the stable deploy-skew state, and caching it avoids a refetch loop.
    Older/unknown schemas and malformed v6 payloads stay uncached so a
    degraded payload self-heals on the next run. A cached envelope from
    before the v5 depth contract lacks depths_by_id, and one from before
    schedule hydration lacks schedules_by_id; both are refetched rather
    than serving depth/schedule-blind quality. --no-cache neither reads nor
    writes. Staleness: generated_at older than QUALITY_CATALOG_MAX_AGE_DAYS
    warns but uses -- as_of makes the staleness visible downstream (the
    check runs only when rows actually hydrate -- warning about the age of
    a catalog we are not using is noise); an unusable generated_at warns
    that staleness cannot be checked and still uses the rows (their as_of
    then shows None rather than a wrong date).
    """
    if not args.no_cache:
        cached = load_cache(QUALITY_CATALOG_CACHE, args.cache_ttl)
        if (
            isinstance(cached, dict)
            and isinstance(cached.get("by_id"), dict)
            and isinstance(cached.get("depths_by_id"), dict)
            and isinstance(cached.get("schedules_by_id"), dict)
            and cached.get("schema_version") == CATALOG_QUALITY_SCHEMA_VERSION
        ):
            return cached
    try:
        doc = fetch_json(QUALITY_CATALOG_URL, timeout=30, retries=3)
    except Exception as exc:
        warn(f"published catalog request failed ({exc})")
        return None
    by_id, depths_by_id = _parse_catalog_v6(doc)
    schedules_by_id = _catalog_schedule_entries(doc)
    generated_at = doc.get("generated_at") if isinstance(doc, dict) else None
    if not isinstance(generated_at, str):
        generated_at = None
    schema_version = doc.get("schema_version") if isinstance(doc, dict) else None
    result = {
        "by_id": by_id,
        "depths_by_id": depths_by_id,
        "generated_at": generated_at,
        "schedules_by_id": schedules_by_id,
    }
    usable = bool(by_id or depths_by_id or schedules_by_id)
    if usable:
        generated = parse_iso_datetime(generated_at)
        if generated is None:
            warn(
                "published catalog generated_at is missing or unparseable; "
                "cannot check staleness"
            )
        else:
            age_days = (datetime.now(timezone.utc) - generated).total_seconds() / 86400
            if age_days > QUALITY_CATALOG_MAX_AGE_DAYS:
                warn(
                    f"published catalog is stale (generated {generated_at}, "
                    f"{age_days:.0f} days old); using it anyway"
                )
    # The previous major schema (reader gate minus one) is the stable
    # deploy-skew state: its envelope is deliberately empty (no hydration),
    # but caching it prevents a refetch-every-run loop. Older/unknown schemas
    # and malformed current-schema payloads still self-heal by staying uncached.
    if usable or (
        type(schema_version) is int
        and schema_version == CATALOG_QUALITY_SCHEMA_VERSION - 1
    ):
        if not args.no_cache:
            result["schema_version"] = CATALOG_QUALITY_SCHEMA_VERSION
            save_cache(QUALITY_CATALOG_CACHE, result)
    return result


# ---------------------------------------------------------------------------
# OpenRouter catalog
# ---------------------------------------------------------------------------


def fetch_openrouter_models(args):
    if not args.no_cache:
        cached = load_cache("openrouter-models", args.cache_ttl)
        if cached:
            return cached, True
    payload = fetch_json(OPENROUTER_MODELS_URL, timeout=30)
    models = payload.get("data") or []
    if not models:
        raise RuntimeError("OpenRouter API returned no models")
    save_cache("openrouter-models", models)
    return models, False


FRONTEND_DISCOUNTS_CACHE = "openrouter-frontend-discounts"
# v2: AA records are keyed by exact catalog slug and carry as_of (#17);
# v1 cached the conflated date-stripped shape, so the old key is never read.
FRONTEND_AA_CACHE = "openrouter-frontend-aa-v2"
# The frontend listing projected to the reasoning fields (slug, permaslug,
# supports_reasoning, reasoning_config). Cached separately so a fully cached
# run keeps the reasoning fallback instead of degrading it to unknown.
FRONTEND_REASONING_CACHE = "openrouter-frontend-reasoning-v1"


def fetch_openrouter_frontend(args):
    """Derive discounts, ZDR ids, OR-published AA benchmarks and reasoning.

    Two loads: the base frontend URL serves discounts, benchmarks and the
    per-model reasoning config in one response; the per-endpoint ZDR list
    serves ZDR and is skipped entirely under --no-zdr. Each payload caches
    independently under its own key and is cached only when non-empty, so a
    degraded payload self-heals on the next run. A base-URL outage never
    blocks the ZDR fetch, and vice versa. Returns
    (discounts, zdr_ids, aa_by_id, cache_hits, frontend_models) where
    cache_hits names the payloads served from cache ("discounts", "zdr",
    "aa"), and frontend_models is the frontend listing projected to the
    fields build_reasoning_metadata needs (slug, permaslug,
    supports_reasoning, reasoning_config).
    """
    discounts = {}
    aa_by_id = {}
    zdr_ids = set()
    frontend_models = []
    cache_hits = set()

    aa_from_cache = False
    if not args.no_cache:
        cached = load_cache(FRONTEND_DISCOUNTS_CACHE, args.cache_ttl)
        if cached:
            discounts = cached
            cache_hits.add("discounts")
        cached = load_cache(FRONTEND_AA_CACHE, args.cache_ttl)
        if cached:
            aa_by_id = cached
            aa_from_cache = True
            cache_hits.add("aa")
        cached = load_cache(FRONTEND_REASONING_CACHE, args.cache_ttl)
        if cached:
            frontend_models = cached

    if not discounts or not aa_from_cache:
        try:
            payload = fetch_json(OPENROUTER_DISCOUNTS_URL, timeout=30)
        except Exception as exc:
            if not discounts:
                warn(f"could not fetch discount data: {exc}")
            if not aa_from_cache:
                warn(f"could not fetch AA benchmark data: {exc}")
        else:
            data = payload.get("data") if isinstance(payload, dict) else None
            entries = data.get("models") if isinstance(data, dict) else None
            fresh_discounts = {}
            for entry in entries or []:
                if not isinstance(entry, dict):
                    continue
                slug = entry.get("slug") or ""
                if not slug or slug.startswith("~"):
                    continue
                endpoint = entry.get("endpoint") or {}
                raw = (endpoint.get("pricing") or {}).get("discount")
                if not isinstance(raw, (int, float)):
                    continue
                variant = endpoint.get("variant") or ""
                key = slug if variant in ("", "standard") else f"{slug}:{variant}"
                fresh_discounts.setdefault(key, float(raw))
            fresh_aa = build_aa_benchmarks(
                data.get("benchmarks") if isinstance(data, dict) else None,
                entries,
            )
            if fresh_discounts:
                discounts = fresh_discounts
                save_cache(FRONTEND_DISCOUNTS_CACHE, discounts)
                cache_hits.discard("discounts")
            elif not discounts:
                # An intact endpoint always yields hundreds of entries
                # (discount: 0 is still an entry); an empty map means the
                # response shape changed -- never cache that, so the next
                # run recovers on its own.
                warn("no discount entries found; treating discounts as unavailable")
            if not aa_from_cache:
                if fresh_aa:
                    aa_by_id = fresh_aa
                    save_cache(FRONTEND_AA_CACHE, aa_by_id)
                else:
                    warn(
                        "no AA benchmark entries found; treating OpenRouter "
                        "benchmarks as unavailable"
                    )
            fresh_reasoning = _frontend_reasoning_entries(entries)
            if fresh_reasoning:
                frontend_models = fresh_reasoning
                save_cache(FRONTEND_REASONING_CACHE, frontend_models)

    if args.no_zdr:
        return discounts, zdr_ids, aa_by_id, cache_hits, frontend_models
    if not args.no_cache:
        # v3: the id set now comes from the per-endpoint ZDR list; v1/v2
        # sets were derived from the models/find response and must never
        # be read.
        cached = load_cache("openrouter-zdr-v3", args.cache_ttl)
        if cached:
            return (
                discounts,
                set(cached),
                aa_by_id,
                cache_hits | {"zdr"},
                frontend_models,
            )
    try:
        payload = fetch_json(OPENROUTER_ZDR_URL, timeout=30)
    except Exception as exc:
        warn(f"could not fetch ZDR data: {exc}")
        return discounts, zdr_ids, aa_by_id, cache_hits, frontend_models
    # One entry per ZDR endpoint, keyed by the exact model id (variants
    # such as :free are listed individually). An id is ZDR iff it has at
    # least one ZDR endpoint. The models/find ?zdr=true filter cannot be
    # used: it is model-level, so it lists non-ZDR variants (NVIDIA's
    # :free endpoints) too. No "~" router aliases occur in this list, so
    # unlike the find parsing there is no "~" skip; any stray id is
    # harmless because build_candidates only matches real model ids.
    # An endpoint counts only when its status is exactly the integer 0.
    # OpenRouter's OpenAPI schema (EndpointStatus) lists 0, -1, -2, -3, -5
    # and -10 without describing them, so the meaning of the non-zero values
    # is unverified; treat anything other than 0, including a missing
    # status, as not serving (fail closed).
    entries = payload.get("data") if isinstance(payload, dict) else None
    ids = set()
    for entry in entries if isinstance(entries, list) else []:
        if not isinstance(entry, dict):
            continue
        status = entry.get("status")
        if type(status) is not int or status != 0:
            continue
        model_id = entry.get("model_id")
        if isinstance(model_id, str) and model_id:
            ids.add(model_id)
    if not ids:
        warn("no ZDR entries found; treating ZDR data as unavailable")
        return discounts, zdr_ids, aa_by_id, cache_hits, frontend_models
    save_cache("openrouter-zdr-v3", sorted(ids))
    return discounts, ids, aa_by_id, cache_hits, frontend_models


def parse_price(value) -> float | None:
    # bools are not prices; a huge JSON int (json.loads yields an exact
    # Python int) overflows float() -- both fail soft to None.
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return None


def parse_non_negative_price(value) -> float | None:
    """A finite, non-negative price; malformed, negative or NaN -> None.

    Cache rates use this fail-soft gate: a bad cache rate is ignored, never
    allowed to drop a model or poison a published artifact.
    """
    price = parse_price(value)
    if price is None or not math.isfinite(price) or price < 0:
        return None
    return price


# ---------------------------------------------------------------------------
# OpenRouter tiered pricing (pricing.overrides)
# ---------------------------------------------------------------------------

# Keys of a pricing.overrides entry (OpenRouter docs, verified 2026-10-09).
# An entry carrying any other key is skipped whole (docs forward-compat rule).
TIER_CONDITION_KEYS = (
    "min_prompt_tokens",
    "utc_start",
    "utc_end",
    "utc_days",
)
TIER_PRICE_KEYS = (
    "prompt",
    "completion",
    "request",
    "image",
    "image_output",
    "audio",
    "audio_output",
    "web_search",
    "internal_reasoning",
    "input_cache_read",
    "input_cache_write",
    "input_cache_write_1h",
    "input_audio_cache",
)
WEEKDAYS = (
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
)
_WEEK_MINUTES = 7 * 24 * 60


def _integral(value) -> int | None:
    """Finite, non-bool, integral number -> int; else None (fail-soft)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    # Magnitude guard first: math.isfinite raises OverflowError on huge ints
    # (json.loads of a 400-digit literal); no real token count or clock
    # comes anywhere near 1e15.
    if abs(value) > 10**15:
        return None
    if not math.isfinite(value) or value != int(value):
        return None
    return int(value)


def _hhmm(value) -> int | None:
    """Integer HHMM clock in 0..2359 with minutes < 60; else None."""
    clock = _integral(value)
    if clock is None or not 0 <= clock <= 2359 or clock % 100 > 59:
        return None
    return clock


def _hhmm_minutes(clock: int) -> int:
    return (clock // 100) * 60 + clock % 100


def _window_minutes(start: int, end: int) -> int:
    """Minutes a HHMM window covers per day (wrap-aware; whole day when equal)."""
    span = _hhmm_minutes(end) - _hhmm_minutes(start)
    if span == 0:
        return 24 * 60
    return span % (24 * 60)


def _window_span(start: int, end: int):
    """The per-day minute values a window matches (wrap-aware)."""
    start_min = _hhmm_minutes(start)
    end_min = _hhmm_minutes(end)
    if start_min == end_min:
        return range(24 * 60)
    if start_min < end_min:
        return range(start_min, end_min)
    return list(range(start_min, 24 * 60)) + list(range(0, end_min))


def _coverage(days_count: int, minutes: int) -> float:
    """Fraction of the 7-day week a window covers, 4 decimals."""
    return round(days_count * minutes / _WEEK_MINUTES, 4)


def _entry_prices(entry):
    """Parse an override entry's prices (spec rule 5).

    Returns {"prompt", "completion", "input_cache_read",
    "input_cache_write"} where None means the key is absent (inheritance).
    A malformed prompt/completion skips the entry whole (returns None); a
    malformed cache rate is ignored on its own, never skips the entry.
    """
    values = {}
    for key in ("prompt", "completion"):
        if key not in entry:
            values[key] = None
            continue
        value = parse_price(entry[key])
        if value is None or not math.isfinite(value) or value < 0:
            return None
        values[key] = value
    for key in ("input_cache_read", "input_cache_write"):
        values[key] = parse_non_negative_price(entry.get(key))
    return values


def _schedule_windows(window_entries):
    """Normalize validated time-window entries (spec rules 6, 9, 10).

    Returns the window list (API order, days-only windows normalized to 0/0,
    utc_days deduplicated in week order, per-window coverage), or None when
    the windows do not tile the 7-day week exactly (gaps or overlaps), or a
    window is missing prompt/completion (rule 7: no top-level inheritance --
    the top level is fetch-time dependent). A window's cache rates stay None
    when it does not set them; effective_pricing resolves those against the
    frozen peak base after the peak window is known.
    """
    windows = []
    covered = set()
    matched = 0
    for entry in window_entries:
        start, end = entry["start"], entry["end"]
        days = entry["days"]  # None, or a list of weekday indices
        if start is None:  # days-only window: the whole UTC day
            start, end = 0, 0
        price_in, price_out = entry["price_in"], entry["price_out"]
        if price_in is None or price_out is None:
            return None
        day_indices = list(range(len(WEEKDAYS))) if days is None else sorted(set(days))
        span = _window_span(start, end)
        for day in day_indices:
            for minute in span:
                covered.add((day, minute))
            matched += len(span)
        windows.append(
            {
                "utc_days": None
                if days is None
                else [WEEKDAYS[i] for i in day_indices],
                "utc_start": start,
                "utc_end": end,
                "coverage": _coverage(len(day_indices), len(span)),
                "price_in": price_in,
                "price_out": price_out,
                "input_cache_read": entry["input_cache_read"],
                "input_cache_write": entry["input_cache_write"],
            }
        )
    if matched != _WEEK_MINUTES or len(covered) != _WEEK_MINUTES:
        return None  # gaps or overlaps: the schedule does not tile the week
    return windows


def effective_pricing(pricing, prompt_tokens, input_share):
    """Resolve pricing.overrides into one effective price (spec section 5.1).

    Prices are USD per token. Returns a dict with the effective and base
    prices (per token), the applied tier threshold, the cumulative tier list,
    the normalized schedule (or None), whether the schedule is invalid, the
    rounded maximum off-peak discount, and the peak window's blended price.
    """
    source = pricing if isinstance(pricing, dict) else {}
    top_in = parse_price(source.get("prompt"))
    top_out = parse_price(source.get("completion"))
    top_cache_read = parse_non_negative_price(source.get("input_cache_read"))
    top_cache_write = parse_non_negative_price(source.get("input_cache_write"))
    base_in, base_out = top_in, top_out
    base_cache_read, base_cache_write = top_cache_read, top_cache_write
    current_cache_read, current_cache_write = base_cache_read, base_cache_write
    overrides = source.get("overrides")

    tier_entries = []  # (threshold, prices dict|None), API order
    window_entries = []  # validated time-window pre-windows, API order
    time_touched = False
    schedule_broken = False

    if isinstance(overrides, list):
        for entry in overrides:
            if not isinstance(entry, dict):
                continue
            time_keys = [
                key for key in ("utc_start", "utc_end", "utc_days") if key in entry
            ]
            if any(
                key not in TIER_CONDITION_KEYS and key not in TIER_PRICE_KEYS
                for key in entry
            ):
                if time_keys:
                    schedule_broken = True  # skipped time window: tiling gap
                continue
            has_threshold = "min_prompt_tokens" in entry
            threshold = _integral(entry["min_prompt_tokens"]) if has_threshold else None
            if has_threshold and (threshold is None or threshold < 0):
                if time_keys:
                    schedule_broken = True
                continue
            if ("utc_start" in entry) != ("utc_end" in entry):
                schedule_broken = True  # exactly one of the pair
                continue
            start = end = None
            if "utc_start" in entry:
                start = _hhmm(entry["utc_start"])
                end = _hhmm(entry["utc_end"])
                if start is None or end is None:
                    schedule_broken = True
                    continue
            days = None
            if "utc_days" in entry:
                names = entry["utc_days"]
                if (
                    not isinstance(names, list)
                    or not names
                    or any(day not in WEEKDAYS for day in names)
                ):
                    schedule_broken = True
                    continue
                days = sorted({WEEKDAYS.index(day) for day in names})
            prices = _entry_prices(entry)
            if prices is None:
                if time_keys:
                    schedule_broken = True
                continue
            if time_keys:
                time_touched = True
                if has_threshold:
                    # combined token+time entry: invalid time window (rule 9)
                    schedule_broken = True
                    continue
                window_entries.append(
                    {
                        "start": start,
                        "end": end,
                        "days": days,
                        "price_in": prices["prompt"],
                        "price_out": prices["completion"],
                        "input_cache_read": prices["input_cache_read"],
                        "input_cache_write": prices["input_cache_write"],
                    }
                )
            elif has_threshold:
                tier_entries.append((threshold, prices))
            # else: conditionless entry (price keys only) -- ignored (plan
            # Review Focus; none live, and honoring an unconditional price
            # swap would be dangerous)

    schedule = None
    peak_blended = None
    max_discount = None
    schedule_error = False

    if time_touched or schedule_broken:
        if schedule_broken:
            schedule_error = True
        else:
            schedule = _schedule_windows(window_entries)
            if schedule is None:
                schedule_error = True
                schedule = None
    if schedule is not None and not schedule_error:
        for window in schedule:
            window["blended"] = (
                input_share * window["price_in"]
                + (1 - input_share) * window["price_out"]
            )
        peak = max(schedule, key=lambda window: window["blended"])
        peak_blended = peak["blended"]  # ties: first in API order
        # Deterministic base (rule 7): freeze at the peak window's prices.
        base_in, base_out = peak["price_in"], peak["price_out"]
        base_cache_read = (
            peak["input_cache_read"]
            if peak["input_cache_read"] is not None
            else top_cache_read
        )
        base_cache_write = (
            peak["input_cache_write"]
            if peak["input_cache_write"] is not None
            else top_cache_write
        )
        # A window that sets no cache rate inherits the frozen base, not the
        # fetch-time top-level value.
        for window in schedule:
            if window["input_cache_read"] is None:
                window["input_cache_read"] = base_cache_read
            if window["input_cache_write"] is None:
                window["input_cache_write"] = base_cache_write
        current_cache_read, current_cache_write = base_cache_read, base_cache_write
        if peak_blended == 0:
            max_discount = 0.0
        else:
            max_discount = round(
                max(1 - window["blended"] / peak_blended for window in schedule), 4
            )

    # Token tiers (rule 8) over the (possibly frozen) base: later entries win
    # per key in array order; a threshold equal to prompt_tokens never applies.
    applied = None
    current_in, current_out = base_in, base_out
    by_threshold = {}
    for threshold, prices in tier_entries:
        tier_in, tier_out = prices["prompt"], prices["completion"]
        tier_cache_read = prices["input_cache_read"]
        tier_cache_write = prices["input_cache_write"]
        if prompt_tokens > threshold:
            if tier_in is not None:
                current_in = tier_in
            if tier_out is not None:
                current_out = tier_out
            if tier_cache_read is not None:
                current_cache_read = tier_cache_read
            if tier_cache_write is not None:
                current_cache_write = tier_cache_write
            applied = threshold
        slot = by_threshold.setdefault(
            threshold,
            {
                "price_in": None,
                "price_out": None,
                "cache_read": None,
                "cache_write": None,
            },
        )
        if tier_in is not None:
            slot["price_in"] = tier_in
        if tier_out is not None:
            slot["price_out"] = tier_out
        if tier_cache_read is not None:
            slot["cache_read"] = tier_cache_read
        if tier_cache_write is not None:
            slot["cache_write"] = tier_cache_write
    tiers = []
    running_in, running_out = base_in, base_out
    running_cache_read, running_cache_write = base_cache_read, base_cache_write
    for threshold in sorted(by_threshold):
        slot = by_threshold[threshold]
        if slot["price_in"] is not None:
            running_in = slot["price_in"]
        if slot["price_out"] is not None:
            running_out = slot["price_out"]
        if slot["cache_read"] is not None:
            running_cache_read = slot["cache_read"]
        if slot["cache_write"] is not None:
            running_cache_write = slot["cache_write"]
        tiers.append(
            {
                "min_prompt_tokens": threshold,
                "price_in": running_in,
                "price_out": running_out,
                "input_cache_read": running_cache_read,
                "input_cache_write": running_cache_write,
            }
        )

    return {
        "price_in": current_in,
        "price_out": current_out,
        # top_*: the RAW OpenRouter top-level prices (spec rule 1) -- the
        # validity gate checks these even for windowed models, whose base_*
        # fields are the frozen peak window (rule 7).
        "top_price_in": top_in,
        "top_price_out": top_out,
        "base_price_in": base_in,
        "base_price_out": base_out,
        "price_cache_read": current_cache_read,
        "price_cache_write": current_cache_write,
        "base_cache_read": base_cache_read,
        "base_cache_write": base_cache_write,
        "tier_prompt_tokens": applied,
        "tiers": tiers,
        "schedule": schedule,
        "schedule_error": schedule_error,
        "max_discount": max_discount,
        "peak_blended": peak_blended,
    }


def fmt_tier_note(threshold):
    """>100k-style marker for the applied token tier; None passes through."""
    if threshold is None:
        return None
    value = int(threshold)
    if value >= 1000:
        return f">{value // 1000}k"
    return f">{value}"


def fmt_sched_note(max_discount):
    """-38%-style marker for a schedule's max off-peak discount."""
    if max_discount is None:
        return None
    formatted = f"{max_discount:.0%}"
    if formatted == "0%":
        return "SCHED"
    return f"-{formatted}"


def fmt_sched_detail(schedule, peak_blended):
    """Human-readable off-peak schedule summary (spec section 5.6 grammar).

    Off-peak windows are those whose blended price is strictly below the
    peak's; schedule windows carry their parser-computed "blended" price.
    """
    if not schedule or peak_blended is None:
        return None
    offpeak = [window for window in schedule if window["blended"] < peak_blended]
    if not offpeak:
        return None

    def _clock(clock):
        return f"{clock // 100:02d}:{clock % 100:02d}"

    groups = {}
    for window in offpeak:
        days = window["utc_days"]
        # All seven weekdays listed explicitly is the same day set as
        # utc_days absent: same group, same "daily" label (spec 5.6).
        if days is None or set(days) == set(WEEKDAYS):
            key = (0, -1, ())
            label = "daily"
        else:
            indices = [WEEKDAYS.index(day) for day in days]
            key = (1, min(indices), tuple(days))
            exact = {WEEKDAYS[i] for i in range(5)}
            if set(days) == exact:
                label = "weekdays"
            elif set(days) == {"saturday", "sunday"}:
                label = "weekends"
            else:
                label = ",".join(day[:3] for day in days)  # week order
        group = groups.setdefault(key, {"label": label, "windows": []})
        group["windows"].append(window)
    parts = []
    for key in sorted(groups):
        group = groups[key]
        bits = []
        for window in sorted(group["windows"], key=lambda w: w["utc_start"]):
            if window["utc_start"] == window["utc_end"]:
                bits.append("all day")
            else:
                bits.append(
                    f"{_clock(window['utc_start'])}-{_clock(window['utc_end'])}"
                )
        parts.append(f"{group['label']} {', '.join(bits)}")
    return "; ".join(parts) + " UTC"


def coerce_int(value, default: int = 0) -> int:
    """Best-effort int coercion for external fields that may be str/float/None."""
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def parse_iso_datetime(value):
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


# ---------------------------------------------------------------------------
# Artificial Analysis intelligence index
# ---------------------------------------------------------------------------

_AA_DISCRETE_EFFORTS = frozenset(("max", "xhigh", "high", "medium", "low", "minimal"))
_AA_TRAILING_PAREN_RE = re.compile(r"\s*\(([^()]*)\)\s*$")
_AA_SLUG_EFFORT_RE = re.compile(
    r"-(max|xhigh|high|medium|low|minimal|non-reasoning|reasoning)$", re.IGNORECASE
)
_AA_SLUG_EFFORT_LABELS = {
    "max": "max",
    "xhigh": "xhigh",
    "high": "high",
    "medium": "medium",
    "low": "low",
    "minimal": "minimal",
    "non-reasoning": "none",
    "reasoning": "reasoning",
}
# Shapes (normalized parenthetical text, or the no-label case) already
# warned about. Spec §6.1: one warning per unknown row shape, not per row.
_AA_WARNED_UNKNOWN_SHAPES: set[str] = set()


def _classify_aa_effort_tokens(raw: str) -> tuple[str | None, bool]:
    """Classify one name parenthetical's comma-separated tokens.

    "Default Fallback" is a qualifier, not a label. A parenthetical that
    names two different states -- two discrete efforts, or non-reasoning
    plus an effort -- is ambiguous: no effort, never guessed.
    """
    labels = set()
    for token in (raw or "").split(","):
        token = token.strip().lower()
        if not token or token == "default fallback":
            continue
        if token in _AA_DISCRETE_EFFORTS:
            labels.add(token)
        elif token == "non-reasoning":
            labels.add("none")
        elif token == "reasoning":
            labels.add("reasoning")
    if len(labels) > 1:
        return None, True
    if labels:
        return labels.pop(), False
    return None, False


def _warn_unknown_aa_shape(parenthetical: str | None) -> None:
    """Warn once per distinct unrecognized effort shape (spec §6.1)."""
    if parenthetical is not None:
        detail = f"parenthetical {parenthetical!r}"
        shape = "parenthetical:" + norm_key(parenthetical)
    else:
        detail = "no effort label"
        shape = "no-effort-label"
    if shape in _AA_WARNED_UNKNOWN_SHAPES:
        return
    _AA_WARNED_UNKNOWN_SHAPES.add(shape)
    warn(f"AA row has an unrecognized effort shape ({detail}); row is display-only")


def parse_aa_effort(name: str, slug: str) -> dict:
    """Split an AA row's name/slug into its base model and effort label.

    The name's trailing parenthetical is authoritative; when it names no
    effort (and is not ambiguous) the slug suffix is the fallback. Returns
    base_name/base_slug with the label stripped, effort (None when unknown
    or ambiguous), and ambiguous=True for contradictory labels such as
    "(Non-reasoning, High)" or "(High, Max)". Ambiguous and unknown rows
    are display-only: callers must never guess an effort for them.
    """
    name = name if isinstance(name, str) else ""
    slug = slug if isinstance(slug, str) else ""

    name_match = _AA_TRAILING_PAREN_RE.search(name)
    if name_match:
        base_name = name[: name_match.start()].strip()
        effort, ambiguous = _classify_aa_effort_tokens(name_match.group(1))
    else:
        base_name = name.strip()
        effort, ambiguous = None, False

    slug_match = _AA_SLUG_EFFORT_RE.search(slug)
    if slug_match:
        base_slug = slug[: slug_match.start()]
        slug_effort = _AA_SLUG_EFFORT_LABELS[slug_match.group(1).lower()]
    else:
        base_slug = slug
        slug_effort = None

    if effort is None and not ambiguous:
        effort = slug_effort

    if effort is None and not ambiguous:
        _warn_unknown_aa_shape(name_match.group(1) if name_match is not None else None)

    return {
        "base_name": base_name,
        "base_slug": base_slug,
        "effort": effort,
        "ambiguous": ambiguous,
    }


def _finite_cost(value) -> float | None:
    """A finite float cost, or None for absent/non-numeric/non-finite input."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    try:
        value = float(value)  # huge ints raise OverflowError
    except OverflowError:
        return None
    return value if math.isfinite(value) else None


def _aa_cost_per_task(item: dict) -> float | None:
    """AA's measured cost per benchmark task, or None when not reported."""
    block = item.get("artificial_analysis_intelligence_index_cost")
    if not isinstance(block, dict):
        return None
    cost_per_task = block.get("cost_per_task")
    if not isinstance(cost_per_task, dict):
        return None
    return _finite_cost(cost_per_task.get("total_cost"))


def aa_api_entries(api_key: str) -> tuple[list, float | int | None]:
    """Intelligence entries from AA's V2 language models endpoint.

    Documented envelope: {tier, intelligence_index_version, pagination,
    data[]} with snake_case fields; the index lives at
    data[].evaluations.artificial_analysis_intelligence_index and the
    measured benchmark cost at
    data[].artificial_analysis_intelligence_index_cost.cost_per_task.total_cost.
    AA returns one row per measured reasoning-effort variant; the effort is
    parsed from the row's name/slug by parse_aa_effort. Items without a
    usable slug/name or without a finite numeric index are skipped (nulls
    mean "not measured"). Rows are deduped by their AA `id` (fallback
    (slug, name)), never by slug alone: variants often share a slug.
    Pagination follows the documented has_more/total_pages contract with a
    25-page hard cap; a failure past page 1 is best-effort (warn, keep
    earlier pages), a page-1 failure propagates.

    Returns (entries, version): version is the envelope's
    intelligence_index_version (number-or-null) -- provenance for the whole
    fetch. Every surviving entry carries `as_of`, one ISO fetch timestamp
    shared by the whole fetch, so the cache and any downstream record can
    date the data even after the envelope's version is lost.
    """
    entries = {}
    version = None
    as_of = datetime.now(timezone.utc).isoformat()
    page = 1
    while page <= 25:  # hard cap: halt after processing page 25
        try:
            payload = fetch_json(
                f"{AA_API_URL}?page={page}",
                headers={"x-api-key": api_key},
                timeout=30,
            )
        except Exception as exc:
            if page == 1:
                raise
            warn(f"AA API page {page} failed ({exc}); using entries collected so far")
            break
        data = payload.get("data") if isinstance(payload, dict) else None
        # The envelope's version is one provenance fact for the whole
        # fetch; the last page's value wins (they are identical in practice).
        raw_version = (
            payload.get("intelligence_index_version")
            if isinstance(payload, dict)
            else None
        )
        if isinstance(raw_version, (int, float)) and not isinstance(raw_version, bool):
            version = raw_version
        for item in data if isinstance(data, list) else []:
            if not isinstance(item, dict):
                continue
            slug = item.get("slug")
            name = item.get("name")
            key = (
                slug
                if isinstance(slug, str) and slug
                else (name if isinstance(name, str) and name else None)
            )
            if not key:
                continue
            evals = item.get("evaluations")
            raw = (
                evals.get("artificial_analysis_intelligence_index")
                if isinstance(evals, dict)
                else None
            )
            if not isinstance(raw, (int, float)) or isinstance(raw, bool):
                continue
            try:
                index = float(raw)  # huge ints raise OverflowError
            except OverflowError:
                continue
            if not math.isfinite(index):
                continue
            entry_id = item.get("id")
            entry_id = entry_id if isinstance(entry_id, str) and entry_id else None
            slug = slug if isinstance(slug, str) and slug else None
            raw_name = name if isinstance(name, str) and name else ""
            parsed = parse_aa_effort(raw_name, slug or "")
            name = raw_name or key
            dedupe_key = entry_id if entry_id else (slug, name)
            if dedupe_key in entries:
                continue
            creator = None
            raw_creator = item.get("model_creator")
            if isinstance(raw_creator, dict):
                raw_creator_name = raw_creator.get("name")
                if isinstance(raw_creator_name, str) and raw_creator_name:
                    creator = raw_creator_name
            entries[dedupe_key] = {
                "key": key,
                "name": name,
                "index": index,
                # One fetch timestamp for the whole fetch: every entry
                # from this run shares it, so the cache payload and any
                # downstream record can date the data exactly.
                "as_of": as_of,
                "id": entry_id,
                "slug": slug,
                "base": parsed["base_slug"] or parsed["base_name"],
                "effort": parsed["effort"],
                "creator": creator,
                "cost_per_task": _aa_cost_per_task(item),
            }
        pag = payload.get("pagination") if isinstance(payload, dict) else None
        if not isinstance(pag, dict):
            break  # undocumented shape: single-page degrade
        if pag.get("has_more") is not True:
            break
        total_pages = pag.get("total_pages")
        if isinstance(total_pages, int) and page >= total_pages:
            break
        page += 1
    return list(entries.values()), version


def fetch_aa_entries(args):
    """AA intelligence entries from the keyed AA API v2, or nothing.

    Returns (entries, source, cached, index_version). With a key, an API
    failure or an empty result warns and yields ([], None, False, None).
    Without a key this is silent and yields ([], None, False, None):
    OpenRouter's benchmarks cover what they cover and the catalog records
    the absence (sources.aa).

    `index_version` is the envelope's intelligence_index_version (number
    or null) -- provenance for the whole fetch. Entries carry `as_of`, one
    ISO fetch timestamp for the whole fetch; a cache hit reuses the
    stamped payload verbatim so cached rows stay dated exactly like fresh
    ones.
    """
    if not args.no_cache:
        # v4: payloads cached under earlier keys predate the per-effort
        # fields (id, base, effort, creator, cost_per_task) and must never
        # be read. Only AA API entries are read; build_catalog rejects any
        # other source as unknown.
        cached = load_cache(AA_ENTRIES_CACHE, args.cache_ttl)
        if isinstance(cached, dict) and cached.get("source") == "AA API v2":
            return (
                cached.get("entries", []),
                cached.get("source"),
                True,
                cached.get("version"),
            )
    api_key = args.aa_api_key or os.environ.get("AA_API_KEY")
    if not api_key:
        return [], None, False, None
    try:
        entries, version = aa_api_entries(api_key)
    except Exception as exc:
        warn(f"AA API request failed ({exc})")
        return [], None, False, None
    if not entries:
        warn("AA API returned no intelligence scores")
        return [], None, False, None
    save_cache(
        AA_ENTRIES_CACHE,
        {"entries": entries, "source": "AA API v2", "version": version},
    )
    return entries, "AA API v2", False, version


# ---------------------------------------------------------------------------
# Matching AA entries to OpenRouter model ids
# ---------------------------------------------------------------------------


def dedupe_aa_entries_by_key(entries):
    """Restore the legacy lookup's one-entry-per-key contract.

    The depth path keeps every effort variant (aa_api_entries dedupes by AA
    id), but build_aa_lookup expects the old slug-keyed selection: for each
    `key`, the first occurrence in list order wins. This is the seam between
    the per-effort ingestion and the unchanged legacy matcher.
    """
    unique = []
    seen = set()
    for entry in entries:
        key = entry.get("key")
        if key:
            if key in seen:
                continue
            seen.add(key)
        unique.append(entry)
    return unique


def build_aa_lookup(entries):
    """Index the AA entries into exact and fuzzy lookup maps.

    Values are (index, as_of) pairs, not bare floats: the as_of travels
    with the match so resolve_quality can date every AA-sourced record
    without re-scanning the raw entry list. Entries without a usable key
    or a finite numeric index are skipped.
    """
    exact = {}
    fuzzy = []
    for entry in entries:
        key = entry.get("key") or ""
        name = entry.get("name") or key
        raw_index = entry.get("index")
        if not isinstance(raw_index, (int, float)) or isinstance(raw_index, bool):
            continue
        try:
            index = float(raw_index)  # huge ints raise OverflowError
        except OverflowError:
            continue
        # Non-finite indexes would serialize as NaN and poison quality.
        if not math.isfinite(index):
            continue
        as_of = entry.get("as_of") if isinstance(entry.get("as_of"), str) else None

        def put(tier, raw):
            normalized = norm_key(raw)
            if normalized:
                exact.setdefault((tier, normalized), (index, as_of))

        if "/" in key:
            put(0, key)
            put(1, key.rsplit("/", 1)[-1])
        else:
            put(1, key)
        put(2, name)
        put(3, PAREN_RE.sub(" ", name))
        tokens = set(norm_key(key).split()) | set(norm_key(name).split())
        if tokens:
            fuzzy.append((tokens, (index, as_of)))
    return exact, fuzzy


def build_aa_depth_lookup(entries):
    """Index AA entries as base key -> effort -> depth record.

    Each record is normalized for downstream publication:
    {"effort", "index", "cost_per_task_usd", "source": "aa", "as_of"}.
    Rows whose effort is absent or ambiguous are display-only and skipped.
    Duplicate (base key, effort) records from different AA rows are stored
    as None (ambiguous: matched as absent), never guessed. Base keys come
    from the name and slug with the effort label stripped, both
    unqualified and creator-qualified ("<creator_norm>:<base_key>"), where
    creator_norm is normalize_provider(model_creator.name).
    """
    lookup = {}
    seen = set()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        entry_id = entry.get("id")
        dedupe_key = (
            entry_id
            if isinstance(entry_id, str) and entry_id
            else (entry.get("slug"), entry.get("name"))
        )
        if dedupe_key in seen:
            continue
        parsed = parse_aa_effort(entry.get("name") or "", entry.get("slug") or "")
        effort = parsed["effort"]
        if not effort or parsed["ambiguous"]:
            continue
        raw_index = entry.get("index")
        if not isinstance(raw_index, (int, float)) or isinstance(raw_index, bool):
            continue
        try:
            index = float(raw_index)  # huge ints raise OverflowError
        except OverflowError:
            continue
        if not math.isfinite(index):
            continue
        seen.add(dedupe_key)
        record = {
            "effort": effort,
            "index": index,
            "cost_per_task_usd": _finite_cost(entry.get("cost_per_task")),
            "source": "aa",
            "as_of": (
                entry.get("as_of") if isinstance(entry.get("as_of"), str) else None
            ),
        }
        base_keys = []
        for base in (parsed["base_name"], parsed["base_slug"]):
            normalized = norm_key(base)
            if normalized and normalized not in base_keys:
                base_keys.append(normalized)
        creator = entry.get("creator")
        if not isinstance(creator, str) or not creator:
            raw_creator = entry.get("model_creator")
            if isinstance(raw_creator, dict) and isinstance(
                raw_creator.get("name"), str
            ):
                creator = raw_creator["name"]
        creator_norm = normalize_provider(creator) if isinstance(creator, str) else ""
        keys = list(base_keys)
        if creator_norm:
            keys.extend(f"{creator_norm}:{base_key}" for base_key in base_keys)
        for base_key in keys:
            efforts = lookup.setdefault(base_key, {})
            if effort in efforts:
                efforts[effort] = None
            else:
                efforts[effort] = record
    return lookup


def build_aa_unlabeled_lookup(entries):
    """Index display-only AA rows that carry no reasoning-effort label.

    AA sometimes ranks a model once, with no effort parenthetical and no
    effort-suffixed slug ("GLM-5.3-Flash"). Such a row cannot fill QUAL:
    the index is not tied to a selectable effort, so it may not be the
    quality the user gets. It is still useful as a reference, so it is
    indexed here, separately from build_aa_depth_lookup's effort-keyed
    groups (which skip these rows entirely).

    Each record is {"index", "cost_per_task_usd", "source": "aa", "as_of"}.
    Rows with an effort label, an ambiguous label, a non-finite index or no
    usable key are skipped. Duplicate base keys are stored as None
    (ambiguous: never guessed), mirroring the depth lookup. Base keys come
    from the name and slug with the effort label stripped, both
    unqualified and creator-qualified ("<creator_norm>:<base_key>").
    """
    lookup = {}
    seen = set()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        entry_id = entry.get("id")
        dedupe_key = (
            entry_id
            if isinstance(entry_id, str) and entry_id
            else (entry.get("slug"), entry.get("name"))
        )
        if dedupe_key in seen:
            continue
        parsed = parse_aa_effort(entry.get("name") or "", entry.get("slug") or "")
        if parsed["effort"] or parsed["ambiguous"]:
            continue
        raw_index = entry.get("index")
        if not isinstance(raw_index, (int, float)) or isinstance(raw_index, bool):
            continue
        try:
            index = float(raw_index)  # huge ints raise OverflowError
        except OverflowError:
            continue
        if not math.isfinite(index):
            continue
        seen.add(dedupe_key)
        record = {
            "index": index,
            "cost_per_task_usd": _finite_cost(entry.get("cost_per_task")),
            "source": "aa",
            "as_of": (
                entry.get("as_of") if isinstance(entry.get("as_of"), str) else None
            ),
        }
        base_keys = []
        for base in (parsed["base_name"], parsed["base_slug"]):
            normalized = norm_key(base)
            if normalized and normalized not in base_keys:
                base_keys.append(normalized)
        creator = entry.get("creator")
        if not isinstance(creator, str) or not creator:
            raw_creator = entry.get("model_creator")
            if isinstance(raw_creator, dict) and isinstance(
                raw_creator.get("name"), str
            ):
                creator = raw_creator["name"]
        creator_norm = normalize_provider(creator) if isinstance(creator, str) else ""
        keys = list(base_keys)
        if creator_norm:
            keys.extend(f"{creator_norm}:{base_key}" for base_key in base_keys)
        for base_key in keys:
            if base_key in lookup:
                lookup[base_key] = None
            else:
                lookup[base_key] = record
    return lookup


def _depth_group(model, lookup):
    """The candidate's matching lookup value, or None.

    Used for both effort-keyed base groups (effort -> record) and the
    unlabeled reference map (base -> record): the tier walk is identical.
    Exact tiers only, in the legacy order: full id, base id, display name,
    display name sans parentheticals. At each tier the creator-qualified
    key (the OpenRouter author prefix normalized with normalize_provider)
    is tried before the unqualified key. The first tier with a present key
    wins and terminates the walk; a tier whose group lacks the requested
    effort yields nothing rather than falling through to a lower tier. The
    walk terminates on a present key even when the stored value is None:
    ambiguous records (a None (base, effort) record from
    build_aa_depth_lookup, a None base value from
    build_aa_unlabeled_lookup) match as a resolved-but-empty tier, so the
    caller sees no record and no lower tier is consulted. Exact-key lookup
    means one tier cannot otherwise resolve to multiple base groups. The
    fuzzy pass is never used here -- it is the #17 conflation bug.
    """
    if not isinstance(lookup, dict):
        return None
    model_id = model.get("id") or ""
    name = model.get("name") or ""
    display = PROVIDER_PREFIX_RE.sub("", name).strip()
    without_variant = model_id.split(":", 1)[0]
    _, _, base = without_variant.partition("/")
    creator = normalize_provider(without_variant.partition("/")[0])
    for raw in (model_id, base, display, PAREN_RE.sub(" ", display)):
        normalized = norm_key(raw)
        if not normalized:
            continue
        if creator:
            qualified_key = f"{creator}:{normalized}"
            if qualified_key in lookup:
                return lookup[qualified_key]
        if normalized in lookup:
            return lookup[normalized]
    return None


def match_depth_quality(model, lookup, effort):
    """The depth record at `effort` for the candidate, or None.

    Strict selection: a matched base group with no record at this exact
    effort yields None. Another effort is never substituted, and the
    caller must not fall through to a lower tier.
    """
    if not effort:
        return None
    group = _depth_group(model, lookup)
    if group is None:
        return None
    return group.get(effort)


def _ordered_depth_records(group):
    """Measured depth records from an effort-keyed group, display-ordered.

    Skips ambiguous/absent entries (a None record) and any record without a
    finite index. Every record carries the internal shape resolve_quality
    attaches to cand["reasoning_depths"]:
    {"effort", "index", "cost_per_task_usd", "source", "as_of"}.
    """
    records = []
    for effort, record in (group or {}).items():
        if not isinstance(record, dict):
            continue
        index = record.get("index")
        if (
            not isinstance(index, (int, float))
            or isinstance(index, bool)
            or not math.isfinite(index)
        ):
            continue
        records.append(
            {
                "effort": record.get("effort") or effort,
                "index": float(index),
                "cost_per_task_usd": record.get("cost_per_task_usd"),
                "source": record.get("source"),
                "as_of": record.get("as_of"),
            }
        )

    def order(record):
        effort = record["effort"]
        try:
            return (AA_EFFORT_ORDER.index(effort), effort)
        except ValueError:
            return (len(AA_EFFORT_ORDER), effort or "")

    records.sort(key=order)
    return records


def match_quality(model, exact, fuzzy, allow_fuzzy=False):
    """Match a candidate to an AA entry; exact tiers only unless fuzzy is on.

    Returns the winning entry's (index, as_of) pair, or None when no tier
    matches. Values come straight out of build_aa_lookup's maps, so the
    as_of is the entry's own fetch stamp, never a re-lookup.
    """
    model_id = model["id"]
    name = model.get("name") or ""
    display = PROVIDER_PREFIX_RE.sub("", name).strip()
    # Assumes model_id contains a "/" (provider/base). build_candidates drops
    # malformed ids upstream, so base is never empty here.
    _, _, base = model_id.split(":", 1)[0].partition("/")

    for tier, raw in (
        (0, model_id),
        (1, base),
        (2, display),
        (3, PAREN_RE.sub(" ", display)),
    ):
        normalized = norm_key(raw)
        if normalized and (tier, normalized) in exact:
            return exact[(tier, normalized)]

    if not allow_fuzzy:
        return None
    own = set(norm_key(base).split()) | set(norm_key(display).split())
    if own:
        best = 0.0
        best_match = None
        for tokens, match in fuzzy:
            overlap = len(own & tokens) / len(own | tokens)
            if overlap > best:
                best = overlap
                best_match = match
        if best >= 0.5 and best_match is not None:
            return best_match
    return None


def base_model_id(model_id: str) -> str:
    """Candidate id without its :variant suffix (OR benchmarks are per base model)."""
    return model_id.split(":", 1)[0]


AA_BENCHMARK_FIELDS = ("intelligence_index", "coding_index", "agentic_index")


def _pair_permaslug_payloads(valid, models, merge=None):
    """Resolve permaslug-keyed payloads to exact catalog slugs.

    ``valid`` maps a permaslug (or an undated catalog id) to
    ``(payload, date)``, where ``date`` is the key's trailing -YYYYMMDD or
    "" for an undated key. ``models`` carries the catalog's own
    slug/permaslug pairing. The three passes are the #17-safe pairing
    documented on build_aa_benchmarks: permaslug pairs first (latest date
    wins per slug), then undated keys self-map, then an unambiguous dated
    strip. Returns ``{slug: (payload, date)}``.

    ``merge`` is for payloads that are turn-range maps (fetch_session_cost),
    not whole records, and unions a slug's maps across all three passes:
    pass 1's same-slug snapshots, an undated key naming an already-paired
    slug (pass 2), and the single unambiguous dated strip onto an
    already-paired id (pass 3) -- each range resolves to the latest date's
    cell (first wins on an equal date), so one slug's data never silently
    loses ranges because it arrived through different passes. An ambiguous
    strip (several candidates) still drops rather than guesses.
    Whole-record payloads must keep the default, which replaces the
    payload wholesale -- a record is self-contained, so mixing snapshots
    would splice fields from different snapshots.
    """
    permaslug_to_slug = {}
    for entry in models or []:
        if not isinstance(entry, dict):
            continue
        slug = entry.get("slug") or ""
        permaslug = entry.get("permaslug") or ""
        if slug and permaslug:
            # standard and batch variants repeat an identical pair; the
            # overwrite is the dedup
            permaslug_to_slug[permaslug] = slug

    paired = {}
    claimed = set()
    seen_range = {}

    # merge mode only: union one turn-range-map payload into slug's map.
    # Each range resolves to the latest date's cell (first wins on an
    # equal date); the paired date tracks the latest contributing date.
    # The merged map is always a fresh copy: an earlier pass may have
    # stored a caller-owned payload by reference, and writing into it
    # would corrupt the caller's ``valid`` payloads.
    def merge_ranges(slug, payload, date):
        merged, prev_date = paired.get(slug, ({}, ""))
        merged = dict(merged)
        for turn_range, median in payload.items():
            rkey = (slug, turn_range)
            if rkey in seen_range and seen_range[rkey] >= date:
                continue
            seen_range[rkey] = date
            merged[turn_range] = median
        paired[slug] = (merged, date if date > prev_date else prev_date)

    # Pass 1: permaslug keys pair with their own slug; same-slug
    # snapshots resolve to the latest date (first wins on an equal date).
    # In merge mode a payload is a turn-range map, not a whole record, so
    # same-slug snapshots union their maps per range instead of replacing.
    seen_date = {}
    for key, (payload, date) in valid.items():
        slug = permaslug_to_slug.get(key)
        if slug is None:
            continue
        if merge:
            claimed.add(slug)
            merge_ranges(slug, payload, date)
            continue
        if slug in seen_date and seen_date[slug] >= date:
            continue
        seen_date[slug] = date
        paired[slug] = (payload, date)
        claimed.add(slug)

    # Pass 2: undated keys self-map unless pass 1 claimed the id.
    # In merge mode a claimed undated key that IS an already-paired slug
    # unions its ranges into that slug's map instead of dropping -- one
    # slug's data arriving through different passes must not lose ranges.
    for key, (payload, date) in valid.items():
        if date or key in permaslug_to_slug:
            continue
        if key in claimed:
            if merge and key in paired:
                merge_ranges(key, payload, date)
            continue
        paired[key] = (payload, date)
        claimed.add(key)

    # Pass 3: unpaired dated keys strip, but only onto an unclaimed id
    # and only when the strip is unambiguous. In merge mode the single
    # unambiguous candidate unions into an already-paired bare id instead
    # of dropping; an ambiguous strip (several candidates) still drops.
    strip_candidates = {}
    for key, (payload, date) in valid.items():
        if not date or key in permaslug_to_slug:
            continue
        stripped = strip_permaslug_date(key)
        strip_candidates.setdefault(stripped, []).append((date, payload))
    for bare, candidates in strip_candidates.items():
        if len(candidates) > 1:
            continue
        date, payload = candidates[0]
        if bare in claimed:
            if merge and bare in paired:
                merge_ranges(bare, payload, date)
            continue
        paired[bare] = (payload, date)
        claimed.add(bare)

    return paired


def build_aa_benchmarks(benchmarks, models=None) -> dict:
    """Map exact catalog slug -> OR-published AA trio from data.benchmarks.

    Benchmark keys are dated permaslugs (deepseek/deepseek-v4.1-flash-
    20261017) whose date-stripped form can collide across listings of one
    model family: the snapshot listing deepseek/deepseek-v4.1-flash-1017
    and the base listing deepseek/deepseek-v4.1-flash both strip to
    deepseek/deepseek-v4.1-flash, so blind stripping with latest-date-wins
    made the base listing borrow the snapshot's score (#17). Keys resolve
    through three explicit passes over the surviving entries, and every
    accepted id is claimed by the pass that resolved it, so later passes
    never overwrite an earlier one:

    1. Primary: a key equal to a models entry's permaslug maps to that
       entry's slug -- the catalog's own pairing, no guessing. Several
       dated permaslugs may legitimately pair with one slug (snapshot
       variants of one listing); among those the latest date wins.
    2. Undated: a key without a -YYYYMMDD suffix maps to itself, unless
       pass 1 claimed the id.
    3. Dated fallback: a dated key the catalog does not pair strips via
       strip_permaslug_date, accepted only when the stripped id is
       unclaimed and no other pass-3 key strips to the same id -- an
       ambiguous strip is the #17 conflation shape, so every candidate
       for that id drops rather than guess.

    Entries whose values fail the finite-number guard are treated as
    absent before the passes run; entries with at least one valid field
    are kept. Records carry "as_of": the YYYYMMDD of the permaslug the
    trio came from, or None for undated keys.
    """
    valid = {}  # benchmark key -> (trio, permaslug date or "")
    for key, node in (benchmarks or {}).items():
        aa = node.get("aa") if isinstance(node, dict) else None
        if not isinstance(aa, dict):
            continue
        trio = {}
        for field in AA_BENCHMARK_FIELDS:
            value = aa.get(field)
            if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
            ):
                trio[field] = float(value)
        if not trio:
            continue
        bare = strip_permaslug_date(key)
        # strip_permaslug_date removed a trailing -YYYYMMDD exactly when
        # the key was dated; the date is then the final 8 characters.
        valid[key] = (trio, "" if bare == key else key[-8:])

    paired = _pair_permaslug_payloads(valid, models)
    return {
        slug: {**trio, "as_of": date or None} for slug, (trio, date) in paired.items()
    }


FRONTEND_REASONING_FIELDS = (
    "slug",
    "permaslug",
    "supports_reasoning",
    "reasoning_config",
)


def _frontend_reasoning_entries(entries):
    """Project frontend model entries to the fields reasoning metadata needs."""
    projected = []
    for entry in entries or []:
        if not isinstance(entry, dict):
            continue
        slug = entry.get("slug") or ""
        if slug.startswith("~"):
            continue
        if not (slug or entry.get("permaslug")):
            continue
        projected.append(
            {field: entry.get(field) for field in FRONTEND_REASONING_FIELDS}
        )
    return projected


def _effort_list(value):
    """Expand a supported-efforts field; null means all gateway efforts."""
    if value is None:
        return list(REASONING_GATEWAY_EFFORTS)
    if not isinstance(value, (list, tuple)):
        return []
    return [effort for effort in value if isinstance(effort, str)]


def _reasoning_field(primary, config, primary_key, frontend_key):
    """Field precedence: models API field, then frontend config, then unknown."""
    if isinstance(primary, dict) and primary_key in primary:
        return primary[primary_key]
    if isinstance(config, dict) and frontend_key in config:
        return config[frontend_key]
    return None


def _optional_bool(value, field):
    """A published boolean field: bool, or null when the source is unknown.

    Spec 9.1 publishes ``supports_max_tokens`` as bool or null. The old
    bool() coercion conflated an omitted/unknown field with false. Explicit
    booleans pass through; common string forms are coerced; anything else
    warns and degrades to null (fail closed, never guess).
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "1", "yes"):
            return True
        if lowered in ("false", "0", "no"):
            return False
    warn(f"unexpected reasoning.{field} value {value!r}; treating it as unknown")
    return None


def _optional_effort(value):
    """A published effort label: a non-empty string, or null when unknown.

    Malformed external metadata must degrade to unknown rather than poison
    the published catalog (the catalog validator rejects a non-string
    default_effort, which would fail the whole artifact).
    """
    if value is None:
        return None
    if isinstance(value, str) and value.strip():
        return value
    warn(f"unexpected reasoning.default_effort value {value!r}; treating it as unknown")
    return None


def _supported_efforts(primary, config):
    """The expanded effort list; an omitted field stays empty.

    The models API object is authoritative for effort selection: when it is
    present but omits supported_efforts, the model has no discrete efforts
    (single-state), even if the frontend's null "all efforts" marker would
    expand. The frontend list is only consulted when the primary object is
    absent entirely.
    """
    if isinstance(primary, dict):
        if "supported_efforts" in primary:
            return _effort_list(primary["supported_efforts"])
        return []
    if isinstance(config, dict) and "supported_reasoning_efforts" in config:
        return _effort_list(config["supported_reasoning_efforts"])
    return []


def effective_default_effort(reasoning: dict | None) -> str | None:
    """The effort a request with no ``reasoning`` field actually runs at (§5.2).

    Single-state/token-budget models (no discrete efforts after expansion)
    resolve to ``reasoning`` or ``none`` by ``default_enabled``; that branch
    runs before the discrete-effort rules. Non-reasoning models are ``none``;
    missing metadata is unknown (None).
    """
    if not isinstance(reasoning, dict):
        return None
    if reasoning.get("is_reasoning") is False:
        return "none"
    if reasoning.get("is_reasoning") is not True:
        return None
    efforts = reasoning.get("supported_efforts") or []
    mandatory = reasoning.get("mandatory")
    default_enabled = reasoning.get("default_enabled")
    if mandatory is True and default_enabled is False:
        warn(
            "contradictory reasoning metadata: mandatory=true with "
            "default_enabled=false; treating the effective default as unknown"
        )
        return None
    if not efforts:
        # Single-state or token-budget model: reasoning is one state.
        return "reasoning" if default_enabled is not False else "none"
    if default_enabled is False:
        return "none"
    default_effort = reasoning.get("default_effort")
    if default_effort == "none":
        return "none"
    if isinstance(default_effort, str) and default_effort:
        return default_effort
    return None


def supports_effort(reasoning: dict | None, effort: str) -> bool:
    """Whether a model can run ``effort`` (the §7 filter rule)."""
    if not isinstance(reasoning, dict):
        return False
    efforts = reasoning.get("supported_efforts") or []
    if effort in efforts:
        return True
    if effort == "none":
        if reasoning.get("is_reasoning") is False:
            return True
        # A non-mandatory single-state model exposes a reasoning toggle.
        return (
            not efforts
            and reasoning.get("is_reasoning") is True
            and reasoning.get("mandatory") is not True
        )
    if effort == "reasoning":
        return not efforts and reasoning.get("is_reasoning") is True
    return False


def build_reasoning_metadata(
    models: list[dict], frontend_models: list[dict] | None = None
) -> dict[str, dict]:
    """Per-model reasoning metadata, models API first and frontend second.

    Each value carries the raw flags for provenance plus the *effective*
    default effort (§5.2) and the expanded supported-effort list. Frontend
    entries are keyed by slug/permaslug, so they attach to a candidate id
    through the same three-pass pairing build_aa_benchmarks uses.
    """
    models = [
        model for model in models or [] if isinstance(model, dict) and model.get("id")
    ]
    frontend_models = _frontend_reasoning_entries(frontend_models)

    valid = {}
    for entry in frontend_models:
        slug = entry.get("slug") or ""
        permaslug = entry.get("permaslug") or ""
        key = permaslug or slug
        if not key:
            continue
        bare = strip_permaslug_date(key)
        valid[key] = (entry, "" if bare == key else key[-8:])
        if slug and slug != key:
            # The undated slug is the exact-id self-map (pass 2). Keep it
            # even when a dated permaslug is present, so a listing whose
            # permaslug is unpaired still resolves by its own id.
            valid[slug] = (entry, "")

    catalog_pairs = catalog_pairing_entries(models)
    frontend_by_id = {
        candidate_id: entry
        for candidate_id, (entry, _date) in _pair_permaslug_payloads(
            valid, catalog_pairs
        ).items()
    }
    # Exact-slug self-map. The permaslug pairing is authoritative when it
    # resolves a listing, but a shared canonical slug (base and :batch) is
    # claimed by only one listing, so the other still needs its own slug.
    frontend_by_slug = {
        entry["slug"]: entry for entry in frontend_models if entry.get("slug")
    }

    metadata = {}
    for model in models:
        model_id = model["id"]
        primary = model.get("reasoning")
        if not isinstance(primary, dict):
            primary = None
        entry = frontend_by_id.get(model_id) or frontend_by_slug.get(model_id)
        config = None
        supports_reasoning = None
        if isinstance(entry, dict):
            config = entry.get("reasoning_config")
            if not isinstance(config, dict):
                config = None
            supports_reasoning = entry.get("supports_reasoning")

        if primary is not None:
            is_reasoning = True
        elif isinstance(supports_reasoning, bool):
            is_reasoning = supports_reasoning
        else:
            is_reasoning = None

        reasoning = {
            "default_effort": _optional_effort(
                _reasoning_field(
                    primary, config, "default_effort", "default_reasoning_effort"
                )
            ),
            "supported_efforts": _supported_efforts(primary, config),
            "default_enabled": _optional_bool(
                _reasoning_field(
                    primary, config, "default_enabled", "default_reasoning_enabled"
                ),
                "default_enabled",
            ),
            "mandatory": _optional_bool(
                _reasoning_field(
                    primary, config, "mandatory", "is_mandatory_reasoning"
                ),
                "mandatory",
            ),
            "supports_max_tokens": _optional_bool(
                _reasoning_field(
                    primary,
                    config,
                    "supports_max_tokens",
                    "supports_reasoning_max_tokens",
                ),
                "supports_max_tokens",
            ),
            "is_reasoning": is_reasoning,
        }
        reasoning["default_effort"] = effective_default_effort(reasoning)
        metadata[model_id] = reasoning
    return metadata


def select_default_depth(cand: dict, depths: dict[str, dict]) -> str | None:
    """Select the highest finite measured effort the candidate can run.

    A non-empty supported-effort list limits measured records to selectable
    efforts. Missing or empty metadata leaves all measured records eligible,
    including the single-state ``reasoning`` pseudo-effort used by older
    candidate mappings. Equal scores prefer the deeper effort.
    """
    supported = cand.get("supported_efforts")
    if supported is None:
        reasoning = cand.get("reasoning")
        supported = (
            reasoning.get("supported_efforts") if isinstance(reasoning, dict) else None
        )
    selectable = set(supported) if supported else None

    best_effort = None
    best_index = None
    best_depth = len(AA_EFFORT_ORDER)
    for effort, record in (depths or {}).items():
        if not isinstance(record, dict):
            continue
        record_effort = record.get("effort") or effort
        if selectable is not None and record_effort not in selectable:
            continue
        index = record.get("index")
        if not isinstance(index, (int, float)) or isinstance(index, bool):
            continue
        try:
            finite_index = math.isfinite(index)
        except OverflowError:
            finite_index = False
        if not finite_index:
            continue
        try:
            depth = AA_EFFORT_ORDER.index(record_effort)
        except ValueError:
            depth = len(AA_EFFORT_ORDER)
        if (
            best_index is None
            or index > best_index
            or (index == best_index and depth < best_depth)
        ):
            best_effort = effort
            best_index = index
            best_depth = depth
    return best_effort


def resolve_quality(
    candidates,
    aa_by_id,
    exact,
    fuzzy,
    aa_source,
    catalog_quality=None,
    *,
    depth_lookup=None,
    unlabeled_lookup=None,
    catalog_depths=None,
    requested_effort="default",
):
    """Per-candidate quality records: AA API primary, catalog next, OR last.

    Depth-aware mode (either new input is not None; run() always passes the
    AA depth lookup) selects the highest-scoring selectable record when
    requested_effort is "default". An explicit --reasoning-effort selects
    exactly that effort. Resolution order per candidate:
      1. AA depth lookup -- best selectable measured effort for a default
         run, or the exact requested effort. source "aa", effort from the
         record.
      2. catalog_depths -- the published catalog's per-model depth map,
         selected by the same rules only when AA supplies no record. source
         "catalog", the record's own as_of.
      3. OpenRouter's benchmark trio -- only for a model known to be
         non-reasoning (is_reasoning False) at effort "none". The trio has
         no effort label, so a reasoning model never uses it as its main
         score; unknown reasoning is not non-reasoning.
      4. No record -- the candidate stays quality-blind.

    A quality-blind candidate may still carry a display-only AA reference:
    rows AA measures without an effort label cannot fill QUAL, so
    ``unlabeled_lookup`` (keyed runs) or the catalog depth map's
    ``unlabeled`` record (unkeyed runs) is attached as
    ``cand["quality_unlabeled"]`` when no main record resolved. It never
    enters quality_by_id and never changes the score.

    Default selection considers only finite records whose efforts are
    selectable when support metadata is non-empty; ties prefer the deeper
    effort. Explicit selection remains strict: another effort is never
    substituted. A reasoning model never falls through to the OR trio.

    The legacy path (both new inputs None) keeps the pre-depth behavior:
    match_quality's exact tiers, then the hydrated catalog_quality by id,
    then the OR trio for any model.

    Every depth-mode record is {"index", "source", "as_of", "effort",
    "drift", "or_index", "or_as_of"}; legacy records omit "effort".
    Drift is computed only when the winner is not OpenRouter and only
    like-for-like: the old |index - or_index| >= 3.0 rule applies at effort
    "none"; any other effort keeps the OR reference but sets drift None.
    cand["reasoning_depths"] carries the full measured depth list (AA or
    catalog group, or the single OR "none" record). The list is single-source:
    it comes from whichever tier supplied the main record, and sources are
    never merged.
    """
    quality_by_id = {}
    source_by_id = {}
    depth_mode = depth_lookup is not None or catalog_depths is not None
    for cand in candidates:
        # A previous pass may have attached a display-only reference; a
        # resolved record makes it stale, so never let it survive.
        cand.pop("quality_unlabeled", None)
        record = None
        if depth_mode:
            default_effort = cand.get("default_effort")
            effort = (
                requested_effort if requested_effort != "default" else default_effort
            )
            measured = []
            group = _depth_group(cand, depth_lookup) if depth_lookup else None
            if group is not None:
                measured = _ordered_depth_records(group)
                selected_effort = (
                    effort
                    if requested_effort != "default"
                    else select_default_depth(cand, group)
                )
                depth_record = group.get(selected_effort) if selected_effort else None
                if isinstance(depth_record, dict):
                    record = {
                        "index": depth_record["index"],
                        "source": "aa",
                        "as_of": depth_record.get("as_of"),
                        "effort": depth_record.get("effort") or selected_effort,
                    }
            if record is None and catalog_depths is not None:
                catalog_entry = catalog_depths.get(cand["id"])
                if isinstance(catalog_entry, dict):
                    catalog_group = catalog_entry.get("depths")
                    if isinstance(catalog_group, dict):
                        measured = _ordered_depth_records(catalog_group)
                        selected_effort = (
                            effort
                            if requested_effort != "default"
                            else select_default_depth(cand, catalog_group)
                        )
                        depth_record = (
                            catalog_group.get(selected_effort)
                            if selected_effort
                            else None
                        )
                        if isinstance(depth_record, dict):
                            record = {
                                "index": depth_record["index"],
                                "source": "catalog",
                                "as_of": depth_record.get("as_of"),
                                "effort": depth_record.get("effort") or selected_effort,
                            }
            reasoning = cand.get("reasoning")
            is_non_reasoning = (
                isinstance(reasoning, dict) and reasoning.get("is_reasoning") is False
            )
            fallback_effort = (
                effort if requested_effort != "default" else default_effort
            )
            if record is None and is_non_reasoning and fallback_effort == "none":
                bench = aa_by_id.get(base_model_id(cand["id"]))
                intelligence = bench.get("intelligence_index") if bench else None
                if intelligence is not None:
                    record = {
                        "index": intelligence,
                        "source": "openrouter",
                        "as_of": bench.get("as_of"),
                        "effort": "none",
                    }
                    measured = [
                        {
                            "effort": "none",
                            "index": intelligence,
                            "cost_per_task_usd": None,
                            "source": "openrouter",
                            "as_of": bench.get("as_of"),
                        }
                    ]
            cand["reasoning_depths"] = measured
        else:
            # Tier 1: AA API exact match. match_quality returns the winning
            # entry's (index, as_of) pair.
            match = match_quality(
                {"id": cand["id"], "name": cand["name"]}, exact, fuzzy
            )
            if match is not None:
                index, as_of = match
                record = {"index": index, "source": "aa", "as_of": as_of}
            # Tier 2: catalog record (unkeyed runs hydrate from the published
            # catalog). The run-level provenance label is the source chain:
            # every hydrated record is tagged "catalog" -- read from our
            # published catalog -- overriding the row's own label, which is
            # the keyed CI run's row-level truth inside the artifact ("aa"/
            # "openrouter"; no real catalog row ever carries "catalog").
            # as_of stays the row's own stamp.
            if record is None and catalog_quality is not None:
                catalog_record = catalog_quality.get(cand["id"])
                if catalog_record is not None:
                    record = {
                        "index": catalog_record["index"],
                        "source": "catalog",
                        "as_of": catalog_record["as_of"],
                    }
            # Tier 3: OpenRouter's per-slug benchmark trio (last resort).
            if record is None:
                bench = aa_by_id.get(base_model_id(cand["id"]))
                intelligence = bench.get("intelligence_index") if bench else None
                if intelligence is not None:
                    record = {
                        "index": intelligence,
                        "source": "openrouter",
                        "as_of": bench.get("as_of"),
                    }
        if record is None:
            # No effort-attributable record: surface an unlabeled AA
            # measurement as a display-only reference. It never fills
            # QUAL, and a resolved record makes it redundant.
            reference = (
                _depth_group(cand, unlabeled_lookup) if unlabeled_lookup else None
            )
            if reference is None and catalog_depths is not None:
                catalog_entry = catalog_depths.get(cand["id"])
                if isinstance(catalog_entry, dict):
                    candidate_reference = catalog_entry.get("unlabeled")
                    if isinstance(candidate_reference, dict):
                        reference = candidate_reference
            reference_index = (
                reference.get("index") if isinstance(reference, dict) else None
            )
            if (
                isinstance(reference_index, (int, float))
                and not isinstance(reference_index, bool)
                and math.isfinite(reference_index)
            ):
                cand["quality_unlabeled"] = {
                    "index": float(reference_index),
                    "cost_per_task_usd": reference.get("cost_per_task_usd"),
                    "source": reference.get("source") or "aa",
                    "as_of": reference.get("as_of"),
                }
            continue

        # Drift: only meaningful when the winner is not OpenRouter itself.
        bench = aa_by_id.get(base_model_id(cand["id"])) or {}
        or_intelligence = bench.get("intelligence_index")
        reasoning = cand.get("reasoning") or {}
        if (
            record["source"] in ("aa", "catalog")
            and isinstance(or_intelligence, (int, float))
            and not isinstance(or_intelligence, bool)
            and math.isfinite(or_intelligence)
        ):
            record["or_index"] = or_intelligence
            record["or_as_of"] = bench.get("as_of")
            # The comparable case is a non-reasoning MODEL at effort "none",
            # not merely an effort label: a reasoning model whose effective
            # default is "none" still has an OR snapshot measured at its
            # reasoning state, so a difference may be a depth difference.
            comparable = (not depth_mode) or (
                reasoning.get("is_reasoning") is False
                and record.get("effort") == "none"
            )
            if comparable:
                record["drift"] = abs(record["index"] - or_intelligence) >= 3.0
            else:
                record["drift"] = None
        else:
            record["or_index"] = None
            record["or_as_of"] = None
            record["drift"] = None

        quality_by_id[cand["id"]] = record
        source_by_id[cand["id"]] = record["source"]
    return quality_by_id, source_by_id


def model_family(model_id: str) -> str | None:
    """Best-effort model family from the base slug (documented heuristic).

    The leading token delimited by dash, underscore or digit of the
    lowercased base slug: glm-5.3 -> glm, gpt-5.2-mini -> gpt,
    deepseek-chat-v4 -> deepseek. Oddballs yield oddballs (o4-mini -> "o");
    a bare letter-run followed by a trailing digit version (k2) yields None.
    """
    base = model_id.split(":", 1)[0].partition("/")[2].lower()
    parts = re.split(r"[-_\d]", base, maxsplit=1)
    token = parts[0]
    if len(parts) > 1 and not parts[1] and base[len(token) : len(token) + 1].isdigit():
        return None
    return token or None


def _per_1m(price_per_token: float) -> float:
    """USD per token -> USD per 1M tokens, 6 decimals."""
    return round(price_per_token * 1_000_000.0, 6)


def build_schedules(
    models,
    args,
    quality_by_id,
    aa_by_id,
    balanced_by_id,
    catalog_quality=None,
    *,
    depth_lookup=None,
    catalog_depths=None,
    reasoning_by_id=None,
    requested_effort="default",
    schedule_entries=None,
):
    """Catalog-wide off-peak index built from the RAW model list (spec 5.4).

    Deliberately unfiltered by the candidate pool: sub-floor-context and
    non-ZDR schedule models are listed (that is the point -- today's live
    schedule models mostly sit below the 1M context floor). Router aliases
    (~ids) and malformed ids are excluded; :batch variants are included as
    distinct deals. Sorted by id.

    Depth-aware mode (run() always supplies the AA depth lookup; an unkeyed
    run supplies the catalog depth map and the published schedule entries)
    follows spec 6.5:
      0. ``schedule_entries`` (unkeyed runs): the published catalog's v6
         schedules array is the source of truth for Deals. Whenever an
         entry exists with a non-null quality AND a non-empty
         ``quality_effort``, its quality and all three provenance fields
         are copied VERBATIM (no run-level relabeling); requested_effort is
         ignored because there is no depth list to re-select from. An entry
         with no effort label is legacy/unknown (spec 6.5): its quality is
         null and the lower tiers are skipped. A null-quality or absent
         entry falls through to the tiers below.
      1. AA depth lookup at the highest-scoring selectable effort for a
         default run, or exactly ``requested_effort`` -- keyed runs only.
      2. Catalog depth map by the same selection rule when AA has no record;
         unkeyed runs use that source without merging its records with AA.
      3. Catalog main-quality map, only when the record carries an effort
         (a record without one is legacy/unknown: quality stays null).
      4. OpenRouter's benchmark trio, only for a model known to be
         non-reasoning (``is_reasoning`` False) at effort "none", when the
         run requests ``default`` (whose effective default is ``none``) or
         explicitly requests ``none``. An explicit request for another
         effort never borrows the unscoped trio. Reasoning models never fall
         through because the trio has no effort label for the measurement.
    Default selection at tiers 1-2 considers finite records supported by
    the live reasoning metadata; equal scores prefer deeper effort. Explicit
    selection remains strict, and another effort is never substituted.

    The legacy path (both new inputs None) keeps the pre-depth fallback
    chain: candidate record, hydrated catalog, then the OR trio. Every row
    carries ``quality_effort`` (string or null) alongside the winning
    record's provenance (quality_source/quality_as_of).
    """
    depth_mode = depth_lookup is not None or catalog_depths is not None
    entries = []
    for model in models:
        model_id = model.get("id") or ""
        if not model_id or model_id.startswith("~") or "/" not in model_id:
            continue
        eff = effective_pricing(
            model.get("pricing") or {}, args.min_context, args.input_share
        )
        top_in, top_out = eff["top_price_in"], eff["top_price_out"]
        if (
            top_in is None
            or top_out is None
            or top_in < 0
            or top_out < 0
            or not math.isfinite(top_in)
            or not math.isfinite(top_out)
        ):
            continue  # invalid top level: not a deal worth listing
        windows = eff["schedule"]
        if eff["schedule_error"] or not windows:
            continue
        peak = max(windows, key=lambda window: window["blended"])
        offpeak = min(windows, key=lambda window: window["blended"])
        # quality_by_id values are records {"index", "source", ...}; the
        # schedule row keeps the bare float index verbatim (never a dict)
        # and echoes the winning record's provenance for the QUAL tooltip.
        quality = None
        quality_source = None
        quality_as_of = None
        quality_effort = None
        schedule_entry = (schedule_entries or {}).get(model_id)
        if (
            isinstance(schedule_entry, dict)
            and schedule_entry.get("quality") is not None
        ):
            effort_label = schedule_entry.get("quality_effort")
            if isinstance(effort_label, str) and effort_label:
                # Unkeyed source of truth: recorded provenance verbatim.
                quality = schedule_entry.get("quality")
                quality_source = schedule_entry.get("quality_source")
                quality_as_of = schedule_entry.get("quality_as_of")
                quality_effort = effort_label
            # else: a recorded entry with no effort label is legacy/unknown
            # (spec 6.5): its quality is null, and the lower tiers are
            # skipped rather than guessing from an unscoped OR snapshot.
        elif depth_mode:
            reasoning = (reasoning_by_id or {}).get(model_id)
            if not isinstance(reasoning, dict):
                reasoning = None
            default_effort = (reasoning or {}).get("default_effort")
            effort = (
                requested_effort if requested_effort != "default" else default_effort
            )
            candidate = {
                "supported_efforts": (reasoning or {}).get("supported_efforts"),
                "reasoning": reasoning,
            }
            # Tier 1: AA depth lookup (keyed runs).
            group = _depth_group(model, depth_lookup) if depth_lookup else None
            selected_effort = (
                effort
                if requested_effort != "default"
                else select_default_depth(candidate, group)
            )
            record = (
                group.get(selected_effort)
                if group is not None and selected_effort
                else None
            )
            if isinstance(record, dict):
                quality = record.get("index")
                quality_source = "aa"
                quality_as_of = record.get("as_of")
                quality_effort = record.get("effort") or selected_effort
            # Tier 2: catalog depth map, consulted only when AA has no record.
            if quality is None and catalog_depths is not None:
                catalog_entry = catalog_depths.get(model_id)
                if isinstance(catalog_entry, dict):
                    group = catalog_entry.get("depths")
                    if isinstance(group, dict):
                        selected_effort = (
                            effort
                            if requested_effort != "default"
                            else select_default_depth(candidate, group)
                        )
                        record = group.get(selected_effort) if selected_effort else None
                        if isinstance(record, dict):
                            quality = record.get("index")
                            quality_source = "catalog"
                            quality_as_of = record.get("as_of")
                            quality_effort = record.get("effort") or selected_effort
            # Tier 3: catalog main-quality map (unkeyed runs). A record with
            # no quality_effort is legacy/unknown: null, never a guess.
            if quality is None and catalog_quality is not None:
                catalog_record = catalog_quality.get(model_id)
                if isinstance(catalog_record, dict) and catalog_record.get("effort"):
                    quality = catalog_record.get("index")
                    quality_source = "catalog"
                    quality_as_of = catalog_record.get("as_of")
                    quality_effort = catalog_record.get("effort")
            # Tier 4: OR trio is eligible only for known non-reasoning models
            # when this run requests default (effective none) or explicit none.
            if (
                quality is None
                and reasoning is not None
                and reasoning.get("is_reasoning") is False
                and (
                    requested_effort == "none"
                    or (requested_effort == "default" and effort == "none")
                )
            ):
                benchmark = aa_by_id.get(base_model_id(model_id)) or {}
                quality = benchmark.get("intelligence_index")
                if quality is not None:
                    quality_source = "openrouter"
                    quality_as_of = benchmark.get("as_of")
                    quality_effort = "none"
        else:
            # Legacy path: candidate record, hydrated catalog, OR trio.
            record = quality_by_id.get(model_id)
            if isinstance(record, dict):
                quality = record.get("index")
                quality_source = record.get("source")
                quality_as_of = record.get("as_of")
                quality_effort = record.get("effort")
            else:
                quality = record
            if quality is None and catalog_quality is not None:
                catalog_record = catalog_quality.get(model_id)
                if catalog_record is not None:
                    quality = catalog_record["index"]
                    quality_source = "catalog"
                    quality_as_of = catalog_record["as_of"]
                    quality_effort = catalog_record.get("effort")
            if quality is None:
                benchmark = aa_by_id.get(base_model_id(model_id)) or {}
                quality = benchmark.get("intelligence_index")
                if quality is not None:
                    quality_source = "openrouter"
                    quality_as_of = benchmark.get("as_of")
                else:
                    quality_source = None
                    quality_as_of = None
        name = model.get("name") or model_id
        entries.append(
            {
                "id": model_id,
                "name": PROVIDER_PREFIX_RE.sub("", name).strip() or name,
                "context": coerce_int(model.get("context_length"), 0),
                "quality": quality,
                # Provenance for the site's QUAL tooltip (null when the
                # quality came without any -- e.g. a bare-float test map).
                "quality_source": quality_source,
                "quality_as_of": quality_as_of,
                "quality_effort": quality_effort,
                "score": balanced_by_id.get(model_id),
                "max_discount": eff["max_discount"],
                "sched_note": fmt_sched_note(eff["max_discount"]),
                "sched_detail": fmt_sched_detail(windows, eff["peak_blended"]),
                "peak": {
                    "input_per_1m": _per_1m(peak["price_in"]),
                    "output_per_1m": _per_1m(peak["price_out"]),
                    "blended_per_1m": _per_1m(peak["blended"]),
                    "input_cache_read_per_1m": (
                        _per_1m(peak["input_cache_read"])
                        if peak["input_cache_read"] is not None
                        else None
                    ),
                    "input_cache_write_per_1m": (
                        _per_1m(peak["input_cache_write"])
                        if peak["input_cache_write"] is not None
                        else None
                    ),
                },
                "offpeak": {
                    "input_per_1m": _per_1m(offpeak["price_in"]),
                    "output_per_1m": _per_1m(offpeak["price_out"]),
                    "blended_per_1m": _per_1m(offpeak["blended"]),
                    "input_cache_read_per_1m": (
                        _per_1m(offpeak["input_cache_read"])
                        if offpeak["input_cache_read"] is not None
                        else None
                    ),
                    "input_cache_write_per_1m": (
                        _per_1m(offpeak["input_cache_write"])
                        if offpeak["input_cache_write"] is not None
                        else None
                    ),
                },
                "schedule": [
                    {
                        "utc_days": window["utc_days"],
                        "utc_start": window["utc_start"],
                        "utc_end": window["utc_end"],
                        "coverage": window["coverage"],
                        "input_per_1m": _per_1m(window["price_in"]),
                        "output_per_1m": _per_1m(window["price_out"]),
                        "blended_per_1m": _per_1m(window["blended"]),
                        "input_cache_read_per_1m": (
                            _per_1m(window["input_cache_read"])
                            if window["input_cache_read"] is not None
                            else None
                        ),
                        "input_cache_write_per_1m": (
                            _per_1m(window["input_cache_write"])
                            if window["input_cache_write"] is not None
                            else None
                        ),
                    }
                    for window in windows
                ],
            }
        )
    entries.sort(key=lambda entry: entry["id"])
    return entries


def catalog_weights(candidates, quality_by_id):
    """Effective per-priority weights for the catalog document.

    Mirrors compute_scores' quality-blind rule: when no candidate has a
    quality score the quality weight is dropped and the rest renormalize,
    so scores.overall in the catalog is exactly reproducible downstream.
    """
    quality_blind = not any(c["id"] in quality_by_id for c in candidates)
    weights = {}
    for priority, base in PRIORITY_WEIGHTS.items():
        effective = dict(base)
        if quality_blind and "quality" in effective:
            effective.pop("quality")
            total = sum(effective.values())
            effective = {name: value / total for name, value in effective.items()}
        weights[priority] = effective
    return weights


def overall_scores(cand, weights=None) -> dict:
    """Per-priority overall scores from rounded components (catalog parity).

    weights is the per-priority map from catalog_weights; callers that must
    match the catalog on quality-blind runs pass it, otherwise the base
    PRIORITY_WEIGHTS apply.
    """
    weights = weights or PRIORITY_WEIGHTS
    scores = {
        "price": round(cand["price_score"], 4),
        "quality": round(cand["quality_score"], 4),
        "context": round(cand["context_score"], 4),
        "age": round(cand["age_score"], 4),
    }
    return {
        priority: round(
            weights[priority].get("quality", 0.0) * scores["quality"]
            + weights[priority].get("price", 0.0) * scores["price"]
            + weights[priority].get("context", 0.0) * scores["context"]
            + weights[priority].get("age", 0.0) * scores["age"],
            4,
        )
        for priority in PRIORITY_WEIGHTS
    }


def off_peak_deals(schedules, limit=10):
    rows = [row for row in schedules if f"{row['max_discount']:.0%}" != "0%"]
    rows.sort(key=lambda row: (-row["max_discount"], row["id"]))
    return rows[:limit]


# ---------------------------------------------------------------------------
# Filtering and scoring
# ---------------------------------------------------------------------------


def build_candidates(
    models,
    args,
    discounts,
    zdr_ids,
    filtered_out=None,
    reasoning_by_id: dict[str, dict] | None = None,
):
    now = time.time()
    require_tools = not args.no_require_tools
    requested_effort = getattr(args, "reasoning_effort", "default")
    dropped = {}
    candidates = []

    def drop(reason, model_id, name):
        dropped[reason] = dropped.get(reason, 0) + 1
        if filtered_out is None:
            return
        # Empty/malformed ids would fail the site validator's filtered-id
        # rule, so they stay counted in dropped only.
        if not model_id or "/" not in model_id:
            return
        filtered_out.append(
            {"id": model_id, "name": name or model_id, "reasons": [reason]}
        )

    for model in models:
        model_id = model.get("id") or ""
        if "/" not in model_id:
            drop("malformed id", model_id, model.get("name"))
            continue
        context = coerce_int(model.get("context_length"), 0)
        if context < args.min_context:
            drop("context", model_id, model.get("name"))
            continue
        pricing = model.get("pricing") or {}
        eff = effective_pricing(pricing, args.min_context, args.input_share)
        top_in, top_out = eff["top_price_in"], eff["top_price_out"]
        # The RAW top-level prices carry the validity drop (spec rules 1+7):
        # windowed models freeze their base at the peak window, so the gate
        # must check the top level itself. A bad override must never drop a
        # model, and effective prices inherit validity from the validated
        # base and override prices.
        if (
            top_in is None
            or top_out is None
            or top_in < 0
            or top_out < 0
            or not math.isfinite(top_in)
            or not math.isfinite(top_out)
        ):
            drop("pricing", model_id, model.get("name"))
            continue
        if eff["schedule_error"]:
            # No valid peak window means no deterministic base (spec rule 9):
            # schedule-only AND mixed models fail closed here.
            drop("schedule", model_id, model.get("name"))
            continue
        # The effort filter runs before the ZDR/discount gates: it is the
        # caller's explicit request, so it is the most useful drop reason
        # when several gates would reject the model.
        reasoning = (reasoning_by_id or {}).get(model_id)
        if not isinstance(reasoning, dict):
            reasoning = None
        # A model can always run at its own effective default (§7 equality
        # carve-out), even when that effort is absent from supported_efforts
        # -- e.g. default_enabled false resolves to "none" while the list
        # omits it.
        if requested_effort != "default" and not (
            supports_effort(reasoning, requested_effort)
            or requested_effort == (reasoning or {}).get("default_effort")
        ):
            drop("reasoning effort", model_id, model.get("name"))
            continue
        price_in, price_out = eff["price_in"], eff["price_out"]
        if args.exclude_free and price_in == 0 and price_out == 0:
            drop("free", model_id, model.get("name"))
            continue
        if not args.include_batch and model_id.endswith(":batch"):
            drop("batch", model_id, model.get("name"))
            continue
        discount = (discounts or {}).get(model_id)
        if args.discount and not has_discount(discount):
            drop("no discount", model_id, model.get("name"))
            continue
        if not args.no_zdr and model_id not in (zdr_ids or set()):
            drop("not ZDR", model_id, model.get("name"))
            continue
        modality = (model.get("architecture") or {}).get("modality") or ""
        output_modality = (
            modality.split("->")[-1].strip() if "->" in modality else "text"
        )
        if output_modality != "text":
            drop("modality", model_id, model.get("name"))
            continue
        params = model.get("supported_parameters") or []
        tool_calling = "tools" in params and "tool_choice" in params
        if require_tools and not tool_calling:
            drop("tool calling", model_id, model.get("name"))
            continue
        expiry = parse_iso_datetime(model.get("expiration_date"))
        expired = bool(expiry and expiry < datetime.now(timezone.utc))
        if expired:
            drop("expired", model_id, model.get("name"))
            continue
        created = model.get("created") or 0
        try:
            created = float(created)
        except (TypeError, ValueError):
            created = 0.0
        age_days = max(0.0, (now - created) / 86400.0) if created > 0 else None
        if args.max_age_days and age_days is not None and age_days > args.max_age_days:
            drop("age", model_id, model.get("name"))
            continue
        price_in_m = price_in * 1_000_000.0
        price_out_m = price_out * 1_000_000.0
        blended_m = (
            args.input_share * price_in_m + (1.0 - args.input_share) * price_out_m
        )
        top_provider = model.get("top_provider")
        max_output = (
            top_provider.get("max_completion_tokens")
            if isinstance(top_provider, dict)
            else None
        )
        if type(max_output) is not int or max_output <= 0:
            max_output = None
        architecture = model.get("architecture")
        input_modalities = (
            architecture.get("input_modalities")
            if isinstance(architecture, dict)
            else None
        )
        if not isinstance(input_modalities, list) or any(
            not isinstance(modality, str) or not modality
            for modality in input_modalities
        ):
            input_modalities = []
        cache_read_per_1m = (
            _per_1m(eff["price_cache_read"])
            if eff["price_cache_read"] is not None
            else None
        )
        cache_write_per_1m = (
            _per_1m(eff["price_cache_write"])
            if eff["price_cache_write"] is not None
            else None
        )
        base_cache_read_per_1m = (
            _per_1m(eff["base_cache_read"])
            if eff["base_cache_read"] is not None
            else None
        )
        base_cache_write_per_1m = (
            _per_1m(eff["base_cache_write"])
            if eff["base_cache_write"] is not None
            else None
        )
        candidates.append(
            {
                "id": model_id,
                "name": model.get("name") or model_id,
                "context": context,
                "max_output_tokens": max_output,
                "input_modalities": input_modalities,
                "price_in": price_in_m,
                "price_out": price_out_m,
                "blended": blended_m,
                "cache_read_per_1m": cache_read_per_1m,
                "cache_write_per_1m": cache_write_per_1m,
                "base_cache_read_per_1m": base_cache_read_per_1m,
                "base_cache_write_per_1m": base_cache_write_per_1m,
                "base_price_in": eff["base_price_in"] * 1_000_000.0,
                "base_price_out": eff["base_price_out"] * 1_000_000.0,
                "tier_prompt_tokens": eff["tier_prompt_tokens"],
                "tiers": eff["tiers"],
                "schedule": eff["schedule"],
                "max_discount": eff["max_discount"],
                "tier_note": fmt_tier_note(eff["tier_prompt_tokens"]),
                "sched_note": fmt_sched_note(eff["max_discount"]),
                "sched_detail": fmt_sched_detail(eff["schedule"], eff["peak_blended"]),
                "age_days": age_days,
                "discount": discount,
                "created": created,
                "tool_calling": tool_calling,
                "zdr": None if args.no_zdr else model_id in (zdr_ids or set()),
                "expired": expired,
                "reasoning": reasoning,
                "default_effort": (reasoning or {}).get("default_effort"),
                "supported_efforts": list(
                    (reasoning or {}).get("supported_efforts") or []
                ),
            }
        )

    return candidates, dropped


def compute_scores(candidates, args, quality_by_id):
    weights = dict(PRIORITY_WEIGHTS[args.priority])
    # Deliberate asymmetry: the quality weight is dropped (and the remaining
    # weights renormalized) only when *no* candidate has a quality score, so a
    # quality-blind pool is ranked purely on price/context/age. When *some*
    # candidates match, the quality weight is kept and unmatched candidates take
    # quality_score = 0 -- they are penalized rather than silently reweighted.
    if not any(c["id"] in quality_by_id for c in candidates):
        weights.pop("quality")
        total = sum(weights.values())
        weights = {name: value / total for name, value in weights.items()}

    prices = [c["blended"] for c in candidates]
    low = math.log10(min(prices) + 0.01)
    high = math.log10(max(prices) + 0.01)
    price_span = (high - low) or 1.0
    floor_ctx = max(args.min_context, 1000)
    cap_ctx = 4.0 * floor_ctx
    ctx_span = math.log(cap_ctx / floor_ctx) or 1.0

    for cand in candidates:
        price = cand["blended"]
        if price <= 0:
            cand["price_score"] = 1.0
        else:
            cand["price_score"] = max(
                0.0, min(1.0, 1.0 - (math.log10(price + 0.01) - low) / price_span)
            )
        quality = quality_by_id.get(cand["id"])
        # quality_by_id values are records {"index", "source", ...}; the
        # candidate keeps the bare float index (that is what ranking_key,
        # build_catalog's quality field and print_json's quality_index all
        # expect) and the full record travels alongside for provenance.
        record = quality if isinstance(quality, dict) else None
        cand["quality"] = record["index"] if record else quality
        cand["quality_record"] = record
        if cand["quality"] is None:
            cand["quality_score"] = 0.0
        else:
            cand["quality_score"] = max(
                0.0, min(1.0, cand["quality"] / args.quality_ref)
            )
        context = cand["context"]
        cand["context_score"] = (
            max(0.0, min(1.0, math.log(context / floor_ctx) / ctx_span))
            if context > floor_ctx
            else 0.0
        )
        age_days = cand["age_days"]
        cand["age_score"] = (
            0.5 if age_days is None else 0.5 ** (age_days / args.recency_half_life)
        )
        cand["score"] = weighted_score(cand, weights)

    candidates.sort(key=lambda c: ranking_key(c["score"], c))
    return weights


def weighted_score(cand, w):
    """The one unrounded weighted score, shared by compute_scores and
    catalog_rankings. Keep the operand order (quality, price, context, age):
    float addition is non-associative, and a reordered sum can differ by one
    ULP and flip a near-tie between the CLI and the catalog."""
    return (
        w.get("quality", 0.0) * cand["quality_score"]
        + w.get("price", 0.0) * cand["price_score"]
        + w.get("context", 0.0) * cand["context_score"]
        + w.get("age", 0.0) * cand["age_score"]
    )


def ranking_key(score, cand):
    """The one sort key: score desc, then raw quality desc, cheaper first, id."""
    return (-score, -(cand["quality"] or 0.0), cand["blended"], cand["id"])


def catalog_rankings(candidates, weights):
    """Per-priority full ordered id lists: the order a `--priority P` run
    produces. Pure: sorted() only, never writes onto the candidate dicts."""
    return {
        priority: [
            c["id"]
            for c in sorted(
                candidates,
                key=lambda c, w=weights[priority]: ranking_key(weighted_score(c, w), c),
            )
        ]
        for priority in PRIORITY_WEIGHTS
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def fmt_price(value: float) -> str:
    if value == 0:
        return "0"
    if value >= 100:
        return f"{value:.0f}"
    if value >= 10:
        return f"{value:.1f}"
    if value >= 1:
        return f"{value:.2f}"
    return f"{value:.3f}"


def fmt_context(value: int) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.0f}K"
    return str(value)


def fmt_age(age_days) -> str:
    if age_days is None:
        return "-"
    if age_days >= 365:
        return f"{age_days / 365:.1f}y"
    if age_days >= 60:
        return f"{age_days / 30.44:.0f}mo"
    return f"{age_days:.0f}d"


def has_discount(value) -> bool:
    # Only discounts that survive .0% rounding count; smaller slivers and
    # malformed (negative) values are treated as no discount everywhere:
    # the DISC column, --json output, and the --discount filter.
    return bool(value) and value > 0 and f"{value:.0%}" != "0%"


def fmt_discount(value) -> str:
    return f"{value:.0%}" if has_discount(value) else "--"


def _display_default_effort(cand):
    """The effort a default run uses, for the DEPTH cell and the depth block.

    Production candidates carry it on both ``cand["default_effort"]`` and the
    reasoning block; partial rows (tests, legacy paths) may carry only one.
    """
    return cand.get("default_effort") or (cand.get("reasoning") or {}).get(
        "default_effort"
    )


def _published_unlabeled_record(cand):
    """The display-only unlabeled AA reference, normalized for publication.

    None when the candidate has no such reference. The record keeps the
    depth record's shape minus ``effort``: it was measured without one, so
    publishing an effort label here would be a lie.
    """
    record = cand.get("quality_unlabeled")
    if not isinstance(record, dict):
        return None
    return {
        "index": record.get("index"),
        "cost_per_task_usd": record.get("cost_per_task_usd"),
        "source": record.get("source"),
        "as_of": record.get("as_of"),
    }


def _published_reasoning_block(cand):
    """The six published reasoning keys plus the measured depth records.

    ``is_reasoning`` is an internal flag: neither catalog v6 nor spec §8.4
    publishes it, so it never appears here. ``aa_unlabeled`` is the
    display-only AA reference (index with no effort label); it is null
    when AA measured every row at a known effort.
    """
    reasoning = cand.get("reasoning") or {}
    return {
        "default_effort": reasoning.get("default_effort"),
        "supported_efforts": reasoning.get("supported_efforts") or [],
        "default_enabled": reasoning.get("default_enabled"),
        "mandatory": reasoning.get("mandatory"),
        "supports_max_tokens": reasoning.get("supports_max_tokens"),
        "depths": cand.get("reasoning_depths") or [],
        "aa_unlabeled": _published_unlabeled_record(cand),
    }


def fmt_depth(cand) -> str:
    """DEPTH cell: the effort QUAL was measured at; ``*`` is informational."""
    effort = (cand.get("quality_record") or {}).get("effort")
    if not effort:
        return "-"
    return f"{effort}*" if effort == _display_default_effort(cand) else effort


def fmt_prec(cand) -> str:
    """PREC cell: the precision the ranked price is quoted at.

    The maker's default precision when disclosed, the best available
    otherwise, "n/a" when none could be derived; "!" flags default routing
    that favors cheaper sub-default endpoints (annotations.risk_flag).
    Candidates that never went through the endpoint stage degrade to "n/a".
    """
    cell = cand.get("default_precision") or "n/a"
    if (cand.get("annotations") or {}).get("risk_flag"):
        cell += "!"
    return cell


def fmt_quant_note(annotations: dict) -> str | None:
    """One-line note on the cheapest sub-default endpoint (PREC tooltip).

    The exact string the site shows (Task 11 renders it verbatim), so the
    format is pinned by test. None when there is no non-eligible live
    endpoint -- nothing cheaper to route to. cheapest_quantized.price is
    stored per-token, the way build_annotations leaves it; the note shows
    USD per 1M like every other price the CLI prints. A missing spread
    (defensive) drops the parenthetical. A null quantization (third-party
    null == "unknown" for eligibility) and a null provider_name on
    malformed data render the design's "unknown" sentinel, never Python's
    None repr.
    """
    quant = (annotations or {}).get("cheapest_quantized")
    if not quant:
        return None
    quantization = quant["quantization"] or "unknown"
    provider = quant["provider"] or "unknown"
    note = (
        f"cheapest non-baseline: {quantization} @ {provider}, "
        f"${fmt_price(quant['price'] * 1_000_000)}/M"
    )
    spread = (annotations or {}).get("price_spread")
    if spread is not None:
        note += f" in ({spread:.2f}x baseline)"
    return note


def _footer_as_of(source, value):
    """Format a record's as_of for its footer block (None when unusable).

    Two stamp shapes circulate. A YYYYMMDD date (OpenRouter's benchmark
    date, and catalog rows hydrated from OR rows) renders as a plain date
    ("as of 2026-10-17") for ANY source -- a date-only stamp never
    fabricates a time of day. ISO stamps: live AA records carry one fetch
    timestamp for the whole fetch (checked today), so their block shows
    the time of day only; other rows carry the full stamp ("as of
    2026-10-23 13:15Z").
    """
    if not value:
        return None
    text = str(value)
    if len(text) == 8 and text.isdigit():
        return f"{text[:4]}-{text[4:6]}-{text[6:]}"
    parsed = parse_iso_datetime(value)
    if parsed is None:
        return None
    if source == "aa":
        return parsed.strftime("%H:%MZ")
    return parsed.strftime("%Y-%m-%d %H:%MZ")


def _as_of_instant(value):
    """Parse an as_of stamp to a comparable UTC instant (None when unusable).

    The pipeline carries ISO stamps and the YYYYMMDD benchmark date; a
    date-only stamp means "that day" and compares as midnight UTC.
    """
    text = str(value)
    if len(text) == 8 and text.isdigit():
        return parse_iso_datetime(f"{text[:4]}-{text[4:6]}-{text[6:]}")
    return parse_iso_datetime(value)


def _later_as_of(value, current) -> bool:
    """True when the value stamp names a later instant than current.

    Parsed instants, never raw strings: a YYYYMMDD stamp must not
    lexicographically beat a fresher ISO stamp ("20261017" sorts above
    every "2026-..." string). Ties keep the earlier pick; an unparseable
    stamp never displaces anything.
    """
    new, old = _as_of_instant(value), _as_of_instant(current)
    if new is None or old is None:
        return new is not None
    return new > old


QUALITY_SOURCE_LABEL = "Quality source: "


def quality_footer_note(quality_by_id, aa_index_version, unmatched):
    """Compose the footer's quality note from the resolved records.

    Returns {"sources": [...], "clauses": [...], "warn": bool}. Each source
    names its model count, its own as_of and the AA index version when
    known, e.g. "AA index v4.3 - live AA (58 models, checked 13:15Z)".
    Clauses carry the unmatched-candidates note and the drift footnote when
    any resolved record disagrees with the OpenRouter check value; the drift
    count covers only comparable (non-reasoning) rows and says so. Returns
    None when no record resolved -- run() falls back to the quality-blind
    note.
    """
    labels = {
        "aa": "live AA",
        "catalog": "our published catalog",
        "openrouter": "OpenRouter snapshot",
    }
    stamp_words = {"aa": "checked", "catalog": "as of", "openrouter": "as of"}
    counts = {}
    latest = {}
    for record in quality_by_id.values():
        source = record.get("source")
        if source not in labels:
            continue
        counts[source] = counts.get(source, 0) + 1
        value = record.get("as_of")
        if value:
            current = latest.get(source)
            if current is None or _later_as_of(value, current):
                latest[source] = value
    blocks = []
    for source in ("aa", "catalog", "openrouter"):
        count = counts.get(source, 0)
        if not count:
            continue
        plural = "" if count == 1 else "s"
        stamp = _footer_as_of(source, latest.get(source))
        if stamp:
            blocks.append(
                f"{labels[source]} ({count} model{plural}, {stamp_words[source]} {stamp})"
            )
        else:
            blocks.append(f"{labels[source]} ({count} model{plural})")
    if not blocks:
        return None
    if aa_index_version is not None and counts.get("aa"):
        # Only the AA block gets the index-version prefix; a run whose first
        # (or only) source is the OpenRouter snapshot must not be mislabelled
        # as an AA index version.
        blocks[0] = f"AA index v{aa_index_version} - {blocks[0]}"
    clauses = []
    if unmatched:
        plural = "" if unmatched == 1 else "s"
        clauses.append(
            f"{unmatched} model{plural} unmatched (no quality data; quality score 0)"
        )
    drift = sum(1 for record in quality_by_id.values() if record.get("drift"))
    if drift:
        rows = "row" if drift == 1 else "rows"
        clauses.append(
            f"drift vs OpenRouter snapshot: {drift} comparable (non-reasoning) {rows}"
        )
    return {"sources": blocks, "clauses": clauses, "warn": bool(unmatched)}


def quality_source_lines(note) -> list[str]:
    """Render the footer as one source per line, 16-space continuation."""
    if not note or not note.get("sources"):
        return [QUALITY_SOURCE_LABEL + "unavailable; ranked on price/context/age"]
    indent = " " * len(QUALITY_SOURCE_LABEL)
    sources = note["sources"]
    clauses = note.get("clauses") or []
    lines = []
    for index, source in enumerate(sources):
        prefix = QUALITY_SOURCE_LABEL if index == 0 else indent
        if index < len(sources) - 1:
            lines.append(prefix + source + " |")
        elif clauses:
            lines.append(prefix + source + ";")
        else:
            lines.append(prefix + source)
    lines.extend(indent + clause for clause in clauses)
    return lines


TABLE_COLUMNS = {
    "expert": (
        "RANK",
        "MODEL",
        "DEPTH",
        "QUAL",
        "$IN/M",
        "$OUT/M",
        "DISC",
        "PREC",
        "TIER",
        "CTX",
        "AGE",
        "SCORE",
    ),
    "simple": ("RANK", "MODEL", "QUALITY", "IN $/M", "OUT $/M", "DISCOUNT", "AGE"),
}

# Legend text carries inline style markers so one rule set covers every
# legend (main, depth, deals): {c:...} = CLI option/typed token (cyan),
# {d:...} = literal value or unknown-data placeholder (dim), {y:...} =
# symbol marker (yellow). Wanting another emphasis means adding a marker
# kind here, never a one-off style call at a call site.
LEGEND_MARKER_RE = re.compile(r"\{([bcdy]):([^{}]*)\}")


def legend_plain(text: str) -> str:
    """Legend text with its style markers stripped (for width maths)."""
    return LEGEND_MARKER_RE.sub(r"\2", text)


def render_legend(text: str, style) -> str:
    """Expand a legend line's inline markers into the active Style."""
    methods = {
        "b": style.bold,
        "c": style.cyan,
        "d": style.dim,
        "y": style.yellow,
    }
    out, pos = [], 0
    for match in LEGEND_MARKER_RE.finditer(text):
        out.append(text[pos : match.start()])
        out.append(methods[match.group(1)](match.group(2)))
        pos = match.end()
    out.append(text[pos:])
    return "".join(out)


LEGENDS = {
    "expert": (
        (
            "QUALITY",
            [
                "AA intelligence index at the model's best measured selectable depth",
                '({d:default}; {c:--reasoning-effort} overrides); {d:"-"} = unknown, scores 0',
            ],
        ),
        (
            "DEPTH",
            [
                'reasoning effort QUAL was measured at; {y:"*"} = OpenRouter\'s default (informational)',
                "{d:-} = no known effort; {c:--depths} shows the depth detail",
            ],
        ),
        (
            "$IN/$OUT",
            [
                "USD per 1M tokens at the cheapest eligible endpoint, blended",
                "{d:{in_pct}} input / {d:{out_pct}} output ({c:--input-share}); differs from the",
                "openrouter.ai default-endpoint price by design. A higher rate",
                "may apply once the prompt outgrows a model's short-context tier.",
            ],
        ),
        ("DISC", ["active discount"]),
        (
            "PREC",
            [
                'quoted precision ({y:"!"} = default routing favors cheaper',
                "sub-default endpoints)",
            ],
        ),
        (
            "TIER",
            [
                "how prices vary: {d:>Nk} = pricier above that prompt size; {d:-N%} =",
                "scheduled off-peak discount (price shown: the standard rate);",
                "{d:SCHED} = schedule with a discount under 1% ({c:--off-peak} lists them)",
            ],
        ),
        ("CTX", ["context window; {b:AGE} = days since listed on OpenRouter"]),
        ("SCORE", ["weighted blend of the four criteria above"]),
    ),
    "simple": (
        ("RANK", ["position in this ranking"]),
        ("MODEL", ["provider/model id as listed; {c:--best} prefixes openrouter/"]),
        (
            "QUALITY",
            [
                "AA intelligence index at the model's best measured selectable depth",
                "{d:'-'} = unknown, scores 0",
                "suffix after · names the effort used for QUAL; this view has no",
                'DEPTH column or {y:"*"} marker',
            ],
        ),
        (
            "IN/OUT $/M",
            [
                "USD per 1M tokens at the cheapest eligible endpoint, blended",
                "{d:{in_pct}} input / {d:{out_pct}} output ({c:--input-share})",
            ],
        ),
        ("DISCOUNT", ["active discount, if any"]),
        ("AGE", ["days since listed on OpenRouter"]),
    ),
}


def fmt_quality(value) -> str:
    return "-" if value is None else f"{value:.1f}"


_CONTROL_CHARS_RE = re.compile(r"[\x00-\x1f\x7f-\x9f]")


def display_text(value) -> str:
    """Return untrusted text safe to print: strip C0/C1 control characters.

    Model ids come from the OpenRouter API. An id carrying ESC (SGR/CSI
    injection) or a newline (forged output lines) must never reach the
    terminal raw. Normal ids pass through unchanged.
    """
    return _CONTROL_CHARS_RE.sub("", str(value))


def table_cells(cand, rank):
    effort = (cand.get("quality_record") or {}).get("effort")
    quality_cell = fmt_quality(cand["quality"])
    if effort and cand["quality"] is not None:
        # Simple view has no DEPTH column; the effort rides on QUALITY.
        quality_cell += f" · {effort}"
    return {
        "RANK": str(rank),
        "MODEL": display_text(cand["id"]),
        "DEPTH": fmt_depth(cand),
        "QUAL": fmt_quality(cand["quality"]),
        "QUALITY": quality_cell,
        "$IN/M": fmt_price(cand["price_in"]),
        "IN $/M": fmt_price(cand["price_in"]),
        "$OUT/M": fmt_price(cand["price_out"]),
        "OUT $/M": fmt_price(cand["price_out"]),
        "DISC": fmt_discount(cand["discount"]),
        "DISCOUNT": fmt_discount(cand["discount"]),
        "PREC": fmt_prec(cand),
        "TIER": cand["tier_note"] or cand["sched_note"] or "--",
        "CTX": fmt_context(cand["context"]),
        "AGE": fmt_age(cand["age_days"]),
        "SCORE": f"{cand['score']:.3f}",
    }


def format_weights(weights) -> str:
    """Render weights as integer percentages that sum to exactly 100.

    Naive per-item ``:.0%`` rounding can print 101% (or 99%) because each
    share rounds independently. Largest-remainder rounding floors every
    share, then hands the leftover points to the largest fractional parts
    (deterministic name tie-break) so the line is always self-consistent.
    """
    items = sorted(weights.items(), reverse=True)
    total = sum(value for _, value in items)
    if total <= 0:
        return ", ".join(f"{name} 0%" for name, _ in items)
    exact = [(name, value / total * 100) for name, value in items]
    floors = [
        (name, math.floor(share), share - math.floor(share)) for name, share in exact
    ]
    remainder = 100 - sum(floor for _, floor, _ in floors)
    order = sorted(range(len(floors)), key=lambda i: (-floors[i][2], floors[i][0]))
    bumped = set(order[:remainder])
    return ", ".join(
        f"{name} {floor + (1 if index in bumped else 0)}%"
        for index, (name, floor, _) in enumerate(floors)
    )


def print_table(
    top,
    weights,
    quality_note,
    detail="expert",
    style=None,
    bare=False,
    input_share=0.75,
    depths=False,
):
    style = style or Style(False)
    columns = TABLE_COLUMNS[detail]
    rows = [table_cells(cand, rank) for rank, cand in enumerate(top, 1)]
    widths = {col: max([len(col)] + [len(row[col]) for row in rows]) for col in columns}

    def pad(col, value):
        if col == "MODEL":
            return value.ljust(widths[col])
        return value.rjust(widths[col])

    header = []
    for col in columns:
        text = col.ljust(widths[col])
        header.append(style.bold(text))
    print("  ".join(header))
    print(style.dim("  ".join("-" * widths[col] for col in columns)))
    for rank, row in enumerate(rows, 1):
        cells = []
        for col in columns:
            text = pad(col, row[col])
            # The risk marker is a symbol marker: yellow in the cell just
            # as the legend paints it, inside or outside the green #1 row.
            risk_marker = col == "PREC" and row[col].endswith("!")
            body = text[:-1] if risk_marker else text
            if col == "RANK":
                body = style.bold(body)
                if rank == 1:
                    body = style.green(body)
            elif rank == 1:
                body = style.green(body)
            elif col in ("QUAL", "QUALITY") and row[col] == "-":
                body = style.dim(body)
            cells.append(body + (style.yellow("!") if risk_marker else ""))
        print("  ".join(cells))
    if bare:
        return
    print()
    print("Weights: " + format_weights(weights))
    warn = not quality_note or quality_note.get("warn")
    for line in quality_source_lines(quality_note):
        print(style.yellow(line) if warn else line)
    # The column legend sits right under the table it explains; the depth
    # details follow it (issue #28).
    print()
    legend = LEGENDS[detail]
    width = max(len(legend_plain(term)) for term, _ in legend)
    in_pct = f"{input_share:.0%}"
    out_pct = f"{1 - input_share:.0%}"
    for term, lines in legend:
        # Percentages are substituted before the style markers are parsed,
        # so {d:{in_pct}} becomes {d:75%}.
        lines = [
            line.replace("{in_pct}", in_pct).replace("{out_pct}", out_pct)
            for line in lines
        ]
        label = legend_plain(term).ljust(width)
        print(style.bold(label) + "  " + render_legend(lines[0], style))
        for extra in lines[1:]:
            print(" " * (width + 2) + render_legend(extra, style))
    if depths and top:
        # The separator only prints when a depth section will follow:
        # print_depth_detail returns early on an empty top.
        print()
        print_depth_detail(top, style)


DEALS_COLUMNS = (
    "MODEL",
    "OFF-PEAK",
    "SCHEDULE",
    "QUAL",
    "$IN/M",
    "$OUT/M",
    "CTX",
    "SCORE",
)

DEALS_LEGEND = (
    ("OFF-PEAK", "largest scheduled discount vs the standard rate"),
    ("SCHEDULE", "when the discounted rate applies (UTC)"),
)


def print_deals(rows, style=None, bare=False):
    style = style or Style(False)
    if not rows:
        if not bare:
            print()
        print("No scheduled off-peak discounts in the current data.")
        return
    cells = [
        {
            "MODEL": display_text(row["id"]),
            "OFF-PEAK": row["sched_note"],
            "SCHEDULE": row["sched_detail"] or "--",
            "QUAL": fmt_quality(row["quality"]),
            "$IN/M": fmt_price(row["offpeak"]["input_per_1m"]),
            "$OUT/M": fmt_price(row["offpeak"]["output_per_1m"]),
            "CTX": fmt_context(row["context"]),
            "SCORE": "-" if row["score"] is None else f"{row['score']:.3f}",
        }
        for row in rows
    ]
    widths = {
        col: max([len(col)] + [len(cell[col]) for cell in cells])
        for col in DEALS_COLUMNS
    }

    def pad(col, value):
        if col in ("MODEL", "SCHEDULE"):
            return value.ljust(widths[col])
        return value.rjust(widths[col])

    if not bare:
        print()
        print(
            style.bold(
                "Off-peak deals (cheapest off-peak window's prices; "
                "the standard rate applies at all other times)"
            )
        )
        print()
    header = []
    for col in DEALS_COLUMNS:
        # Every deals header is left-aligned (matching the main table and
        # the frozen mockup); only the data cells keep numeric right-align.
        text = col.ljust(widths[col])
        header.append(style.bold(text))
    print("  ".join(header))
    print(style.dim("  ".join("-" * widths[col] for col in DEALS_COLUMNS)))
    for cell in cells:
        out = []
        for col in DEALS_COLUMNS:
            text = pad(col, cell[col])
            if col == "OFF-PEAK":
                text = style.bold(text)
            elif col == "QUAL" and cell[col] == "-":
                text = style.dim(text)
            out.append(text)
        print("  ".join(out))
    if bare:
        return
    print()
    width = max(len(legend_plain(term)) for term, _ in DEALS_LEGEND)
    for term, text in DEALS_LEGEND:
        label = legend_plain(term).ljust(width)
        print(style.bold(label) + "  " + render_legend(text, style))


DEPTH_LEGEND = (
    ("effort", ["reasoning effort this row measures (deepest first)"]),
    (
        "AA idx",
        ['AA intelligence index at that effort; {d:"-"} = not measured'],
    ),
    (
        "AA $/task",
        [
            "AA's measured benchmark cost at AA's reference price, not",
            "the OR per-token price; it never enters the weighted score",
        ],
    ),
    ("vs default", ["AA $/task as a ratio of the default effort's cost"]),
    (
        "select",
        [
            'the OpenCode variant suffix that pins the depth; {y:"-"} = not',
            'selectable; {y:"*"} marks OpenRouter\'s default, and appending',
            "{c:#effort} to the model id pins it in OpenCode",
        ],
    ),
    ("OR $/M", ["constant across depths (the title line's per-listing price)"]),
    (
        "(no label)",
        [
            "an AA index measured without an effort label is a reference",
            "only; it never fills QUAL",
        ],
    ),
)


def print_depth_detail(top, style=None):
    """Per-model AA depth detail: measured scores and selectable variants.

    One block per model in the given (rank) order, listing the union of
    OpenRouter-supported and AA-measured efforts, deepest first. Models with
    no measured depths get a single ``no AA depth data`` line, unless AA
    measured them without an effort label: that index is shown with a
    reference-only caveat instead. Display only: AA $/task never enters the
    weighted score.
    """
    if not top:
        return

    headers = ["effort", "AA idx", "AA $/task", "vs default", "select"]

    def finite(value):
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
        ):
            return value
        return None

    def order(effort):
        try:
            return (AA_EFFORT_ORDER.index(effort), effort)
        except ValueError:
            return (len(AA_EFFORT_ORDER), effort or "")

    def unlabeled_note(record):
        """One line for an AA index measured without an effort label.

        Display only: the number cannot fill QUAL, so the note says what it
        is and warns that a selectable effort may score differently.
        """
        if not isinstance(record, dict):
            return None
        index = finite(record.get("index"))
        if index is None:
            return None
        cost = finite(record.get("cost_per_task_usd"))
        text = f"AA index {index:.1f} without an effort label"
        if cost is not None:
            text += f" (AA ${cost:.2f} per task)"
        return (
            text + "; reference only, may not be the quality you get at a "
            "selectable effort."
        )

    blocks = []
    for rank, cand in enumerate(top, 1):
        reasoning = cand.get("reasoning") or {}
        supported = list(reasoning.get("supported_efforts") or [])
        records = [
            record
            for record in (cand.get("reasoning_depths") or [])
            if isinstance(record, dict) and record.get("effort")
        ]
        measured = {record["effort"]: record for record in records}
        default = _display_default_effort(cand)
        is_non_reasoning = reasoning.get("is_reasoning") is False
        # A non-mandatory single-state model exposes only the "none" toggle.
        single_state_none = (
            reasoning.get("is_reasoning") is True
            and reasoning.get("mandatory") is not True
            and not supported
        )
        default_cost = (
            finite((measured.get(default) or {}).get("cost_per_task_usd"))
            if default
            else None
        )
        efforts = sorted(set(supported) | set(measured), key=order)
        rows = []
        for effort in efforts:
            record = measured.get(effort) or {}
            index = finite(record.get("index"))
            cost = finite(record.get("cost_per_task_usd"))
            if effort == "reasoning":
                # Single-state reasoning row: the bare id is the only form.
                select = "-"
            elif effort == "none" and is_non_reasoning:
                # A non-reasoning model's bare id is the only form.
                select = "-"
            elif effort in supported:
                select = f"#{effort}"
            elif effort == "none" and single_state_none:
                select = "#none"
            else:
                select = "- not selectable"
            ratio = "-"
            if cost and default_cost:
                value = cost / default_cost
                ratio = f"{value:.1f}x" if value >= 1 else f"{value:.2f}x"
            rows.append(
                [
                    f"{effort}*" if effort == default else effort,
                    "-" if index is None else f"{index:.1f}",
                    "-" if cost is None else f"{cost:.2f}",
                    ratio,
                    select,
                ]
            )
        has_measured = any(
            finite(record.get("index")) is not None for record in measured.values()
        )
        blocks.append(
            {
                "rank": rank,
                "cand": cand,
                "default": default,
                "rows": rows,
                "has_records": bool(records),
                "missing_default": (
                    bool(default)
                    and default in efforts
                    and finite((measured.get(default) or {}).get("index")) is None
                    and has_measured
                ),
            }
        )

    widths = [len(header) for header in headers]
    for block in blocks:
        for row in block["rows"]:
            for i, cell in enumerate(row):
                widths[i] = max(widths[i], len(cell))

    style = style or Style(False)
    # Columns whose "-" means "no data" get dim; select's "-" is a value,
    # not a placeholder, so that column is left plain (issue #28).
    value_columns = {1, 2, 3}

    def pad_cells(cells):
        return [
            cell.ljust(width) if index < len(widths) - 1 else cell
            for index, (cell, width) in enumerate(zip(cells, widths))
        ]

    def emit_header():
        print("   " + "   ".join(style.bold(cell) for cell in pad_cells(headers)))

    def emit_rule():
        print("   " + style.dim("   ".join("-" * width for width in widths)))

    def emit_row(row):
        out = []
        for index, (cell, padded) in enumerate(zip(row, pad_cells(row))):
            if index == 0 and cell.endswith("*"):
                # The default-effort marker is a symbol marker (yellow),
                # like the main legend's.
                body = padded[: len(cell) - 1]
                out.append(body + style.yellow("*") + padded[len(cell) :])
            elif index in value_columns and cell.strip() == "-":
                out.append(style.dim(padded))
            else:
                out.append(padded)
        print("   " + "   ".join(out))

    print(style.bold("Depth details"))
    for position, block in enumerate(blocks):
        if position:
            print()
        cand = block["cand"]
        pricing = f"OR ${cand['price_in']:.2f} in / ${cand['price_out']:.2f} out per 1M"
        print(
            f"#{block['rank']} {style.bold(cand['id'])}   "
            f"{style.dim(pricing)}   "
            f"{style.dim('default:')} {block['default'] or '-'}"
        )
        note = unlabeled_note(cand.get("quality_unlabeled"))
        if note:
            print("   " + style.dim(note))
        if not block["has_records"]:
            if not note:
                print("   " + style.dim("no AA depth data"))
            continue
        if block["missing_default"]:
            print(f"   AA has no {block['default']} record for this model.")
        emit_header()
        emit_rule()
        for row in block["rows"]:
            emit_row(row)

    print()
    width = max(len(legend_plain(term)) for term, _ in DEPTH_LEGEND)
    for term, lines in DEPTH_LEGEND:
        label = legend_plain(term).ljust(width)
        print(style.bold(label) + "  " + render_legend(lines[0], style))
        for extra in lines[1:]:
            print(" " * (width + 2) + render_legend(extra, style))


def print_best(top, args):
    """Print the top candidate as an OpenCode model id.

    A non-default ``--reasoning-effort`` appends ``#<effort>`` only when the
    model exposes that variant; otherwise the bare id selects the default.
    """
    cand = top[0]
    model_id = opencode_model_id(cand["id"])
    effort = args.reasoning_effort
    reasoning = cand.get("reasoning")
    if (
        effort != "default"
        and isinstance(reasoning, dict)
        and reasoning.get("is_reasoning") is not False
        and supports_effort(reasoning, effort)
    ):
        model_id += f"#{effort}"
    print(display_text(model_id))


def print_json(top):
    payload = [
        {
            "model": cand["id"],
            "opencode_model": opencode_model_id(cand["id"]),
            "name": cand["name"],
            "score": round(cand["score"], 4),
            "quality_index": cand["quality"],
            # Provenance block, verbatim from the quality record resolved
            # in compute_scores: the winning tier's source and as_of plus
            # the drift check against OpenRouter's benchmarks. Rows without
            # a record (pre-endpoint test paths) degrade to None.
            "quality_source": (cand.get("quality_record") or {}).get("source"),
            "quality_as_of": (cand.get("quality_record") or {}).get("as_of"),
            "quality_drift": (cand.get("quality_record") or {}).get("drift"),
            "quality_or_index": (cand.get("quality_record") or {}).get("or_index"),
            "quality_or_as_of": (cand.get("quality_record") or {}).get("or_as_of"),
            # The effort the quality record was measured at. Legacy records
            # (pre-depth paths) carry no "effort" key, so this degrades to
            # None exactly like a missing record.
            "quality_effort": (cand.get("quality_record") or {}).get("effort"),
            # The model's effective default effort, duplicated at top level:
            # data.json rows must keep it after the nested reasoning block is
            # dropped downstream. Same resolver as the DEPTH cell.
            "default_effort": _display_default_effort(cand),
            # The six published reasoning fields plus the measured depth
            # records; the internal is_reasoning flag never leaves the
            # process (catalog v6 and spec §8.4 publish only these keys).
            "reasoning": _published_reasoning_block(cand),
            "input_usd_per_m": round(cand["price_in"], 6),
            "output_usd_per_m": round(cand["price_out"], 6),
            "blended_usd_per_m": round(cand["blended"], 6),
            "discount": round(cand["discount"], 4)
            if has_discount(cand["discount"])
            else None,
            # Display string from the DISC column's formatter, so the site
            # shows exactly what the CLI prints instead of re-rounding.
            "discount_pct": fmt_discount(cand["discount"]),
            # Tier/schedule fields follow the same precedent: the strings the
            # site shows are these, never re-formatted downstream.
            "pricing_tier_prompt_tokens": cand["tier_prompt_tokens"],
            "base_input_usd_per_m": round(cand["base_price_in"], 6),
            "base_output_usd_per_m": round(cand["base_price_out"], 6),
            "tier_note": cand["tier_note"],
            "sched_note": cand["sched_note"],
            "sched_detail": cand["sched_detail"],
            "time_schedule": (
                cand["sched_note"] + " " + cand["sched_detail"]
                if cand["sched_detail"]
                else cand["sched_note"]
            ),
            "context_tokens": cand["context"],
            "age_days": round(cand["age_days"], 1)
            if cand["age_days"] is not None
            else None,
            # Additive keys (Task 9's ROW_KEYS extension pins this exact
            # key set and order). Risk/spread come from the annotations
            # block; session_cost_usd is run()'s per-candidate slice of the
            # Task 7 envelope, shown for the display turn range (run()
            # attaches it before any output path). Candidates that never
            # went through either stage (historical test paths) degrade to
            # null, as does a model the dataset covers only in other turn
            # ranges.
            "default_precision": cand.get("default_precision"),
            "risk_flag": (cand.get("annotations") or {}).get("risk_flag"),
            "price_spread": (cand.get("annotations") or {}).get("price_spread"),
            "session_cost_usd": (cand.get("session_cost") or {}).get(
                SESSION_COST_DISPLAY_TURN_RANGE
            ),
            # The cheapest-sub-default note, exactly as the CLI prints it
            # (Task 11 shows it verbatim). Emit-first: Task 9's ROW_KEYS
            # pins this key last; pre-endpoint candidates degrade to None.
            "quant_note": fmt_quant_note(cand.get("annotations")),
            # One extra row key AFTER quant_note -- steering is NOT in
            # ROW_KEYS. build_site_data's row check is missing-key-only, so
            # extras pass through (tier_note precedent) and the site
            # ignores it. The OpenRouter provider routing object (design
            # section 4.7) the plugin contract reads; pre-endpoint
            # candidates degrade to None.
            "steering": cand.get("steering"),
        }
        for cand in top
    ]
    print(json.dumps(payload, indent=2))


def build_catalog(
    args,
    models,
    candidates,
    dropped,
    filtered,
    discounts,
    quality_by_id,
    aa_source,
    aa_by_id,
    quality_source_by_id,
    session_cost=None,
    session_cost_mode="skipped",
    index_version=None,
    catalog_generated_at=None,
    catalog_quality=None,
    *,
    depth_lookup=None,
    catalog_depths=None,
    reasoning_by_id=None,
    requested_effort="default",
    schedule_entries=None,
):
    """Assemble the full-catalog document (see README "Catalog output").

    Pure: reads candidates post-compute_scores (which already carry the
    component scores) and never mutates them. Deterministic apart from
    generated_at: models sort by (-overall.balanced, id), filtered by id,
    and age uses date precision so two runs in the same UTC day match.

    session_cost is the Task 7 fetch envelope (verbatim, medians included)
    or None; session_cost_mode is run()'s resolution of the key-presence
    fact ("ok" / "unavailable" / "skipped") -- build_catalog stays pure and
    never fetches or inspects the environment itself. index_version is the
    AA envelope's intelligence_index_version (number or null) and
    catalog_generated_at the published catalog's generated_at when this
    run hydrated from it (else None); both pass verbatim into sources.aa.
    catalog_quality is the hydrated record map (or None): candidates read
    it through resolve_quality, and build_schedules falls back to it for
    non-candidates.
    The keyword-only depth inputs feed build_schedules (spec 6.5); run()
    always supplies them, direct callers may omit them for the legacy path.
    """
    now = datetime.now(timezone.utc)
    weights = catalog_weights(candidates, quality_by_id)
    if aa_source is not None and aa_source != "AA API v2":
        # Never publish a document whose AA source we cannot name: the
        # consumers' mode enum has no slot for it.
        raise ValueError(f"unknown AA source: {aa_source!r}")
    # The three counts are disjoint label buckets: resolve_quality's strict
    # rank gives every candidate at most one record, so their sum is exactly
    # the number of entries with a non-null quality_match -- the invariant
    # the site validator enforces. A label outside the buckets would break
    # it silently, so it fails loudly here.
    matched_aa = sum(1 for s in quality_source_by_id.values() if s == "aa")
    matched_catalog = sum(1 for s in quality_source_by_id.values() if s == "catalog")
    matched_openrouter = sum(
        1 for s in quality_source_by_id.values() if s == "openrouter"
    )
    unknown = sorted(
        {
            s
            for s in quality_source_by_id.values()
            if s not in ("aa", "catalog", "openrouter")
        }
    )
    if unknown:
        raise ValueError(f"unknown quality source: {unknown[0]!r}")
    # Depth counts are candidate-scoped and separate from the disjoint label
    # buckets above (spec 9.1): schedule entries carry their own provenance
    # and never enter these counts or the pool invariants. matched_default
    # is the number of main quality records under strict selection -- exactly
    # the label buckets' sum, kept explicit for the publish gate.
    matched_default = matched_aa + matched_catalog + matched_openrouter
    depth_rows = sum(len(cand.get("reasoning_depths") or []) for cand in candidates)
    matched_depth = sum(1 for cand in candidates if cand.get("reasoning_depths"))
    # mode is the run's quality chain, not a match outcome: "aa" when the
    # AA API returned usable entries, "catalog" when unkeyed runs hydrated
    # the published catalog, "none" when neither tier produced data.
    # OpenRouter's benchmarks are the chain's shared last resort, never a
    # mode of their own -- matched_openrouter carries their count. The
    # catalog branch also fires on the counts alone: hydrated records are
    # proof of a usable catalog even when its generated_at was unusable.
    if aa_source is not None:
        aa_mode = "aa"
    elif catalog_generated_at is not None or matched_catalog or matched_aa:
        aa_mode = "catalog"
    else:
        aa_mode = "none"

    entries = []
    for cand in candidates:
        provider, _, _base = cand["id"].partition("/")
        created = cand["created"] or 0.0
        listed_date = (
            datetime.fromtimestamp(created, tz=timezone.utc).date()
            if created > 0
            else None
        )
        # Round the component scores first, then derive overall from the
        # rounded values so scores.overall is exactly reproducible from the
        # document's own numbers.
        scores = {
            "price": round(cand["price_score"], 4),
            "quality": round(cand["quality_score"], 4),
            "context": round(cand["context_score"], 4),
            "age": round(cand["age_score"], 4),
        }
        overall = overall_scores(cand, weights)
        bench = aa_by_id.get(base_model_id(cand["id"])) or {}
        # The quality record map is already a parameter; the winning
        # record's as_of and drift verdict become entry fields verbatim.
        # Unmatched candidates degrade to null -- same shape as the v3
        # additive blocks below. The candidate's own record wins (it is the
        # same object compute_scores attached), with the map as the fallback
        # for direct callers that skip compute_scores.
        record = cand.get("quality_record")
        if not isinstance(record, dict):
            record = quality_by_id.get(cand["id"])
            record = record if isinstance(record, dict) else None
        reasoning = cand.get("reasoning")
        reasoning = reasoning if isinstance(reasoning, dict) else {}
        depths = [
            {
                "effort": depth.get("effort"),
                "index": depth.get("index"),
                "cost_per_task_usd": depth.get("cost_per_task_usd"),
                "source": depth.get("source"),
                "as_of": depth.get("as_of"),
            }
            for depth in (cand.get("reasoning_depths") or [])
            if isinstance(depth, dict)
        ]
        entries.append(
            {
                "id": cand["id"],
                "name": PROVIDER_PREFIX_RE.sub("", cand["name"]).strip()
                or cand["name"],
                "provider": provider,
                "family": model_family(cand["id"]),
                "pricing": {
                    "input_per_1m": round(cand["price_in"], 6),
                    "output_per_1m": round(cand["price_out"], 6),
                    "blended_per_1m": round(cand["blended"], 6),
                    "input_cache_read_per_1m": cand.get("cache_read_per_1m"),
                    "input_cache_write_per_1m": cand.get("cache_write_per_1m"),
                    "base": {
                        "input_per_1m": round(cand["base_price_in"], 6),
                        "output_per_1m": round(cand["base_price_out"], 6),
                        "blended_per_1m": round(
                            args.input_share * cand["base_price_in"]
                            + (1 - args.input_share) * cand["base_price_out"],
                            6,
                        ),
                        "input_cache_read_per_1m": cand.get("base_cache_read_per_1m"),
                        "input_cache_write_per_1m": cand.get("base_cache_write_per_1m"),
                    },
                    "tiers": [
                        {
                            "min_prompt_tokens": tier["min_prompt_tokens"],
                            "input_per_1m": _per_1m(tier["price_in"]),
                            "output_per_1m": _per_1m(tier["price_out"]),
                            "blended_per_1m": _per_1m(
                                args.input_share * tier["price_in"]
                                + (1 - args.input_share) * tier["price_out"]
                            ),
                            "input_cache_read_per_1m": (
                                _per_1m(tier["input_cache_read"])
                                if tier["input_cache_read"] is not None
                                else None
                            ),
                            "input_cache_write_per_1m": (
                                _per_1m(tier["input_cache_write"])
                                if tier["input_cache_write"] is not None
                                else None
                            ),
                        }
                        for tier in cand["tiers"]
                    ],
                    "schedule": (
                        [
                            {
                                "utc_days": window["utc_days"],
                                "utc_start": window["utc_start"],
                                "utc_end": window["utc_end"],
                                "coverage": window["coverage"],
                                "input_per_1m": _per_1m(window["price_in"]),
                                "output_per_1m": _per_1m(window["price_out"]),
                                "blended_per_1m": _per_1m(window["blended"]),
                                "input_cache_read_per_1m": (
                                    _per_1m(window["input_cache_read"])
                                    if window["input_cache_read"] is not None
                                    else None
                                ),
                                "input_cache_write_per_1m": (
                                    _per_1m(window["input_cache_write"])
                                    if window["input_cache_write"] is not None
                                    else None
                                ),
                            }
                            for window in cand["schedule"]
                        ]
                        if cand["schedule"]
                        else None
                    ),
                },
                "context": cand["context"],
                "max_output_tokens": cand.get("max_output_tokens"),
                "input_modalities": cand.get("input_modalities") or [],
                "listed_at": listed_date.isoformat() if listed_date else None,
                "age_days": max(0, (now.date() - listed_date).days)
                if listed_date
                else None,
                "tool_calling": cand["tool_calling"],
                "zdr": cand["zdr"],
                "discount": round(cand["discount"], 4)
                if has_discount(cand["discount"])
                else None,
                "expired": cand["expired"],
                "aa": {
                    "intelligence_index": bench.get("intelligence_index"),
                    "coding_index": bench.get("coding_index"),
                    "agentic_index": bench.get("agentic_index"),
                },
                "quality": cand["quality"],
                "quality_match": quality_source_by_id.get(cand["id"]),
                # Provenance (schema v4), verbatim from the record: the
                # winning source's own as_of and the drift verdict against
                # OpenRouter's check value.
                "quality_as_of": record.get("as_of") if record else None,
                "quality_drift": record.get("drift") if record else None,
                # Schema v5 depth contract: the effort the main quality was
                # measured at, and the model's reasoning metadata plus every
                # measured depth record (record shape: effort/index/
                # cost_per_task_usd/source/as_of -- `index`, never
                # `quality_index`).
                "quality_effort": record.get("effort") if record else None,
                "reasoning": {
                    "default_effort": reasoning.get("default_effort"),
                    "supported_efforts": reasoning.get("supported_efforts") or [],
                    "default_enabled": reasoning.get("default_enabled"),
                    "mandatory": reasoning.get("mandatory"),
                    "supports_max_tokens": reasoning.get("supports_max_tokens"),
                    "depths": depths,
                    "aa_unlabeled": _published_unlabeled_record(cand),
                },
                "scores": {**scores, "overall": overall},
                # Additive provider-aware blocks (schema v3), verbatim from
                # the candidate. Candidates that never went through the
                # endpoint stage (historical test paths) degrade to null --
                # .get() with null defaults for these NEW keys only; the
                # pre-existing keys above stay direct-indexed.
                "endpoints": cand.get("endpoints"),
                "default_precision": cand.get("default_precision"),
                "advertised_pricing": cand.get("advertised_pricing"),
                "annotations": cand.get("annotations"),
                # The OpenRouter provider routing object (design section
                # 4.7) the plugin contract reads -- same null-default
                # degradation shape as the blocks above.
                "steering": cand.get("steering"),
            }
        )
    entries.sort(key=lambda e: (-e["scores"]["overall"]["balanced"], e["id"]))

    balanced_by_id = {
        entry["id"]: entry["scores"]["overall"]["balanced"] for entry in entries
    }

    return {
        "schedules": build_schedules(
            models,
            args,
            quality_by_id,
            aa_by_id,
            balanced_by_id,
            catalog_quality,
            depth_lookup=depth_lookup,
            catalog_depths=catalog_depths,
            reasoning_by_id=reasoning_by_id,
            requested_effort=requested_effort,
            schedule_entries=schedule_entries,
        ),
        "schema_version": CATALOG_SCHEMA_VERSION,
        "tool": "model-compare",
        "generated_at": now.isoformat(timespec="seconds"),
        "parameters": {
            "input_share": args.input_share,
            "quality_ref": args.quality_ref,
            "min_context": args.min_context,
            "recency_half_life": args.recency_half_life,
            "max_age_days": args.max_age_days,
            "zdr_required": not args.no_zdr,
            "require_tools": not args.no_require_tools,
            "exclude_free": args.exclude_free,
            "include_batch": args.include_batch,
            # The requested effort documents the semantic of every model's
            # quality value (default = best measured selectable effort).
            "reasoning_effort": getattr(args, "reasoning_effort", "default"),
            "weights": weights,
        },
        "sources": {
            "openrouter": "ok",
            "aa": {
                "mode": aa_mode,
                "matched_aa": matched_aa,
                "matched_catalog": matched_catalog,
                "matched_openrouter": matched_openrouter,
                # Depth coverage (schema v5), candidate-scoped like the
                # label buckets above. Schedule entries never contribute.
                "depth_rows": depth_rows,
                "matched_depth": matched_depth,
                "matched_default": matched_default,
                # Provenance meta (schema v4): the published catalog's stamp
                # when this run hydrated from it, and the AA envelope's
                # intelligence index version when the AA API ran. Both null
                # when their source was not consulted.
                "catalog_generated_at": catalog_generated_at,
                "intelligence_index_version": index_version,
            },
            "zdr": "skipped" if args.no_zdr else "ok",
            "discounts": "ok" if discounts else "unavailable",
            # The endpoint stage always ran before build_catalog is reached;
            # its failures surface as pool.dropped counts
            # (endpoint_fetch_failed / no_baseline_eligible_endpoint), never
            # as source state -- so this is deterministic "ok".
            "endpoints": "ok",
            # "ok" (envelope fetched) / "unavailable" (key configured, fetch
            # failed) / "skipped" (no key) -- resolved in run().
            "session_cost": session_cost_mode,
        },
        # Per-task session-cost medians (Task 7), verbatim from the fetch
        # envelope. Omitted entirely when unavailable: consumers branch on
        # key presence, never on a null shape.
        **({"session_cost": session_cost} if session_cost is not None else {}),
        "pool": {
            "listed": len(models),
            "candidates": len(candidates),
            "dropped": {
                reason: dropped.get(reason, 0) for reason in CATALOG_DROP_REASONS
            },
        },
        "models": entries,
        # The single ranking authority: every site artifact (data.json rows,
        # history tabs, the highlights diff, best.txt) projects from or is
        # checked against it. models keeps its own balanced sort.
        "rankings": catalog_rankings(candidates, weights),
        "filtered": sorted(filtered, key=lambda e: e["id"]),
    }


def print_catalog(document):
    print(json.dumps(document, indent=2))


# --- plain-text styling -----------------------------------------------------
# SGR attributes only: the terminal theme supplies the hues, so the output
# stays readable on light and dark themes alike.
SGR_RESET = "\x1b[0m"
SGR_BOLD = "\x1b[1m"
SGR_DIM = "\x1b[2m"
SGR_GREEN = "\x1b[32m"
SGR_YELLOW = "\x1b[33m"
# CLI options own an attribute of their own: bold is already the term
# label/header treatment, so a bold flag reads as another label.
SGR_CYAN = "\x1b[36m"


class Style:
    """Minimal SGR wrapper; every method is a no-op when disabled."""

    def __init__(self, enabled: bool):
        self.enabled = enabled

    def _wrap(self, text: str, code: str) -> str:
        return f"{code}{text}{SGR_RESET}" if self.enabled else text

    def bold(self, text: str) -> str:
        return self._wrap(text, SGR_BOLD)

    def dim(self, text: str) -> str:
        return self._wrap(text, SGR_DIM)

    def green(self, text: str) -> str:
        return self._wrap(text, SGR_GREEN)

    def yellow(self, text: str) -> str:
        return self._wrap(text, SGR_YELLOW)

    def cyan(self, text: str) -> str:
        return self._wrap(text, SGR_CYAN)


def color_enabled(mode: str, stream) -> bool:
    """Resolve --color for one stream. Explicit "on" beats NO_COLOR."""
    if mode == "off":
        return False
    if mode == "on":
        return True
    if os.environ.get("NO_COLOR"):
        return False
    return bool(getattr(stream, "isatty", lambda: False)())


class PagerInterrupted(Exception):
    """Ctrl-C reached the parent while the pager was running."""


def resolve_pager(args) -> list[str] | None:
    """The pager argv for the human output path, or None to print directly.

    None when --no-pager is set or stdout is not a TTY. $PAGER is used
    verbatim (shlex-split, no injected flags; an empty split counts as
    unset); the fallback pages only what exceeds one screen, keeps the SGR
    codes and skips the alt-screen init so the stderr notes survive
    quitting. POSIX terminals only.
    """
    if args.no_pager:
        return None
    if not getattr(sys.stdout, "isatty", lambda: False)():
        return None
    raw = os.environ.get("PAGER")
    argv = []
    if raw is not None:
        try:
            argv = shlex.split(raw)
        except ValueError:
            # Malformed $PAGER (unbalanced quote, trailing backslash):
            # pager startup problems never crash the run or lose output --
            # degrade to the fallback, exactly like an unset $PAGER.
            argv = []
    if not argv:
        argv = ["less", "--quit-if-one-screen", "--RAW-CONTROL-CHARS", "--no-init"]
    return argv


@contextmanager
def paged_stdout(argv: list[str]):
    """Run argv as a pager taking over sys.stdout; yield True when paged.

    False (direct output) when the pager cannot be started -- startup
    problems never lose output. Restores sys.stdout in a finally, closes
    the pager's stdin guarded against EPIPE and reaps the child. A
    KeyboardInterrupt from the block, the close or the reap terminates the
    pager (plain less ignores SIGINT without -K) and surfaces as
    PagerInterrupted; the pager's own exit status is never propagated.
    """
    if not argv:
        yield False
        return
    # Colors through a verbatim $PAGER: less reads $LESS, so a bare
    # `PAGER=less` would otherwise show raw SGR as escape junk. Git's
    # convention: hand the child LESS=FRX (raw control chars,
    # quit-if-one-screen, no alt-screen init) when the user has none;
    # their own $LESS always wins.
    child_env = {**os.environ, "LESS": "FRX"} if "LESS" not in os.environ else None
    # Captured before any try so the finally below can always restore it;
    # the swap itself is the try's first statement, so a Ctrl-C landing
    # anywhere from the spawn onward still runs the cleanup.
    original = sys.stdout
    interrupted = False
    try:
        proc = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=sys.stdout,
            text=True,
            env=child_env,
            # Write the pipe with the encoding (and error handler) the direct
            # path would have used: Popen's text-mode default comes from the
            # locale, which can disagree with sys.stdout.encoding (e.g.
            # PYTHONIOENCODING=utf-8 over a latin-1 locale) and then a
            # non-ASCII model name crashes only the paged path.
            encoding=getattr(sys.stdout, "encoding", None) or "utf-8",
            errors=getattr(sys.stdout, "errors", None),
        )
    except OSError:
        yield False
        return
    try:
        sys.stdout = proc.stdin
        yield True
    except KeyboardInterrupt:
        interrupted = True
    finally:
        sys.stdout = original
        try:
            proc.stdin.close()
        except KeyboardInterrupt:
            interrupted = True
        except OSError:
            pass
        try:
            if interrupted and proc.poll() is None:
                proc.terminate()
            proc.wait()
        except KeyboardInterrupt:
            interrupted = True
            if proc.poll() is None:
                proc.terminate()
            try:
                proc.wait()
            except KeyboardInterrupt:
                pass
    if interrupted:
        raise PagerInterrupted


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="model-compare",
        description="Pick the best value-for-money LLM on OpenRouter for coding-agent work.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(__doc__ or "").split("Examples:", 1)[-1],
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {VERSION}",
    )
    output = parser.add_argument_group("output")
    ranking = parser.add_argument_group("ranking")
    filters = parser.add_argument_group("filters")
    sources = parser.add_argument_group("data sources & cache")
    filters.add_argument(
        "--min-context",
        type=int,
        default=1_000_000,
        help="hard minimum context window in tokens (default: 1000000)",
    )
    ranking.add_argument(
        "--priority",
        choices=sorted(PRIORITY_WEIGHTS),
        default="balanced",
        help="ranking emphasis (default: balanced)",
    )
    ranking.add_argument(
        "--top", type=int, default=5, help="how many models to list (default: 5)"
    )
    output.add_argument(
        "--best",
        action="store_true",
        help="print only the #1 model as an opencode id (openrouter/<model>), for scripting",
    )
    output.add_argument(
        "--json", action="store_true", help="print the ranked candidates as JSON"
    )
    output.add_argument(
        "--catalog",
        action="store_true",
        help="print the full model catalog (ranked candidates and filtered-out models with reasons) as one JSON document; ignores --top and --priority, cannot be combined with --best or --json",
    )
    output.add_argument(
        "--detail",
        choices=("simple", "expert"),
        default="expert",
        help="table density (default: expert; simple hides PREC, TIER, CTX and "
        "SCORE, mimicking the web page's simple view)",
    )
    output.add_argument(
        "--off-peak",
        action="store_true",
        help="also list scheduled off-peak deals, ranked by discount size",
    )
    output.add_argument(
        "--bare",
        action="store_true",
        help="print just the table(s), no summary or legend",
    )
    output.add_argument(
        "--depths",
        action="store_true",
        help=(
            "also print the per-model depth detail below the table (the DEPTH "
            "and QUAL columns are always shown)"
        ),
    )
    output.add_argument(
        "--color",
        choices=("auto", "on", "off"),
        default="auto",
        help="colorize the table output (default: auto; honors NO_COLOR; the "
        "built-in pager keeps color on under auto, and on also forces color "
        "for manual pipes like less -R)",
    )
    output.add_argument(
        "--no-pager",
        action="store_true",
        help="do not page the table output even on a terminal",
    )
    ranking.add_argument(
        "--input-share",
        type=float,
        default=0.75,
        help="input-token share used for the blended price, 0-1 (default: 0.75)",
    )
    ranking.add_argument(
        "--recency-half-life",
        type=float,
        default=120.0,
        help="age decay half-life in days (default: 120)",
    )
    ranking.add_argument(
        "--max-age-days",
        type=float,
        default=0.0,
        help="drop models listed more than this many days ago (0 = off)",
    )
    ranking.add_argument(
        "--quality-ref",
        type=float,
        default=70.0,
        help="AA intelligence index that counts as full quality score (default: 70)",
    )
    ranking.add_argument(
        "--reasoning-effort",
        choices=REASONING_EFFORT_CHOICES,
        default="default",
        help=(
            "reasoning effort used for quality and filtering: a specific "
            "effort drops models that cannot run it (default scores each model "
            "at its best measured selectable depth)"
        ),
    )
    sources.add_argument(
        "--aa-api-key",
        default=None,
        help="Artificial Analysis API key (or set AA_API_KEY; free at artificialanalysis.ai)",
    )
    sources.add_argument(
        "--openrouter-api-key",
        default=None,
        help="OpenRouter API key (or set OPENROUTER_API_KEY; free at openrouter.ai/keys)",
    )
    filters.add_argument(
        "--no-require-tools",
        action="store_true",
        help="do not require tool-calling support (coding agents want it)",
    )
    filters.add_argument(
        "--exclude-free",
        action="store_true",
        help="drop zero-cost listings such as rate-limited :free variants",
    )
    filters.add_argument(
        "--include-batch",
        action="store_true",
        help="keep ':batch' variants (asynchronous completion; cheaper but unsuitable for interactive agents)",
    )
    filters.add_argument(
        "--discount",
        action="store_true",
        help="only list models with an active discount",
    )
    filters.add_argument(
        "--no-zdr",
        action="store_true",
        help="consider all models, not just zero-data-retention (ZDR) ones",
    )
    sources.add_argument(
        "--no-cache",
        action="store_true",
        help="bypass the local response cache and refetch",
    )
    sources.add_argument(
        "--cache-ttl",
        type=int,
        default=6 * 3600,
        help="cache lifetime in seconds (default: 21600 = 6h)",
    )
    args = parser.parse_args(argv)
    if not 0.0 <= args.input_share <= 1.0:
        parser.error("--input-share must be between 0 and 1")
    if args.top < 1:
        parser.error("--top must be at least 1")
    if args.recency_half_life <= 0 or args.quality_ref <= 0:
        parser.error("--recency-half-life and --quality-ref must be positive")
    if args.catalog and (args.best or args.json):
        parser.error("--catalog cannot be combined with --best or --json")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        models, or_cached = fetch_openrouter_models(args)
    except Exception as exc:
        warn(f"could not fetch the OpenRouter catalog: {exc}")
        return 1
    try:
        return run(args, models, or_cached)
    except Exception as exc:
        warn(f"could not rank models: {exc}")
        return 1


def run(args, models, or_cached) -> int:
    (
        discounts,
        zdr_ids,
        aa_by_id,
        frontend_cache_hits,
        frontend_models,
    ) = fetch_openrouter_frontend(args)
    if not args.no_zdr and not zdr_ids:
        warn(
            "ZDR data unavailable; refusing to rank possibly non-ZDR models "
            "(--no-zdr to override)"
        )
        return 1
    # Mode-scoped quality chains: a keyed run consults the AA API only; an
    # unkeyed run hydrates from the published catalog only. The catalog
    # fetcher must NEVER run on keyed builds -- a CI/site build hydrating
    # from the published catalog would consume its own output (a
    # self-reference loop). Both chains end in OpenRouter's benchmarks as
    # the last resort, then nothing.
    if args.aa_api_key or os.environ.get("AA_API_KEY"):
        aa_entries, aa_source, aa_cached, aa_index_version = fetch_aa_entries(args)
        catalog_quality = None
        catalog_depths = None
        catalog_generated_at = None
        # Keyed runs never hydrate schedules from the published catalog.
        schedule_entries = None
    else:
        aa_entries, aa_source, aa_cached, aa_index_version = [], None, False, None
        # None (transport failure) and empty maps (fetched-but-unusable
        # artifact) both degrade to no catalog tier -- resolve_quality and
        # the footer treat them identically as "catalog unavailable".
        catalog_doc = fetch_quality_catalog(args)
        catalog_quality = (catalog_doc or {}).get("by_id") or None
        catalog_depths = (catalog_doc or {}).get("depths_by_id") or None
        # Spec 6.5: the published schedules array is the unkeyed Deals
        # source of truth -- schedule-only models never appear in models.
        schedule_entries = (catalog_doc or {}).get("schedules_by_id") or None
        # The catalog meta only means something when hydration actually
        # produced rows: an unusable artifact contributes no meta, so the
        # sources block and the mode never credit a catalog that yielded
        # nothing.
        catalog_generated_at = (
            (catalog_doc or {}).get("generated_at")
            if (catalog_quality or catalog_depths)
            else None
        )
    exact, fuzzy = build_aa_lookup(dedupe_aa_entries_by_key(aa_entries))
    depth_lookup = build_aa_depth_lookup(aa_entries)
    unlabeled_lookup = build_aa_unlabeled_lookup(aa_entries)

    filtered = []
    reasoning_by_id = build_reasoning_metadata(models, frontend_models)
    candidates, dropped = build_candidates(
        models,
        args,
        discounts,
        zdr_ids,
        filtered,
        reasoning_by_id=reasoning_by_id,
    )
    # The provider-aware stage: every surviving candidate is repriced at its
    # baseline endpoint (or dropped fail-closed) before any scoring runs.
    candidates, extra_dropped = apply_endpoint_pricing(candidates, args, filtered)
    dropped.update(extra_dropped)
    if not candidates:
        # When the endpoint stage wiped the pool, naming the filter knobs is
        # misleading (they passed; the drops came after) -- warn with the
        # endpoint-stage drop counts instead (both reasons, zero-filled).
        counts = {r: extra_dropped.get(r, 0) for r in ENDPOINT_DROP_REASONS}
        if any(counts.values()):
            bits = ", ".join(f"{k}={v}" for k, v in counts.items())
            warn(f"no models after the endpoint stage: {bits}")
        else:
            warn(
                f"no models satisfy the filters (min-context={args.min_context}, "
                f"tools={'off' if args.no_require_tools else 'required'}); "
                "relax them and retry"
            )
        return 2

    quality_by_id, quality_source_by_id = resolve_quality(
        candidates,
        aa_by_id,
        exact,
        fuzzy,
        aa_source,
        catalog_quality,
        depth_lookup=depth_lookup,
        unlabeled_lookup=unlabeled_lookup,
        catalog_depths=catalog_depths,
        requested_effort=args.reasoning_effort,
    )
    weights = compute_scores(candidates, args, quality_by_id)

    # Per-task session-cost medians (Task 7): one fetch per invocation, but
    # only on the output paths that surface it (--catalog block, --json
    # rows; publish runs --json per priority, so the cache makes the repeat
    # cost free). A keyed --best/plain run skips the paginated fetch
    # entirely -- the medians never reach those outputs. Keyless,
    # fetch_session_cost returns None before any network or cache access.
    if args.catalog or args.json:
        session_cost = fetch_session_cost(args, models)
        # The key-presence fact distinguishes "skipped" from "unavailable";
        # same expression fetch_session_cost uses for its own guard.
        session_cost_mode = (
            "skipped"
            if not (args.openrouter_api_key or os.environ.get("OPENROUTER_API_KEY"))
            else ("ok" if session_cost else "unavailable")
        )
    else:
        session_cost = None
        session_cost_mode = "skipped"
    # run() owns the per-candidate attachment (build_catalog stays pure);
    # print_json reads it for the session_cost_usd display cell.
    for cand in candidates:
        cand["session_cost"] = (session_cost or {}).get("medians", {}).get(cand["id"])

    if args.catalog:
        print_catalog(
            build_catalog(
                args,
                models,
                candidates,
                dropped,
                filtered,
                discounts,
                quality_by_id,
                aa_source,
                aa_by_id,
                quality_source_by_id,
                session_cost,
                session_cost_mode,
                aa_index_version,
                catalog_generated_at,
                catalog_quality,
                depth_lookup=depth_lookup,
                catalog_depths=catalog_depths,
                reasoning_by_id=reasoning_by_id,
                requested_effort=args.reasoning_effort,
                schedule_entries=schedule_entries,
            )
        )
        return 0

    limit = 1 if args.best else args.top
    top = candidates[:limit]

    if args.best:
        print_best(top, args)
    elif args.json:
        print_json(top)
    else:
        pager_argv = resolve_pager(args)
        style = Style(color_enabled(args.color, sys.stdout))
        err_style = Style(color_enabled(args.color, sys.stderr))
        # The summary is stderr by contract (scripts pipe stdout), but a
        # pager owns the screen: notes printed to stderr before it starts
        # are painted over and never reach scrollback. So when paging, the
        # whole human stream -- summary included -- goes through the pager
        # (issue #28). Resolved here, printed below.
        summary = []
        if not args.bare:
            source_note = []
            if or_cached:
                source_note.append("catalog cached")
            if "discounts" in frontend_cache_hits:
                source_note.append("discounts cached")
            if "zdr" in frontend_cache_hits:
                source_note.append("ZDR cached")
            if "aa" in frontend_cache_hits:
                source_note.append("AA benchmarks cached")
            if aa_cached and aa_source:
                source_note.append("quality cached")
            if source_note:
                summary.append(
                    (
                        f"note: {', '.join(source_note)} (use --no-cache to refresh)",
                        False,
                    )
                )
            summary.append(
                (f"{len(candidates)} of {len(models)} listed models ranked", False)
            )
            if dropped:
                bits = ", ".join(
                    f"{v} {k}"
                    for k, v in sorted(dropped.items(), key=lambda kv: -kv[1])
                )
                summary.append((f"(dropped: {bits})", True))
        if not pager_argv and summary:
            for text, is_dim in summary:
                print(err_style.dim(text) if is_dim else text, file=sys.stderr)
            print(file=sys.stderr)
        # The footer lists per-source counts, each source's own as_of and
        # the AA index version when known; unmatched candidates and drift
        # get trailing clauses. No record resolved -> print_table's
        # quality-blind fallback line.
        unmatched = len(candidates) - len(quality_by_id)
        quality_note = quality_footer_note(quality_by_id, aa_index_version, unmatched)
        try:
            with paged_stdout(pager_argv):
                # Paged runs carry the summary above the table; the pager's
                # stream is stdout, so re-style dim lines for it.
                if pager_argv:
                    for text, is_dim in summary:
                        print(style.dim(text) if is_dim else text)
                    if summary:
                        print()
                print_table(
                    top,
                    weights,
                    quality_note,
                    detail=args.detail,
                    style=style,
                    bare=args.bare,
                    input_share=args.input_share,
                    depths=bool(args.depths) and not args.bare,
                )
                if args.off_peak:
                    catalog_w = catalog_weights(candidates, quality_by_id)
                    # Use the exact unrounded weighted score (not
                    # overall_scores, whose rounded components can disagree
                    # by 0.001) so a candidate in both tables always prints
                    # the same SCORE under balanced.
                    balanced_w = catalog_w["balanced"]
                    balanced = {
                        cand["id"]: weighted_score(cand, balanced_w)
                        for cand in candidates
                    }
                    schedules = build_schedules(
                        models,
                        args,
                        quality_by_id,
                        aa_by_id,
                        balanced,
                        catalog_quality,
                    )
                    # build_schedules resolves quality for any listed model;
                    # the web page's deals table shows that resolved value
                    # even for models outside the ranked pool, so keep it
                    # here too. Only SCORE is blanked for non-candidates
                    # (balanced is None for them).
                    print_deals(off_peak_deals(schedules), style=style, bare=args.bare)
        except BrokenPipeError:
            # Pager quit early: stop writing quietly (spec §Failure modes) --
            # deliberate on the direct path too: EPIPE from a closed consumer
            # leaves the exit status unchanged (CPython's shutdown flush).
            pass
        except PagerInterrupted:
            return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
