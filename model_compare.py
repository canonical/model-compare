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
                 70), clamped to [0, 1]; unmatched models score 0.
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
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
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
VERSION = "0.2.7"
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

CATALOG_SCHEMA_VERSION = 1

# The drop-reason keys build_candidates counts, verbatim -- keep in sync with
# its drop() call sites. pool.dropped lists all of them zero-filled so the
# contract is self-describing.
CATALOG_DROP_REASONS = (
    "malformed id",
    "context",
    "pricing",
    "free",
    "batch",
    "no discount",
    "not ZDR",
    "modality",
    "tool calling",
    "expired",
    "age",
)

PAREN_RE = re.compile(r"\([^)]*\)")
PROVIDER_PREFIX_RE = re.compile(r"^[^:\s]{1,30}:\s*")


def warn(msg: str) -> None:
    print(f"warning: {msg}", file=sys.stderr)


def norm_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


# ---------------------------------------------------------------------------
# HTTP + cache
# ---------------------------------------------------------------------------


class StripApiKeyRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Drop the AA API key when a redirect leaves the original origin.

    urllib copies custom request headers onto the redirected request even
    across hosts. The key is kept only when scheme, host and port all match,
    so a cross-host redirect or an https->http downgrade never carries it.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None:
            old, dest = (
                urllib.parse.urlsplit(req.full_url),
                urllib.parse.urlsplit(new.full_url),
            )
            same_origin = (old.scheme, old.hostname, old.port) == (
                dest.scheme,
                dest.hostname,
                dest.port,
            )
            if not same_origin:
                # Request stores header names capitalize()d: "X-api-key".
                new.remove_header("X-api-key")
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
FRONTEND_AA_CACHE = "openrouter-frontend-aa"


def fetch_openrouter_frontend(args):
    """Derive discounts, ZDR ids and OR-published AA benchmarks.

    Two loads: the base frontend URL serves discounts and benchmarks in one
    response; the per-endpoint ZDR list serves ZDR and is skipped entirely
    under --no-zdr. Each payload caches independently under its own key and is
    cached only when non-empty, so a degraded payload self-heals on the
    next run. A base-URL outage never blocks the ZDR fetch, and vice
    versa. Returns (discounts, zdr_ids, aa_by_id, cache_hits) where
    cache_hits names the payloads served from cache ("discounts", "zdr",
    "aa").
    """
    discounts = {}
    aa_by_id = {}
    zdr_ids = set()
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
                data.get("benchmarks") if isinstance(data, dict) else None
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

    if args.no_zdr:
        return discounts, zdr_ids, aa_by_id, cache_hits
    if not args.no_cache:
        # v3: the id set now comes from the per-endpoint ZDR list; v1/v2
        # sets were derived from the models/find response and must never
        # be read.
        cached = load_cache("openrouter-zdr-v3", args.cache_ttl)
        if cached:
            return discounts, set(cached), aa_by_id, cache_hits | {"zdr"}
    try:
        payload = fetch_json(OPENROUTER_ZDR_URL, timeout=30)
    except Exception as exc:
        warn(f"could not fetch ZDR data: {exc}")
        return discounts, zdr_ids, aa_by_id, cache_hits
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
        return discounts, zdr_ids, aa_by_id, cache_hits
    save_cache("openrouter-zdr-v3", sorted(ids))
    return discounts, ids, aa_by_id, cache_hits


def parse_price(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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


def aa_api_entries(api_key: str) -> list:
    """Intelligence entries from AA's V2 language models endpoint.

    Documented envelope: {tier, intelligence_index_version, pagination,
    data[]} with snake_case fields; the index lives at
    data[].evaluations.artificial_analysis_intelligence_index. Items
    without a usable slug/name or without a finite numeric index are
    skipped (nulls mean "not measured"). Pagination follows the documented
    has_more/total_pages contract with a 25-page hard cap; a failure past
    page 1 is best-effort (warn, keep earlier pages), a page-1 failure
    propagates.
    """
    entries = {}
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
            entries.setdefault(
                key,
                {
                    "key": key,
                    "name": name if isinstance(name, str) and name else key,
                    "index": index,
                },
            )
        pag = payload.get("pagination") if isinstance(payload, dict) else None
        if not isinstance(pag, dict):
            break  # undocumented shape: single-page degrade
        if pag.get("has_more") is not True:
            break
        total_pages = pag.get("total_pages")
        if isinstance(total_pages, int) and page >= total_pages:
            break
        page += 1
    return list(entries.values())


def fetch_aa_entries(args):
    """AA intelligence entries from the keyed AA API v2, or nothing.

    Returns (entries, source, cached). With a key, an API failure or an
    empty result warns and yields ([], None, False). Without a key this is
    silent and yields ([], None, False): OpenRouter's benchmarks are the
    primary source and the catalog records the absence (sources.aa).
    """
    if not args.no_cache:
        # v2: entries cached under the old key predate the V2 endpoint
        # migration and must never be read. Only AA API entries are read;
        # build_catalog rejects any other source as unknown.
        cached = load_cache("aa-intelligence-v2", args.cache_ttl)
        if isinstance(cached, dict) and cached.get("source") == "AA API v2":
            return cached.get("entries", []), cached.get("source"), True
    api_key = args.aa_api_key or os.environ.get("AA_API_KEY")
    if not api_key:
        return [], None, False
    try:
        entries = aa_api_entries(api_key)
    except Exception as exc:
        warn(f"AA API request failed ({exc})")
        return [], None, False
    if not entries:
        warn("AA API returned no intelligence scores")
        return [], None, False
    save_cache("aa-intelligence-v2", {"entries": entries, "source": "AA API v2"})
    return entries, "AA API v2", False


# ---------------------------------------------------------------------------
# Matching AA entries to OpenRouter model ids
# ---------------------------------------------------------------------------


def build_aa_lookup(entries):
    exact = {}
    fuzzy = []
    for entry in entries:
        key = entry.get("key") or ""
        name = entry.get("name") or key
        raw_index = entry.get("index")
        if not isinstance(raw_index, (int, float)) or isinstance(raw_index, bool):
            continue
        index = float(raw_index)
        # Non-finite indexes would serialize as NaN and poison quality.
        if not math.isfinite(index):
            continue

        def put(tier, raw):
            normalized = norm_key(raw)
            if normalized:
                exact.setdefault((tier, normalized), index)

        if "/" in key:
            put(0, key)
            put(1, key.rsplit("/", 1)[-1])
        else:
            put(1, key)
        put(2, name)
        put(3, PAREN_RE.sub(" ", name))
        tokens = set(norm_key(key).split()) | set(norm_key(name).split())
        if tokens:
            fuzzy.append((tokens, index))
    return exact, fuzzy


def match_quality(model, exact, fuzzy, allow_fuzzy=False):
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
        best_index = None
        for tokens, index in fuzzy:
            overlap = len(own & tokens) / len(own | tokens)
            if overlap > best:
                best = overlap
                best_index = index
        if best >= 0.5 and best_index is not None:
            return best_index
    return None


def base_model_id(model_id: str) -> str:
    """Candidate id without its :variant suffix (OR benchmarks are per base model)."""
    return model_id.split(":", 1)[0]


AA_BENCHMARK_FIELDS = ("intelligence_index", "coding_index", "agentic_index")


def build_aa_benchmarks(benchmarks) -> dict:
    """Map bare OpenRouter id -> OR-published AA trio from data.benchmarks.

    Keys are dated permaslugs (z-ai/glm-5.3-flash-20260826): the trailing
    -YYYYMMDD is stripped and undated keys map to themselves; when two keys
    strip to the same id the latest date wins. Values failing the
    finite-number guard are treated as absent; entries with at least one
    valid field are kept.
    """
    aa_by_id = {}
    seen_date = {}

    def accept(bare, date, trio):
        if bare in seen_date and seen_date[bare] >= date:
            return
        seen_date[bare] = date
        aa_by_id[bare] = trio

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
        match = re.match(r"^(.*)-(\d{8})$", key)
        if match:
            accept(match.group(1), match.group(2), trio)
        else:
            accept(key, "", trio)
    return aa_by_id


def resolve_quality(candidates, aa_by_id, exact, fuzzy, aa_source):
    """Per-candidate quality: OR benchmarks first, AA exact fallback, else None.

    OR's per-slug values cannot mispair; the AA fallback is exact-tier only
    (match_quality's fuzzy tier is the proven variant-conflation bug and
    stays off in production). Source values match the catalog contract:
    "openrouter", "api".
    """
    fallback_source = {"AA API v2": "api"}.get(aa_source)
    quality_by_id = {}
    source_by_id = {}
    for cand in candidates:
        bench = aa_by_id.get(base_model_id(cand["id"]))
        intelligence = bench.get("intelligence_index") if bench else None
        if intelligence is not None:
            quality_by_id[cand["id"]] = intelligence
            source_by_id[cand["id"]] = "openrouter"
            continue
        index = match_quality({"id": cand["id"], "name": cand["name"]}, exact, fuzzy)
        if index is not None:
            quality_by_id[cand["id"]] = index
            source_by_id[cand["id"]] = fallback_source
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


# ---------------------------------------------------------------------------
# Filtering and scoring
# ---------------------------------------------------------------------------


def build_candidates(models, args, discounts, zdr_ids, filtered_out=None):
    now = time.time()
    require_tools = not args.no_require_tools
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
        price_in = parse_price(pricing.get("prompt"))
        price_out = parse_price(pricing.get("completion"))
        if price_in is None or price_out is None or price_in < 0 or price_out < 0:
            drop("pricing", model_id, model.get("name"))
            continue
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
        candidates.append(
            {
                "id": model_id,
                "name": model.get("name") or model_id,
                "context": context,
                "price_in": price_in_m,
                "price_out": price_out_m,
                "blended": blended_m,
                "age_days": age_days,
                "discount": discount,
                "created": created,
                "tool_calling": tool_calling,
                "zdr": None if args.no_zdr else model_id in (zdr_ids or set()),
                "expired": expired,
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
        cand["quality"] = quality
        if quality is None:
            cand["quality_score"] = 0.0
        else:
            cand["quality_score"] = max(0.0, min(1.0, quality / args.quality_ref))
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


def print_table(top, total_candidates, weights, quality_note):
    headers = [
        "RANK",
        "MODEL",
        "QUAL",
        "$IN/M",
        "$OUT/M",
        "DISC",
        "CTX",
        "AGE",
        "SCORE",
    ]
    rows = []
    for rank, cand in enumerate(top, 1):
        rows.append(
            [
                str(rank),
                cand["id"],
                "-" if cand["quality"] is None else f"{cand['quality']:.1f}",
                fmt_price(cand["price_in"]),
                fmt_price(cand["price_out"]),
                fmt_discount(cand["discount"]),
                fmt_context(cand["context"]),
                fmt_age(cand["age_days"]),
                f"{cand['score']:.3f}",
            ]
        )
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def emit(row):
        if row[1] == "MODEL":
            line = "  ".join(h.ljust(w) for h, w in zip(row, widths))
            print(line)
            print("  ".join("-" * w for w in widths))
        else:
            cells = [row[0].rjust(widths[0]), row[1].ljust(widths[1])]
            cells += [c.rjust(w) for c, w in zip(row[2:], widths[2:])]
            print("  ".join(cells))

    emit(headers)
    for row in rows:
        emit(row)
    weight_bits = ", ".join(
        f"{k} {v:.0%}" for k, v in sorted(weights.items(), reverse=True)
    )
    print()
    print(
        f"pool: {total_candidates} candidates | weights: {weight_bits} | {quality_note}"
    )
    print(
        "quality: Artificial Analysis intelligence index (- = unknown); prices in USD per 1M tokens; "
        "DISC = active discount; AGE = time since listed on OpenRouter"
    )


def print_json(top):
    payload = [
        {
            "model": cand["id"],
            "opencode_model": opencode_model_id(cand["id"]),
            "name": cand["name"],
            "score": round(cand["score"], 4),
            "quality_index": cand["quality"],
            "input_usd_per_m": round(cand["price_in"], 6),
            "output_usd_per_m": round(cand["price_out"], 6),
            "blended_usd_per_m": round(cand["blended"], 6),
            "discount": round(cand["discount"], 4)
            if has_discount(cand["discount"])
            else None,
            # Display string from the DISC column's formatter, so the site
            # shows exactly what the CLI prints instead of re-rounding.
            "discount_pct": fmt_discount(cand["discount"]),
            "context_tokens": cand["context"],
            "age_days": round(cand["age_days"], 1)
            if cand["age_days"] is not None
            else None,
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
):
    """Assemble the full-catalog document (see README "Catalog output").

    Pure: reads candidates post-compute_scores (which already carry the
    component scores) and never mutates them. Deterministic apart from
    generated_at: models sort by (-overall.balanced, id), filtered by id,
    and age uses date precision so two runs in the same UTC day match.
    """
    now = datetime.now(timezone.utc)
    weights = catalog_weights(candidates, quality_by_id)
    aa_modes = {"AA API v2": "api"}
    if aa_source is not None and aa_source not in aa_modes:
        # Never claim mode "none" for a source we do not know: the document
        # would contradict itself (mode none with matched quality scores).
        raise ValueError(f"unknown AA source: {aa_source!r}")
    matched_openrouter = sum(
        1 for s in quality_source_by_id.values() if s == "openrouter"
    )
    # fallback is what the AA API yielded on its own; mode
    # hides it behind "openrouter" as soon as one OpenRouter match exists.
    aa_fallback = aa_modes.get(aa_source, "none")
    aa_mode = "openrouter" if matched_openrouter else aa_fallback

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
        overall = {}
        for priority in PRIORITY_WEIGHTS:
            w = weights[priority]
            overall[priority] = round(
                w.get("quality", 0.0) * scores["quality"]
                + w.get("price", 0.0) * scores["price"]
                + w.get("context", 0.0) * scores["context"]
                + w.get("age", 0.0) * scores["age"],
                4,
            )
        bench = aa_by_id.get(base_model_id(cand["id"])) or {}
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
                },
                "context": cand["context"],
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
                "scores": {**scores, "overall": overall},
            }
        )
    entries.sort(key=lambda e: (-e["scores"]["overall"]["balanced"], e["id"]))

    return {
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
            "weights": weights,
        },
        "sources": {
            "openrouter": "ok",
            "aa": {
                "mode": aa_mode,
                "fallback": aa_fallback,
                "matched": len(quality_by_id),
                "matched_openrouter": matched_openrouter,
            },
            "zdr": "skipped" if args.no_zdr else "ok",
            "discounts": "ok" if discounts else "unavailable",
        },
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
    parser.add_argument(
        "--min-context",
        type=int,
        default=1_000_000,
        help="hard minimum context window in tokens (default: 1000000)",
    )
    parser.add_argument(
        "--priority",
        choices=sorted(PRIORITY_WEIGHTS),
        default="balanced",
        help="ranking emphasis (default: balanced)",
    )
    parser.add_argument(
        "--top", type=int, default=5, help="how many models to list (default: 5)"
    )
    parser.add_argument(
        "--best",
        action="store_true",
        help="print only the #1 model as an opencode id (openrouter/<model>), for scripting",
    )
    parser.add_argument(
        "--json", action="store_true", help="print the ranked candidates as JSON"
    )
    parser.add_argument(
        "--catalog",
        action="store_true",
        help="print the full model catalog (ranked candidates and filtered-out models with reasons) as one JSON document; ignores --top and --priority, cannot be combined with --best or --json",
    )
    parser.add_argument(
        "--input-share",
        type=float,
        default=0.75,
        help="input-token share used for the blended price, 0-1 (default: 0.75)",
    )
    parser.add_argument(
        "--recency-half-life",
        type=float,
        default=120.0,
        help="age decay half-life in days (default: 120)",
    )
    parser.add_argument(
        "--max-age-days",
        type=float,
        default=0.0,
        help="drop models listed more than this many days ago (0 = off)",
    )
    parser.add_argument(
        "--quality-ref",
        type=float,
        default=70.0,
        help="AA intelligence index that counts as full quality score (default: 70)",
    )
    parser.add_argument(
        "--aa-api-key",
        default=None,
        help="Artificial Analysis API key (or set AA_API_KEY; free at artificialanalysis.ai)",
    )
    parser.add_argument(
        "--no-require-tools",
        action="store_true",
        help="do not require tool-calling support (coding agents want it)",
    )
    parser.add_argument(
        "--exclude-free",
        action="store_true",
        help="drop zero-cost listings such as rate-limited :free variants",
    )
    parser.add_argument(
        "--include-batch",
        action="store_true",
        help="keep ':batch' variants (asynchronous completion; cheaper but unsuitable for interactive agents)",
    )
    parser.add_argument(
        "--discount",
        action="store_true",
        help="only list models with an active discount",
    )
    parser.add_argument(
        "--no-zdr",
        action="store_true",
        help="consider all models, not just zero-data-retention (ZDR) ones",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="bypass the local response cache and refetch",
    )
    parser.add_argument(
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
    discounts, zdr_ids, aa_by_id, frontend_cache_hits = fetch_openrouter_frontend(args)
    if not args.no_zdr and not zdr_ids:
        warn(
            "ZDR data unavailable; refusing to rank possibly non-ZDR models "
            "(--no-zdr to override)"
        )
        return 1
    aa_entries, aa_source, aa_cached = fetch_aa_entries(args)
    exact, fuzzy = build_aa_lookup(aa_entries)

    filtered = []
    candidates, dropped = build_candidates(models, args, discounts, zdr_ids, filtered)
    if not candidates:
        warn(
            f"no models satisfy the filters (min-context={args.min_context}, "
            f"tools={'off' if args.no_require_tools else 'required'}); relax them and retry"
        )
        return 2

    quality_by_id, quality_source_by_id = resolve_quality(
        candidates, aa_by_id, exact, fuzzy, aa_source
    )
    weights = compute_scores(candidates, args, quality_by_id)

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
            )
        )
        return 0

    limit = 1 if args.best else args.top
    top = candidates[:limit]

    if args.best:
        print(opencode_model_id(top[0]["id"]))
    elif args.json:
        print_json(top)
    else:
        drop_note = ""
        if dropped:
            bits = ", ".join(
                f"{v} {k}" for k, v in sorted(dropped.items(), key=lambda kv: -kv[1])
            )
            drop_note = f" (dropped: {bits})"
        matched_or = sum(1 for s in quality_source_by_id.values() if s == "openrouter")
        matched_aa = len(quality_by_id) - matched_or
        unmatched = len(candidates) - len(quality_by_id)
        bits = []
        if matched_or:
            bits.append(f"OpenRouter benchmarks ({matched_or})")
        if matched_aa:
            bits.append(f"{aa_source} exact ({matched_aa})")
        if bits:
            quality_note = (
                f"quality via {' + '.join(bits)}: matched "
                f"{len(quality_by_id)}/{len(candidates)} candidates"
            )
            if unmatched:
                quality_note += "; unmatched candidates score 0 on quality"
        else:
            quality_note = "quality data unavailable, ranked on price/context/age"
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
            print(
                f"note: {', '.join(source_note)} (use --no-cache to refresh)",
                file=sys.stderr,
            )
        print(
            f"pool: {len(models)} listed -> {len(candidates)} candidates{drop_note}",
            file=sys.stderr,
        )
        print_table(top, len(candidates), weights, quality_note)
    return 0


if __name__ == "__main__":
    sys.exit(main())
