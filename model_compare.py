#!/usr/bin/env python3
"""model-compare: pick the best value-for-money LLM on OpenRouter.

Run it before launching a coding agent to auto-select the model that gives
the most capability per dollar today. Data sources:

  * OpenRouter public API (/api/v1/models) - catalog, pricing (input and
    output), context window, release date and capabilities.
  * Artificial Analysis intelligence index - model quality, obtained via
    the AA API v2 when an API key is available (--aa-api-key or the
    AA_API_KEY environment variable; free key at artificialanalysis.ai),
    falling back to a best-effort scrape of artificialanalysis.ai/models.
    If neither works, ranking continues on price/context/age alone.

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
import urllib.request
from datetime import datetime, timezone

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
OPENROUTER_DISCOUNTS_URL = (
    "https://openrouter.ai/api/frontend/v1/models/find?output_modalities=text"
)
OPENROUTER_ZDR_URL = (
    "https://openrouter.ai/api/frontend/v1/models/find?output_modalities=text&zdr=true"
)
AA_MODELS_PAGE_URL = "https://artificialanalysis.ai/models"
AA_API_URL = "https://artificialanalysis.ai/api/v2/data/llms/models"
USER_AGENT = "model-compare/1.0 (https://github.com/rkratky/model-compare)"

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


def http_get(url: str, headers: dict | None = None, timeout: int = 30) -> bytes:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept-Encoding": "gzip",
            **(headers or {}),
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
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


def fetch_discount_map(args):
    """Map of public model id -> current discount fraction (e.g. 0.5).

    Comes from the OpenRouter frontend models API -- the same data the
    website's ?discount=true filter uses. It is undocumented and may change;
    on any failure the map is empty and discounts simply show as "--".
    """
    if not args.no_cache:
        cached = load_cache("openrouter-discounts", args.cache_ttl)
        if cached:
            return cached, True
    try:
        payload = fetch_json(OPENROUTER_DISCOUNTS_URL, timeout=30)
    except Exception as exc:
        warn(f"could not fetch discount data: {exc}")
        return {}, False
    data = payload.get("data") if isinstance(payload, dict) else None
    entries = data.get("models") if isinstance(data, dict) else None
    discounts = {}
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
        discounts.setdefault(key, float(raw))
    if not discounts:
        # An intact endpoint always yields hundreds of entries (discount: 0 is
        # still an entry); an empty map means the response shape changed --
        # never cache that, so the next run recovers on its own.
        warn("no discount entries found; treating discounts as unavailable")
        return {}, False
    save_cache("openrouter-discounts", discounts)
    return discounts, False


def fetch_zdr_set(args):
    """Set of public model ids whose endpoint is zero-data-retention.

    Sourced from the same frontend models API as the website's ?zdr=true
    filter (undocumented). An empty result means the data is unavailable:
    the caller must then fail closed rather than silently considering
    non-ZDR models. The fetch is skipped entirely under --no-zdr.
    """
    if args.no_zdr:
        return set(), False
    if not args.no_cache:
        cached = load_cache("openrouter-zdr", args.cache_ttl)
        if cached:
            return set(cached), True
    try:
        payload = fetch_json(OPENROUTER_ZDR_URL, timeout=30)
    except Exception as exc:
        warn(f"could not fetch ZDR data: {exc}")
        return set(), False
    data = payload.get("data") if isinstance(payload, dict) else None
    entries = data.get("models") if isinstance(data, dict) else None
    ids = set()
    for entry in entries or []:
        if not isinstance(entry, dict):
            continue
        slug = entry.get("slug") or ""
        if not slug or slug.startswith("~"):
            continue
        endpoint = entry.get("endpoint") or {}
        variant = endpoint.get("variant") or ""
        ids.add(slug if variant in ("", "standard") else f"{slug}:{variant}")
    if not ids:
        warn("no ZDR entries found; treating ZDR data as unavailable")
        return set(), False
    save_cache("openrouter-zdr", sorted(ids))
    return ids, False


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
    payload = fetch_json(AA_API_URL, headers={"x-api-key": api_key}, timeout=30)
    found = []

    def pick_index(node: dict):
        # Prefer the canonical field name; only fall back to a substring match
        # so a future sibling metric like "estimatedIntelligenceCost" can't win
        # by sheer dict-ordering luck.
        canonical = node.get("artificialAnalysisIntelligenceIndex")
        if isinstance(canonical, (int, float)):
            return float(canonical)
        for key, val in node.items():
            key_l = key.lower()
            if (
                isinstance(val, (int, float))
                and "intelligence" in key_l
                and "estimated" not in key_l
                and "cost" not in key_l
            ):
                return float(val)
        return None

    def walk(node, depth=0):
        if depth > 100:
            return
        if isinstance(node, dict):
            ident = (
                node.get("id")
                or node.get("slug")
                or node.get("model")
                or node.get("name")
            )
            index = pick_index(node)
            if ident and index is not None:
                found.append(
                    {
                        "key": str(ident),
                        "name": str(node.get("name") or ident),
                        "index": index,
                    }
                )
            for val in node.values():
                walk(val, depth + 1)
        elif isinstance(node, list):
            for val in node:
                walk(val, depth + 1)

    walk(payload)
    deduped = {}
    for entry in found:
        deduped.setdefault(entry["key"], entry)
    return list(deduped.values())


def aa_scrape_entries() -> list:
    html = http_get(
        AA_MODELS_PAGE_URL, headers={"Accept": "text/html"}, timeout=45
    ).decode("utf-8", "replace")
    entries = {}

    def add(key, name, index, estimated):
        if not key or index is None:
            return
        current = entries.get(key)
        if current is None or (current["estimated"] and not estimated):
            entries[key] = {
                "key": key,
                "name": name,
                "index": index,
                "estimated": estimated,
            }

    for block in re.findall(
        r'<script type="application/ld\+json">(.*?)</script>', html, re.S
    ):
        try:
            doc = json.loads(block)
        except ValueError:
            continue
        for node in doc if isinstance(doc, list) else [doc]:
            if not isinstance(node, dict):
                continue
            for item in node.get("data") or []:
                if isinstance(item, dict) and isinstance(
                    item.get("artificialAnalysisIntelligenceIndex"), (int, float)
                ):
                    slug = (item.get("detailsUrl") or "").rsplit("/", 1)[-1]
                    add(
                        slug,
                        item.get("label") or slug,
                        float(item["artificialAnalysisIntelligenceIndex"]),
                        False,
                    )

    return list(entries.values())


def fetch_aa_entries(args):
    if not args.no_cache:
        cached = load_cache("aa-intelligence", args.cache_ttl)
        if isinstance(cached, dict):
            return cached.get("entries", []), cached.get("source"), True
    api_key = args.aa_api_key or os.environ.get("AA_API_KEY")
    if api_key:
        try:
            entries = aa_api_entries(api_key)
            if entries:
                save_cache(
                    "aa-intelligence", {"entries": entries, "source": "AA API v2"}
                )
                return entries, "AA API v2", False
            warn("AA API returned no intelligence scores; falling back to page scrape")
        except Exception as exc:
            warn(f"AA API request failed ({exc}); falling back to page scrape")
    try:
        entries = aa_scrape_entries()
    except Exception as exc:
        warn(f"could not fetch AA intelligence data: {exc}")
        return [], None, False
    if not entries:
        warn("no intelligence scores found on the AA page")
        return [], None, False
    save_cache("aa-intelligence", {"entries": entries, "source": "AA page scrape"})
    return entries, "AA page scrape", False


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
        cand["score"] = (
            weights.get("quality", 0.0) * cand["quality_score"]
            + weights.get("price", 0.0) * cand["price_score"]
            + weights.get("context", 0.0) * cand["context_score"]
            + weights.get("age", 0.0) * cand["age_score"]
        )

    candidates.sort(
        key=lambda c: (-c["score"], -(c["quality"] or 0.0), c["blended"], c["id"])
    )
    return weights


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
            "context_tokens": cand["context"],
            "age_days": round(cand["age_days"], 1)
            if cand["age_days"] is not None
            else None,
        }
        for cand in top
    ]
    print(json.dumps(payload, indent=2))


def build_catalog(
    args, models, candidates, dropped, filtered, discounts, quality_by_id, aa_source
):
    """Assemble the full-catalog document (see README "Catalog output").

    Pure: reads candidates post-compute_scores (which already carry the
    component scores) and never mutates them. Deterministic apart from
    generated_at: models sort by (-overall.balanced, id), filtered by id,
    and age uses date precision so two runs in the same UTC day match.
    """
    now = datetime.now(timezone.utc)
    weights = catalog_weights(candidates, quality_by_id)
    aa_modes = {"AA API v2": "api", "AA page scrape": "scrape"}
    if aa_source is not None and aa_source not in aa_modes:
        # Never claim mode "none" for a source we do not know: the document
        # would contradict itself (mode none with matched quality scores).
        raise ValueError(f"unknown AA source: {aa_source!r}")
    aa_mode = aa_modes.get(aa_source, "none")
    quality_match = aa_modes.get(aa_source)

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
                "quality": cand["quality"] if aa_source else None,
                "quality_match": quality_match if cand["id"] in quality_by_id else None,
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
            "aa": {"mode": aa_mode, "matched": len(quality_by_id)},
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
    discounts, disc_cached = fetch_discount_map(args)
    zdr_ids, zdr_cached = fetch_zdr_set(args)
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

    quality_by_id = {}
    for cand in candidates:
        index = match_quality({"id": cand["id"], "name": cand["name"]}, exact, fuzzy)
        if index is not None:
            quality_by_id[cand["id"]] = index
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
        unmatched = sum(1 for c in candidates if c["id"] not in quality_by_id)
        if aa_source:
            matched = len(candidates) - unmatched
            quality_note = (
                f"quality via {aa_source}{' (cached)' if aa_cached else ''}: "
                f"{len(aa_entries)} scores, matched {matched}/{len(candidates)} candidates"
            )
            if unmatched:
                quality_note += "; unmatched candidates score 0 on quality"
        else:
            quality_note = "quality data unavailable, ranked on price/context/age"
        source_note = []
        if or_cached:
            source_note.append("catalog cached")
        if disc_cached:
            source_note.append("discounts cached")
        if zdr_cached:
            source_note.append("ZDR cached")
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
