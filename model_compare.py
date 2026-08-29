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
  model_compare.py --best                   print only "#1 model id" (parseable)
  model_compare.py --priority price --top 3 three cheapest-sensible picks
  MODEL=$(model_compare.py --best)          shell integration
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
AA_MODELS_PAGE_URL = "https://artificialanalysis.ai/models"
AA_API_URL = "https://artificialanalysis.ai/api/v2/data/llms/models"
USER_AGENT = "model-compare/1.0 (https://github.com/rkratky/model-compare)"

PRIORITY_WEIGHTS = {
    "balanced": {"quality": 0.40, "price": 0.40, "context": 0.10, "age": 0.10},
    "price": {"quality": 0.20, "price": 0.60, "context": 0.10, "age": 0.10},
    "quality": {"quality": 0.60, "price": 0.20, "context": 0.10, "age": 0.10},
}

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
        with open(path, "w") as fh:
            json.dump({"fetched_at": time.time(), "payload": payload}, fh)
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


def parse_price(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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

    def walk(node):
        if isinstance(node, dict):
            ident = (
                node.get("id")
                or node.get("slug")
                or node.get("model")
                or node.get("name")
            )
            index = None
            for key, val in node.items():
                key_l = key.lower()
                if (
                    isinstance(val, (int, float))
                    and "intelligence" in key_l
                    and "estimated" not in key_l
                    and "cost" not in key_l
                ):
                    index = float(val)
                    break
            if ident and index is not None:
                found.append(
                    {
                        "key": str(ident),
                        "name": str(node.get("name") or ident),
                        "index": index,
                    }
                )
            for val in node.values():
                walk(val)
        elif isinstance(node, list):
            for val in node:
                walk(val)

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

    stream = []
    for chunk in re.finditer(
        r'self\.__next_f\.push\(\[1,\s*("(?:[^"\\]|\\.)*")\]\)', html
    ):
        try:
            stream.append(json.loads(chunk.group(1)))
        except ValueError:
            pass
    flight = "".join(stream)
    for match in re.finditer(
        r'"slug":"([^"]+)"|"name":"([^"]+)"|"intelligenceIndex":(-?\d+(?:\.\d+)?)',
        flight,
    ):
        slug, name, index = match.group(1), match.group(2), match.group(3)
        if index is not None:
            estimated = (
                '"intelligenceIndexIsEstimated":true'
                in flight[match.end() : match.end() + 80]
            )
            add(slug, name, float(index), estimated)

    return list(entries.values())


def fetch_aa_entries(args):
    if not args.no_cache:
        cached = load_cache("aa-intelligence", args.cache_ttl)
        if cached is not None:
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
        index = float(entry["index"])

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


def match_quality(model, exact, fuzzy):
    model_id = model["id"]
    name = model.get("name") or ""
    display = PROVIDER_PREFIX_RE.sub("", name).strip()
    provider, _, base = model_id.split(":", 1)[0].partition("/")

    for tier, raw in (
        (0, model_id),
        (1, base),
        (2, display),
        (3, PAREN_RE.sub(" ", display)),
    ):
        normalized = norm_key(raw)
        if normalized and (tier, normalized) in exact:
            return exact[(tier, normalized)]

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


# ---------------------------------------------------------------------------
# Filtering and scoring
# ---------------------------------------------------------------------------


def build_candidates(models, args):
    now = time.time()
    require_tools = not args.no_require_tools
    dropped = {}
    candidates = []

    def drop(reason):
        dropped[reason] = dropped.get(reason, 0) + 1

    for model in models:
        model_id = model.get("id") or ""
        if "/" not in model_id:
            drop("malformed id")
            continue
        context = model.get("context_length") or 0
        if context < args.min_context:
            drop("context")
            continue
        pricing = model.get("pricing") or {}
        price_in = parse_price(pricing.get("prompt"))
        price_out = parse_price(pricing.get("completion"))
        if price_in is None or price_out is None or price_in < 0 or price_out < 0:
            drop("pricing")
            continue
        if args.exclude_free and price_in == 0 and price_out == 0:
            drop("free")
            continue
        modality = (model.get("architecture") or {}).get("modality") or ""
        output_modality = (
            modality.split("->")[-1].strip() if "->" in modality else "text"
        )
        if output_modality != "text":
            drop("modality")
            continue
        params = model.get("supported_parameters") or []
        if require_tools and not ("tools" in params and "tool_choice" in params):
            drop("tool calling")
            continue
        expiry = parse_iso_datetime(model.get("expiration_date"))
        if expiry and expiry < datetime.now(timezone.utc):
            drop("expired")
            continue
        created = model.get("created") or 0
        try:
            created = float(created)
        except (TypeError, ValueError):
            created = 0.0
        age_days = max(0.0, (now - created) / 86400.0) if created > 0 else None
        if args.max_age_days and age_days is not None and age_days > args.max_age_days:
            drop("age")
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
            }
        )

    return candidates, dropped


def compute_scores(candidates, args, quality_by_id):
    weights = dict(PRIORITY_WEIGHTS[args.priority])
    if not any(c["id"] in quality_by_id for c in candidates):
        weights.pop("quality")
        total = sum(weights.values())
        weights = {name: value / total for name, value in weights.items()}

    prices = [c["blended"] for c in candidates]
    low = math.log10(min(prices) + 0.01)
    high = math.log10(max(prices) + 0.01)
    price_span = (high - low) or 1.0
    floor_ctx = max(args.min_context, 1000)
    cap_ctx = max(4.0 * floor_ctx, 1_000_000.0)
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


def print_table(top, total_candidates, weights, quality_note):
    headers = ["RANK", "MODEL", "QUAL", "$IN/M", "$OUT/M", "CTX", "AGE", "SCORE"]
    rows = []
    for rank, cand in enumerate(top, 1):
        rows.append(
            [
                str(rank),
                cand["id"],
                "-" if cand["quality"] is None else f"{cand['quality']:.1f}",
                fmt_price(cand["price_in"]),
                fmt_price(cand["price_out"]),
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
        "AGE = time since listed on OpenRouter"
    )


def print_json(top):
    payload = [
        {
            "model": cand["id"],
            "name": cand["name"],
            "score": round(cand["score"], 4),
            "quality_index": cand["quality"],
            "input_usd_per_m": round(cand["price_in"], 6),
            "output_usd_per_m": round(cand["price_out"], 6),
            "blended_usd_per_m": round(cand["blended"], 6),
            "context_tokens": cand["context"],
            "age_days": round(cand["age_days"], 1)
            if cand["age_days"] is not None
            else None,
        }
        for cand in top
    ]
    print(json.dumps(payload, indent=2))


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
        help="print only the #1 model id, nothing else, for scripting",
    )
    parser.add_argument(
        "--json", action="store_true", help="print the ranked candidates as JSON"
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
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        models, or_cached = fetch_openrouter_models(args)
    except Exception as exc:
        warn(f"could not fetch the OpenRouter catalog: {exc}")
        return 1
    aa_entries, aa_source, aa_cached = fetch_aa_entries(args)
    exact, fuzzy = build_aa_lookup(aa_entries)

    candidates, dropped = build_candidates(models, args)
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

    limit = 1 if args.best else args.top
    top = candidates[:limit]

    if args.best:
        print(top[0]["id"])
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
