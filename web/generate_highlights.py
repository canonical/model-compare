#!/usr/bin/env python3
"""Generate the site's three highlight sections from weekly history data.

Computes a deterministic numeric diff of today's catalog against the
snapshot dated exactly 7 days back, then either reuses recent LLM output,
asks OpenRouter for three grounded sentences, or falls back to templates.
A broken LLM never fails the deploy: the fallback is always available.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from datetime import date, datetime, timedelta, timezone

HIGHLIGHTS_SCHEMA_VERSION = 1
SECTION_KEYS = ("week", "intelligence", "prices")
DIFF_PRIORITY_KEYS = ("balanced", "price", "quality")
CATALOG_PRICING_KEYS = ("input_per_1m", "output_per_1m", "blended_per_1m")
MAX_LLM_MODELS = 5
OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
FRONTEND_MODELS_URL = (
    "https://openrouter.ai/api/frontend/v1/models/find?output_modalities=text"
)
CATALOG_MODELS_URL = "https://openrouter.ai/api/v1/models"
USER_AGENT = "model-compare/1.0 (https://github.com/rkratky/model-compare)"
PROMPT_RULES = (
    "You write deployment notes for a model-ranking site. The user message is"
    " a JSON diff of the last 7 days. Reply with ONE JSON object with keys"
    ' "week", "intelligence", "prices" -- each value 1-2 short declarative'
    " sentences (max ~30 words each). State facts only from the diff; invent"
    " nothing; no hype or filler adjectives; wrap every model id in"
    " backticks; no markdown besides those backticks. If baseline_present is"
    " false, say in one sentence that weekly data collection is still"
    " building up."
)
HIGHLIGHTS_MAX_AGE_HOURS = 24


def seven_days_before(today: str) -> str:
    return (date.fromisoformat(today) - timedelta(days=7)).isoformat()


def _num(value):
    """Return value unchanged when it is a finite non-bool number, else None."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value if math.isfinite(value) else None


def _overall_score(entry, key):
    scores = entry.get("scores") if isinstance(entry, dict) else None
    overall = scores.get("overall") if isinstance(scores, dict) else None
    return _num(overall.get(key)) if isinstance(overall, dict) else None


def _prev_ranks(baseline_tabs, key):
    """id -> rank map from a tabs list, skipping unusable rows.

    The baseline is a previously deployed file, so it is read defensively:
    anything that would crash the diff arithmetic is treated as absent.
    """
    rows = baseline_tabs.get(key) if isinstance(baseline_tabs, dict) else None
    ranks = {}
    for row in rows or []:
        if (
            isinstance(row, dict)
            and isinstance(row.get("id"), str)
            and isinstance(row.get("rank"), int)
            and not isinstance(row["rank"], bool)
        ):
            ranks[row["id"]] = row["rank"]
    return ranks


def build_diff(catalog, history) -> dict:
    """Numeric-only diff of today's catalog vs the snapshot 7 days ago.

    Deterministic ordering everywhere; all mover lists are bounded by
    top-10 history membership (see the design spec).
    """
    today = catalog["generated_at"][:10]
    snapshots = history.get("snapshots") if isinstance(history, dict) else None
    baseline = (
        snapshots.get(seven_days_before(today)) if isinstance(snapshots, dict) else None
    )
    diff = {"baseline_present": baseline is not None}
    if baseline is None:
        diff["tabs"] = {
            key: {"entries": [], "new_ids": []} for key in DIFF_PRIORITY_KEYS
        }
        diff["new_pool_ids_count"] = 0
        diff["new_pool_id_sample"] = []
        diff["aa_movers"] = {"up": [], "down": []}
        diff["price_moves"] = {"down": [], "up": []}
        diff["discounts"] = {"appeared": [], "vanished": []}
        return diff

    prev_pool = {
        model_id
        for model_id in baseline.get("pool_ids") or []
        if isinstance(model_id, str)
    }
    now_pool = set(
        [e["id"] for e in catalog["models"]] + [e["id"] for e in catalog["filtered"]]
    )
    new_pool = sorted(now_pool - prev_pool)

    tabs = {}
    for key in DIFF_PRIORITY_KEYS:
        prev_ranks = _prev_ranks(baseline.get("tabs"), key)
        ranked = sorted(
            (
                entry
                for entry in catalog["models"]
                if isinstance(entry, dict)
                and isinstance(entry.get("id"), str)
                and _overall_score(entry, key) is not None
            ),
            key=lambda e: (-e["scores"]["overall"][key], e["id"]),
        )[:10]
        entries = []
        new_ids = []
        for i, entry in enumerate(ranked, start=1):
            prev = prev_ranks.get(entry["id"])
            entries.append(
                {
                    "id": entry["id"],
                    "rank": i,
                    "prev_rank": prev,
                    "delta": None if prev is None else prev - i,
                }
            )
            if prev is None:
                new_ids.append(entry["id"])
        tabs[key] = {"entries": entries, "new_ids": new_ids}
    diff["tabs"] = tabs

    diff["new_pool_ids_count"] = len(new_pool)
    diff["new_pool_id_sample"] = new_pool[:5]

    prev_aa = baseline.get("aa") if isinstance(baseline.get("aa"), dict) else {}
    movers = []
    for entry in catalog["models"]:
        prev = _num(prev_aa.get(entry["id"]))
        now = _num((entry.get("aa") or {}).get("intelligence_index"))
        if prev is None or now is None:
            continue
        movers.append({"id": entry["id"], "delta": round(now - prev, 2), "value": now})
    movers.sort(key=lambda m: (-m["delta"], m["id"]))
    diff["aa_movers"] = {
        "up": movers[:3],
        "down": sorted(movers, key=lambda m: (m["delta"], m["id"]))[:3],
    }

    prev_prices = (
        baseline.get("prices") if isinstance(baseline.get("prices"), dict) else {}
    )
    price_moves = []
    appeared, vanished = [], []
    prev_discounts = {
        model_id: row[3]
        for model_id, row in prev_prices.items()
        if isinstance(row, list) and len(row) == 4
    }
    for entry in catalog["models"]:
        old = prev_prices.get(entry["id"])
        if not (isinstance(old, list) and len(old) >= 3 and _num(old[2]) is not None):
            continue
        new_blended = _num((entry.get("pricing") or {}).get("blended_per_1m"))
        if new_blended is None:
            continue
        price_moves.append(
            {
                "id": entry["id"],
                "old": old[2],
                "new": new_blended,
                "delta": round(new_blended - old[2], 4),
            }
        )
        old_disc, new_disc = prev_discounts.get(entry["id"]), entry.get("discount")
        if old_disc is None and new_disc is not None:
            appeared.append(entry["id"])
        elif old_disc is not None and new_disc is None:
            vanished.append(entry["id"])
    price_moves.sort(key=lambda m: (m["delta"], m["id"]))
    diff["price_moves"] = {
        "down": price_moves[:3],
        "up": sorted(price_moves, key=lambda m: (-m["delta"], m["id"]))[:3],
    }
    diff["discounts"] = {"appeared": sorted(appeared), "vanished": sorted(vanished)}
    return diff


def fallback_texts(diff) -> dict:
    """Deterministic template sentences (same backtick convention as the LLM)."""
    if not diff["baseline_present"]:
        return {
            "week": (
                "Weekly comparison data is still building up; movement tracking "
                "starts once seven days of snapshots exist."
            ),
            "intelligence": (
                "No weekly intelligence baseline yet; AA movements will be "
                "summarised once history covers seven days."
            ),
            "prices": (
                "No weekly price baseline yet; price and discount movements "
                "will be summarised once history covers seven days."
            ),
        }
    week_bits = []
    if diff["new_pool_ids_count"]:
        week_bits.append(f"{diff['new_pool_ids_count']} new model(s) on OpenRouter")
    balanced = diff["tabs"]["balanced"]
    movers = [e for e in balanced["entries"] if e["delta"]]
    for entry in movers[:2]:
        arrow = "climbed" if entry["delta"] > 0 else "fell"
        week_bits.append(f"`{entry['id']}` {arrow} {abs(entry['delta'])} spot(s)")
    if balanced["new_ids"]:
        week_bits.append(f"`{balanced['new_ids'][0]}` entered the balanced top ten")
    week = (
        "This week: " + "; ".join(week_bits) + "."
        if week_bits
        else "This week: no top-ten changes."
    )

    intel_bits = [
        f"`{m['id']}` {'+' if m['delta'] > 0 else ''}{m['delta']}"
        for m in diff["aa_movers"]["up"][:2] + diff["aa_movers"]["down"][:2]
    ]
    intelligence = (
        "AA intelligence moves: " + "; ".join(intel_bits) + "."
        if intel_bits
        else "No AA intelligence changes this week."
    )

    price_bits = [
        f"`{m['id']}` {m['old']} -> {m['new']}" for m in diff["price_moves"]["down"][:2]
    ]
    if diff["discounts"]["appeared"]:
        price_bits.append(
            "discount appeared for "
            + ", ".join(f"`{i}`" for i in diff["discounts"]["appeared"][:2])
        )
    prices = (
        "Blended price moves: " + "; ".join(price_bits) + "."
        if price_bits
        else "No notable price moves this week."
    )
    return {"week": week, "intelligence": intelligence, "prices": prices}


def _post_chat(model, body, api_key):
    import urllib.request

    request = urllib.request.Request(
        OPENROUTER_CHAT_URL,
        data=json.dumps(body).encode(),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as resp:
        return json.load(resp)


def _fetch_json(url):
    import urllib.request

    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=30) as resp:
        return json.load(resp)


def resolve_llm_models() -> list[str]:
    """Currently-listed :free models, best-ranked first.

    The free lineup rotates, so the chain is discovered at publish time
    instead of hardcoded. Primary source: the frontend models API, whose
    benchmarks payload carries OpenRouter's published AA intelligence per
    model -- free variants are the endpoint.variant == "free" entries
    (id = slug + ":free"), ranked by that intelligence. If that endpoint
    fails or lists nothing, fall back to the public catalog's :free ids
    (sorted by id). An empty result makes generate_with_llm skip
    straight to the template fallback.
    """
    try:
        data = (_fetch_json(FRONTEND_MODELS_URL).get("data")) or {}
        intelligence = {}
        for key, node in (data.get("benchmarks") or {}).items():
            index = (node or {}).get("aa", {}).get("intelligence_index")
            if isinstance(index, (int, float)) and not isinstance(index, bool):
                intelligence[re.sub(r"-\d{8}$", "", key)] = float(index)
        candidates = []
        for entry in data.get("models") or []:
            if not isinstance(entry, dict):
                continue
            slug = entry.get("slug") or ""
            endpoint = entry.get("endpoint") or {}
            if endpoint.get("variant") != "free" or not slug or slug.startswith("~"):
                continue
            bare = re.sub(r"-\d{8}$", "", slug)
            model_id = bare + ":free"
            score = intelligence.get(bare)
            candidates.append((-(score if score is not None else -1.0), model_id))
        ranked = [model_id for _, model_id in sorted(candidates)]
        if ranked:
            return ranked
    except Exception:
        pass
    try:
        models = (_fetch_json(CATALOG_MODELS_URL).get("data")) or []
        free_ids = sorted(
            m.get("id")
            for m in models
            if isinstance(m, dict) and str(m.get("id", "")).endswith(":free")
        )
        return [model_id for model_id in free_ids if model_id]
    except Exception:
        return []


def generate_with_llm(diff, api_key):
    """One grounded call; first model in the chain that answers wins.

    Returns the three texts, or None on no key / all-models failure /
    unparseable output -- the caller then falls back to templates.
    """
    if not api_key:
        return None
    models = resolve_llm_models()
    if not models:
        return None
    body = {
        "messages": [
            {"role": "system", "content": PROMPT_RULES},
            {"role": "user", "content": json.dumps(diff)},
        ],
        "temperature": 0,
        "max_tokens": 300,
    }
    for model in models[:MAX_LLM_MODELS]:
        try:
            payload = _post_chat(model, {**body, "model": model}, api_key)
            content = payload["choices"][0]["message"]["content"]
            stripped = content.strip().removeprefix("```json").removeprefix("```")
            stripped = stripped.removesuffix("```").strip()
            parsed = json.loads(stripped)
        except Exception:
            continue
        if isinstance(parsed, dict) and all(
            isinstance(parsed.get(key), str) and parsed[key].strip()
            for key in SECTION_KEYS
        ):
            return {key: parsed[key] for key in SECTION_KEYS}
    return None


def _validate_catalog_for_diff(catalog) -> None:
    """Check the numeric fields build_diff reads, with a readable error.

    generate_highlights runs before build_site_data validates the catalog,
    so a regression emitting a string where a number belongs would
    otherwise surface as a raw TypeError deep in build_diff instead of a
    loud, clean failure.
    """
    if not isinstance(catalog, dict):
        raise ValueError("catalog document is not an object")
    generated_at = catalog.get("generated_at")
    if not isinstance(generated_at, str) or not generated_at:
        raise ValueError("catalog generated_at must be a non-empty string")
    for key in ("models", "filtered"):
        if not isinstance(catalog.get(key), list):
            raise ValueError(f"catalog {key} must be a list")
    for entry in catalog["models"] + catalog["filtered"]:
        if (
            not isinstance(entry, dict)
            or not isinstance(entry.get("id"), str)
            or not entry["id"]
        ):
            raise ValueError("catalog model entries need non-empty string ids")
    for entry in catalog["models"]:
        pricing = entry.get("pricing")
        if not isinstance(pricing, dict):
            raise ValueError(f"catalog model {entry['id']} pricing must be an object")
        for key in CATALOG_PRICING_KEYS:
            value = _num(pricing.get(key))
            if value is None or value < 0:
                raise ValueError(
                    f"catalog model {entry['id']} pricing.{key} must be a "
                    "non-negative number"
                )
        for key in DIFF_PRIORITY_KEYS:
            if _overall_score(entry, key) is None:
                raise ValueError(
                    f"catalog model {entry['id']} scores.overall.{key} must be a number"
                )
        aa = entry.get("aa")
        if not isinstance(aa, dict) or (
            aa.get("intelligence_index") is not None
            and _num(aa.get("intelligence_index")) is None
        ):
            raise ValueError(
                f"catalog model {entry['id']} aa.intelligence_index must be a "
                "number or null"
            )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate highlight sections from weekly history data"
    )
    parser.add_argument("--catalog", required=True, help="today's catalog.json")
    parser.add_argument(
        "--history", required=True, help="previous history.json (may be absent/empty)"
    )
    parser.add_argument(
        "--prev-highlights", required=True, help="previous highlights.json"
    )
    parser.add_argument("--output", required=True, help="highlights.json destination")
    args = parser.parse_args(argv)
    try:
        with open(args.catalog) as fh:
            catalog = json.load(fh)
        _validate_catalog_for_diff(catalog)
    except (OSError, ValueError) as exc:
        print(f"error: invalid catalog: {exc}", file=sys.stderr)
        return 1
    try:
        with open(args.history) as fh:
            history = json.load(fh)
    except (OSError, ValueError):
        history = {}
    diff = build_diff(catalog, history)

    prev = None
    try:
        with open(args.prev_highlights) as fh:
            prev = json.load(fh)
    except (OSError, ValueError):
        prev = None

    def _prev_is_recent_llm():
        if not isinstance(prev, dict) or prev.get("source") != "openrouter":
            return False
        # Reusing a structurally invalid document would fail validation
        # downstream and wedge every publish until generated_at ages out,
        # so the reuse rule mirrors validate_highlights' shape contract.
        if prev.get("schema_version") != HIGHLIGHTS_SCHEMA_VERSION:
            return False
        sections = prev.get("sections")
        if not isinstance(sections, dict) or any(
            not isinstance(sections.get(key), str) or not sections[key].strip()
            for key in SECTION_KEYS
        ):
            return False
        try:
            generated = datetime.fromisoformat(str(prev.get("generated_at")))
            if generated.tzinfo is None:
                generated = generated.replace(tzinfo=timezone.utc)
        except ValueError:
            return False
        age = datetime.now(timezone.utc) - generated
        return age < timedelta(hours=HIGHLIGHTS_MAX_AGE_HOURS)

    if _prev_is_recent_llm():
        document = prev
    else:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        texts = generate_with_llm(diff, api_key)
        source = "openrouter" if texts is not None else "fallback"
        if texts is None:
            texts = fallback_texts(diff)
        document = {
            "schema_version": HIGHLIGHTS_SCHEMA_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source": source,
            "sections": texts,
        }
    try:
        with open(args.output, "w") as fh:
            json.dump(document, fh, indent=2)
            fh.write("\n")
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
