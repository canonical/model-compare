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
import os
import sys
from datetime import date, datetime, timedelta, timezone

HIGHLIGHTS_SCHEMA_VERSION = 1
SECTION_KEYS = ("week", "intelligence", "prices")
OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
LLM_MODELS = (
    "z-ai/glm-5.3-flash:free",
    "deepseek/deepseek-chat-v3.1:free",
    "meta-llama/llama-3.3-70b-instruct:free",
)
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
            key: {"entries": [], "new_ids": []}
            for key in ("balanced", "price", "quality")
        }
        diff["new_pool_ids_count"] = 0
        diff["new_pool_id_sample"] = []
        diff["aa_movers"] = {"up": [], "down": []}
        diff["price_moves"] = {"down": [], "up": []}
        diff["discounts"] = {"appeared": [], "vanished": []}
        return diff

    prev_pool = set(baseline.get("pool_ids") or [])
    now_pool = set(
        [e["id"] for e in catalog["models"]] + [e["id"] for e in catalog["filtered"]]
    )
    new_pool = sorted(now_pool - prev_pool)

    tabs = {}
    for key in ("balanced", "price", "quality"):
        prev_ranks = {row["id"]: row["rank"] for row in baseline["tabs"].get(key) or []}
        ranked = sorted(
            catalog["models"],
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

    prev_aa = baseline.get("aa") or {}
    movers = []
    for entry in catalog["models"]:
        prev = prev_aa.get(entry["id"])
        now = entry["aa"]["intelligence_index"]
        if prev is None or now is None:
            continue
        movers.append({"id": entry["id"], "delta": round(now - prev, 2), "value": now})
    movers.sort(key=lambda m: (-m["delta"], m["id"]))
    diff["aa_movers"] = {
        "up": movers[:3],
        "down": sorted(movers, key=lambda m: (m["delta"], m["id"]))[:3],
    }

    prev_prices = baseline.get("prices") or {}
    price_moves = []
    appeared, vanished = [], []
    prev_discounts = {
        model_id: row[3] for model_id, row in (baseline.get("prices") or {}).items()
    }
    for entry in catalog["models"]:
        old = prev_prices.get(entry["id"])
        if old is None:
            continue
        new_blended = entry["pricing"]["blended_per_1m"]
        price_moves.append(
            {
                "id": entry["id"],
                "old": old[2],
                "new": new_blended,
                "delta": round(new_blended - old[2], 4),
            }
        )
        old_disc, new_disc = prev_discounts.get(entry["id"]), entry["discount"]
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


def generate_with_llm(diff, api_key):
    """One grounded call; first model in the chain that answers wins.

    Returns the three texts, or None on no key / all-models failure /
    unparseable output -- the caller then falls back to templates.
    """
    if not api_key:
        return None
    body = {
        "messages": [
            {"role": "system", "content": PROMPT_RULES},
            {"role": "user", "content": json.dumps(diff)},
        ],
        "temperature": 0,
        "max_tokens": 300,
    }
    for model in LLM_MODELS:
        try:
            payload = _post_chat(model, body, api_key)
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
    except (OSError, ValueError) as exc:
        print(f"error: could not read catalog: {exc}", file=sys.stderr)
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
    with open(args.output, "w") as fh:
        json.dump(document, fh, indent=2)
        fh.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
