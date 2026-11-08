#!/usr/bin/env python3
"""Build the LiteLLM rate card, steering companion and merged map.

Reads and validates the published catalog, writes a pure LiteLLM map
fragment, a keyed steering companion, a merged upstream map (when one can be
built safely) and a merge-status document. Fails loudly on a catalog or
mapping defect; a bad upstream fetch or a key collision only freezes the
merged map.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from build_site_data import PRECISION_LADDER, validate_catalog

RATE_CARD_NAME = "litellm_rate_card.json"
STEERING_NAME = "litellm_steering.json"
MERGED_NAME = "litellm.json"
STATUS_NAME = "litellm_merge_status.json"
STATUS_SCHEMA_VERSION = 1
MIN_UPSTREAM_ENTRIES = 1000
UPSTREAM_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/main/"
    "model_prices_and_context_window.json"
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


def _per_token(price_per_1m) -> float:
    return price_per_1m / 1_000_000.0


def _entry_key(model_id: str) -> str:
    if not isinstance(model_id, str) or not model_id:
        raise ValueError(f"model id is not a non-empty string: {model_id!r}")
    if model_id.startswith("openrouter/"):
        raise ValueError(
            f"model id already carries the openrouter/ prefix: {model_id!r}"
        )
    return "openrouter/" + model_id


def _positive_int(value) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def _reasoning_flags(model: dict) -> dict:
    reasoning = model.get("reasoning")
    reasoning = reasoning if isinstance(reasoning, dict) else {}
    supported = []
    for effort in reasoning.get("supported_efforts") or []:
        if isinstance(effort, str) and effort and effort not in supported:
            supported.append(effort)
    default = reasoning.get("default_effort")
    if not isinstance(default, str) or not default:
        default = None
    mandatory = reasoning.get("mandatory") is True
    flags = {
        "supports_reasoning": (
            reasoning.get("default_enabled") is True
            or any(effort != "none" for effort in supported)
        ),
        "supports_none_reasoning_effort": "none" in supported and not mandatory,
    }
    if supported:
        flags["reasoning_effort_levels"] = supported
    if default is not None:
        flags["default_reasoning_effort"] = default
    return flags


def _range_entry(start: int, end: int, tier: dict) -> dict:
    entry = {
        "range": [start, end],
        "input_cost_per_token": _per_token(tier["input_per_1m"]),
        "output_cost_per_token": _per_token(tier["output_per_1m"]),
    }
    cache_read = tier.get("input_cache_read_per_1m")
    if cache_read is not None:
        entry["cache_read_input_token_cost"] = _per_token(cache_read)
    cache_write = tier.get("input_cache_write_per_1m")
    if cache_write is not None:
        entry["cache_creation_input_token_cost"] = _per_token(cache_write)
    return entry


def _tiered_pricing(model: dict) -> list[dict] | None:
    tiers = model["pricing"].get("tiers") or []
    if not tiers:
        return None
    context = model.get("context")
    if isinstance(context, bool) or not isinstance(context, int) or context <= 0:
        raise ValueError(f"{model['id']}: context is not a positive integer")
    thresholds = []
    for tier in tiers:
        threshold = tier.get("min_prompt_tokens")
        if isinstance(threshold, bool) or not isinstance(threshold, int):
            raise ValueError(f"{model['id']}: tier threshold is not an integer")
        thresholds.append(threshold)
    if thresholds[0] <= 0:
        raise ValueError(f"{model['id']}: first tier threshold must be positive")
    if any(b <= a for a, b in zip(thresholds, thresholds[1:])):
        raise ValueError(f"{model['id']}: tier thresholds must be strictly ascending")
    if thresholds[-1] >= context:
        raise ValueError(f"{model['id']}: last tier threshold must be below context")
    base = model["pricing"]["base"]
    ranges = [_range_entry(0, thresholds[0], base)]
    for index in range(len(thresholds) - 1):
        ranges.append(
            _range_entry(thresholds[index], thresholds[index + 1], tiers[index])
        )
    ranges.append(_range_entry(thresholds[-1], context, tiers[-1]))
    return ranges


def _hours(window: dict) -> str:
    start = window["utc_start"]
    end = window["utc_end"]
    return f"{start // 100:02d}:{start % 100:02d}-{end // 100:02d}:{end % 100:02d}"


def _off_peak_pricing(model: dict, warnings: list[str]) -> dict | None:
    pricing = model["pricing"]
    schedule = pricing.get("schedule")
    if not schedule:
        return None
    if pricing.get("tiers"):
        warnings.append(
            f"{model['id']}: tiered and scheduled pricing; off_peak_pricing omitted"
        )
        return None
    base = pricing["base"]
    peak = (
        base["input_per_1m"],
        base["output_per_1m"],
        base.get("input_cache_read_per_1m"),
        base.get("input_cache_write_per_1m"),
    )
    off_peak = []
    levels = set()
    for window in schedule:
        level = (
            window["input_per_1m"],
            window["output_per_1m"],
            window.get("input_cache_read_per_1m"),
            window.get("input_cache_write_per_1m"),
        )
        if level == peak:
            continue
        off_peak.append((window, level))
        levels.add(level)
    if not off_peak:
        return None
    if len(levels) > 1:
        warnings.append(
            f"{model['id']}: multiple off-peak price levels; off_peak_pricing omitted"
        )
        return None
    _, level = off_peak[0]
    block = {
        "input_cost_per_token": _per_token(level[0]),
        "output_cost_per_token": _per_token(level[1]),
    }
    if level[2] is not None:
        block["cache_read_input_token_cost"] = _per_token(level[2])
    if level[3] is not None:
        block["cache_creation_input_token_cost"] = _per_token(level[3])
    every_day = []
    day_windows = []
    all_days = set(WEEKDAYS)
    for window, _ in off_peak:
        hours = _hours(window)
        days = window.get("utc_days")
        if days is None or set(days) == all_days:
            every_day.append(hours)
        else:
            day_windows.append(
                {
                    "hours_utc": [hours],
                    "weekdays": [WEEKDAYS.index(day) + 1 for day in days],
                }
            )
    if every_day:
        block["hours_utc"] = every_day
    if day_windows:
        block["windows"] = day_windows
    return block


def _rate_entry(model: dict, warnings: list[str]) -> dict:
    pricing = model["pricing"]
    base = pricing["base"]
    context = model.get("context")
    if isinstance(context, bool) or not isinstance(context, int) or context <= 0:
        raise ValueError(f"{model['id']}: context is not a positive integer")
    modalities = model.get("input_modalities") or []
    entry = {
        "litellm_provider": "openrouter",
        "mode": "chat",
        "input_cost_per_token": _per_token(base["input_per_1m"]),
        "output_cost_per_token": _per_token(base["output_per_1m"]),
        "max_input_tokens": context,
        "supports_function_calling": bool(model.get("tool_calling")),
        "supports_vision": "image" in modalities,
    }
    max_output = _positive_int(model.get("max_output_tokens"))
    if max_output is not None:
        entry["max_output_tokens"] = max_output
    cache_read = base.get("input_cache_read_per_1m")
    if cache_read is not None:
        entry["cache_read_input_token_cost"] = _per_token(cache_read)
    cache_write = base.get("input_cache_write_per_1m")
    if cache_write is not None:
        entry["cache_creation_input_token_cost"] = _per_token(cache_write)
    entry.update(_reasoning_flags(model))
    tiers = _tiered_pricing(model)
    if tiers is not None:
        entry["tiered_pricing"] = tiers
    off_peak = _off_peak_pricing(model, warnings)
    if off_peak is not None:
        entry["off_peak_pricing"] = off_peak
    return entry


def build_rate_card(catalog: dict) -> tuple[dict, list[str]]:
    """Return (rate_card, sorted warnings) for a validated catalog."""
    validate_catalog(catalog)
    if not catalog.get("models"):
        raise ValueError("catalog has no models")
    card = {}
    warnings = []
    for model in catalog["models"]:
        key = _entry_key(model["id"])
        if key in card:
            raise ValueError(f"duplicate rate-card key: {key}")
        card[key] = _rate_entry(model, warnings)
    return card, sorted(set(warnings))


def serialize_readable(doc: dict) -> bytes:
    return (json.dumps(doc, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode(
        "utf-8"
    )


def serialize_compact(doc: dict) -> bytes:
    return (
        json.dumps(
            doc,
            separators=(",", ":"),
            sort_keys=True,
            ensure_ascii=True,
        )
        + "\n"
    ).encode("utf-8")


def _reasoning(model: dict) -> dict:
    reasoning = model.get("reasoning")
    reasoning = reasoning if isinstance(reasoning, dict) else {}
    supported = []
    for effort in reasoning.get("supported_efforts") or []:
        if isinstance(effort, str) and effort and effort not in supported:
            supported.append(effort)
    default = reasoning.get("default_effort")
    if not isinstance(default, str) or not default:
        default = None
    preferred = model.get("quality_effort")
    if not isinstance(preferred, str) or not preferred:
        preferred = default
    return {
        "preferred": {"effort": preferred} if preferred else None,
        "model_default": {"effort": default} if default else None,
        "supported_efforts": supported,
        "mandatory": reasoning.get("mandatory") is True,
        "supports_max_tokens": reasoning.get("supports_max_tokens"),
    }


def build_steering(
    catalog: dict, rate_card_bytes: bytes, revision: str, warnings: list[str]
) -> dict:
    validate_catalog(catalog)
    entries = {}
    for model in catalog["models"]:
        key = _entry_key(model["id"])
        steering = model.get("steering")
        if not isinstance(steering, dict):
            raise ValueError(f"{model['id']}: missing steering block")
        provider = steering.get("provider")
        if not isinstance(provider, dict):
            raise ValueError(f"{model['id']}: steering.provider is not an object")
        order = provider.get("order")
        if (
            not isinstance(order, list)
            or not order
            or any(not isinstance(tag, str) or not tag for tag in order)
        ):
            raise ValueError(f"{model['id']}: steering.provider.order is unusable")
        copied_provider = {}
        for name in ("order", "ignore", "quantizations", "allow_fallbacks"):
            if name in provider:
                copied_provider[name] = provider[name]
        base = model["pricing"]["base"]
        strict = {
            "order": [order[0]],
            "allow_fallbacks": False,
            "max_price": {
                "prompt": base["input_per_1m"],
                "completion": base["output_per_1m"],
            },
        }
        baseline_endpoint = next(
            (
                endpoint
                for endpoint in (model.get("endpoints") or [])
                if isinstance(endpoint, dict) and endpoint.get("tag") == order[0]
            ),
            None,
        )
        quantization = (
            baseline_endpoint.get("quantization")
            if isinstance(baseline_endpoint, dict)
            else None
        )
        if quantization in PRECISION_LADDER:
            strict["quantizations"] = [quantization]
        entries[key] = {
            "provider": copied_provider,
            "provider_strict": strict,
            "reasoning": _reasoning(model),
        }
    meta = {
        "schema_version": STATUS_SCHEMA_VERSION,
        "generated_at": catalog["generated_at"],
        "source_revision": revision,
        "catalog_schema_version": catalog["schema_version"],
        "rate_card_sha256": hashlib.sha256(rate_card_bytes).hexdigest(),
        "entries": len(entries),
        "warnings": sorted(set(warnings)),
    }
    return {"_meta": meta, **entries}


def parse_upstream(raw: bytes | None) -> dict | None:
    if raw is None:
        return None
    try:
        doc = json.loads(raw)
    except (ValueError, TypeError):
        return None
    if not isinstance(doc, dict) or len(doc) < MIN_UPSTREAM_ENTRIES:
        return None
    if any(not isinstance(value, dict) for value in doc.values()):
        return None
    return doc


def _previous_status(prev_status: dict | None) -> dict:
    return prev_status if isinstance(prev_status, dict) else {}


def _previous_merged(
    prev_merged_bytes: bytes | None,
) -> tuple[dict | None, str | None, int | None]:
    if prev_merged_bytes is None:
        return None, None, None
    digest = hashlib.sha256(prev_merged_bytes).hexdigest()
    try:
        doc = json.loads(prev_merged_bytes)
    except (ValueError, TypeError):
        return None, digest, None
    if not isinstance(doc, dict):
        return None, digest, None
    return doc, digest, len(doc)


def _status_doc(catalog: dict, revision: str) -> dict:
    return {
        "_meta": {
            "schema_version": STATUS_SCHEMA_VERSION,
            "generated_at": catalog["generated_at"],
            "source_revision": revision,
            "catalog_schema_version": catalog["schema_version"],
        }
    }


def build_merge_status(
    catalog: dict,
    revision: str,
    rate_card_bytes: bytes,
    upstream_status: str,
    upstream_bytes: bytes | None,
    prev_merged_bytes: bytes | None,
    prev_status: dict | None,
) -> tuple[bytes | None, dict]:
    """Return (merged bytes or None, status document)."""
    previous = _previous_status(prev_status)
    previous_failures = previous.get("upstream_fetch_failures")
    if not isinstance(previous_failures, int) or isinstance(previous_failures, bool):
        previous_failures = 0
    prev_doc, prev_sha, prev_entries = _previous_merged(prev_merged_bytes)
    prev_rate_card_sha = None
    previous_merged_status = previous.get("merged")
    if isinstance(previous_merged_status, dict) and (
        "rate_card_sha256" in previous_merged_status
    ):
        candidate = previous_merged_status.get("rate_card_sha256")
        prev_rate_card_sha = candidate if isinstance(candidate, str) else None
    else:
        # Older status files had no merged block; their rate_card hash is
        # the only provenance available.
        previous_rate_card = previous.get("rate_card")
        if isinstance(previous_rate_card, dict):
            candidate = previous_rate_card.get("sha256")
            if isinstance(candidate, str):
                prev_rate_card_sha = candidate

    upstream = parse_upstream(upstream_bytes) if upstream_status == "ok" else None
    effective_status = upstream_status
    if upstream_status == "ok" and upstream is None:
        effective_status = "failed"
    if effective_status == "failed":
        failures = previous_failures + 1
        reason = (
            "upstream_invalid" if upstream_status == "ok" else "upstream_fetch_failed"
        )
    elif effective_status == "skipped":
        failures = previous_failures
        reason = "not_attempted"
    else:
        failures = 0
        reason = None

    merged_bytes = None
    merged_sha = None
    merged_entries = None
    merged_rate_card_sha = None
    status = "skipped"
    collisions = []

    if upstream is not None:
        rate_card = json.loads(rate_card_bytes)
        collisions = sorted(set(upstream) & set(rate_card))
        if not collisions:
            merged = dict(upstream)
            merged.update(rate_card)
            if len(merged) != len(upstream) + len(rate_card):
                raise ValueError("merged map entry count does not add up")
            merged_bytes = serialize_compact(merged)
            merged_sha = hashlib.sha256(merged_bytes).hexdigest()
            merged_entries = len(merged)
            merged_rate_card_sha = hashlib.sha256(rate_card_bytes).hexdigest()
            status = "ok"
        else:
            reason = "collision"

    if status != "ok":
        if prev_doc is not None:
            merged_bytes = prev_merged_bytes
            merged_sha = prev_sha
            merged_entries = prev_entries
            merged_rate_card_sha = prev_rate_card_sha
            status = "frozen"
        else:
            status = "skipped"
            reason = reason or "no_previous_map"

    report_needed = bool(collisions) or (
        effective_status != "skipped" and failures >= 3
    )
    report_reason = None
    if collisions:
        report_reason = "collision"
    elif effective_status != "skipped" and failures >= 3:
        report_reason = "upstream_fetch_failures"

    doc = _status_doc(catalog, revision)
    doc.update(
        {
            "rate_card": {
                "sha256": hashlib.sha256(rate_card_bytes).hexdigest(),
                "entries": len(json.loads(rate_card_bytes)),
            },
            "upstream": {
                "url": UPSTREAM_URL,
                "sha256": (
                    hashlib.sha256(upstream_bytes).hexdigest()
                    if upstream is not None
                    else None
                ),
                "entries": len(upstream) if upstream is not None else None,
            },
            "merged": {
                "status": status,
                "reason": reason,
                "sha256": merged_sha,
                "entries": merged_entries,
                "rate_card_sha256": merged_rate_card_sha,
            },
            "collisions": collisions,
            "upstream_fetch_failures": failures,
            "report": {
                "needed": report_needed,
                "reason": report_reason,
            },
        }
    )
    return merged_bytes, doc


def write_artifacts(
    output_dir: Path,
    rate_card_bytes: bytes,
    steering: dict,
    merged_bytes: bytes | None,
    status: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / RATE_CARD_NAME).write_bytes(rate_card_bytes)
    (output_dir / STEERING_NAME).write_bytes(serialize_readable(steering))
    (output_dir / STATUS_NAME).write_bytes(serialize_readable(status))
    if merged_bytes is not None:
        (output_dir / MERGED_NAME).write_bytes(merged_bytes)
    elif (output_dir / MERGED_NAME).exists():
        (output_dir / MERGED_NAME).unlink()


def _read(path: str | None) -> bytes | None:
    if not path:
        return None
    try:
        return Path(path).read_bytes()
    except OSError:
        return None


def _read_json(path: str | None):
    raw = _read(path)
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="build_litellm_artifacts.py")
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument(
        "--upstream-status", choices=("ok", "failed", "skipped"), default="skipped"
    )
    parser.add_argument("--upstream-map")
    parser.add_argument("--upstream-url", default=UPSTREAM_URL)
    parser.add_argument("--prev-merged")
    parser.add_argument("--prev-status")
    args = parser.parse_args(argv)

    raw_catalog = _read(args.catalog)
    if raw_catalog is None:
        raise RuntimeError(f"unreadable catalog: {args.catalog}")
    try:
        catalog = json.loads(raw_catalog)
    except ValueError as exc:
        raise RuntimeError(f"catalog is not JSON: {exc}") from exc
    validate_catalog(catalog)

    rate_card, warnings = build_rate_card(catalog)
    rate_card_bytes = serialize_readable(rate_card)
    upstream_bytes = _read(args.upstream_map)
    if args.upstream_status == "ok" and upstream_bytes is None:
        upstream_status = "failed"
    else:
        upstream_status = args.upstream_status
    merged_bytes, status = build_merge_status(
        catalog,
        args.revision,
        rate_card_bytes,
        upstream_status,
        upstream_bytes,
        _read(args.prev_merged),
        _read_json(args.prev_status),
    )
    status["upstream"]["url"] = args.upstream_url
    steering = build_steering(catalog, rate_card_bytes, args.revision, warnings)
    write_artifacts(
        Path(args.output_dir), rate_card_bytes, steering, merged_bytes, status
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
