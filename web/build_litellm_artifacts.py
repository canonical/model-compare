#!/usr/bin/env python3
"""Build the LiteLLM rate card, steering companion and merged map.

Reads and validates the published catalog, writes a pure LiteLLM map
fragment, a keyed steering companion, a merged upstream map (when one can be
built safely) and a merge-status document. Fails loudly on a catalog or
mapping defect; a bad upstream fetch or a price-conflicting key collision
only freezes the merged map. Collisions whose price schedules agree are merged
automatically (keeping upstream's richer entry), and a reviewed override file
can resolve conflicts in favor of either the fragment or upstream.
"""

from __future__ import annotations

import argparse
from decimal import Decimal
import hashlib
import json
import math
import re
import sys
from pathlib import Path

from build_site_data import PRECISION_LADDER, validate_catalog

RATE_CARD_NAME = "litellm_rate_card.json"
STEERING_NAME = "litellm_steering.json"
MERGED_NAME = "litellm.json"
STATUS_NAME = "litellm_merge_status.json"
STATUS_SCHEMA_VERSION = 1
OVERRIDES_NAME = "litellm_collision_overrides.json"
MIN_UPSTREAM_ENTRIES = 1000
PRICE_FIELDS = (
    "input_cost_per_token",
    "output_cost_per_token",
    "cache_read_input_token_cost",
    "cache_creation_input_token_cost",
)
ABOVE_RE = re.compile(
    r"^(" + "|".join(PRICE_FIELDS) + r")_above_(\d+(?:\.\d+)?)(k?)_tokens$"
)
WEEKDAY_ALIASES = {
    "mon": 1,
    "monday": 1,
    "tue": 2,
    "tues": 2,
    "tuesday": 2,
    "wed": 3,
    "wednesday": 3,
    "thu": 4,
    "thur": 4,
    "thurs": 4,
    "thursday": 4,
    "fri": 5,
    "friday": 5,
    "sat": 6,
    "saturday": 6,
    "sun": 7,
    "sunday": 7,
}
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


def _numeric(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _same_price(left, right) -> bool:
    return math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-12)


def _token_bound(value) -> int | None:
    if not _numeric(value) or not math.isfinite(value) or value < 0:
        return None
    if isinstance(value, float) and not value.is_integer():
        return None
    return int(value)


def _parse_above_threshold(number_text: str, suffix: str) -> int | None:
    if not math.isfinite(float(number_text)):
        return None
    scaled = Decimal(number_text) * (1000 if suffix == "k" else 1)
    rounded = round(scaled)
    if scaled != rounded:
        return None
    return int(rounded)


def _tiered_schedule(entry: dict) -> list[tuple[int, dict]] | None:
    tiers = entry.get("tiered_pricing")
    if not isinstance(tiers, list) or not tiers:
        return None
    top_output = (
        entry["output_cost_per_token"]
        if _numeric(entry.get("output_cost_per_token"))
        else 0.0
    )
    points: dict[int, dict] = {}
    for tier in tiers:
        if not isinstance(tier, dict):
            continue
        span = tier.get("range")
        if not isinstance(span, (list, tuple)) or len(span) != 2:
            continue
        start = _token_bound(span[0])
        if start is None or not _numeric(span[1]):
            continue
        tier_input = (
            tier["input_cost_per_token"]
            if _numeric(tier.get("input_cost_per_token"))
            else 0.0
        )
        tier_output = (
            tier["output_cost_per_token"]
            if _numeric(tier.get("output_cost_per_token"))
            else top_output
        )
        tier_cache_read = (
            tier["cache_read_input_token_cost"]
            if _numeric(tier.get("cache_read_input_token_cost"))
            else tier_input
        )
        tier_cache_creation = (
            tier["cache_creation_input_token_cost"]
            if _numeric(tier.get("cache_creation_input_token_cost"))
            else tier_input
        )
        points[start] = {
            "input_cost_per_token": tier_input,
            "output_cost_per_token": tier_output,
            "cache_read_input_token_cost": tier_cache_read,
            "cache_creation_input_token_cost": tier_cache_creation,
        }
    if not points:
        return None
    return [(start, points[start]) for start in sorted(points)]


def _price_schedule(entry: dict) -> list[tuple[int, dict]]:
    """Return (start tokens, effective price fields) points for an entry.

    LiteLLM's two threshold encodings have distinct lookup and fallback rules:

    - A non-empty `tiered_pricing` table with at least one valid range takes
      precedence over base and `*_above_*` fields. Each tier resolves omitted
      rates independently (input -> 0.0, output -> top-level output -> 0.0,
      cache read/creation -> that tier's input), without carrying values forward
      from earlier tiers.
    - Otherwise, base fields and `*_above_<N>[k]_tokens` thresholds are read
      and carried forward across boundaries.
    """
    tiered = _tiered_schedule(entry)
    if tiered is not None:
        return tiered

    points: dict[int, dict] = {}

    def add(start: int, values: dict) -> None:
        points.setdefault(start, {}).update(values)

    add(
        0, {field: entry[field] for field in PRICE_FIELDS if _numeric(entry.get(field))}
    )
    for field, value in entry.items():
        if not _numeric(value):
            continue
        match = ABOVE_RE.match(field)
        if match is not None:
            threshold = _parse_above_threshold(match.group(2), match.group(3))
            if threshold is not None:
                add(threshold, {match.group(1): value})

    schedule = []
    current: dict = {}
    for start in sorted(points):
        current.update(points[start])
        schedule.append((start, dict(current)))
    return schedule


def _price_at(schedule: list[tuple[int, dict]], start: int) -> dict:
    current: dict = {}
    for point_start, prices in schedule:
        if point_start > start:
            break
        current.update(prices)
    return current


def _canonical_hours(value) -> tuple[str, ...] | None:
    if value is None:
        return ()
    if isinstance(value, str):
        cleaned = value.strip()
        return (cleaned,) if cleaned else None
    if isinstance(value, (list, tuple)):
        items = []
        for item in value:
            if not isinstance(item, str) or not item.strip():
                return None
            items.append(item.strip())
        return tuple(sorted(set(items)))
    return None


def _canonical_weekdays(value) -> tuple[int, ...] | None:
    if value is None:
        return tuple(range(1, 8))
    if isinstance(value, (list, tuple)):
        days = []
        for item in value:
            if isinstance(item, bool):
                return None
            if isinstance(item, int):
                if not 1 <= item <= 7:
                    return None
                days.append(item)
            elif isinstance(item, str):
                day = WEEKDAY_ALIASES.get(item.strip().lower())
                if day is None:
                    return None
                days.append(day)
            else:
                return None
        return tuple(sorted(set(days)))
    return None


def _canonical_off_peak(block) -> dict | None:
    if not isinstance(block, dict):
        return None
    allowed = set(PRICE_FIELDS) | {"hours_utc", "windows", "weekday_timezone"}
    if any(key not in allowed for key in block):
        return None
    prices: dict = {}
    for field in PRICE_FIELDS:
        if field not in block or block[field] is None:
            continue
        if not _numeric(block[field]):
            return None
        prices[field] = block[field]
    hours = _canonical_hours(block.get("hours_utc"))
    if hours is None:
        return None
    raw_windows = block.get("windows")
    grouped: dict[tuple[int, ...], set[str]] = {}
    if raw_windows is not None:
        if not isinstance(raw_windows, (list, tuple)):
            return None
        for rule in raw_windows:
            if not isinstance(rule, dict):
                return None
            if any(key not in ("hours_utc", "weekdays") for key in rule):
                return None
            rule_hours = _canonical_hours(rule.get("hours_utc"))
            rule_days = _canonical_weekdays(rule.get("weekdays"))
            if rule_hours is None or rule_days is None:
                return None
            if rule_hours and rule_days:
                grouped.setdefault(rule_days, set()).update(rule_hours)
    tz = block.get("weekday_timezone")
    if tz is None:
        canonical_tz = "UTC"
    elif isinstance(tz, str) and tz.strip():
        canonical_tz = "UTC" if tz.strip().upper() == "UTC" else tz.strip()
    else:
        return None
    return {
        "prices": prices,
        "hours_utc": hours,
        "windows": tuple(
            sorted((days, tuple(sorted(hrs))) for days, hrs in grouped.items())
        ),
        "weekday_timezone": canonical_tz,
    }


def _off_peak_equal(ours_off_peak, upstream_off_peak) -> bool:
    if ours_off_peak == upstream_off_peak:
        return True
    ours_canon = _canonical_off_peak(ours_off_peak)
    upstream_canon = _canonical_off_peak(upstream_off_peak)
    if ours_canon is None or upstream_canon is None:
        return False
    if (
        ours_canon["hours_utc"] != upstream_canon["hours_utc"]
        or ours_canon["windows"] != upstream_canon["windows"]
        or ours_canon["weekday_timezone"] != upstream_canon["weekday_timezone"]
    ):
        return False
    ours_prices = ours_canon["prices"]
    upstream_prices = upstream_canon["prices"]
    if set(ours_prices) != set(upstream_prices):
        return False
    return all(
        _same_price(ours_prices[field], upstream_prices[field]) for field in ours_prices
    )


def _off_peak_difference(ours: dict, upstream: dict) -> dict | None:
    ours_off_peak = ours.get("off_peak_pricing")
    upstream_off_peak = upstream.get("off_peak_pricing")
    if ours_off_peak is None and upstream_off_peak is None:
        return None
    if (
        ours_off_peak is not None
        and upstream_off_peak is not None
        and _off_peak_equal(ours_off_peak, upstream_off_peak)
    ):
        return None
    return {
        "field": "off_peak_pricing",
        "from_tokens": "off-peak",
        "model_compare": "present" if ours_off_peak is not None else "absent",
        "upstream": "present" if upstream_off_peak is not None else "absent",
    }


def _price_differences(
    ours: dict, upstream: dict
) -> tuple[list[dict], int, str | None]:
    """Return (differences, comparable field count, note).

    A collision is price-equivalent only when both base `input_cost_per_token`
    and `output_cost_per_token` are present and match, every price present in
    ours at any tier boundary is present and matches upstream, and any off-peak
    schedule agrees. Representation differences (`tiered_pricing` vs
    `*_above_...`) are not differences. Off-peak schedules are compared
    semantically on prices/hours/weekdays; other structural differences are
    reported conservatively rather than guessed at.
    """
    ours_schedule = _price_schedule(ours)
    upstream_schedule = _price_schedule(upstream)
    ours_base = _price_at(ours_schedule, 0)
    upstream_base = _price_at(upstream_schedule, 0)
    if not (
        "input_cost_per_token" in ours_base
        and "output_cost_per_token" in ours_base
        and "input_cost_per_token" in upstream_base
        and "output_cost_per_token" in upstream_base
    ):
        return [], 0, "no comparable base price"

    starts = sorted(
        {start for start, _ in ours_schedule}
        | {start for start, _ in upstream_schedule}
    )
    differences = []
    comparable = 0
    for start in starts:
        ours_prices = _price_at(ours_schedule, start)
        upstream_prices = _price_at(upstream_schedule, start)
        for field in PRICE_FIELDS:
            if field not in ours_prices:
                continue
            comparable += 1
            if field not in upstream_prices:
                differences.append(
                    {
                        "field": field,
                        "from_tokens": start,
                        "model_compare": ours_prices[field],
                        "upstream": "absent",
                    }
                )
                continue
            if not _same_price(ours_prices[field], upstream_prices[field]):
                differences.append(
                    {
                        "field": field,
                        "from_tokens": start,
                        "model_compare": ours_prices[field],
                        "upstream": upstream_prices[field],
                    }
                )
    note = None
    off_peak = _off_peak_difference(ours, upstream)
    if off_peak is not None:
        differences.append(off_peak)
        comparable += 1
        note = "off-peak pricing differs"

    def field_order(field: str) -> int:
        return PRICE_FIELDS.index(field) if field in PRICE_FIELDS else len(PRICE_FIELDS)

    def start_order(value) -> tuple[int, int | str]:
        if isinstance(value, int) and not isinstance(value, bool):
            return (0, value)
        return (1, str(value))

    differences.sort(
        key=lambda item: (field_order(item["field"]), start_order(item["from_tokens"]))
    )
    return differences, comparable, note


def _validate_overrides(overrides) -> dict:
    if not isinstance(overrides, dict):
        raise ValueError("collision overrides must be an object")
    for key, value in overrides.items():
        if not isinstance(key, str) or not key.startswith("openrouter/"):
            raise ValueError(
                f"collision override key is not an openrouter/ key: {key!r}"
            )
        if not isinstance(value, dict):
            raise ValueError(f"collision override {key} is not an object")
        if value.get("resolution") not in ("fragment", "upstream"):
            raise ValueError(
                f"collision override {key} resolution must be fragment or upstream"
            )
        reason = value.get("reason")
        if reason is not None and not isinstance(reason, str):
            raise ValueError(f"collision override {key} reason is not a string")
    return overrides


def load_collision_overrides(path: Path | str | None) -> dict:
    """Load the reviewed collision overrides, returning {} when absent."""
    if path is None:
        return {}
    try:
        raw = Path(path).read_bytes()
    except FileNotFoundError:
        return {}
    except OSError as exc:
        raise ValueError(f"cannot read collision overrides: {exc}") from exc
    try:
        doc = json.loads(raw)
    except ValueError as exc:
        raise ValueError(f"collision overrides are not JSON: {exc}") from exc
    if not isinstance(doc, dict):
        raise ValueError("collision overrides must be a JSON object")
    schema_version = doc.get("schema_version")
    if type(schema_version) is not int or schema_version != 1:
        raise ValueError("collision overrides schema_version must be 1")
    overrides = doc.get("overrides")
    if not isinstance(overrides, dict):
        raise ValueError("collision overrides lacks an overrides object")
    return _validate_overrides(overrides)


def _collision_detail(
    key: str, ours: dict, upstream: dict, override: dict | None
) -> dict:
    differences, comparable, note = _price_differences(ours, upstream)
    price_equal = comparable > 0 and not differences
    resolution = "upstream"
    overridden = False
    override_reason = None
    if override is not None:
        if not isinstance(override, dict):
            raise ValueError(f"collision override {key} is not an object")
        resolution = override.get("resolution")
        if resolution not in ("fragment", "upstream"):
            raise ValueError(
                f"collision override {key} resolution must be fragment or upstream"
            )
        overridden = True
        reason = override.get("reason")
        if isinstance(reason, str) and reason:
            override_reason = reason
    elif not price_equal:
        resolution = "blocked"
    detail = {
        "key": key,
        "resolution": resolution,
        "price_equal": price_equal,
        "overridden": overridden,
        "differences": differences,
    }
    if note is not None:
        detail["note"] = note
    if override_reason is not None:
        detail["override_reason"] = override_reason
    return detail


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
    overrides: dict | None = None,
) -> tuple[bytes | None, dict]:
    """Return (merged bytes or None, status document)."""
    overrides = {} if overrides is None else _validate_overrides(overrides)
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
    collision_details = []
    blocking = []

    if upstream is not None:
        rate_card = json.loads(rate_card_bytes)
        collisions = sorted(set(upstream) & set(rate_card))
        collision_details = [
            _collision_detail(key, rate_card[key], upstream[key], overrides.get(key))
            for key in collisions
        ]
        blocking = [
            detail for detail in collision_details if detail["resolution"] == "blocked"
        ]
        if not blocking:
            details_by_key = {detail["key"]: detail for detail in collision_details}
            merged = dict(upstream)
            for key, entry in rate_card.items():
                detail = details_by_key.get(key)
                if detail is None or detail["resolution"] == "fragment":
                    merged[key] = entry
            expected = len(upstream) + len(rate_card) - len(collisions)
            if len(merged) != expected:
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

    report_needed = bool(blocking) or (effective_status != "skipped" and failures >= 3)
    report_reason = None
    if blocking:
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
            "collision_details": collision_details,
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
    parser.add_argument(
        "--collision-overrides",
        default=str(Path(__file__).with_name(OVERRIDES_NAME)),
    )
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
        load_collision_overrides(args.collision_overrides),
    )
    status["upstream"]["url"] = args.upstream_url
    steering = build_steering(catalog, rate_card_bytes, args.revision, warnings)
    write_artifacts(
        Path(args.output_dir), rate_card_bytes, steering, merged_bytes, status
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
