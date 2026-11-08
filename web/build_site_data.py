#!/usr/bin/env python3
"""Build data.json for the published model-compare site.

Reads the JSON output of `model_compare.py --priority P --json --top 20` for
each priority plus the `--best` output, validates everything, and writes a
single data.json consumed by site/index.html. Fails loudly on anything
unexpected so a broken run never deploys a broken site.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from datetime import date, datetime, timedelta, timezone

PRIORITIES = ("balanced", "price", "quality")
ROW_KEYS = (
    "model",
    "opencode_model",
    "name",
    "score",
    "quality_index",
    "input_usd_per_m",
    "output_usd_per_m",
    "blended_usd_per_m",
    "discount",
    "discount_pct",
    "context_tokens",
    "age_days",
    # data.json v2 row keys (Task 9): exactly the additive keys
    # model_compare.print_json appends (default_precision first,
    # quant_note last). print_json always emits them -- pre-endpoint
    # candidates degrade to null -- so the missing-key check below
    # demands them on every row.
    "default_precision",
    "risk_flag",
    "price_spread",
    "session_cost_usd",
    "quant_note",
    # data.json v2 additive provenance (quality source reconciliation):
    # print_json emits the quality record's source label, its as_of, the
    # drift verdict against OpenRouter's check value and the check pair
    # itself; rows without a record degrade to null. The missing-key check
    # demands them like the v2 keys above; the schema stays 2 (additive).
    "quality_source",
    "quality_as_of",
    "quality_drift",
    "quality_or_index",
    "quality_or_as_of",
    # data.json v3 row keys (thinking depth): the effort the main quality
    # was measured at and the model's effective default effort. print_json
    # always emits both -- unknown efforts degrade to null -- so the
    # missing-key check demands them on every row. The nested reasoning
    # block is NOT a row key: build_data drops it (the site's depth table
    # joins it from catalog.json).
    "quality_effort",
    "default_effort",
)
# provider-qualified opencode id (openrouter/<provider/model>), the plain
# catalog id (<provider/model>), each with an optional :variant suffix.
MODEL_ID_RE = re.compile(
    r"^(?:[a-zA-Z0-9_.-]+/)?[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+"
    r"(?::[a-zA-Z0-9_.-]+)?$"
)
# A git commit hash in short (7) to full (40) lowercase hex form -- the
# source_revision stamp the site footer links to. Mirrors publish.py's
# REVISION_RE (build_site_data is subprocess-isolated from publish by
# convention, so the contract is duplicated here like the constants
# above): whatever is stamped into data.json must be a hash, never
# arbitrary text that would ship a dead footer link.
REVISION_RE = re.compile(r"[0-9a-f]{7,40}")

CATALOG_SCHEMA_VERSION = 6

# Duplicated from model_compare.py: build_site_data is subprocess-isolated
# from the CLI, and the catalog constants are duplicated here by convention
# (same precedent as CATALOG_ENTRY_KEYS below). The ladder keys are the only
# quantization labels a recorded endpoint may name besides "unknown" and
# null; default_precision is restricted to the ladder keys alone -- the
# derivation never yields the "unknown" sentinel.
PRECISION_LADDER = {
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
QUANTIZATION_ENUM = frozenset(PRECISION_LADDER) | {"unknown"}

CATALOG_ENTRY_KEYS = (
    "id",
    "name",
    "provider",
    "family",
    "pricing",
    "context",
    "listed_at",
    "age_days",
    "tool_calling",
    "zdr",
    "discount",
    "expired",
    "quality",
    "aa",
    "quality_match",
    # Schema v4 provenance (quality source reconciliation): the record's
    # own as_of and drift verdict, emitted on every entry and null when
    # the candidate matched no quality source.
    "quality_as_of",
    "quality_drift",
    # Schema v5 (thinking depth): the effort the main quality was measured
    # at, and the model's reasoning metadata plus every measured depth
    # record (the site's depth table and the plugin read the block).
    "quality_effort",
    "reasoning",
    "scores",
    # Schema v3 additive blocks (Task 8/9): the producer emits all four on
    # every entry, null when the candidate skipped the endpoint stage.
    "endpoints",
    "default_precision",
    "advertised_pricing",
    "annotations",
)
# Keys of one parsed endpoint (model_compare.parse_endpoints' output shape).
CATALOG_ENDPOINT_KEYS = (
    "provider_name",
    "tag",
    "quantization",
    "pricing",
    "context_length",
    "status",
    "service_tier",
)
# Keys of the per-model annotations block (design section 4.6).
CATALOG_ANNOTATION_KEYS = (
    "price_spread",
    "cheapest_quantized",
    "risk_flag",
    "baseline_context_below_catalog",
)
# Keys of the per-model reasoning block (schema v5): the metadata fields the
# site and plugin read plus the measured depth records. Only the three keys
# the depth contract depends on are required; the producer's other metadata
# fields are tolerated.
CATALOG_REASONING_KEYS = ("default_effort", "supported_efforts", "depths")
# One measured depth record: effort label, AA intelligence index at that
# effort, AA cost per benchmark task (null when unreported), source label and
# its as_of. `index`, never `quality_index`.
CATALOG_DEPTH_KEYS = ("effort", "index", "cost_per_task_usd", "source", "as_of")
# One display-only unlabeled AA reference: the intelligence index and AA's
# benchmark cost, both measured without a reasoning-effort label, plus the
# source label and its as_of. Deliberately no `effort` key.
CATALOG_UNLABELED_KEYS = ("index", "cost_per_task_usd", "source", "as_of")
# The OpenRouter session-cost dataset's four turn ranges (brainstorm
# section 2.6): the only keys a medians cell dict may carry.
SESSION_COST_TURN_RANGES = ("1-turn", "2-9-turns", "10-49-turns", "50-plus-turns")
CATALOG_PRICING_KEYS = ("input_per_1m", "output_per_1m", "blended_per_1m")
CATALOG_BASE_KEYS = ("input_per_1m", "output_per_1m", "blended_per_1m")
CATALOG_WEEKDAYS = (
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
)
CATALOG_SCHEDULE_ENTRY_KEYS = (
    "id",
    "name",
    "context",
    "quality",
    # Provenance for the site's QUAL tooltip (data.json v2 additive): the
    # winning quality tier's label and stamp, null when the row's quality
    # came without any.
    "quality_source",
    "quality_as_of",
    # Schema v5: the effort the scheduled row's quality was measured at
    # (data.json v3 additive), null when unknown.
    "quality_effort",
    "score",
    "max_discount",
    "sched_note",
    "sched_detail",
    "peak",
    "offpeak",
    "schedule",
)
CATALOG_SCORE_KEYS = ("price", "quality", "context", "age")
CATALOG_OVERALL_KEYS = ("balanced", "price", "quality")
CATALOG_AA_KEYS = ("intelligence_index", "coding_index", "agentic_index")
CATALOG_QUALITY_MATCH_VALUES = ("aa", "catalog", "openrouter")
# sources.aa.mode is the run's quality chain, not a match outcome: "aa"
# when the AA API returned usable entries (a keyed run whose fetch failed
# publishes "none"), "catalog" when unkeyed runs hydrated the published
# catalog, "none" when neither tier produced data. OpenRouter's benchmarks
# are the chain's shared last resort and carry their own count
# (matched_openrouter), never a mode.
CATALOG_AA_MODES = ("aa", "catalog", "none")


def _is_score(value) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and 0.0 <= value <= 1.0
    )


def _is_aa_value(value) -> bool:
    if value is None:
        return True
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and 0.0 <= value <= 100.0
    )


def _validate_rankings(rankings, model_ids) -> None:
    """rankings is the single ranking authority every projection reads: per
    priority, a permutation of the models ids (each exactly once)."""
    if not isinstance(rankings, dict) or set(rankings) != set(CATALOG_OVERALL_KEYS):
        raise ValueError(
            "catalog rankings must have exactly the keys: "
            + ", ".join(CATALOG_OVERALL_KEYS)
        )
    expected = set(model_ids)
    for priority in CATALOG_OVERALL_KEYS:
        ranking = rankings[priority]
        if not isinstance(ranking, list) or not all(
            isinstance(model_id, str) and model_id for model_id in ranking
        ):
            raise ValueError(
                f"catalog rankings.{priority} must be a list of non-empty id strings"
            )
        if len(set(ranking)) != len(ranking):
            raise ValueError(f"catalog rankings.{priority} contains duplicate ids")
        if set(ranking) != expected or len(ranking) != len(model_ids):
            missing = sorted(expected - set(ranking))
            unknown = sorted(set(ranking) - expected)
            raise ValueError(
                f"catalog rankings.{priority} is not a permutation of models ids"
                f" (missing: {missing}, unknown: {unknown})"
            )


def _validate_window(window, label) -> None:
    """One schedule window: HHMM clocks, weekday names, coverage, prices."""
    if not isinstance(window, dict):
        raise ValueError(f"{label} window is not an object")
    for key in ("utc_start", "utc_end"):
        value = window.get(key)
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or not 0 <= value <= 2359
            or value % 100 > 59
        ):
            raise ValueError(
                f"{label} {key} must be an HHMM int in 0..2359 with minutes < 60"
            )
    days = window.get("utc_days")
    if days is not None and (
        not isinstance(days, list)
        or not days
        or any(day not in CATALOG_WEEKDAYS for day in days)
    ):
        raise ValueError(f"{label} utc_days must be null or weekday names")
    coverage = window.get("coverage")
    if not _is_number(coverage) or not 0 < coverage <= 1:
        raise ValueError(f"{label} coverage must be a number in (0, 1]")
    for key in ("input_per_1m", "output_per_1m", "blended_per_1m"):
        value = window.get(key)
        if not _is_number(value) or value < 0:
            raise ValueError(f"{label} {key} must be a non-negative number")
    _validate_optional_cache_rates(window, label)


def _validate_schedule_list(schedule, label) -> None:
    """Null, or a non-empty window list whose coverages sum to 1.

    Tolerance: max(1e-3, n * 5e-5) for n windows (per-window 4-decimal
    rounding in the producer).
    """
    if schedule is None:
        return
    if not isinstance(schedule, list) or not schedule:
        raise ValueError(f"{label} must be null or a non-empty list")
    total = 0.0
    for i, window in enumerate(schedule):
        _validate_window(window, f"{label}[{i}]")
        total += window["coverage"]
    # The producer rounds each window's coverage to 4 decimals (up to 5e-5
    # error per window), so the tolerance grows with the window count; the
    # 1e-3 floor keeps short schedules strict.
    tolerance = max(1e-3, len(schedule) * 5e-5 + 1e-9)
    if abs(total - 1.0) > tolerance:
        raise ValueError(f"{label} coverages must sum to 1 (got {total})")


def _validate_optional_cache_rates(block, label) -> None:
    """Optional per-1M cache rates: null or a finite non-negative number."""
    for key in ("input_cache_read_per_1m", "input_cache_write_per_1m"):
        if key not in block:
            continue
        value = block[key]
        if value is not None and (not _is_number(value) or value < 0):
            raise ValueError(f"{label} {key} must be a non-negative number or null")


def _validate_price_block(block, label) -> None:
    if not isinstance(block, dict) or any(
        key not in block for key in CATALOG_BASE_KEYS
    ):
        raise ValueError(f"{label} is incomplete")
    bad = [
        key for key in CATALOG_BASE_KEYS if not _is_number(block[key]) or block[key] < 0
    ]
    if bad:
        raise ValueError(f"{label} must hold non-negative numbers: " + ", ".join(bad))
    _validate_optional_cache_rates(block, label)


def _validate_catalog_endpoints(endpoints, label) -> None:
    """Schema v3 endpoint list: null, or parsed endpoint dicts.

    parse_endpoints (Task 3) emits exactly these seven keys with prices
    parsed to numbers or null -- the shape downstream arithmetic (baseline
    pricing, annotations) indexes into, so a deviation fails here.
    """
    if endpoints is None:
        return
    if not isinstance(endpoints, list):
        raise ValueError(f"{label} must be a list or null")
    for j, endpoint in enumerate(endpoints):
        sub = f"{label}[{j}]"
        if not isinstance(endpoint, dict):
            raise ValueError(f"{sub} is not an object")
        missing = [key for key in CATALOG_ENDPOINT_KEYS if key not in endpoint]
        if missing:
            raise ValueError(f"{sub} is missing keys: {', '.join(missing)}")
        for key in ("provider_name", "tag", "service_tier"):
            value = endpoint[key]
            if value is not None and not isinstance(value, str):
                raise ValueError(f"{sub} {key} must be a string or null")
        quantization = endpoint["quantization"]
        if quantization is not None and (
            # isinstance guard first: an unhashable value must fail the
            # ValueError contract, never escape as a TypeError traceback.
            not isinstance(quantization, str) or quantization not in QUANTIZATION_ENUM
        ):
            raise ValueError(f"{sub} has unknown quantization: {quantization!r}")
        pricing = endpoint["pricing"]
        if not isinstance(pricing, dict) or any(
            key not in pricing for key in ("prompt", "completion")
        ):
            raise ValueError(f"{sub} pricing is incomplete")
        bad = [
            key
            for key in ("prompt", "completion")
            if pricing[key] is not None
            and (not _is_number(pricing[key]) or pricing[key] < 0)
        ]
        if bad:
            raise ValueError(
                f"{sub} pricing must hold non-negative numbers or null: "
                + ", ".join(bad)
            )
        for key in ("context_length", "status"):
            value = endpoint[key]
            if value is not None and (
                not isinstance(value, int) or isinstance(value, bool)
            ):
                raise ValueError(f"{sub} {key} must be an integer or null")
        max_completion = endpoint.get("max_completion_tokens")
        if max_completion is not None and (
            not isinstance(max_completion, int)
            or isinstance(max_completion, bool)
            or max_completion <= 0
        ):
            raise ValueError(
                f"{sub} max_completion_tokens must be a positive integer or null"
            )
        for key in (
            "input_cache_read",
            "input_cache_write",
            "input_cache_write_1h",
        ):
            if key not in pricing:
                continue
            value = pricing[key]
            if value is not None and (not _is_number(value) or value < 0):
                raise ValueError(
                    f"{sub} pricing.{key} must be a non-negative number or null"
                )


def _validate_catalog_advertised_pricing(advertised, label) -> None:
    """Schema v3 advertised-pricing block: null, or the models-API scalars.

    The endpoint stage captures the candidate's pre-mutation models-API
    prices (source-tagged "models-api"); the site shows them next to the
    baseline prices, so the trio must stay numeric-or-null and the tag
    exact.
    """
    if advertised is None:
        return
    if not isinstance(advertised, dict):
        raise ValueError(f"{label} is not an object")
    missing = [key for key in CATALOG_BASE_KEYS if key not in advertised]
    if missing:
        raise ValueError(f"{label} is missing keys: {', '.join(missing)}")
    bad = [
        key
        for key in CATALOG_BASE_KEYS
        if advertised[key] is not None
        and (not _is_number(advertised[key]) or advertised[key] < 0)
    ]
    if bad:
        raise ValueError(
            f"{label} must hold non-negative numbers or null: " + ", ".join(bad)
        )
    if advertised.get("source") != "models-api":
        raise ValueError(f"{label} source must be 'models-api'")


def _validate_catalog_annotations(annotations, label) -> None:
    """Schema v3 annotations block (design section 4.6): null, or the four
    annotation fields the CLI and site read."""
    if annotations is None:
        return
    if not isinstance(annotations, dict):
        raise ValueError(f"{label} is not an object")
    missing = [key for key in CATALOG_ANNOTATION_KEYS if key not in annotations]
    if missing:
        raise ValueError(f"{label} is missing keys: {', '.join(missing)}")
    spread = annotations["price_spread"]
    if spread is not None and not _is_number(spread):
        raise ValueError(f"{label} price_spread must be a number or null")
    cheapest = annotations["cheapest_quantized"]
    if cheapest is not None:
        if not isinstance(cheapest, dict) or any(
            key not in cheapest for key in ("provider", "quantization", "price")
        ):
            raise ValueError(f"{label} cheapest_quantized is incomplete")
        for key in ("provider", "quantization"):
            value = cheapest[key]
            if value is not None and not isinstance(value, str):
                raise ValueError(
                    f"{label} cheapest_quantized {key} must be a string or null"
                )
        if not _is_number(cheapest["price"]):
            raise ValueError(f"{label} cheapest_quantized price must be a number")
    for key in ("risk_flag", "baseline_context_below_catalog"):
        if not isinstance(annotations[key], bool):
            raise ValueError(f"{label} {key} must be a boolean")


def _validate_catalog_reasoning(reasoning, label) -> None:
    """Schema v5 reasoning block: metadata plus the measured depth records.

    The producer emits the block on every entry (never null), and downstream
    consumers index straight into it: the site's depth table, the plugin's
    depth injection and unkeyed runs' depth re-selection. A malformed block
    fails closed here rather than degrading a published score.

    Depth records reuse the quality record's shape, so the score field is
    ``index``; ``quality_index`` is the pre-release draft key and is rejected
    as an unknown key. Effort labels are unique per model (a duplicate is
    ambiguous, never last-wins), the index is finite, and the AA cost is
    finite or null when unreported. ``aa_unlabeled`` is optional (older
    catalogs lack it): null, or a display-only AA reference measured without
    an effort label.
    """
    if not isinstance(reasoning, dict):
        raise ValueError(f"{label} is not an object")
    missing = [key for key in CATALOG_REASONING_KEYS if key not in reasoning]
    if missing:
        raise ValueError(f"{label} is missing keys: {', '.join(missing)}")
    default_effort = reasoning["default_effort"]
    if default_effort is not None and (
        not isinstance(default_effort, str) or not default_effort
    ):
        raise ValueError(f"{label} default_effort must be a non-empty string or null")
    for flag in ("mandatory", "default_enabled", "supports_max_tokens"):
        value = reasoning[flag]
        if value is not None and not isinstance(value, bool):
            raise ValueError(f"{label} {flag} must be a boolean or null")
    supported = reasoning["supported_efforts"]
    if not isinstance(supported, list) or any(
        not isinstance(effort, str) or not effort for effort in supported
    ):
        raise ValueError(
            f"{label} supported_efforts must be a list of non-empty strings"
        )
    depths = reasoning["depths"]
    if not isinstance(depths, list):
        raise ValueError(f"{label} depths must be a list")
    seen = set()
    for j, depth in enumerate(depths):
        sub = f"{label}.depths[{j}]"
        if not isinstance(depth, dict):
            raise ValueError(f"{sub} is not an object")
        unknown = [key for key in depth if key not in CATALOG_DEPTH_KEYS]
        if unknown:
            raise ValueError(f"{sub} has unknown keys: " + ", ".join(unknown))
        missing = [key for key in CATALOG_DEPTH_KEYS if key not in depth]
        if missing:
            raise ValueError(f"{sub} is missing keys: " + ", ".join(missing))
        effort = depth["effort"]
        if not isinstance(effort, str) or not effort:
            raise ValueError(f"{sub} effort must be a non-empty string")
        if effort in seen:
            raise ValueError(f"{sub} duplicates effort: {effort!r}")
        seen.add(effort)
        if not _is_number(depth["index"]):
            raise ValueError(f"{sub} index must be a finite number")
        cost = depth["cost_per_task_usd"]
        if cost is not None and not _is_number(cost):
            raise ValueError(f"{sub} cost_per_task_usd must be a finite number or null")
        source = depth["source"]
        if not isinstance(source, str) or not source:
            raise ValueError(f"{sub} source must be a non-empty string")
        as_of = depth["as_of"]
        if as_of is not None and not isinstance(as_of, str):
            raise ValueError(f"{sub} as_of must be a string or null")
    unlabeled = reasoning.get("aa_unlabeled")
    if unlabeled is not None:
        sub = f"{label}.aa_unlabeled"
        if not isinstance(unlabeled, dict):
            raise ValueError(f"{sub} is not an object")
        unknown = [key for key in unlabeled if key not in CATALOG_UNLABELED_KEYS]
        if unknown:
            raise ValueError(f"{sub} has unknown keys: " + ", ".join(unknown))
        missing = [key for key in CATALOG_UNLABELED_KEYS if key not in unlabeled]
        if missing:
            raise ValueError(f"{sub} is missing keys: " + ", ".join(missing))
        if not _is_number(unlabeled["index"]):
            raise ValueError(f"{sub} index must be a finite number")
        cost = unlabeled["cost_per_task_usd"]
        if cost is not None and not _is_number(cost):
            raise ValueError(f"{sub} cost_per_task_usd must be a finite number or null")
        source = unlabeled["source"]
        if not isinstance(source, str) or not source:
            raise ValueError(f"{sub} source must be a non-empty string")
        as_of = unlabeled["as_of"]
        if as_of is not None and not isinstance(as_of, str):
            raise ValueError(f"{sub} as_of must be a string or null")


def _validate_catalog_steering(steering, label) -> None:
    """Schema v3 steering block (design section 4.7): null, or the
    OpenRouter provider routing object the plugin contract reads.

    build_steering emits {"model": id, "provider": {order, ignore,
    quantizations?, allow_fallbacks}} -- provider's keys are a subset of
    the four the design pins, with quantizations OMITTED entirely when
    default_precision is null (never an empty list), so it is optional
    here. Unknown keys fail: the site reads these shapes verbatim.
    """
    if steering is None:
        return
    if not isinstance(steering, dict):
        raise ValueError(f"{label} is not an object")
    # Unknown keys fail at the top level too -- the docstring's contract
    # (the site and the plugin read this shape verbatim).
    unknown_top = [key for key in steering if key not in ("model", "provider")]
    if unknown_top:
        raise ValueError(f"{label} has unknown keys: " + ", ".join(unknown_top))
    model = steering.get("model")
    if not isinstance(model, str) or not model:
        raise ValueError(f"{label} model must be a non-empty string")
    provider = steering.get("provider")
    if not isinstance(provider, dict):
        raise ValueError(f"{label} provider is not an object")
    allowed = ("order", "ignore", "quantizations", "allow_fallbacks")
    unknown = [key for key in provider if key not in allowed]
    if unknown:
        raise ValueError(f"{label} provider has unknown keys: " + ", ".join(unknown))
    for key in ("order", "ignore"):
        if key not in provider:
            continue
        tags = provider[key]
        if not isinstance(tags, list) or any(
            not isinstance(tag, str) or not tag for tag in tags
        ):
            raise ValueError(
                f"{label} provider.{key} must be a list of non-empty strings"
            )
    if "quantizations" in provider:
        quants = provider["quantizations"]
        if not isinstance(quants, list) or any(
            # isinstance guard first: unhashable entries must raise the
            # ValueError contract, not a TypeError.
            not isinstance(quant, str) or quant not in PRECISION_LADDER
            for quant in quants
        ):
            raise ValueError(
                f"{label} provider.quantizations must be precision-ladder keys"
            )
    if "allow_fallbacks" in provider and not isinstance(
        provider["allow_fallbacks"], bool
    ):
        raise ValueError(f"{label} provider.allow_fallbacks must be a boolean")


def _validate_catalog_session_cost(session_cost) -> None:
    """Top-level session-cost envelope (Task 7), present only when the
    fetch succeeded: harness attribution, window meta and per-model medians
    over the dataset's four turn ranges."""
    if not isinstance(session_cost, dict):
        raise ValueError("catalog session_cost is not an object")
    for key in ("attribution", "app_slug", "as_of", "window_end_date"):
        value = session_cost.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"catalog session_cost {key} must be a non-empty string")
    window_days = session_cost.get("window_days")
    if (
        not isinstance(window_days, int)
        or isinstance(window_days, bool)
        or window_days <= 0
    ):
        raise ValueError("catalog session_cost window_days must be a positive integer")
    medians = session_cost.get("medians")
    if not isinstance(medians, dict):
        raise ValueError("catalog session_cost medians must be an object")
    for model_id, cells in medians.items():
        if not isinstance(model_id, str) or not model_id:
            raise ValueError(
                "catalog session_cost medians keys must be non-empty id strings"
            )
        if not isinstance(cells, dict):
            raise ValueError(
                f"catalog session_cost medians {model_id!r} must be an object"
            )
        for turn_range, median in cells.items():
            if turn_range not in SESSION_COST_TURN_RANGES:
                raise ValueError(
                    f"catalog session_cost medians {model_id!r} has unknown "
                    f"turn range: {turn_range!r}"
                )
            if median is not None and not _is_number(median):
                raise ValueError(
                    f"catalog session_cost medians {model_id!r} {turn_range} "
                    "must be a number or null"
                )


def validate_catalog(document) -> None:
    """Validate a raw model_compare.py --catalog document.

    Raises ValueError on any breach so a broken run fails the build and
    never deploys. The contract is additive-only: unknown schema_version
    or missing fields fail here rather than downstream.
    """
    if not isinstance(document, dict):
        raise ValueError("catalog document is not an object")
    if document.get("schema_version") != CATALOG_SCHEMA_VERSION:
        raise ValueError(
            f"unknown catalog schema_version: {document.get('schema_version')!r}"
        )
    if document.get("tool") != "model-compare":
        raise ValueError(f"unexpected catalog tool: {document.get('tool')!r}")
    for key in (
        "generated_at",
        "parameters",
        "sources",
        "pool",
        "models",
        "rankings",
        "schedules",
        "filtered",
    ):
        if key not in document:
            raise ValueError(f"catalog missing key: {key}")
    # build_snapshot/merge_history slice and parse this stamp; reject a
    # malformed one here instead of crashing (or keying history) downstream.
    generated_at = document["generated_at"]
    if not isinstance(generated_at, str) or not generated_at:
        raise ValueError(
            f"catalog generated_at must be an ISO-8601 string: {generated_at!r}"
        )
    try:
        datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
    except ValueError:
        raise ValueError(
            f"catalog generated_at is not ISO-8601: {generated_at!r}"
        ) from None
    parameters = document["parameters"]
    if not isinstance(parameters, dict) or "zdr_required" not in parameters:
        raise ValueError("catalog parameters must set zdr_required")
    # scores.overall is documented as reproducible from parameters.weights
    # alone, so the weights must cover every priority and sum to 1.
    weights = parameters.get("weights")
    if not isinstance(weights, dict) or any(
        key not in weights for key in CATALOG_OVERALL_KEYS
    ):
        raise ValueError(
            "catalog parameters.weights must cover: " + ", ".join(CATALOG_OVERALL_KEYS)
        )
    for priority in CATALOG_OVERALL_KEYS:
        w = weights[priority]
        if not isinstance(w, dict) or not w:
            raise ValueError(
                f"catalog parameters.weights.{priority} must map score names to weights"
            )
        bad_weights = [
            name for name, value in w.items() if not _is_score(value) or value == 0
        ]
        if bad_weights or abs(sum(w.values()) - 1.0) > 1e-6:
            raise ValueError(
                f"catalog parameters.weights.{priority} must be positive weights summing to 1"
            )
    pool = document["pool"]
    if not isinstance(pool, dict) or not isinstance(pool.get("candidates"), int):
        raise ValueError("catalog pool.candidates must be an integer")
    dropped = pool.get("dropped")
    if not isinstance(dropped, dict) or not all(
        isinstance(value, int) for value in dropped.values()
    ):
        raise ValueError("catalog pool.dropped must map reasons to integers")
    listed = pool.get("listed")
    if not isinstance(listed, int):
        raise ValueError("catalog pool.listed must be an integer")
    if sum(dropped.values()) + pool["candidates"] != listed:
        raise ValueError(
            "catalog pool accounting mismatch: "
            f"{sum(dropped.values())} dropped + {pool['candidates']} candidates"
            f" != {listed} listed"
        )
    sources = document["sources"]
    aa_sources = sources.get("aa") if isinstance(sources, dict) else None
    if (
        not isinstance(aa_sources, dict)
        or aa_sources.get("mode") not in CATALOG_AA_MODES
    ):
        raise ValueError("catalog sources.aa.mode is unknown")
    # The v4 counts are disjoint label buckets (strict rank counts each
    # model once); the sum-vs-entries invariant is checked below, after the
    # entries themselves have been validated. The v3 two-bucket shape
    # (fallback/matched) has no slot here and must not validate.
    for key in ("matched_aa", "matched_catalog", "matched_openrouter"):
        value = aa_sources.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"catalog sources.aa.{key} must be a non-negative integer")
    catalog_generated_at = aa_sources.get("catalog_generated_at")
    if catalog_generated_at is not None:
        if not isinstance(catalog_generated_at, str) or not catalog_generated_at:
            raise ValueError(
                "catalog sources.aa.catalog_generated_at must be an ISO-8601 "
                f"string or null: {catalog_generated_at!r}"
            )
        try:
            datetime.fromisoformat(catalog_generated_at.replace("Z", "+00:00"))
        except ValueError:
            raise ValueError(
                "catalog sources.aa.catalog_generated_at is not ISO-8601: "
                f"{catalog_generated_at!r}"
            ) from None
    version = aa_sources.get("intelligence_index_version")
    # AA types the index version as a number ("v4.3" is a display form);
    # bools are not versions.
    if version is not None and not _is_number(version):
        raise ValueError(
            "catalog sources.aa.intelligence_index_version must be a number or null"
        )
    models = document["models"]
    if not isinstance(models, list):
        raise ValueError("catalog models must be a list")
    if len(models) != pool["candidates"]:
        raise ValueError(
            f"catalog has {len(models)} models but pool.candidates is {pool['candidates']}"
        )
    for i, entry in enumerate(models):
        if not isinstance(entry, dict):
            raise ValueError(f"models[{i}] is not an object")
        missing = [key for key in CATALOG_ENTRY_KEYS if key not in entry]
        if missing:
            raise ValueError(f"models[{i}] is missing keys: {', '.join(missing)}")
        pricing = entry["pricing"]
        if not isinstance(pricing, dict) or any(
            key not in pricing for key in CATALOG_PRICING_KEYS
        ):
            raise ValueError(f"models[{i}] pricing is incomplete")
        bad_pricing = [
            key
            for key in CATALOG_PRICING_KEYS
            if not _is_number(pricing[key]) or pricing[key] < 0
        ]
        if bad_pricing:
            raise ValueError(
                f"models[{i}] pricing must hold non-negative numbers: "
                + ", ".join(bad_pricing)
            )
        _validate_optional_cache_rates(pricing, f"models[{i}] pricing")
        max_output = entry.get("max_output_tokens")
        if max_output is not None and (
            not isinstance(max_output, int)
            or isinstance(max_output, bool)
            or max_output <= 0
        ):
            raise ValueError(
                f"models[{i}] max_output_tokens must be a positive integer or null"
            )
        modalities = entry.get("input_modalities")
        if modalities is not None and (
            not isinstance(modalities, list)
            or any(
                not isinstance(modality, str) or not modality for modality in modalities
            )
        ):
            raise ValueError(
                f"models[{i}] input_modalities must be a list of non-empty strings"
            )
        missing_tier_keys = [
            key for key in ("base", "tiers", "schedule") if key not in pricing
        ]
        if missing_tier_keys:
            raise ValueError(
                f"models[{i}] pricing is missing keys: {', '.join(missing_tier_keys)}"
            )
        _validate_price_block(pricing["base"], f"models[{i}] pricing.base")
        tiers = pricing["tiers"]
        if not isinstance(tiers, list):
            raise ValueError(f"models[{i}] pricing.tiers must be a list")
        previous_threshold = None
        for j, tier in enumerate(tiers):
            if not isinstance(tier, dict) or "min_prompt_tokens" not in tier:
                raise ValueError(f"models[{i}] pricing.tiers[{j}] is incomplete")
            threshold = tier["min_prompt_tokens"]
            if not _is_number(threshold) or threshold < 0:
                raise ValueError(
                    f"models[{i}] pricing.tiers[{j}] min_prompt_tokens must be "
                    "a non-negative number"
                )
            _validate_price_block(tier, f"models[{i}] pricing.tiers[{j}]")
            if previous_threshold is not None and threshold <= previous_threshold:
                raise ValueError(
                    f"models[{i}] pricing.tiers must be strictly ascending "
                    f"(tiers[{j}] back at {threshold})"
                )
            previous_threshold = threshold
        _validate_schedule_list(pricing["schedule"], f"models[{i}] pricing.schedule")
        if entry["discount"] is not None and not _is_number(entry["discount"]):
            raise ValueError(f"models[{i}] discount must be a number or null")
        if entry["context"] is not None and not _is_number(entry["context"]):
            raise ValueError(f"models[{i}] context must be a number or null")
        scores = entry["scores"]
        if not isinstance(scores, dict):
            raise ValueError(f"models[{i}] scores is not an object")
        bad = [key for key in CATALOG_SCORE_KEYS if not _is_score(scores.get(key))]
        if bad:
            raise ValueError(f"models[{i}] scores out of range: {', '.join(bad)}")
        overall = scores.get("overall")
        if not isinstance(overall, dict) or any(
            key not in overall for key in CATALOG_OVERALL_KEYS
        ):
            raise ValueError(f"models[{i}] scores.overall is incomplete")
        bad_overall = [
            key for key in CATALOG_OVERALL_KEYS if not _is_score(overall[key])
        ]
        if bad_overall:
            raise ValueError(
                f"models[{i}] scores.overall out of range: {', '.join(bad_overall)}"
            )
        aa = entry["aa"]
        if not isinstance(aa, dict) or any(key not in aa for key in CATALOG_AA_KEYS):
            raise ValueError(f"models[{i}] aa is incomplete")
        bad_aa = [key for key in CATALOG_AA_KEYS if not _is_aa_value(aa[key])]
        if bad_aa:
            raise ValueError(f"models[{i}] aa out of range: {', '.join(bad_aa)}")
        # The same 0..100 range the schedules rows enforce: quality is an
        # AA intelligence index, never a negative or astronomically large
        # number.
        if entry["quality"] is not None and not _is_aa_value(entry["quality"]):
            raise ValueError(f"models[{i}] quality must be a number in 0..100 or null")
        if (
            entry["quality_match"] is not None
            and entry["quality_match"] not in CATALOG_QUALITY_MATCH_VALUES
        ):
            raise ValueError(f"models[{i}] has unknown quality_match")
        # Schema v4 provenance: presence-checked via CATALOG_ENTRY_KEYS,
        # value types pinned here -- the stamp is a string or null, the
        # drift verdict a boolean or null.
        if entry["quality_as_of"] is not None and not isinstance(
            entry["quality_as_of"], str
        ):
            raise ValueError(f"models[{i}] quality_as_of must be a string or null")
        if entry["quality_drift"] is not None and not isinstance(
            entry["quality_drift"], bool
        ):
            raise ValueError(f"models[{i}] quality_drift must be a boolean or null")
        # Schema v5 (thinking depth): the main quality's effort label is a
        # non-empty string or null (unknown), and the reasoning block is
        # validated in full -- malformed depth data fails the build.
        quality_effort = entry["quality_effort"]
        if quality_effort is not None and (
            not isinstance(quality_effort, str) or not quality_effort
        ):
            raise ValueError(
                f"models[{i}] quality_effort must be a non-empty string or null"
            )
        _validate_catalog_reasoning(entry["reasoning"], f"models[{i}] reasoning")
        if parameters["zdr_required"] and entry["zdr"] is not True:
            raise ValueError(f"models[{i}] is not zdr=true under zdr_required")
        # Schema v3 additive blocks: the producer emits all four (null when
        # the candidate skipped the endpoint stage); the shapes downstream
        # consumers index into are pinned here.
        default_precision = entry["default_precision"]
        if default_precision is not None and (
            # isinstance guard first: an unhashable value must fail the
            # ValueError contract, never escape as a TypeError traceback.
            not isinstance(default_precision, str)
            or default_precision not in PRECISION_LADDER
        ):
            raise ValueError(
                f"models[{i}] default_precision must be a precision-ladder key "
                f"or null: {default_precision!r}"
            )
        _validate_catalog_endpoints(entry["endpoints"], f"models[{i}] endpoints")
        _validate_catalog_advertised_pricing(
            entry["advertised_pricing"], f"models[{i}] advertised_pricing"
        )
        _validate_catalog_annotations(entry["annotations"], f"models[{i}] annotations")
        # Steering is optional here (old fixtures lack it; not in CATALOG_ENTRY_KEYS).
        if "steering" in entry:
            _validate_catalog_steering(entry["steering"], f"models[{i}] steering")
    # The sources.aa counts are disjoint buckets over the quality labels:
    # strict rank gives every matched entry exactly one label, so their sum
    # must equal the number of entries with a non-null quality_match. An
    # inflated bucket would let the document claim more matched models
    # than it actually lists.
    matched_sum = (
        aa_sources["matched_aa"]
        + aa_sources["matched_catalog"]
        + aa_sources["matched_openrouter"]
    )
    matched_entries = sum(1 for entry in models if entry["quality_match"] is not None)
    if matched_sum != matched_entries:
        raise ValueError(
            "catalog sources.aa counts are not disjoint buckets: "
            f"{matched_sum} counted vs {matched_entries} entries with quality_match"
        )
    # ids double as the document's primary key for downstream consumers:
    # unique within models, and never also present in filtered.
    model_ids = [entry["id"] for entry in models]
    if not all(isinstance(model_id, str) for model_id in model_ids):
        raise ValueError("catalog models id must be a string")
    if len(set(model_ids)) != len(model_ids):
        raise ValueError("catalog models contain duplicate ids")
    _validate_rankings(document["rankings"], model_ids)
    filtered_ids = []
    for i, entry in enumerate(document["filtered"]):
        if not isinstance(entry, dict):
            raise ValueError(f"filtered[{i}] is not an object")
        if not isinstance(entry.get("id"), str) or not entry["id"]:
            raise ValueError(f"filtered[{i}] has no id")
        if not isinstance(entry.get("reasons"), list) or not entry["reasons"]:
            raise ValueError(f"filtered[{i}] has no reasons")
        filtered_ids.append(entry["id"])
    in_both = set(model_ids) & set(filtered_ids)
    if in_both:
        raise ValueError(
            f"catalog id in both models and filtered: {sorted(in_both)[0]}"
        )
    schedules = document["schedules"]
    if not isinstance(schedules, list):
        raise ValueError("catalog schedules must be a list")
    schedule_ids = []
    for i, entry in enumerate(schedules):
        if not isinstance(entry, dict):
            raise ValueError(f"schedules[{i}] is not an object")
        missing = [key for key in CATALOG_SCHEDULE_ENTRY_KEYS if key not in entry]
        if missing:
            raise ValueError(f"schedules[{i}] is missing keys: {', '.join(missing)}")
        model_id = entry["id"]
        if not isinstance(model_id, str) or not MODEL_ID_RE.fullmatch(model_id):
            raise ValueError(f"schedules[{i}] id looks wrong: {model_id!r}")
        if entry["name"] is not None and not isinstance(entry["name"], str):
            raise ValueError(f"schedules[{i}] name must be a string or null")
        context = entry["context"]
        if context is not None and (not _is_number(context) or context < 0):
            raise ValueError(
                f"schedules[{i}] context must be a non-negative number or null"
            )
        # Every entry has a valid schedule, so its note is always computable;
        # the site's deals table reads it unguarded.
        if not isinstance(entry["sched_note"], str) or not entry["sched_note"]:
            raise ValueError(f"schedules[{i}] sched_note must be a non-empty string")
        discount = entry["max_discount"]
        if not _is_number(discount) or not 0 <= discount <= 1:
            raise ValueError(f"schedules[{i}] max_discount must be in [0, 1]")
        if entry["quality"] is not None and not _is_aa_value(entry["quality"]):
            raise ValueError(
                f"schedules[{i}] quality must be a number in 0..100 or null"
            )
        # Provenance (data.json v2 additive): presence-checked via
        # CATALOG_SCHEDULE_ENTRY_KEYS, value types pinned here -- the label
        # is one of the quality match values or null, the stamp a string or
        # null (same shapes as models[].quality_match/quality_as_of).
        if (
            entry["quality_source"] is not None
            and entry["quality_source"] not in CATALOG_QUALITY_MATCH_VALUES
        ):
            raise ValueError(f"schedules[{i}] has unknown quality_source")
        if entry["quality_as_of"] is not None and not isinstance(
            entry["quality_as_of"], str
        ):
            raise ValueError(f"schedules[{i}] quality_as_of must be a string or null")
        # Schema v5: the effort label the scheduled row's quality was
        # measured at, a non-empty string or null (unknown).
        quality_effort = entry["quality_effort"]
        if quality_effort is not None and (
            not isinstance(quality_effort, str) or not quality_effort
        ):
            raise ValueError(
                f"schedules[{i}] quality_effort must be a non-empty string or null"
            )
        if entry["score"] is not None and not _is_score(entry["score"]):
            raise ValueError(f"schedules[{i}] score must be a score in [0, 1] or null")
        if entry["sched_detail"] is not None and not isinstance(
            entry["sched_detail"], str
        ):
            raise ValueError(f"schedules[{i}] sched_detail must be a string or null")
        _validate_price_block(entry["peak"], f"schedules[{i}] peak")
        _validate_price_block(entry["offpeak"], f"schedules[{i}] offpeak")
        _validate_schedule_list(entry["schedule"], f"schedules[{i}].schedule")
        schedule_ids.append(entry["id"])
    if len(set(schedule_ids)) != len(schedule_ids):
        raise ValueError("catalog schedules contain duplicate ids")
    # Session-cost envelope (Task 7): present only when the fetch succeeded
    # (consumers branch on key presence, never on a null shape).
    if "session_cost" in document:
        _validate_catalog_session_cost(document["session_cost"])


HISTORY_SCHEMA_VERSION = 5
HISTORY_RETENTION = 10
HISTORY_TOP_N = 20


def _is_number(value) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _is_iso_date(value) -> bool:
    text = str(value)
    if not _ISO_DATE_RE.fullmatch(text):
        return False
    try:
        date.fromisoformat(text)
    except ValueError:
        return False
    return True


def _has_snapshot_shape(snap) -> bool:
    """True when a snapshot satisfies every read build_diff performs.

    rows must be dicts with a non-empty string id and rank == position, aa
    values numeric or null, prices rows 4-element numeric/null lists, and
    pool_ids strings -- anything weaker could crash the 7-day baseline
    arithmetic downstream.
    """
    if not isinstance(snap, dict):
        return False
    generated_at = snap.get("generated_at")
    if not isinstance(generated_at, str) or not generated_at:
        return False
    pool_ids = snap.get("pool_ids")
    if not isinstance(pool_ids, list) or not all(
        isinstance(model_id, str) for model_id in pool_ids
    ):
        return False
    tabs = snap.get("tabs")
    if not isinstance(tabs, dict):
        return False
    for key in CATALOG_OVERALL_KEYS:
        rows = tabs.get(key)
        if not isinstance(rows, list):
            return False
        for i, row in enumerate(rows):
            if not isinstance(row, dict) or row.get("rank") != i + 1:
                return False
            if not isinstance(row.get("id"), str) or not row["id"]:
                return False
    aa = snap.get("aa")
    if not isinstance(aa, dict) or not all(
        value is None or _is_number(value) for value in aa.values()
    ):
        return False
    prices = snap.get("prices")
    if not isinstance(prices, dict):
        return False
    for row in prices.values():
        if not isinstance(row, list) or len(row) != 4:
            return False
        if not all(value is None or _is_number(value) for value in row):
            return False
    return True


def build_snapshot(catalog) -> dict:
    """Project a validated catalog document into one daily history snapshot.

    tabs are a projection of the catalog's rankings (the single ranking
    authority, the same order the data.json rows follow): per priority the
    top `HISTORY_TOP_N` ids (20), rank 1-based, quality/blended merged from
    the models entry. No sort happens here. aa and prices cover candidates only -- filtered
    entries carry neither.
    """
    models = catalog["models"]
    by_id = {entry["id"]: entry for entry in models}
    tabs = {}
    for priority in CATALOG_OVERALL_KEYS:
        tabs[priority] = [
            {
                "id": model_id,
                "rank": i + 1,
                "quality": by_id[model_id]["quality"],
                "blended": by_id[model_id]["pricing"]["blended_per_1m"],
            }
            for i, model_id in enumerate(catalog["rankings"][priority][:HISTORY_TOP_N])
        ]
    return {
        "generated_at": catalog["generated_at"],
        "pool_ids": sorted(
            set([e["id"] for e in models] + [e["id"] for e in catalog["filtered"]])
        ),
        "tabs": tabs,
        "aa": {
            e["id"]: e["aa"]["intelligence_index"]
            for e in models
            if e["aa"]["intelligence_index"] is not None
        },
        "prices": {
            e["id"]: [
                e["pricing"]["input_per_1m"],
                e["pricing"]["output_per_1m"],
                e["pricing"]["blended_per_1m"],
                e["discount"],
            ]
            for e in models
        },
    }


def merge_history(prev, snapshot) -> dict:
    """Upsert the snapshot under its UTC date and prune to the newest 10.

    A previous document with an unknown schema_version starts fresh (an
    absent one is tolerated for back-compat); previous snapshots are
    shape-filtered per date, so a date is carried forward only when its
    key is a strict extended-format ISO date no more than a day ahead of
    the snapshot being written and the snapshot is a dict with a
    non-empty generated_at string, string pool_ids, a tabs dict holding
    id-bearing rank-consistent lists for each of balanced/price/quality,
    a numeric-or-null aa dict and a 4-element numeric/null prices dict.
    Same-day upserts are last-write-wins. updated_at is always the written
    snapshot's generated_at (the catalog's stamp), never a retained
    future-dated key's, so publish's stamp gate cannot wedge on skewed data.
    """
    today = snapshot["generated_at"][:10]
    horizon = (date.fromisoformat(today) + timedelta(days=1)).isoformat()
    snapshots = {}
    if (
        isinstance(prev, dict)
        and prev.get("schema_version", HISTORY_SCHEMA_VERSION) == HISTORY_SCHEMA_VERSION
        and isinstance(prev.get("snapshots"), dict)
    ):
        for date_key, snap in prev["snapshots"].items():
            if (
                _is_iso_date(date_key)
                and date_key <= horizon
                and _has_snapshot_shape(snap)
            ):
                snapshots[date_key] = snap
    snapshots[today] = snapshot
    kept = sorted(snapshots)[-HISTORY_RETENTION:]
    return {
        "schema_version": HISTORY_SCHEMA_VERSION,
        "updated_at": snapshot["generated_at"],
        "snapshots": {date_key: snapshots[date_key] for date_key in kept},
    }


def validate_history(document) -> None:
    """Validate the merged history document (strict identity, <=10 dates)."""
    if not isinstance(document, dict):
        raise ValueError("history document is not an object")
    if document.get("schema_version") != HISTORY_SCHEMA_VERSION:
        raise ValueError(
            f"unknown history schema_version: {document.get('schema_version')!r}"
        )
    snapshots = document.get("snapshots")
    if not isinstance(snapshots, dict) or not snapshots:
        raise ValueError("history snapshots must be a non-empty object")
    if not all(_is_iso_date(d) for d in snapshots):
        raise ValueError("history snapshot keys must be ISO dates")
    if len(snapshots) > HISTORY_RETENTION:
        raise ValueError(
            f"history holds {len(snapshots)} snapshots (max {HISTORY_RETENTION})"
        )
    for date_key, snap in snapshots.items():
        if not isinstance(snap, dict):
            raise ValueError(f"history snapshot {date_key} is not an object")
        for key in ("generated_at", "pool_ids", "tabs", "aa", "prices"):
            if key not in snap:
                raise ValueError(f"history snapshot {date_key} missing key: {key}")
        for priority in CATALOG_OVERALL_KEYS:
            rows = snap["tabs"].get(priority)
            if not isinstance(rows, list):
                raise ValueError(f"history snapshot {date_key} tabs.{priority} missing")
            for i, row in enumerate(rows):
                if not isinstance(row, dict) or row.get("rank") != i + 1:
                    raise ValueError(
                        f"history snapshot {date_key} tabs.{priority} rank mismatch"
                    )


HIGHLIGHTS_SCHEMA_VERSION = 1
HIGHLIGHTS_SECTION_KEYS = ("week", "intelligence", "prices")
HIGHLIGHTS_SOURCES = ("openrouter", "fallback")


def validate_highlights(document) -> None:
    """Validate the highlights document (strict identity, tolerant extensions)."""
    if not isinstance(document, dict):
        raise ValueError("highlights document is not an object")
    if document.get("schema_version") != HIGHLIGHTS_SCHEMA_VERSION:
        raise ValueError(
            f"unknown highlights schema_version: {document.get('schema_version')!r}"
        )
    if document.get("source") not in HIGHLIGHTS_SOURCES:
        raise ValueError(f"unknown highlights source: {document.get('source')!r}")
    generated_at = document.get("generated_at")
    if not isinstance(generated_at, str) or not generated_at:
        raise ValueError("highlights generated_at must be an ISO-8601 string")
    try:
        datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(
            f"highlights generated_at is not ISO-8601: {generated_at!r}"
        ) from exc
    sections = document.get("sections")
    if not isinstance(sections, dict):
        raise ValueError("highlights sections must be an object")
    for key in HIGHLIGHTS_SECTION_KEYS:
        text = sections.get(key)
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"highlights section {key} must be a non-empty string")


def order_rows(rows, ranking) -> list:
    """Reorder CLI rows to follow the catalog ranking (a projection, no sort).

    Fails loudly unless the rows are exactly the ranking's top len(rows):
    the `--priority P --json` and `--catalog` invocations must have ranked
    the same models. Falling back to the CLI order would silently bring back
    the table-vs-history divergence the single ranking exists to remove.
    Duplicated row ids get their own message (unreachable via print_json,
    which cannot emit duplicates; diagnostic hardening for hand-made rows).
    """
    ids = [row.get("model") for row in rows]
    if not all(isinstance(model_id, str) for model_id in ids):
        raise ValueError("row model ids must be strings")
    dupes = sorted({model_id for model_id in ids if ids.count(model_id) > 1})
    if dupes:
        raise ValueError(f"rows contain duplicate model ids: {dupes}")
    top = ranking[: len(rows)]
    if set(ids) != set(top) or len(ids) != len(top):
        missing = sorted(set(top) - set(ids))
        extra = sorted(set(ids) - set(top))
        raise ValueError(
            f"rows disagree with the catalog ranking top {len(rows)}"
            f" (missing: {missing}, extra: {extra})"
        )
    by_id = {row["model"]: row for row in rows}
    return [by_id[model_id] for model_id in top]


# data.json's own contract version: additive changes are free; renaming,
# removing, or changing the meaning of a field bumps it (mirrors the catalog
# and history documents). v2: the five provider-aware row keys became part
# of ROW_KEYS' required set (print_json emits them on every row). v3
# (thinking depth): quality_effort and default_effort join the required row
# keys; the nested reasoning block stays catalog-only and is dropped here.
DATA_SCHEMA_VERSION = 3


def project_schedules(catalog) -> list:
    """data.json projection of catalog.schedules: the Off-peak deals rows.

    Excludes schedules whose (already 4-decimal-rounded) max_discount formats
    as 0% -- the SCHED rule; sorted by discount descending then id; capped at
    10. Prices are the cheapest off-peak window's: the table sells the deal.
    The row's quality provenance (source, as_of, effort) rides through
    verbatim for the Deals QUAL tooltip.
    """
    decorated = []
    for entry in catalog.get("schedules") or []:
        if f"{entry['max_discount']:.0%}" == "0%":
            continue
        decorated.append(
            (
                -entry["max_discount"],
                entry["id"],
                {
                    "model": entry["id"],
                    "sched_note": entry["sched_note"],
                    "sched_detail": entry["sched_detail"],
                    "quality": entry["quality"],
                    # Provenance for the deals QUAL tooltip (presence is
                    # pinned by validate_catalog; the site degrades rows
                    # without it silently).
                    "quality_source": entry["quality_source"],
                    "quality_as_of": entry["quality_as_of"],
                    "quality_effort": entry["quality_effort"],
                    "score": entry["score"],
                    "input_per_1m": entry["offpeak"]["input_per_1m"],
                    "output_per_1m": entry["offpeak"]["output_per_1m"],
                    "context": entry["context"],
                },
            )
        )
    decorated.sort(key=lambda item: (item[0], item[1]))
    return [item[2] for item in decorated[:10]]


def build_data(
    best, priorities, now=None, generated_at=None, catalog=None, revision=None
) -> dict:
    """Validate the rows and wrap them into the data.json document.

    generated_at, when given, is used verbatim (main passes the catalog's
    timestamp so the site's 7-day baseline and the history snapshot keys
    share one date); otherwise the document is stamped with now/wall-clock.
    catalog, when given (already validated), is the ranking authority: each
    priority's rows are reordered to catalog["rankings"] via order_rows.
    Without it the rows pass through verbatim.
    revision, when given, is the short commit hash of the source that built
    this document; it is stamped as source_revision (an additive field, the
    contract stays at schema_version 2) so the site is traceable to the code
    that produced it without versioning the web itself. Validated against
    REVISION_RE -- the site footer links straight to it, so arbitrary text
    must never reach the document. Omitted when absent.
    """
    if not isinstance(best, str) or not MODEL_ID_RE.fullmatch(best.strip()):
        raise ValueError(f"best model id looks wrong: {best!r}")
    best = best.strip()
    if revision is not None:
        if not isinstance(revision, str) or not REVISION_RE.fullmatch(revision.strip()):
            raise ValueError(
                f"revision must be a 7-40 char lowercase hex commit hash: {revision!r}"
            )
        revision = revision.strip()
    if generated_at is not None:
        if not isinstance(generated_at, str) or not generated_at:
            raise ValueError(
                f"generated_at must be an ISO-8601 string: {generated_at!r}"
            )
        try:
            datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
        except ValueError:
            raise ValueError(
                f"generated_at is not ISO-8601: {generated_at!r}"
            ) from None
    else:
        generated_at = (now or datetime.now(timezone.utc)).isoformat(timespec="seconds")
    for name in PRIORITIES:
        rows = priorities.get(name)
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"no rows for priority {name!r}")
        for i, row in enumerate(rows):
            if not isinstance(row, dict):
                raise ValueError(f"{name}[{i}] is not an object")
            missing = [key for key in ROW_KEYS if key not in row]
            if missing:
                raise ValueError(f"{name}[{i}] is missing keys: {', '.join(missing)}")
    # data.json rows stay lean: the producer's nested reasoning block is
    # catalog-only (the site's depth table joins it from catalog.json), so it
    # is dropped before the rows are projected into the document.
    rows_by_priority = {
        name: [
            {key: value for key, value in row.items() if key != "reasoning"}
            for row in priorities[name]
        ]
        for name in PRIORITIES
    }
    if catalog is not None:
        for name in PRIORITIES:
            try:
                rows_by_priority[name] = order_rows(
                    rows_by_priority[name], catalog["rankings"][name]
                )
            except ValueError as exc:
                raise ValueError(f"priority {name!r}: {exc}") from None
    document = {
        "schema_version": DATA_SCHEMA_VERSION,
        "generated_at": generated_at,
        "best": best,
        "priorities": rows_by_priority,
        "schedules": project_schedules(catalog) if catalog is not None else [],
    }
    if revision is not None:
        document["source_revision"] = revision
    return document


def parse_priority_pair(pair):
    name, sep, path = pair.partition("=")
    if not sep or name not in PRIORITIES:
        raise ValueError(
            f"bad --priority {pair!r} (want NAME=FILE, NAME one of {PRIORITIES})"
        )
    with open(path) as fh:
        rows = json.load(fh)
    return name, rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Build data.json for the published model-compare site"
    )
    parser.add_argument(
        "--best-file", required=True, help="file holding the `--best` output"
    )
    parser.add_argument("--output", required=True, help="data.json destination path")
    parser.add_argument(
        "--catalog-file",
        help="raw model_compare.py --catalog output; validated and written as catalog.json next to --output",
    )
    parser.add_argument(
        "--history-prev-file",
        help="previous history.json (fetched from the live site); merged, pruned and rewritten as history.json next to --output",
    )
    parser.add_argument(
        "--highlights-file",
        help="generate_highlights.py output; validated and written as highlights.json next to --output",
    )
    parser.add_argument(
        "--priority",
        action="append",
        required=True,
        metavar="NAME=FILE",
        help="priority name and its model_compare.py --json output file",
    )
    parser.add_argument(
        "--revision",
        help="short commit hash of the source that built this document; "
        "stamped into data.json as source_revision (omitted when not given)",
    )
    args = parser.parse_args(argv)
    catalog = None
    if args.catalog_file:
        try:
            with open(args.catalog_file) as fh:
                catalog = json.load(fh)
            validate_catalog(catalog)
        except (OSError, ValueError) as exc:
            print(f"error: invalid catalog: {exc}", file=sys.stderr)
            return 1
    highlights = None
    if args.highlights_file:
        try:
            with open(args.highlights_file) as fh:
                highlights = json.load(fh)
            validate_highlights(highlights)
        except (OSError, ValueError) as exc:
            print(f"error: invalid highlights: {exc}", file=sys.stderr)
            return 1
    try:
        with open(args.best_file) as fh:
            best = fh.read()
        priorities = {}
        for pair in args.priority:
            name, rows = parse_priority_pair(pair)
            if name in priorities:
                raise ValueError(
                    f"duplicate --priority {name!r} (want each priority once)"
                )
            priorities[name] = rows
        data = build_data(
            best,
            priorities,
            generated_at=catalog["generated_at"] if catalog is not None else None,
            catalog=catalog,
            revision=args.revision,
        )
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    history = None
    if args.history_prev_file is not None:
        if catalog is None:
            print(
                "error: --history-prev-file requires --catalog-file",
                file=sys.stderr,
            )
            return 1
        prev = None
        if args.history_prev_file:
            try:
                with open(args.history_prev_file) as fh:
                    prev = json.load(fh)
            except (OSError, ValueError):
                prev = None  # malformed/missing previous: start fresh
        history = merge_history(prev, build_snapshot(catalog))
        try:
            validate_history(history)
        except ValueError as exc:
            print(f"error: invalid history: {exc}", file=sys.stderr)
            return 1
    try:
        with open(args.output, "w") as fh:
            json.dump(data, fh, indent=2)
            fh.write("\n")
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if catalog is not None:
        catalog_path = os.path.join(
            os.path.dirname(os.path.abspath(args.output)), "catalog.json"
        )
        try:
            with open(catalog_path, "w") as fh:
                json.dump(catalog, fh, indent=2)
                fh.write("\n")
        except OSError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
    if history is not None:
        history_path = os.path.join(
            os.path.dirname(os.path.abspath(args.output)), "history.json"
        )
        try:
            with open(history_path, "w") as fh:
                json.dump(history, fh, indent=2)
                fh.write("\n")
        except OSError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
    if highlights is not None:
        highlights_path = os.path.join(
            os.path.dirname(os.path.abspath(args.output)), "highlights.json"
        )
        try:
            with open(highlights_path, "w") as fh:
                json.dump(highlights, fh, indent=2)
                fh.write("\n")
        except OSError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
