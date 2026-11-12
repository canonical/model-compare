"""Fast contract checks for the LiteLLM artifacts (no LiteLLM dependency).

These tests validate the artifact shapes over the serialized bytes a
consumer would fetch -- not producer internals: flat map, required LiteLLM
keys, tier ranges, off-peak windows, steering parity/hash and the intended
steering application. The real-LiteLLM loader and cost checks live in
test_litellm_compat.py and skip when LiteLLM cannot be imported.
"""

from __future__ import annotations

import hashlib
import json
import re

import pytest

import build_litellm_artifacts as bla
from test_build_litellm_artifacts import _basic_catalog, _upstream_bytes

HOURS_RE = re.compile(r"^\d{2}:\d{2}-\d{2}:\d{2}$")


def _artifacts():
    """Build the artifacts and hand back consumer-visible serialized forms.

    ``card``, ``steering`` and ``merged`` are parsed from the exact bytes a
    consumer would fetch; ``card_bytes`` stays raw bytes for the hash
    assertion.
    """
    catalog = _basic_catalog()
    card, warnings = bla.build_rate_card(catalog)
    card_bytes = bla.serialize_readable(card)
    steering = bla.build_steering(catalog, card_bytes, "deadbee", warnings)
    merged_bytes, status = bla.build_merge_status(
        catalog, "deadbee", card_bytes, "ok", _upstream_bytes(), None, None
    )
    return (
        catalog,
        json.loads(bla.serialize_readable(card)),
        card_bytes,
        json.loads(bla.serialize_readable(steering)),
        json.loads(merged_bytes),
        status,
    )


def resolve_key(model: str, steering: dict) -> str | None:
    """Resolve a request model id to a steering key (bare alias aware)."""
    if model in steering:
        return model
    candidate = "openrouter/" + model
    return candidate if candidate in steering else None


def apply_steering(
    data: dict, steering: dict, model: str, strict: bool = False
) -> dict:
    """Apply steering as the consumer contract dictates.

    Injects provider (and the preferred reasoning effort) into
    ``extra_body`` unless the caller already set them. Never injects
    effort ``none`` for a model with mandatory reasoning. Returns ``data``
    unchanged when the model has no steering entry.
    """
    key = resolve_key(model, steering)
    if key is None:
        return data
    entry = steering[key]
    provider = entry["provider_strict"] if strict else entry["provider"]
    extra = data.setdefault("extra_body", {})
    extra.setdefault("provider", provider)
    preferred = entry["reasoning"]["preferred"]
    if preferred is not None:
        effort = preferred["effort"]
        if not (entry["reasoning"]["mandatory"] and effort == "none"):
            extra.setdefault("reasoning", preferred)
    return data


def test_rate_card_is_a_flat_litellm_map():
    _, card, _, _, _, _ = _artifacts()
    assert isinstance(card, dict) and card
    for key, entry in card.items():
        assert key.startswith("openrouter/")
        assert isinstance(entry, dict)
        assert entry["litellm_provider"] == "openrouter"
        assert entry["mode"] == "chat"
        assert "input_cost_per_token" in entry
        assert "output_cost_per_token" in entry
        assert "max_input_tokens" in entry
        assert "_meta" not in entry
        assert not ({"rank", "score", "quality", "reasons"} & set(entry))


def test_tier_ranges_cover_the_input_domain():
    _, card, _, _, _, _ = _artifacts()
    tiered = [entry for entry in card.values() if "tiered_pricing" in entry]
    assert tiered
    for entry in tiered:
        ranges = entry["tiered_pricing"]
        assert ranges[0]["range"][0] == 0
        for previous, current in zip(ranges, ranges[1:]):
            assert previous["range"][1] == current["range"][0]
        assert ranges[-1]["range"][1] == entry["max_input_tokens"]


def test_off_peak_shape_is_consumable():
    _, card, _, _, _, _ = _artifacts()
    off_peak = [
        entry["off_peak_pricing"]
        for entry in card.values()
        if "off_peak_pricing" in entry
    ]
    assert off_peak
    for block in off_peak:
        assert block["input_cost_per_token"] >= 0
        assert block["output_cost_per_token"] >= 0
        for hours in block.get("hours_utc", []):
            assert HOURS_RE.fullmatch(hours)
        for window in block.get("windows", []):
            assert window["weekdays"]
            assert all(1 <= day <= 7 for day in window["weekdays"])
            for hours in window["hours_utc"]:
                assert HOURS_RE.fullmatch(hours)


def test_steering_keys_and_hash_match_the_rate_card():
    _, card, card_bytes, steering, _, _ = _artifacts()
    meta = steering["_meta"]
    assert set(steering) - {"_meta"} == set(card)
    assert meta["entries"] == len(card)
    assert meta["rate_card_sha256"] == hashlib.sha256(card_bytes).hexdigest()
    assert meta["catalog_schema_version"] == 6


def test_steering_strict_profile_is_single_endpoint():
    _, _, _, steering, _, _ = _artifacts()
    for key, entry in steering.items():
        if key == "_meta":
            continue
        strict = entry["provider_strict"]
        assert len(strict["order"]) == 1
        assert strict["allow_fallbacks"] is False
        assert set(strict["max_price"]) == {"prompt", "completion"}


def test_merged_map_is_a_pure_superset():
    _, card, _, _, merged, status = _artifacts()
    assert status["merged"]["status"] == "ok"
    assert set(card) <= set(merged)
    assert "_meta" not in merged
    assert all(isinstance(entry, dict) for entry in merged.values())


def test_intended_steering_application():
    """Pin the consumer contract: caller wins, otherwise inject provider+effort."""
    _, _, _, steering, _, _ = _artifacts()
    key = "openrouter/z-ai/glm-5.3"

    preference = apply_steering({"model": key}, steering, key)
    assert preference["extra_body"]["provider"] == steering[key]["provider"]
    assert preference["extra_body"]["reasoning"] == {"effort": "max"}

    strict = apply_steering({"model": key}, steering, key, strict=True)
    assert strict["extra_body"]["provider"] == steering[key]["provider_strict"]
    assert strict["extra_body"]["provider"]["allow_fallbacks"] is False

    caller = apply_steering(
        {"model": key, "extra_body": {"provider": {"order": ["mine"]}}},
        steering,
        key,
    )
    assert caller["extra_body"]["provider"] == {"order": ["mine"]}


def test_apply_steering_resolves_bare_alias():
    _, _, _, steering, _, _ = _artifacts()
    key = "openrouter/z-ai/glm-5.3"
    bare = apply_steering({"model": "z-ai/glm-5.3"}, steering, "z-ai/glm-5.3")
    full = apply_steering({"model": key}, steering, key)
    assert bare["extra_body"] == full["extra_body"]
    assert bare["extra_body"]["provider"] == steering[key]["provider"]
    assert bare["extra_body"]["reasoning"] == steering[key]["reasoning"]["preferred"]


def test_steering_does_not_send_none_for_a_mandatory_model():
    """Mandatory reasoning suppresses effort none; non-mandatory none is sent."""
    _, _, _, steering, _, _ = _artifacts()
    key = "openrouter/z-ai/glm-5.3"
    provider = steering[key]["provider"]
    provider_strict = steering[key]["provider_strict"]

    mandatory_key = "openrouter/synthetic/mandatory-none"
    mandatory = {
        mandatory_key: {
            "provider": provider,
            "provider_strict": provider_strict,
            "reasoning": {"preferred": {"effort": "none"}, "mandatory": True},
        }
    }
    out = apply_steering({"model": mandatory_key}, mandatory, mandatory_key)
    assert out["extra_body"]["provider"] == provider
    assert "reasoning" not in out["extra_body"]

    optional_key = "openrouter/synthetic/optional-none"
    optional = {
        optional_key: {
            "provider": provider,
            "provider_strict": provider_strict,
            "reasoning": {"preferred": {"effort": "none"}, "mandatory": False},
        }
    }
    out = apply_steering({"model": optional_key}, optional, optional_key)
    assert out["extra_body"]["provider"] == provider
    assert out["extra_body"]["reasoning"] == {"effort": "none"}


def test_readable_artifacts_are_deterministic():
    catalog = _basic_catalog()
    card, _ = bla.build_rate_card(catalog)
    first = bla.serialize_readable(card)
    second = bla.serialize_readable(bla.build_rate_card(catalog)[0])
    assert first == second
    assert first.endswith(b"\n")
