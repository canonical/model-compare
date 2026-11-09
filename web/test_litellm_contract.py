"""Fast contract checks for the LiteLLM artifacts (no LiteLLM dependency).

These tests validate the published shapes as an independent consumer would:
flat map, required LiteLLM keys, tier ranges, off-peak windows, steering
parity/hash and the intended steering application. The real-LiteLLM loader
and cost checks live in test_litellm_compat.py and skip when LiteLLM is not
installed.
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
    catalog = _basic_catalog()
    card, warnings = bla.build_rate_card(catalog)
    card_bytes = bla.serialize_readable(card)
    steering = bla.build_steering(catalog, card_bytes, "deadbee", warnings)
    merged_bytes, status = bla.build_merge_status(
        catalog, "deadbee", card_bytes, "ok", _upstream_bytes(), None, None
    )
    return catalog, card, card_bytes, steering, merged_bytes, status


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
    _, card, _, _, merged_bytes, status = _artifacts()
    merged = json.loads(merged_bytes)
    assert status["merged"]["status"] == "ok"
    assert set(card) <= set(merged)
    assert "_meta" not in merged
    assert all(isinstance(entry, dict) for entry in merged.values())


def test_intended_steering_application():
    """Pin the consumer contract: caller wins, otherwise inject provider+effort."""

    def apply(data, steering, key, strict=False):
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

    _, _, _, steering, _, _ = _artifacts()
    key = "openrouter/z-ai/glm-5.3"

    preference = apply({"model": key}, steering, key)
    assert preference["extra_body"]["provider"] == steering[key]["provider"]
    assert preference["extra_body"]["reasoning"] == {"effort": "max"}

    strict = apply({"model": key}, steering, key, strict=True)
    assert strict["extra_body"]["provider"] == steering[key]["provider_strict"]
    assert strict["extra_body"]["provider"]["allow_fallbacks"] is False

    caller = apply(
        {"model": key, "extra_body": {"provider": {"order": ["mine"]}}},
        steering,
        key,
    )
    assert caller["extra_body"]["provider"] == {"order": ["mine"]}


def test_steering_does_not_send_none_for_a_mandatory_model():
    _, _, _, steering, _, _ = _artifacts()
    entry = steering["openrouter/z-ai/glm-5.3"]
    assert entry["reasoning"]["preferred"] == {"effort": "max"}
    assert entry["reasoning"]["mandatory"] is False
    # The fixture is not mandatory; this pins the guard the consumer must use.
    mandatory = dict(entry["reasoning"], mandatory=True, preferred={"effort": "none"})
    preferred = mandatory["preferred"]
    send = preferred is not None and not (
        mandatory["mandatory"] and preferred["effort"] == "none"
    )
    assert send is False


def test_readable_artifacts_are_deterministic():
    catalog = _basic_catalog()
    card, _ = bla.build_rate_card(catalog)
    first = bla.serialize_readable(card)
    second = bla.serialize_readable(bla.build_rate_card(catalog)[0])
    assert first == second
    assert first.endswith(b"\n")
