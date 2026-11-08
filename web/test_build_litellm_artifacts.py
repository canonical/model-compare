"""Tests for web/build_litellm_artifacts.py (LiteLLM artifacts)."""

import hashlib
import json

import pytest

import build_litellm_artifacts as bla
from test_build_site_data import make_catalog, make_catalog_entry


def _entry(
    model_id,
    *,
    quality_effort=None,
    default_effort=None,
    supported=None,
    mandatory=False,
    default_enabled=None,
    max_output=64000,
    modalities=("text", "image"),
    base=(0.022, 0.4, 0.005, 0.625),
    tiers=None,
    schedule=None,
    precision="fp8",
    order=("morph/fp8", "deepinfra/fp8"),
    ignore=("relace/fp4",),
    quantizations=("fp8",),
):
    entry = make_catalog_entry(id=model_id, quality_effort=quality_effort)
    entry["max_output_tokens"] = max_output
    entry["input_modalities"] = list(modalities)
    entry["context"] = 1_048_576
    entry["tool_calling"] = True
    entry["default_precision"] = precision
    base_in, base_out, cache_read, cache_write = base
    entry["pricing"] = {
        "input_per_1m": base_in,
        "output_per_1m": base_out,
        "blended_per_1m": 0.1165,
        "input_cache_read_per_1m": cache_read,
        "input_cache_write_per_1m": cache_write,
        "base": {
            "input_per_1m": base_in,
            "output_per_1m": base_out,
            "blended_per_1m": 0.1165,
            "input_cache_read_per_1m": cache_read,
            "input_cache_write_per_1m": cache_write,
        },
        "tiers": tiers or [],
        "schedule": schedule,
    }
    entry["reasoning"] = {
        "default_effort": default_effort,
        "supported_efforts": list(supported or []),
        "default_enabled": default_enabled,
        "mandatory": mandatory,
        "supports_max_tokens": None,
        "depths": [],
    }
    provider = {
        "order": list(order),
        "ignore": list(ignore),
        "allow_fallbacks": True,
    }
    if quantizations is not None:
        provider["quantizations"] = list(quantizations)
    entry["steering"] = {"model": model_id, "provider": provider}
    entry["endpoints"] = [
        {
            "provider_name": "Morph",
            "tag": order[0],
            "quantization": precision,
            "pricing": {
                "prompt": 0.00000001,
                "completion": 0.0000005,
                "input_cache_read": None,
                "input_cache_write": None,
                "input_cache_write_1h": None,
                "overrides": None,
            },
            "context_length": 1_048_576,
            "max_completion_tokens": max_output,
            "status": 0,
            "service_tier": None,
        }
    ]
    return entry


def _catalog(entries):
    doc = make_catalog()
    doc["pool"] = {"listed": len(entries), "candidates": len(entries), "dropped": {}}
    doc["models"] = entries
    doc["rankings"] = {
        p: [e["id"] for e in entries] for p in ("balanced", "price", "quality")
    }
    doc["filtered"] = []
    doc["schedules"] = []
    doc["sources"]["aa"].update(
        matched_aa=0,
        matched_catalog=0,
        matched_openrouter=len(entries),
        depth_rows=0,
        matched_depth=0,
        matched_default=len(entries),
    )
    return doc


def _basic_catalog():
    tier = {
        "min_prompt_tokens": 100_000,
        "input_per_1m": 0.5,
        "output_per_1m": 2.5,
        "blended_per_1m": 1.0,
        "input_cache_read_per_1m": 0.05,
        "input_cache_write_per_1m": 0.625,
    }
    schedule = [
        {
            "utc_days": None,
            "utc_start": 1600,
            "utc_end": 0,
            "coverage": 0.3333,
            "input_per_1m": 0.0825,
            "output_per_1m": 0.33,
            "blended_per_1m": 0.144375,
            "input_cache_read_per_1m": 0.05,
            "input_cache_write_per_1m": 0.625,
        },
        {
            "utc_days": None,
            "utc_start": 0,
            "utc_end": 1600,
            "coverage": 0.6667,
            "input_per_1m": 0.132,
            "output_per_1m": 0.528,
            "blended_per_1m": 0.231,
            "input_cache_read_per_1m": 0.1,
            "input_cache_write_per_1m": 1.0,
        },
    ]
    return _catalog(
        [
            _entry(
                "z-ai/glm-5.3",
                quality_effort="max",
                default_effort="high",
                supported=("max", "high", "low", "none"),
                default_enabled=True,
                tiers=[tier],
            ),
            _entry(
                "tencent/hy3",
                max_output=None,
                modalities=("text",),
                base=(0.132, 0.528, 0.1, 1.0),
                schedule=schedule,
                supported=("none",),
                default_effort="none",
            ),
            _entry(
                "deepseek/deepseek-v4.1-flash",
                quality_effort="max",
                default_effort="high",
                supported=("max", "high", "low", "none"),
                default_enabled=True,
            ),
        ]
    )


def _upstream_bytes(count=1000):
    return json.dumps({f"up/{i}": {} for i in range(count)}).encode()


def _upstream_with_collision():
    data = {f"up/{i}": {} for i in range(1000)}
    data["openrouter/z-ai/glm-5.3"] = {}
    return json.dumps(data).encode()


def test_rate_card_keys_and_purity():
    card, _ = bla.build_rate_card(_basic_catalog())
    assert set(card) == {
        "openrouter/z-ai/glm-5.3",
        "openrouter/tencent/hy3",
        "openrouter/deepseek/deepseek-v4.1-flash",
    }
    assert all("_meta" not in e for e in card.values())
    assert all(
        not ({"rank", "score", "quality", "reasons"} & set(e)) for e in card.values()
    )


def test_rate_card_base_cost_and_capabilities():
    card, _ = bla.build_rate_card(_basic_catalog())
    e = card["openrouter/z-ai/glm-5.3"]
    assert e["litellm_provider"] == "openrouter"
    assert e["mode"] == "chat"
    assert e["input_cost_per_token"] == pytest.approx(0.022 / 1e6)
    assert e["output_cost_per_token"] == pytest.approx(0.4 / 1e6)
    assert e["max_input_tokens"] == 1_048_576
    assert e["max_output_tokens"] == 64_000
    assert e["supports_function_calling"] is True
    assert e["supports_vision"] is True
    assert e["cache_read_input_token_cost"] == pytest.approx(0.005 / 1e6)
    assert e["cache_creation_input_token_cost"] == pytest.approx(0.625 / 1e6)


def test_rate_card_tier_ranges_share_boundaries():
    card, _ = bla.build_rate_card(_basic_catalog())
    tiers = card["openrouter/z-ai/glm-5.3"]["tiered_pricing"]
    assert [t["range"] for t in tiers] == [[0, 100_000], [100_000, 1_048_576]]
    assert tiers[0]["input_cost_per_token"] == pytest.approx(0.022 / 1e6)
    assert tiers[1]["input_cost_per_token"] == pytest.approx(0.5 / 1e6)
    assert tiers[0]["cache_read_input_token_cost"] == pytest.approx(0.005 / 1e6)
    assert tiers[1]["cache_read_input_token_cost"] == pytest.approx(0.05 / 1e6)
    assert tiers[1]["cache_creation_input_token_cost"] == pytest.approx(0.625 / 1e6)


def test_rate_card_off_peak_single_level():
    card, _ = bla.build_rate_card(_basic_catalog())
    e = card["openrouter/tencent/hy3"]
    assert e["off_peak_pricing"]["input_cost_per_token"] == pytest.approx(0.0825 / 1e6)
    assert e["off_peak_pricing"]["hours_utc"] == ["16:00-00:00"]
    assert e["off_peak_pricing"]["cache_read_input_token_cost"] == pytest.approx(
        0.05 / 1e6
    )


def test_rate_card_off_peak_weekday_windows():
    catalog = _basic_catalog()
    schedule = catalog["models"][1]["pricing"]["schedule"]
    schedule[0]["utc_days"] = ["saturday", "sunday"]
    schedule[0]["utc_start"] = 0
    schedule[0]["utc_end"] = 0
    schedule[0]["coverage"] = 2 / 7
    schedule[1]["utc_days"] = [
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
    ]
    schedule[1]["utc_start"] = 0
    schedule[1]["utc_end"] = 0
    schedule[1]["coverage"] = 5 / 7
    card, _ = bla.build_rate_card(catalog)
    off = card["openrouter/tencent/hy3"]["off_peak_pricing"]
    assert off["windows"] == [{"hours_utc": ["00:00-00:00"], "weekdays": [6, 7]}]


def test_rate_card_reasoning_flags():
    card, _ = bla.build_rate_card(_basic_catalog())
    e = card["openrouter/deepseek/deepseek-v4.1-flash"]
    assert e["supports_reasoning"] is True
    assert e["supports_none_reasoning_effort"] is True
    assert e["reasoning_effort_levels"] == ["max", "high", "low", "none"]
    assert e["default_reasoning_effort"] == "high"


def test_rate_card_non_reasoning_flags():
    card, _ = bla.build_rate_card(_basic_catalog())
    e = card["openrouter/tencent/hy3"]
    assert e["supports_reasoning"] is False
    assert e["supports_none_reasoning_effort"] is True


def test_rate_card_off_peak_multiple_levels_omitted():
    catalog = _basic_catalog()
    catalog["models"][1]["pricing"]["schedule"] = [
        {
            "utc_days": None,
            "utc_start": 0,
            "utc_end": 1000,
            "coverage": 0.5,
            "input_per_1m": 0.132,
            "output_per_1m": 0.528,
            "blended_per_1m": 0.231,
            "input_cache_read_per_1m": 0.1,
            "input_cache_write_per_1m": 1.0,
        },
        {
            "utc_days": None,
            "utc_start": 1000,
            "utc_end": 1100,
            "coverage": 0.25,
            "input_per_1m": 0.09,
            "output_per_1m": 0.33,
            "blended_per_1m": 0.15,
            "input_cache_read_per_1m": 0.05,
            "input_cache_write_per_1m": 0.5,
        },
        {
            "utc_days": None,
            "utc_start": 1100,
            "utc_end": 0,
            "coverage": 0.25,
            "input_per_1m": 0.08,
            "output_per_1m": 0.3,
            "blended_per_1m": 0.135,
            "input_cache_read_per_1m": 0.04,
            "input_cache_write_per_1m": 0.4,
        },
    ]
    card, warnings = bla.build_rate_card(catalog)
    assert "off_peak_pricing" not in card["openrouter/tencent/hy3"]
    assert any("multiple off-peak" in w for w in warnings)


def test_rate_card_tiers_and_schedule_keeps_tiers_omits_offpeak():
    catalog = _basic_catalog()
    catalog["models"][0]["pricing"]["schedule"] = [
        {
            "utc_days": None,
            "utc_start": 0,
            "utc_end": 0,
            "coverage": 1.0,
            "input_per_1m": 0.01,
            "output_per_1m": 0.2,
            "blended_per_1m": 0.05,
            "input_cache_read_per_1m": None,
            "input_cache_write_per_1m": None,
        }
    ]
    card, warnings = bla.build_rate_card(catalog)
    e = card["openrouter/z-ai/glm-5.3"]
    assert "tiered_pricing" in e
    assert "off_peak_pricing" not in e
    assert any("tiered and scheduled" in w for w in warnings)


@pytest.mark.parametrize(
    "tiers",
    [
        [{"min_prompt_tokens": 0, "input_per_1m": 1, "output_per_1m": 2}],
        [
            {"min_prompt_tokens": 200, "input_per_1m": 1, "output_per_1m": 2},
            {"min_prompt_tokens": 100, "input_per_1m": 1, "output_per_1m": 2},
        ],
        [{"min_prompt_tokens": 1_048_576, "input_per_1m": 1, "output_per_1m": 2}],
    ],
)
def test_rate_card_bad_tiers_fail(tiers):
    catalog = _basic_catalog()
    catalog["models"][0]["pricing"]["tiers"] = tiers
    with pytest.raises(ValueError):
        bla.build_rate_card(catalog)


def test_rate_card_missing_optional_fields():
    catalog = _basic_catalog()
    entry = catalog["models"][0]
    entry.pop("max_output_tokens")
    entry.pop("input_modalities")
    entry["pricing"].pop("input_cache_read_per_1m")
    entry["pricing"].pop("input_cache_write_per_1m")
    entry["pricing"]["base"].pop("input_cache_read_per_1m")
    entry["pricing"]["base"].pop("input_cache_write_per_1m")
    card, _ = bla.build_rate_card(catalog)
    e = card["openrouter/z-ai/glm-5.3"]
    assert "max_output_tokens" not in e
    assert e["supports_vision"] is False
    assert "cache_read_input_token_cost" not in e
    assert "cache_creation_input_token_cost" not in e


def test_rate_card_rejects_empty_catalog():
    catalog = _basic_catalog()
    catalog["models"] = []
    catalog["pool"] = {"listed": 0, "candidates": 0, "dropped": {}}
    catalog["rankings"] = {p: [] for p in ("balanced", "price", "quality")}
    catalog["sources"]["aa"].update(
        matched_aa=0,
        matched_catalog=0,
        matched_openrouter=0,
        matched_default=0,
    )
    with pytest.raises(ValueError, match="no models"):
        bla.build_rate_card(catalog)


def test_rate_card_duplicate_keys_fail():
    catalog = _basic_catalog()
    catalog["models"].append(dict(catalog["models"][0]))
    with pytest.raises(ValueError):
        bla.build_rate_card(catalog)


def test_serializers_deterministic():
    doc = {"b": 1, "a": 2}
    assert bla.serialize_readable(doc) == b'{\n  "a": 2,\n  "b": 1\n}\n'
    assert bla.serialize_compact(doc) == b'{"a":2,"b":1}\n'


def _rate_card_bytes():
    card, warnings = bla.build_rate_card(_basic_catalog())
    return bla.serialize_readable(card), warnings


def test_steering_copies_provider_without_model():
    raw, warnings = _rate_card_bytes()
    steering = bla.build_steering(_basic_catalog(), raw, "deadbee", warnings)
    entry = steering["openrouter/z-ai/glm-5.3"]
    assert "model" not in entry["provider"]
    assert entry["provider"]["order"] == ["morph/fp8", "deepinfra/fp8"]
    assert entry["provider"]["allow_fallbacks"] is True


def test_steering_strict_profile():
    raw, warnings = _rate_card_bytes()
    steering = bla.build_steering(_basic_catalog(), raw, "deadbee", warnings)
    strict = steering["openrouter/z-ai/glm-5.3"]["provider_strict"]
    assert strict["order"] == ["morph/fp8"]
    assert strict["allow_fallbacks"] is False
    assert strict["max_price"] == {"prompt": 0.022, "completion": 0.4}
    assert strict["quantizations"] == ["fp8"]


def test_steering_strict_quantization_comes_from_baseline_endpoint():
    catalog = _basic_catalog()
    entry = catalog["models"][0]
    entry["default_precision"] = "fp8"
    entry["steering"]["provider"]["order"] = ["morph/bf16", "deepinfra/fp8"]
    entry["endpoints"][0]["tag"] = "morph/bf16"
    entry["endpoints"][0]["quantization"] = "bf16"
    raw, warnings = _rate_card_bytes()
    steering = bla.build_steering(catalog, raw, "deadbee", warnings)
    strict = steering["openrouter/z-ai/glm-5.3"]["provider_strict"]
    assert strict["quantizations"] == ["bf16"]


def test_steering_strict_omits_unknown_baseline_quantization():
    catalog = _basic_catalog()
    entry = catalog["models"][0]
    entry["default_precision"] = "fp8"
    entry["steering"]["provider"]["order"] = ["morph", "deepinfra/fp8"]
    entry["endpoints"][0]["tag"] = "morph"
    entry["endpoints"][0]["quantization"] = None
    raw, warnings = _rate_card_bytes()
    steering = bla.build_steering(catalog, raw, "deadbee", warnings)
    strict = steering["openrouter/z-ai/glm-5.3"]["provider_strict"]
    assert "quantizations" not in strict


def test_steering_reasoning_prefers_rank_effort():
    raw, warnings = _rate_card_bytes()
    steering = bla.build_steering(_basic_catalog(), raw, "deadbee", warnings)
    r = steering["openrouter/z-ai/glm-5.3"]["reasoning"]
    assert r["preferred"] == {"effort": "max"}
    assert r["model_default"] == {"effort": "high"}
    assert r["supported_efforts"] == ["max", "high", "low", "none"]
    assert r["mandatory"] is False


def test_steering_meta_hash_and_entries():
    raw, warnings = _rate_card_bytes()
    steering = bla.build_steering(_basic_catalog(), raw, "deadbee", warnings)
    meta = steering["_meta"]
    assert meta["rate_card_sha256"] == hashlib.sha256(raw).hexdigest()
    assert meta["entries"] == 3
    assert meta["catalog_schema_version"] == 6
    assert meta["warnings"] == sorted(meta["warnings"])


def test_steering_missing_steering_fails():
    catalog = _basic_catalog()
    catalog["models"][0]["steering"] = None
    raw, warnings = _rate_card_bytes()
    with pytest.raises(ValueError):
        bla.build_steering(catalog, raw, "deadbee", warnings)


def test_merge_clean():
    raw, warnings = _rate_card_bytes()
    merged, status = bla.build_merge_status(
        _basic_catalog(), "deadbee", raw, "ok", _upstream_bytes(), None, None
    )
    data = json.loads(merged)
    assert len(data) == 1003
    assert data["openrouter/z-ai/glm-5.3"]["litellm_provider"] == "openrouter"
    assert status["merged"]["status"] == "ok"
    assert status["merged"]["entries"] == 1003
    assert status["merged"]["rate_card_sha256"] == hashlib.sha256(raw).hexdigest()
    assert status["report"]["needed"] is False


def test_merge_collision_freezes_previous():
    raw, _ = _rate_card_bytes()
    prev_data = {f"old/{i}": {} for i in range(1000)}
    prev_data["openrouter/z-ai/glm-5.3"] = {"x": 1}
    prev_merged = json.dumps(prev_data).encode()
    upstream_data = {f"up/{i}": {} for i in range(1000)}
    upstream_data["openrouter/z-ai/glm-5.3"] = {"x": 2}
    upstream = json.dumps(upstream_data).encode()
    merged, status = bla.build_merge_status(
        _basic_catalog(), "deadbee", raw, "ok", upstream, prev_merged, None
    )
    assert merged == prev_merged
    assert status["merged"]["status"] == "frozen"
    assert status["merged"]["reason"] == "collision"
    assert status["collisions"] == ["openrouter/z-ai/glm-5.3"]
    assert status["report"]["needed"] is True


def test_fetch_failure_counter_and_report_threshold():
    raw, _ = _rate_card_bytes()
    merged, status = bla.build_merge_status(
        _basic_catalog(),
        "deadbee",
        raw,
        "failed",
        None,
        None,
        {"upstream_fetch_failures": 2},
    )
    assert merged is None
    assert status["merged"]["status"] == "skipped"
    assert status["upstream_fetch_failures"] == 3
    assert status["report"]["needed"] is True


@pytest.mark.parametrize(
    "previous_failures, expected_count, expected_report",
    [
        (0, 1, False),
        (1, 2, False),
        (2, 3, True),
        (3, 4, True),
        (4, 5, True),
    ],
)
def test_fetch_failure_report_threshold_ladder(
    previous_failures, expected_count, expected_report
):
    raw, _ = _rate_card_bytes()
    merged, status = bla.build_merge_status(
        _basic_catalog(),
        "deadbee",
        raw,
        "failed",
        None,
        None,
        {"upstream_fetch_failures": previous_failures},
    )
    assert merged is None
    assert status["merged"]["status"] == "skipped"
    assert status["upstream_fetch_failures"] == expected_count
    assert status["report"]["needed"] is expected_report


def test_invalid_upstream_counts_as_failure():
    raw, _ = _rate_card_bytes()
    merged, status = bla.build_merge_status(
        _basic_catalog(), "deadbee", raw, "ok", b"<html>", None, None
    )
    assert status["merged"]["status"] == "skipped"
    assert status["merged"]["reason"] == "upstream_invalid"
    assert status["upstream_fetch_failures"] == 1


def test_success_resets_failures():
    raw, _ = _rate_card_bytes()
    merged, status = bla.build_merge_status(
        _basic_catalog(),
        "deadbee",
        raw,
        "ok",
        _upstream_bytes(),
        None,
        {"upstream_fetch_failures": 5},
    )
    assert status["upstream_fetch_failures"] == 0


def test_frozen_status_keeps_explicit_null_rate_card_hash():
    raw, _ = _rate_card_bytes()
    prev_merged = json.dumps({f"up/{i}": {} for i in range(1000)}).encode()
    prev_status = {
        "merged": {"rate_card_sha256": None},
        "rate_card": {"sha256": "d" * 64},
    }
    merged, status = bla.build_merge_status(
        _basic_catalog(), "deadbee", raw, "failed", None, prev_merged, prev_status
    )
    assert merged == prev_merged
    assert status["merged"]["status"] == "frozen"
    assert status["merged"]["rate_card_sha256"] is None


def test_collision_still_resets_failures():
    raw, _ = _rate_card_bytes()
    upstream = _upstream_with_collision()
    merged, status = bla.build_merge_status(
        _basic_catalog(),
        "deadbee",
        raw,
        "ok",
        upstream,
        None,
        {"upstream_fetch_failures": 5},
    )
    assert status["upstream_fetch_failures"] == 0


def test_first_run_collision_is_skipped():
    raw, _ = _rate_card_bytes()
    upstream = _upstream_with_collision()
    merged, status = bla.build_merge_status(
        _basic_catalog(), "deadbee", raw, "ok", upstream, None, None
    )
    assert merged is None
    assert status["merged"]["status"] == "skipped"


def test_no_previous_and_fetch_failure_is_skipped():
    raw, _ = _rate_card_bytes()
    merged, status = bla.build_merge_status(
        _basic_catalog(), "deadbee", raw, "failed", None, None, None
    )
    assert merged is None
    assert status["merged"]["status"] == "skipped"
    assert status["report"]["needed"] is False


def test_skipped_upstream_preserves_failure_count():
    raw, _ = _rate_card_bytes()
    merged, status = bla.build_merge_status(
        _basic_catalog(),
        "deadbee",
        raw,
        "skipped",
        None,
        None,
        {"upstream_fetch_failures": 2},
    )
    assert status["upstream_fetch_failures"] == 2
    assert status["report"]["needed"] is False


def test_skipped_upstream_does_not_report_even_at_threshold():
    raw, _ = _rate_card_bytes()
    merged, status = bla.build_merge_status(
        _basic_catalog(),
        "deadbee",
        raw,
        "skipped",
        None,
        None,
        {"upstream_fetch_failures": 5},
    )
    assert status["upstream_fetch_failures"] == 5
    assert status["report"]["needed"] is False


def test_cli_writes_all_artifacts(tmp_path):
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps(_basic_catalog()))
    upstream = tmp_path / "upstream.json"
    upstream.write_bytes(_upstream_bytes())
    out = tmp_path / "out"
    assert (
        bla.main(
            [
                "--catalog",
                str(catalog),
                "--output-dir",
                str(out),
                "--revision",
                "deadbee",
                "--upstream-status",
                "ok",
                "--upstream-map",
                str(upstream),
            ]
        )
        == 0
    )
    assert (out / bla.RATE_CARD_NAME).is_file()
    assert (out / bla.STEERING_NAME).is_file()
    assert (out / bla.MERGED_NAME).is_file()
    assert (out / bla.STATUS_NAME).is_file()
    status = json.loads((out / bla.STATUS_NAME).read_text())
    assert status["merged"]["status"] == "ok"


def test_cli_skips_merged_map_without_upstream(tmp_path):
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps(_basic_catalog()))
    out = tmp_path / "out"
    assert (
        bla.main(
            [
                "--catalog",
                str(catalog),
                "--output-dir",
                str(out),
                "--revision",
                "deadbee",
                "--upstream-status",
                "failed",
            ]
        )
        == 0
    )
    assert (out / bla.RATE_CARD_NAME).is_file()
    assert (out / bla.STEERING_NAME).is_file()
    assert not (out / bla.MERGED_NAME).exists()
    status = json.loads((out / bla.STATUS_NAME).read_text())
    assert status["merged"]["status"] == "skipped"
