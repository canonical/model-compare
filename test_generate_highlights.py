"""Tests for generate_highlights.py -- diff builder and fallback templates."""

import pytest

import generate_highlights as gh


def make_diff_history(**snap_overrides):
    base = {
        "generated_at": "2026-08-26T09:15:00+00:00",
        "pool_ids": ["acme/gone", "acme/model-a", "acme/model-b", "acme/old"],
        "tabs": {
            "balanced": [
                {"id": "acme/model-a", "rank": 1, "quality": 55.0, "blended": 1.25},
                {"id": "acme/model-b", "rank": 2, "quality": 68.4, "blended": 2.5},
            ],
            "price": [],
            "quality": [],
        },
        "aa": {"acme/model-a": 55.0, "acme/model-b": 60.0},
        "prices": {
            "acme/model-a": [1.0, 2.0, 1.25, None],
            "acme/model-b": [2.0, 4.0, 2.5, None],
        },
    }
    base.update(snap_overrides)
    return {"snapshots": {"2026-08-26": base}}


def make_diff_catalog():
    return {
        "generated_at": "2026-09-02T09:15:00+00:00",
        "models": [
            {
                "id": "acme/model-b",
                "quality": 68.4,
                "scores": {
                    "overall": {"balanced": 68.4, "price": 68.4, "quality": 68.4}
                },
                "aa": {
                    "intelligence_index": 68.4,
                    "coding_index": 74.8,
                    "agentic_index": 59.1,
                },
                "pricing": {
                    "input_per_1m": 1.6,
                    "output_per_1m": 3.2,
                    "blended_per_1m": 2.6,
                },
                "discount": None,
            },
            {
                "id": "acme/model-a",
                "quality": 55.0,
                "scores": {
                    "overall": {"balanced": 55.0, "price": 55.0, "quality": 55.0}
                },
                "aa": {
                    "intelligence_index": 55.0,
                    "coding_index": None,
                    "agentic_index": None,
                },
                "pricing": {
                    "input_per_1m": 0.8,
                    "output_per_1m": 1.6,
                    "blended_per_1m": 1.0,
                },
                "discount": 0.5,
            },
            {
                "id": "acme/fresh",
                "quality": 40.0,
                "scores": {
                    "overall": {"balanced": 40.0, "price": 40.0, "quality": 40.0}
                },
                "aa": {
                    "intelligence_index": 40.0,
                    "coding_index": None,
                    "agentic_index": None,
                },
                "pricing": {
                    "input_per_1m": 0.5,
                    "output_per_1m": 1.0,
                    "blended_per_1m": 0.62,
                },
                "discount": None,
            },
        ],
        "filtered": [{"id": "acme/gone", "name": "Gone", "reasons": ["context"]}],
    }


def test_seven_days_before():
    assert gh.seven_days_before("2026-09-02") == "2026-08-26"


def test_build_diff_against_exact_baseline():
    diff = gh.build_diff(make_diff_catalog(), make_diff_history())
    assert diff["baseline_present"] is True
    assert diff["new_pool_ids_count"] == 1
    assert diff["new_pool_id_sample"] == ["acme/fresh"]
    btab = diff["tabs"]["balanced"]
    by_id = {e["id"]: e for e in btab["entries"]}
    assert by_id["acme/model-b"]["rank"] == 1
    assert by_id["acme/model-b"]["prev_rank"] == 2
    assert by_id["acme/model-b"]["delta"] == 1
    assert by_id["acme/model-a"]["rank"] == 2
    assert by_id["acme/model-a"]["prev_rank"] == 1
    assert by_id["acme/model-a"]["delta"] == -1
    assert "acme/fresh" in btab["new_ids"]
    assert diff["aa_movers"]["up"][0]["id"] == "acme/model-b"
    assert diff["aa_movers"]["up"][0]["delta"] == pytest.approx(8.4)
    assert diff["price_moves"]["down"][0]["id"] == "acme/model-a"
    assert diff["discounts"]["appeared"] == ["acme/model-a"]


def test_build_diff_missing_baseline():
    diff = gh.build_diff(make_diff_catalog(), {"snapshots": {}})
    assert diff["baseline_present"] is False
    assert diff["tabs"]["balanced"]["entries"] == []


def test_fallback_texts_grounding():
    diff = gh.build_diff(make_diff_catalog(), make_diff_history())
    texts = gh.fallback_texts(diff)
    assert set(texts) == {"week", "intelligence", "prices"}
    for text in texts.values():
        assert "`acme/" in text or len(diff["tabs"]["balanced"]["entries"]) == 0
        assert text.strip()  # non-empty


def test_fallback_texts_first_week():
    diff = gh.build_diff(make_diff_catalog(), {"snapshots": {}})
    texts = gh.fallback_texts(diff)
    assert "building up" in texts["week"]
