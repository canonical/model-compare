"""Tests for generate_highlights.py -- diff builder and fallback templates."""

import datetime
import json

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


# ---------------------------------------------------------------------------
# LLM client + reuse rule
# ---------------------------------------------------------------------------


def make_prev_highlights(source, generated_at):
    return {
        "schema_version": 1,
        "generated_at": generated_at,
        "source": source,
        "sections": {"week": "w", "intelligence": "i", "prices": "p"},
    }


def test_reuses_recent_llm_output(monkeypatch, tmp_path):
    prev = make_prev_highlights(
        "openrouter",
        datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    )
    prev_file = tmp_path / "prev.json"
    prev_file.write_text(json.dumps(prev))
    monkeypatch.setattr(
        gh, "generate_with_llm", lambda *a, **k: pytest.fail("must not call LLM")
    )
    out = tmp_path / "out.json"
    assert (
        gh.main(
            [
                "--catalog",
                str(_write_catalog(tmp_path)),
                "--history",
                str(_write_empty_history(tmp_path)),
                "--prev-highlights",
                str(prev_file),
                "--output",
                str(out),
            ]
        )
        == 0
    )
    written = json.loads(out.read_text())
    assert written["sections"] == prev["sections"]
    assert written["source"] == "openrouter"
    assert written["generated_at"] == prev["generated_at"]


def test_regenerates_fallback_output_regardless_of_age(monkeypatch, tmp_path):
    prev = make_prev_highlights("fallback", "2026-08-01T00:00:00+00:00")
    prev_file = tmp_path / "prev.json"
    prev_file.write_text(json.dumps(prev))
    monkeypatch.setattr(
        gh,
        "generate_with_llm",
        lambda diff, key: {"week": "w2", "intelligence": "i2", "prices": "p2"},
    )
    out = tmp_path / "out.json"
    assert (
        gh.main(
            [
                "--catalog",
                str(_write_catalog(tmp_path)),
                "--history",
                str(_write_empty_history(tmp_path)),
                "--prev-highlights",
                str(prev_file),
                "--output",
                str(out),
            ]
        )
        == 0
    )
    written = json.loads(out.read_text())
    assert written["source"] == "openrouter"
    assert written["sections"]["week"] == "w2"


def test_llm_failure_falls_back_to_templates(monkeypatch, tmp_path):
    prev = make_prev_highlights("fallback", "2026-08-01T00:00:00+00:00")
    prev_file = tmp_path / "prev.json"
    prev_file.write_text(json.dumps(prev))
    monkeypatch.setattr(gh, "generate_with_llm", lambda diff, key: None)
    out = tmp_path / "out.json"
    assert (
        gh.main(
            [
                "--catalog",
                str(_write_catalog(tmp_path)),
                "--history",
                str(_write_empty_history(tmp_path)),
                "--prev-highlights",
                str(prev_file),
                "--output",
                str(out),
            ]
        )
        == 0
    )
    written = json.loads(out.read_text())
    assert written["source"] == "fallback"
    assert set(written["sections"]) == {"week", "intelligence", "prices"}


def test_generate_with_llm_model_fallback_chain(monkeypatch):
    # All but the last model fail so the chain is walked end to end: the
    # first model that answers wins, so a mid-chain success would stop the
    # loop before the final entry and captured would be shorter than
    # LLM_MODELS (which the assertion below requires to be fully walked).
    responses = {m: RuntimeError("boom") for m in gh.LLM_MODELS[:-1]}
    captured = []

    def fake_post(model, body, api_key):
        captured.append(model)
        if model in responses and isinstance(responses[model], Exception):
            raise responses[model]
        # _post_chat returns the parsed response body (json.load), so the
        # fake must return an OpenRouter-shaped dict, not a bare string.
        return {
            "choices": [
                {
                    "message": {
                        "content": '{"week": "w", "intelligence": "i", "prices": "p"}'
                    }
                }
            ]
        }

    monkeypatch.setattr(gh, "_post_chat", fake_post)
    texts = gh.generate_with_llm({"baseline_present": False}, "key")
    assert texts == {"week": "w", "intelligence": "i", "prices": "p"}
    assert captured == list(gh.LLM_MODELS)


def test_generate_with_llm_rejects_unparseable(monkeypatch):
    monkeypatch.setattr(
        gh,
        "_post_chat",
        lambda model, body, key: {
            "choices": [{"message": {"content": "not json at all"}}]
        },
    )
    assert gh.generate_with_llm({"baseline_present": False}, "key") is None


def test_generate_with_llm_no_key_returns_none():
    assert gh.generate_with_llm({"baseline_present": False}, None) is None


def _write_catalog(tmp_path):
    f = tmp_path / "catalog.json"
    f.write_text(json.dumps(make_diff_catalog()))
    return f


def _write_empty_history(tmp_path):
    f = tmp_path / "history.json"
    f.write_text(json.dumps({"snapshots": {}}))
    return f
