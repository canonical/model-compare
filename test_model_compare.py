"""Unit tests for model_compare scoring, matching, and filtering logic.

These cover pure functions only -- no network access is performed. Run with:

    pytest -q
"""

from __future__ import annotations

import json
import time
from argparse import Namespace
from datetime import datetime, timezone

import pytest

import build_site_data as bsd
import model_compare as mc


def make_args(**overrides):
    """Build an args namespace with the defaults build_candidates/compute_scores expect."""
    base = dict(
        min_context=0,
        priority="balanced",
        top=5,
        best=False,
        json=False,
        catalog=False,
        input_share=0.75,
        recency_half_life=120.0,
        max_age_days=0.0,
        quality_ref=70.0,
        aa_api_key=None,
        no_require_tools=True,
        exclude_free=False,
        include_batch=False,
        discount=False,
        no_zdr=False,
        no_cache=True,
        cache_ttl=0,
    )
    base.update(overrides)
    return Namespace(**base)


def make_model(**overrides):
    model = {
        "id": "acme/model-a",
        "name": "Acme: Model A",
        "context_length": 2_000_000,
        "pricing": {"prompt": "0.000001", "completion": "0.000002"},
        "architecture": {"modality": "text->text"},
        "supported_parameters": ["tools", "tool_choice"],
        "created": 0,
    }
    model.update(overrides)
    return model


# ---------------------------------------------------------------------------
# coerce_int (Fix #1)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        (128000, 128000),
        ("128000", 128000),
        (128000.9, 128000),
        (None, 0),
        ("", 0),
        ("nan-ish", 0),
        ([], 0),
    ],
)
def test_coerce_int(value, expected):
    assert mc.coerce_int(value) == expected


def test_coerce_int_custom_default():
    assert mc.coerce_int(None, default=-1) == -1


def test_build_candidates_survives_string_context_length():
    # Regression: string context_length must not raise a TypeError.
    models = [make_model(context_length="2000000")]
    candidates, dropped = mc.build_candidates(
        models, make_args(min_context=1_000_000), {}, {"acme/model-a"}
    )
    assert len(candidates) == 1
    assert candidates[0]["context"] == 2_000_000
    assert "context" not in dropped


# ---------------------------------------------------------------------------
# build_candidates filtering
# ---------------------------------------------------------------------------


def test_build_candidates_drops_malformed_id():
    models = [make_model(id="no-slash")]
    candidates, dropped = mc.build_candidates(models, make_args(), {}, {"acme/model-a"})
    assert candidates == []
    assert dropped["malformed id"] == 1


def test_build_candidates_drops_low_context():
    models = [make_model(context_length=1000)]
    candidates, dropped = mc.build_candidates(
        models, make_args(min_context=1_000_000), {}, {"acme/model-a"}
    )
    assert candidates == []
    assert dropped["context"] == 1


def test_build_candidates_drops_bad_pricing():
    models = [make_model(pricing={"prompt": "x", "completion": "0.1"})]
    candidates, dropped = mc.build_candidates(models, make_args(), {}, {"acme/model-a"})
    assert candidates == []
    assert dropped["pricing"] == 1


def test_build_candidates_drops_negative_pricing():
    models = [make_model(pricing={"prompt": "-1", "completion": "0.1"})]
    candidates, dropped = mc.build_candidates(models, make_args(), {}, {"acme/model-a"})
    assert candidates == []
    assert dropped["pricing"] == 1


def test_build_candidates_exclude_free():
    models = [make_model(pricing={"prompt": "0", "completion": "0"})]
    candidates, dropped = mc.build_candidates(
        models, make_args(exclude_free=True), {}, {"acme/model-a"}
    )
    assert candidates == []
    assert dropped["free"] == 1


def test_build_candidates_drops_non_text_output():
    models = [make_model(architecture={"modality": "text->image"})]
    candidates, dropped = mc.build_candidates(models, make_args(), {}, {"acme/model-a"})
    assert candidates == []
    assert dropped["modality"] == 1


def test_build_candidates_requires_tools_when_asked():
    models = [make_model(supported_parameters=["tools"])]  # missing tool_choice
    candidates, dropped = mc.build_candidates(
        models, make_args(no_require_tools=False), {}, {"acme/model-a"}
    )
    assert candidates == []
    assert dropped["tool calling"] == 1


def test_build_candidates_blended_price():
    models = [make_model(pricing={"prompt": "0.000001", "completion": "0.000005"})]
    candidates, _ = mc.build_candidates(
        models, make_args(input_share=0.75), {}, {"acme/model-a"}
    )
    cand = candidates[0]
    assert cand["price_in"] == pytest.approx(1.0)
    assert cand["price_out"] == pytest.approx(5.0)
    # 0.75*1 + 0.25*5 = 2.0
    assert cand["blended"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# batch variants and discounts
# ---------------------------------------------------------------------------


def test_build_candidates_drops_batch_by_default():
    models = [make_model(id="acme/model-a:batch")]
    candidates, dropped = mc.build_candidates(models, make_args(), {}, {"acme/model-a"})
    assert candidates == []
    assert dropped["batch"] == 1


def test_build_candidates_keeps_batch_with_include_batch():
    models = [make_model(id="acme/model-a:batch")]
    candidates, dropped = mc.build_candidates(
        models, make_args(include_batch=True), {}, {"acme/model-a:batch"}
    )
    assert [c["id"] for c in candidates] == ["acme/model-a:batch"]
    assert not dropped


def test_build_candidates_discount_filter_keeps_discounted_only():
    models = [make_model(id="acme/discounted"), make_model(id="acme/normal")]
    candidates, dropped = mc.build_candidates(
        models, make_args(discount=True), {"acme/discounted": 0.5}, {"acme/discounted"}
    )
    assert [c["id"] for c in candidates] == ["acme/discounted"]
    assert dropped["no discount"] == 1
    assert candidates[0]["discount"] == 0.5


def test_build_candidates_discount_filter_without_data_drops_all():
    models = [make_model()]
    candidates, dropped = mc.build_candidates(
        models, make_args(discount=True), {}, {"acme/model-a"}
    )
    assert candidates == []
    assert dropped["no discount"] == 1


def test_build_candidates_discount_filter_ignores_negligible_discount():
    models = [make_model(id="acme/sliver")]
    candidates, dropped = mc.build_candidates(
        models, make_args(discount=True), {"acme/sliver": 0.004}, {"acme/sliver"}
    )
    assert candidates == []
    assert dropped["no discount"] == 1


def test_build_candidates_stores_discount():
    models = [make_model(id="acme/model-a")]
    candidates, _ = mc.build_candidates(
        models, make_args(), {"acme/model-a": 0.75}, {"acme/model-a"}
    )
    assert candidates[0]["discount"] == 0.75


# ---------------------------------------------------------------------------
# discount data source (OpenRouter frontend models API)
# ---------------------------------------------------------------------------


def test_fetch_discount_map_builds_variant_aware_keys(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    payload = {
        "data": {
            "models": [
                {
                    "slug": "acme/a",
                    "endpoint": {
                        "variant": "standard",
                        "pricing": {"discount": 0.5, "prompt": "0.1"},
                    },
                },
                {
                    "slug": "acme/b",
                    "endpoint": {"variant": "free", "pricing": {"discount": 0}},
                },
                {
                    "slug": "acme/c",
                    "endpoint": {
                        "variant": "batch",
                        "pricing": {"discount": 0.75},
                    },
                },
                {
                    "slug": "~acme/private",
                    "endpoint": {
                        "variant": "standard",
                        "pricing": {"discount": 0.9},
                    },
                },
                {"slug": "acme/no-endpoint", "endpoint": None},
                {
                    "slug": "acme/no-discount-field",
                    "endpoint": {"variant": "standard", "pricing": {}},
                },
            ]
        }
    }
    monkeypatch.setattr(mc, "fetch_json", lambda *a, **k: payload)
    result, cached = mc.fetch_discount_map(make_args(no_cache=True))
    assert cached is False
    assert result == {"acme/a": 0.5, "acme/b:free": 0.0, "acme/c:batch": 0.75}


def test_fetch_discount_map_caches_result(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    calls = []

    def fake_fetch(*a, **k):
        calls.append(1)
        return {
            "data": {
                "models": [
                    {
                        "slug": "acme/a",
                        "endpoint": {
                            "variant": "standard",
                            "pricing": {"discount": 0.25},
                        },
                    }
                ]
            }
        }

    monkeypatch.setattr(mc, "fetch_json", fake_fetch)
    args = make_args(no_cache=False, cache_ttl=3600)
    first, cached1 = mc.fetch_discount_map(args)
    second, cached2 = mc.fetch_discount_map(args)
    assert len(calls) == 1
    assert (cached1, cached2) == (False, True)
    assert first == second == {"acme/a": 0.25}


def test_fetch_discount_map_fetch_failure_returns_empty(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))

    def boom(*a, **k):
        raise RuntimeError("network down")

    monkeypatch.setattr(mc, "fetch_json", boom)
    result, cached = mc.fetch_discount_map(make_args(no_cache=True))
    assert result == {}
    assert cached is False


def test_fetch_discount_map_does_not_cache_empty(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    calls = []

    def fake_fetch(*a, **k):
        calls.append(1)
        return {"data": {"models": []}}

    monkeypatch.setattr(mc, "fetch_json", fake_fetch)
    args = make_args(no_cache=False, cache_ttl=3600)
    first, cached1 = mc.fetch_discount_map(args)
    second, cached2 = mc.fetch_discount_map(args)
    assert first == second == {}
    assert (cached1, cached2) == (False, False)
    assert len(calls) == 2
    assert not (tmp_path / "model-compare" / "openrouter-discounts.json").exists()


def test_fetch_discount_map_treats_empty_cached_map_as_miss(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    calls = []

    def fake_fetch(*a, **k):
        calls.append(1)
        return {
            "data": {
                "models": [
                    {
                        "slug": "acme/a",
                        "endpoint": {
                            "variant": "standard",
                            "pricing": {"discount": 0.25},
                        },
                    }
                ]
            }
        }

    monkeypatch.setattr(mc, "fetch_json", fake_fetch)
    cache_dir = tmp_path / "model-compare"
    cache_dir.mkdir(parents=True)
    (cache_dir / "openrouter-discounts.json").write_text(
        json.dumps({"fetched_at": time.time(), "payload": {}})
    )
    result, cached = mc.fetch_discount_map(make_args(no_cache=False, cache_ttl=3600))
    assert len(calls) == 1
    assert cached is False
    assert result == {"acme/a": 0.25}


# ---------------------------------------------------------------------------
# ZDR (zero data retention) filter
# ---------------------------------------------------------------------------


def test_build_candidates_drops_non_zdr_by_default():
    models = [make_model(id="acme/zdr"), make_model(id="acme/plain")]
    candidates, dropped = mc.build_candidates(models, make_args(), {}, {"acme/zdr"})
    assert [c["id"] for c in candidates] == ["acme/zdr"]
    assert dropped["not ZDR"] == 1


def test_build_candidates_no_zdr_considers_all():
    models = [make_model(id="acme/zdr"), make_model(id="acme/plain")]
    candidates, dropped = mc.build_candidates(models, make_args(no_zdr=True), {}, set())
    assert len(candidates) == 2
    assert not dropped


def test_build_candidates_zdr_variant_keys():
    # a :batch variant counts as ZDR only if that variant itself is listed
    models = [make_model(id="acme/m:batch"), make_model(id="acme/n:batch")]
    candidates, dropped = mc.build_candidates(
        models, make_args(include_batch=True), {}, {"acme/m:batch"}
    )
    assert [c["id"] for c in candidates] == ["acme/m:batch"]
    assert dropped["not ZDR"] == 1


def test_build_candidates_collects_filtered_entries():
    models = [
        make_model(id="acme/model-a"),
        make_model(id="acme/model-b", name="B corp: Model B", context_length=100),
        make_model(id="no-slash", name="No Slash"),
    ]
    filtered = []
    candidates, dropped = mc.build_candidates(
        models, make_args(min_context=1_000_000), {}, {"acme/model-a"}, filtered
    )
    assert len(candidates) == 1
    # "no-slash" counts under malformed id but is withheld from filtered,
    # which only ever carries valid provider/model ids.
    assert filtered == [
        {"id": "acme/model-b", "name": "B corp: Model B", "reasons": ["context"]}
    ]
    assert dropped == {"context": 1, "malformed id": 1}


def test_build_candidates_empty_id_counted_not_listed():
    # An empty id counts under "malformed id" but is never emitted into
    # filtered, where it would fail the site validator's no-empty-id rule.
    args = make_args(min_context=0)
    models = [make_model(id=""), make_model(id="acme/model-a")]
    filtered = []
    candidates, dropped = mc.build_candidates(
        models, args, {}, {"acme/model-a"}, filtered
    )
    assert dropped == {"malformed id": 1}
    assert filtered == []
    mc.compute_scores(candidates, args, {})
    doc = mc.build_catalog(args, models, candidates, dropped, filtered, {}, {}, None)
    assert doc["filtered"] == []
    assert [e["id"] for e in doc["models"]] == ["acme/model-a"]


def test_build_candidates_without_collector_matches_old_signature():
    # 2-tuple unpacking stays valid; no filtered list, no behavior change.
    candidates, dropped = mc.build_candidates(
        [make_model(id="acme/model-b", context_length=100)],
        make_args(min_context=1_000_000),
        {},
        {"acme/model-a"},
    )
    assert candidates == []
    assert dropped == {"context": 1}


def test_build_candidates_additive_fields():
    created = 1_750_000_000
    candidates, _ = mc.build_candidates(
        [make_model(id="acme/model-a", created=created)],
        make_args(),
        {},
        {"acme/model-a"},
    )
    cand = candidates[0]
    assert cand["created"] == float(created)
    assert cand["tool_calling"] is True
    assert cand["zdr"] is True
    assert cand["expired"] is False


def test_build_candidates_zdr_null_under_no_zdr():
    candidates, _ = mc.build_candidates(
        [make_model(id="acme/model-a")], make_args(no_zdr=True), {}, set()
    )
    assert candidates[0]["zdr"] is None


def test_build_candidates_tool_calling_false_when_relaxed():
    candidates, _ = mc.build_candidates(
        [make_model(supported_parameters=[])],
        make_args(no_require_tools=True),
        {},
        {"acme/model-a"},
    )
    assert candidates[0]["tool_calling"] is False


# ---------------------------------------------------------------------------
# model_family / catalog_weights (catalog helpers)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model_id,expected",
    [
        ("z-ai/glm-5.3", "glm"),
        ("openai/gpt-5.2-mini", "gpt"),
        ("anthropic/claude-opus-4.6", "claude"),
        ("alibaba/qwen3-max", "qwen"),
        ("deepseek/deepseek-chat-v4", "deepseek"),
        ("amazon/nova-pro-2", "nova"),
        ("google/gemini_2_5_pro", "gemini"),
        ("x-ai/o4-mini", "o"),  # documented oddball
        ("kimi/k2", None),
        ("acme/model-a", "model"),
        ("acme/model-a:variant", "model"),
        ("weird/model", "model"),
    ],
)
def test_model_family(model_id, expected):
    assert mc.model_family(model_id) == expected


def test_catalog_weights_match_base_when_quality_present():
    weights = mc.catalog_weights([{"id": "a/b"}], {"a/b": 50.0})
    assert weights == mc.PRIORITY_WEIGHTS


def test_catalog_weights_renormalize_when_quality_blind():
    weights = mc.catalog_weights([{"id": "a/b"}], {})
    for priority, base in mc.PRIORITY_WEIGHTS.items():
        assert "quality" not in weights[priority]
        total = sum(weights[priority].values())
        assert total == pytest.approx(1.0)
        for name, value in base.items():
            if name != "quality":
                assert weights[priority][name] == pytest.approx(
                    value / (sum(base.values()) - base["quality"])
                )


def test_fetch_zdr_set_builds_variant_keys(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    payload = {
        "data": {
            "models": [
                {
                    "slug": "acme/a",
                    "endpoint": {
                        "variant": "standard",
                        "pricing": {"prompt": "0.1"},
                    },
                },
                {"slug": "acme/b", "endpoint": {"variant": "batch", "pricing": {}}},
                {"slug": "~acme/private", "endpoint": {"variant": "standard"}},
                {"slug": "acme/no-endpoint", "endpoint": None},
            ]
        }
    }
    monkeypatch.setattr(mc, "fetch_json", lambda *a, **k: payload)
    ids, cached = mc.fetch_zdr_set(make_args(no_cache=True))
    assert cached is False
    # membership comes from the zdr=true filter itself: entries are kept even
    # without endpoint details, keyed by bare slug as the best available id
    assert ids == {"acme/a", "acme/b:batch", "acme/no-endpoint"}


def test_fetch_zdr_set_skips_fetch_when_disabled(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    calls = []

    def fake_fetch(*a, **k):
        calls.append(1)
        return {"data": {"models": []}}

    monkeypatch.setattr(mc, "fetch_json", fake_fetch)
    ids, cached = mc.fetch_zdr_set(make_args(no_cache=True, no_zdr=True))
    assert ids == set()
    assert cached is False
    assert calls == []


def test_fetch_zdr_set_caches_result(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    calls = []

    def fake_fetch(*a, **k):
        calls.append(1)
        return {
            "data": {
                "models": [
                    {
                        "slug": "acme/a",
                        "endpoint": {
                            "variant": "standard",
                            "pricing": {"prompt": "0.1"},
                        },
                    }
                ]
            }
        }

    monkeypatch.setattr(mc, "fetch_json", fake_fetch)
    args = make_args(no_cache=False, cache_ttl=3600)
    first, cached1 = mc.fetch_zdr_set(args)
    second, cached2 = mc.fetch_zdr_set(args)
    assert len(calls) == 1
    assert (cached1, cached2) == (False, True)
    assert first == second == {"acme/a"}


def test_fetch_zdr_set_does_not_cache_empty(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    calls = []

    def fake_fetch(*a, **k):
        calls.append(1)
        return {"data": {"models": []}}

    monkeypatch.setattr(mc, "fetch_json", fake_fetch)
    args = make_args(no_cache=False, cache_ttl=3600)
    first, cached1 = mc.fetch_zdr_set(args)
    second, cached2 = mc.fetch_zdr_set(args)
    assert first == second == set()
    assert (cached1, cached2) == (False, False)
    assert len(calls) == 2
    assert not (tmp_path / "model-compare" / "openrouter-zdr.json").exists()


def test_fetch_zdr_set_treats_empty_cached_set_as_miss(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    calls = []

    def fake_fetch(*a, **k):
        calls.append(1)
        return {
            "data": {
                "models": [
                    {
                        "slug": "acme/a",
                        "endpoint": {
                            "variant": "standard",
                            "pricing": {"prompt": "0.1"},
                        },
                    }
                ]
            }
        }

    monkeypatch.setattr(mc, "fetch_json", fake_fetch)
    cache_dir = tmp_path / "model-compare"
    cache_dir.mkdir(parents=True)
    (cache_dir / "openrouter-zdr.json").write_text(
        json.dumps({"fetched_at": time.time(), "payload": []})
    )
    result, cached = mc.fetch_zdr_set(make_args(no_cache=False, cache_ttl=3600))
    assert len(calls) == 1
    assert cached is False
    assert result == {"acme/a"}


def test_fetch_zdr_set_fetch_failure_returns_empty(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))

    def boom(*a, **k):
        raise RuntimeError("network down")

    monkeypatch.setattr(mc, "fetch_json", boom)
    result, cached = mc.fetch_zdr_set(make_args(no_cache=True))
    assert result == set()
    assert cached is False


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, "--"),
        (0, "--"),
        (0.0, "--"),
        (0.004, "--"),
        (0.005, "--"),
        (0.006, "1%"),
        (0.049, "5%"),
        (0.5, "50%"),
        (0.561, "56%"),
        (0.75, "75%"),
        (-0.1, "--"),
    ],
)
def test_fmt_discount(value, expected):
    assert mc.fmt_discount(value) == expected


def test_print_table_shows_disc_column(capsys):
    top = [_cand("a/x", 1.0)]
    top[0].update({"score": 0.9, "quality": None, "discount": 0.5})
    mc.print_table(top, 1, {"price": 1.0}, "note")
    out = capsys.readouterr().out
    assert "DISC" in out
    assert "50%" in out


def test_print_table_dash_without_discount(capsys):
    top = [_cand("a/x", 1.0)]
    top[0].update({"score": 0.9, "quality": None, "discount": None})
    mc.print_table(top, 1, {"price": 1.0}, "note")
    out = capsys.readouterr().out
    data_row = next(line for line in out.splitlines() if line.strip().startswith("1"))
    assert "DISC" in out
    assert "--" in data_row


def test_print_json_includes_discount(capsys):
    rows = [_cand("a/x", 1.0)]
    rows[0].update({"score": 0.9, "discount": 0.25})
    mc.print_json(rows)
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["discount"] == 0.25


def test_print_json_discount_null_when_absent(capsys):
    rows = [_cand("a/x", 1.0)]
    rows[0].update({"score": 0.9, "discount": None})
    mc.print_json(rows)
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["discount"] is None


def test_print_json_discount_null_for_negligible(capsys):
    rows = [_cand("a/x", 1.0)]
    rows[0].update({"score": 0.9, "discount": 0.004})
    mc.print_json(rows)
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["discount"] is None


# ---------------------------------------------------------------------------
# opencode model namespace
# ---------------------------------------------------------------------------


def test_opencode_model_id():
    assert mc.opencode_model_id("z-ai/glm-5.3-flash") == "openrouter/z-ai/glm-5.3-flash"
    assert mc.opencode_model_id("nvidia/x:free") == "openrouter/nvidia/x:free"


def test_print_json_includes_opencode_model(capsys):
    rows = [_cand("a/x", 1.0)]
    rows[0].update({"score": 0.9})
    mc.print_json(rows)
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["model"] == "a/x"
    assert payload[0]["opencode_model"] == "openrouter/a/x"


def test_best_prints_provider_qualified_id(monkeypatch, capsys):
    monkeypatch.setattr(
        mc, "fetch_openrouter_models", lambda a: ([make_model()], False)
    )
    monkeypatch.setattr(mc, "fetch_discount_map", lambda a: ({}, False))
    monkeypatch.setattr(mc, "fetch_zdr_set", lambda a: ({"acme/model-a"}, False))
    monkeypatch.setattr(mc, "fetch_aa_entries", lambda a: ([], None, False))
    argv = ["--best", "--no-cache", "--min-context", "0", "--no-require-tools"]
    assert mc.main(argv) == 0
    assert capsys.readouterr().out.strip() == "openrouter/acme/model-a"


# ---------------------------------------------------------------------------
# compute_scores
# ---------------------------------------------------------------------------


def _cand(model_id, blended, context=2_000_000, age_days: float | None = 0.0):
    return {
        "id": model_id,
        "name": model_id,
        "context": context,
        "price_in": blended,
        "price_out": blended,
        "blended": blended,
        "age_days": age_days,
        "quality": None,
        "discount": None,
        "score": 0.0,
    }


def test_compute_scores_cheapest_gets_top_price_score():
    candidates = [_cand("a/cheap", 1.0), _cand("a/pricey", 100.0)]
    mc.compute_scores(candidates, make_args(priority="price"), {})
    by_id = {c["id"]: c for c in candidates}
    assert by_id["a/cheap"]["price_score"] == pytest.approx(1.0)
    assert by_id["a/pricey"]["price_score"] < by_id["a/cheap"]["price_score"]
    # price priority -> cheap should rank first
    assert candidates[0]["id"] == "a/cheap"


def test_compute_scores_drops_quality_weight_when_all_unmatched():
    candidates = [_cand("a/x", 1.0), _cand("a/y", 2.0)]
    weights = mc.compute_scores(candidates, make_args(), {})
    assert "quality" not in weights
    assert sum(weights.values()) == pytest.approx(1.0)


def test_compute_scores_keeps_quality_weight_on_partial_match():
    candidates = [_cand("a/x", 1.0), _cand("a/y", 2.0)]
    weights = mc.compute_scores(candidates, make_args(), {"a/x": 70.0})
    assert "quality" in weights
    by_id = {c["id"]: c for c in candidates}
    # unmatched candidate is penalized with quality_score 0, not reweighted
    assert by_id["a/y"]["quality_score"] == 0.0
    assert by_id["a/x"]["quality_score"] == pytest.approx(1.0)


def test_compute_scores_quality_ref_clamps():
    candidates = [_cand("a/x", 1.0)]
    mc.compute_scores(candidates, make_args(quality_ref=70.0), {"a/x": 140.0})
    assert candidates[0]["quality_score"] == 1.0


def test_compute_scores_context_score_bounds():
    floor = 1_000_000
    candidates = [
        _cand("a/at-floor", 1.0, context=floor),
        _cand("a/4x", 1.0, context=4 * floor),
        _cand("a/8x", 1.0, context=8 * floor),
    ]
    mc.compute_scores(candidates, make_args(min_context=floor), {})
    by_id = {c["id"]: c for c in candidates}
    assert by_id["a/at-floor"]["context_score"] == 0.0
    assert by_id["a/4x"]["context_score"] == pytest.approx(1.0)
    assert by_id["a/8x"]["context_score"] == 1.0  # capped


def test_compute_scores_age_score_decay():
    candidates = [
        _cand("a/fresh", 1.0, age_days=0.0),
        _cand("a/half", 1.0, age_days=120.0),
        _cand("a/unknown", 1.0, age_days=None),
    ]
    mc.compute_scores(candidates, make_args(recency_half_life=120.0), {})
    by_id = {c["id"]: c for c in candidates}
    assert by_id["a/fresh"]["age_score"] == pytest.approx(1.0)
    assert by_id["a/half"]["age_score"] == pytest.approx(0.5)
    assert by_id["a/unknown"]["age_score"] == pytest.approx(0.5)


def test_compute_scores_single_candidate_pool():
    candidates = [_cand("a/only", 5.0)]
    weights = mc.compute_scores(candidates, make_args(), {})
    assert candidates[0]["price_score"] == pytest.approx(1.0)
    assert sum(weights.values()) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# AA lookup + matching
# ---------------------------------------------------------------------------


def test_build_aa_lookup_skips_non_numeric_index():
    entries = [
        {"key": "acme/model-a", "name": "Model A", "index": "not-a-number"},
        {"key": "acme/model-b", "name": "Model B", "index": 55.0},
    ]
    exact, fuzzy = mc.build_aa_lookup(entries)
    # only the numeric entry survives into the fuzzy list
    assert len(fuzzy) == 1
    assert (
        mc.match_quality({"id": "acme/model-b", "name": "Model B"}, exact, fuzzy)
        == 55.0
    )


def test_build_aa_lookup_skips_non_finite_index():
    entries = [
        {"key": "acme/model-a", "name": "Model A", "index": float("nan")},
        {"key": "acme/model-b", "name": "Model B", "index": 55.0},
    ]
    exact, fuzzy = mc.build_aa_lookup(entries)
    # a NaN index would serialize as JSON NaN and score 1.0; never match it
    assert (
        mc.match_quality({"id": "acme/model-a", "name": "Acme: Model A"}, exact, fuzzy)
        is None
    )
    assert (
        mc.match_quality({"id": "acme/model-b", "name": "Model B"}, exact, fuzzy)
        == 55.0
    )


def test_match_quality_exact_by_id():
    entries = [{"key": "acme/model-a", "name": "Model A", "index": 60.0}]
    exact, fuzzy = mc.build_aa_lookup(entries)
    assert (
        mc.match_quality({"id": "acme/model-a", "name": "Acme: Model A"}, exact, fuzzy)
        == 60.0
    )


def test_match_quality_fuzzy_overlap():
    entries = [{"key": "model-a", "name": "Model A", "index": 55.0}]
    exact, fuzzy = mc.build_aa_lookup(entries)
    result = mc.match_quality(
        {"id": "acme/model-a-extra", "name": "Model A Extra"},
        exact,
        fuzzy,
        allow_fuzzy=True,
    )
    assert result == 55.0


def test_match_quality_no_match_returns_none():
    entries = [{"key": "acme/model-a", "name": "Model A", "index": 60.0}]
    exact, fuzzy = mc.build_aa_lookup(entries)
    assert (
        mc.match_quality({"id": "other/unrelated-xyz", "name": "Zzz"}, exact, fuzzy)
        is None
    )


def test_match_quality_fuzzy_disabled_by_default():
    # Regression: the scrape+fuzzy path paired z-ai/glm-5.3-flash with the
    # single AA entry glm-5-3 (Jaccard 0.6 >= 0.5). Exact-only must refuse.
    entries = [{"key": "glm-5-3", "name": "GLM-5.3 (max)", "index": 59.5}]
    exact, fuzzy = mc.build_aa_lookup(entries)
    model = {"id": "z-ai/glm-5.3-flash", "name": "Z.AI: GLM 5.3 Flash"}
    assert mc.match_quality(model, exact, fuzzy) is None
    assert mc.match_quality(model, exact, fuzzy, allow_fuzzy=True) == 59.5


# ---------------------------------------------------------------------------
# aa_api_entries walk (Fix #3 canonical key, Fix #4 depth guard)
# ---------------------------------------------------------------------------


def test_aa_api_prefers_canonical_index_key(monkeypatch):
    payload = {
        "data": [
            {
                "id": "acme/m",
                "name": "M",
                # a decoy metric that also contains "intelligence"
                "otherIntelligenceScore": 999.0,
                "artificialAnalysisIntelligenceIndex": 42.0,
            }
        ]
    }
    monkeypatch.setattr(mc, "fetch_json", lambda *a, **k: payload)
    entries = mc.aa_api_entries("dummy-key")
    assert len(entries) == 1
    assert entries[0]["index"] == 42.0


def test_aa_api_substring_fallback(monkeypatch):
    payload = {"data": [{"id": "acme/m", "name": "M", "intelligenceIndex": 33.0}]}
    monkeypatch.setattr(mc, "fetch_json", lambda *a, **k: payload)
    entries = mc.aa_api_entries("dummy-key")
    assert entries[0]["index"] == 33.0


def test_aa_api_ignores_estimated_and_cost(monkeypatch):
    payload = {
        "data": [
            {
                "id": "acme/m",
                "name": "M",
                "estimatedIntelligence": 1.0,
                "intelligenceCost": 2.0,
            }
        ]
    }
    monkeypatch.setattr(mc, "fetch_json", lambda *a, **k: payload)
    entries = mc.aa_api_entries("dummy-key")
    assert entries == []


def test_aa_api_depth_guard_does_not_recurse_forever(monkeypatch):
    # Build a payload deeper than the guard threshold; must return without RecursionError.
    node = {"id": "acme/deep", "name": "Deep", "intelligenceIndex": 10.0}
    for _ in range(500):
        node = {"child": node}
    monkeypatch.setattr(mc, "fetch_json", lambda *a, **k: node)
    entries = mc.aa_api_entries("dummy-key")
    # The scored leaf is deeper than the guard, so it is not collected -- but crucially
    # the call completes instead of raising RecursionError.
    assert isinstance(entries, list)


# ---------------------------------------------------------------------------
# aa_scrape_entries (JSON-LD page scrape fallback)
# ---------------------------------------------------------------------------


def _aa_html(*models):
    """Build a minimal artificialanalysis.ai page with JSON-LD benchmark data."""
    data = [
        {
            "label": label,
            "detailsUrl": f"https://artificialanalysis.ai/models/{slug}",
            "artificialAnalysisIntelligenceIndex": index,
        }
        for slug, label, index in models
    ]
    block = json.dumps({"@type": "ItemList", "data": data})
    return (
        "<html><head>"
        '<script type="application/ld+json">' + block + "</script>"
        "</head><body></body></html>"
    ).encode("utf-8")


def test_aa_scrape_extracts_index(monkeypatch):
    html = _aa_html(
        ("gpt-4o", "GPT-4o", 60.0),
        ("claude-3-5-sonnet", "Claude 3.5 Sonnet", 55.0),
    )
    monkeypatch.setattr(mc, "http_get", lambda *a, **k: html)
    entries = mc.aa_scrape_entries()
    by_key = {e["key"]: e for e in entries}
    assert by_key["gpt-4o"]["index"] == 60.0
    assert by_key["gpt-4o"]["name"] == "GPT-4o"
    assert by_key["claude-3-5-sonnet"]["index"] == 55.0


def test_aa_scrape_skips_entries_without_index(monkeypatch):
    html = _aa_html(("gpt-4o", "GPT-4o", 60.0))
    # Add a second JSON-LD block without the index field; must be ignored.
    noise = b'<script type="application/ld+json">{"@type":"ItemList","data":[{"label":"No Index"}]}</script>'
    monkeypatch.setattr(mc, "http_get", lambda *a, **k: html + noise)
    entries = mc.aa_scrape_entries()
    assert [e["key"] for e in entries] == ["gpt-4o"]


def test_aa_scrape_bad_json_block_is_tolerated(monkeypatch):
    html = _aa_html(("gpt-4o", "GPT-4o", 60.0))
    bad = b'<script type="application/ld+json">{not valid json}</script>'
    monkeypatch.setattr(mc, "http_get", lambda *a, **k: html + bad)
    entries = mc.aa_scrape_entries()
    assert [e["key"] for e in entries] == ["gpt-4o"]


# ---------------------------------------------------------------------------
# realistic OpenRouter frontend payload (discount + ZDR key shape)
# ---------------------------------------------------------------------------


def _realistic_openrouter_payload():
    """Mirror the real frontend `models/find` response shape with real slugs."""
    return {
        "data": {
            "models": [
                {
                    "slug": "openai/gpt-4o",
                    "endpoint": {
                        "variant": "standard",
                        "pricing": {"prompt": "0.1", "discount": 0.5},
                    },
                },
                {
                    "slug": "openai/gpt-4o",
                    "endpoint": {
                        "variant": "free",
                        "pricing": {"prompt": "0", "discount": 0},
                    },
                },
                {
                    "slug": "anthropic/claude-sonnet-4-20250514",
                    "endpoint": {
                        "variant": "batch",
                        "pricing": {"prompt": "0.1", "discount": 0.25},
                    },
                },
                {
                    "slug": "~acme/private",
                    "endpoint": {
                        "variant": "standard",
                        "pricing": {"prompt": "0.1", "discount": 0.9},
                    },
                },
            ]
        }
    }


def test_fetch_discount_map_realistic_slugs(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr(
        mc, "fetch_json", lambda *a, **k: _realistic_openrouter_payload()
    )
    result, cached = mc.fetch_discount_map(make_args(no_cache=True))
    assert cached is False
    # Real catalog ids use "provider/model" and ":variant" suffixes.
    assert result == {
        "openai/gpt-4o": 0.5,
        "openai/gpt-4o:free": 0.0,
        "anthropic/claude-sonnet-4-20250514:batch": 0.25,
    }


def test_fetch_zdr_set_realistic_slugs(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr(
        mc, "fetch_json", lambda *a, **k: _realistic_openrouter_payload()
    )
    ids, cached = mc.fetch_zdr_set(make_args(no_cache=True))
    assert cached is False
    assert ids == {
        "openai/gpt-4o",
        "openai/gpt-4o:free",
        "anthropic/claude-sonnet-4-20250514:batch",
    }


# ---------------------------------------------------------------------------
# formatting helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [(0.0, "0"), (0.123, "0.123"), (1.5, "1.50"), (12.3, "12.3"), (150.0, "150")],
)
def test_fmt_price(value, expected):
    assert mc.fmt_price(value) == expected


@pytest.mark.parametrize(
    "value,expected",
    [(500, "500"), (128_000, "128K"), (2_000_000, "2.00M")],
)
def test_fmt_context(value, expected):
    assert mc.fmt_context(value) == expected


def test_fmt_age():
    assert mc.fmt_age(None) == "-"
    assert mc.fmt_age(10) == "10d"
    assert mc.fmt_age(90).endswith("mo")
    assert mc.fmt_age(400).endswith("y")


# ---------------------------------------------------------------------------
# parse helpers
# ---------------------------------------------------------------------------


def test_parse_price():
    assert mc.parse_price("0.5") == 0.5
    assert mc.parse_price(None) is None
    assert mc.parse_price("x") is None


def test_parse_iso_datetime_naive_gets_utc():
    dt = mc.parse_iso_datetime("2024-01-01T00:00:00")
    assert dt is not None
    assert dt.tzinfo is not None


def test_parse_iso_datetime_bad():
    assert mc.parse_iso_datetime("not-a-date") is None
    assert mc.parse_iso_datetime(None) is None


# ---------------------------------------------------------------------------
# catalog document
# ---------------------------------------------------------------------------


# Derived from the site validator's contract so the key set lives in exactly
# two places: the producer (build_catalog) and the validator it feeds.
CATALOG_ENTRY_KEYS = set(bsd.CATALOG_ENTRY_KEYS)


def catalog_pool(**overrides):
    """Run the real pipeline over a small fixed pool and hand back everything
    build_catalog needs."""
    args = make_args(min_context=0, **overrides)
    models = [
        make_model(
            id="acme/model-a",
            name="Acme: Model A",
            created=1_700_000_000,
        ),
        make_model(
            id="acme/model-b",
            name="B corp: Model B",
            pricing={"prompt": "0.000002", "completion": "0.000004"},
            context_length=4_000_000,
            created=1_750_000_000,
        ),
        # context_length is negative on purpose: the envelope test pins
        # min_context to 0 (the make_args default), so the "context" drop
        # reason -- which fires only for context < min_context -- needs a
        # negative context to land on acme/small instead of "not ZDR".
        make_model(
            id="acme/small",
            name="Small",
            context_length=-5,
        ),
    ]
    filtered = []
    candidates, dropped = mc.build_candidates(
        models, args, {"acme/model-a": 0.5}, {"acme/model-a", "acme/model-b"}, filtered
    )
    quality_by_id = {"acme/model-b": 68.4}
    mc.compute_scores(candidates, args, quality_by_id)
    return (
        args,
        models,
        candidates,
        dropped,
        filtered,
        {"acme/model-a": 0.5},
        quality_by_id,
    )


def build_doc(**overrides):
    aa_source = overrides.pop("aa_source", "AA API v2")
    args, models, candidates, dropped, filtered, discounts, quality_by_id = (
        catalog_pool(**overrides)
    )
    return mc.build_catalog(
        args, models, candidates, dropped, filtered, discounts, quality_by_id, aa_source
    )


def test_catalog_envelope():
    doc = build_doc()
    assert doc["schema_version"] == 1
    assert doc["tool"] == "model-compare"
    datetime.fromisoformat(doc["generated_at"])  # ISO with offset, raises if not
    p = doc["parameters"]
    assert p["input_share"] == 0.75
    assert p["quality_ref"] == 70.0
    assert p["min_context"] == 0
    assert p["recency_half_life"] == 120.0
    assert p["max_age_days"] == 0.0
    assert p["zdr_required"] is True
    assert p["require_tools"] is False  # make_args default
    assert p["exclude_free"] is False
    assert p["include_batch"] is False
    assert set(p["weights"]) == {"balanced", "price", "quality"}
    assert p["weights"]["balanced"] == mc.PRIORITY_WEIGHTS["balanced"]
    assert doc["sources"] == {
        "openrouter": "ok",
        "aa": {"mode": "api", "matched": 1},
        "zdr": "ok",
        "discounts": "ok",
    }
    assert doc["pool"]["listed"] == 3
    assert doc["pool"]["candidates"] == 2
    assert doc["pool"]["dropped"]["context"] == 1
    assert set(doc["pool"]["dropped"]) == set(mc.CATALOG_DROP_REASONS)
    assert sum(doc["pool"]["dropped"].values()) == 3 - doc["pool"]["candidates"]


def test_catalog_entry_shape():
    doc = build_doc()
    assert {e["id"] for e in doc["models"]} == {"acme/model-a", "acme/model-b"}
    for entry in doc["models"]:
        assert set(entry) == CATALOG_ENTRY_KEYS
        assert set(entry["pricing"]) == {
            "input_per_1m",
            "output_per_1m",
            "blended_per_1m",
        }
        assert set(entry["scores"]) == {"price", "quality", "context", "age", "overall"}
        assert set(entry["scores"]["overall"]) == {"balanced", "price", "quality"}
    a = next(e for e in doc["models"] if e["id"] == "acme/model-a")
    assert a["name"] == "Model A"  # vendor prefix stripped
    assert a["provider"] == "acme"
    assert a["family"] == "model"
    assert a["pricing"] == {
        "input_per_1m": 1.0,
        "output_per_1m": 2.0,
        "blended_per_1m": 1.25,
    }
    assert a["listed_at"] == "2023-11-14"  # utc date of 1_700_000_000
    assert isinstance(a["age_days"], int) and a["age_days"] >= 0
    assert a["tool_calling"] is True
    assert a["zdr"] is True
    assert a["discount"] == 0.5
    assert a["expired"] is False
    assert a["quality"] is None
    assert a["quality_match"] is None  # matched nothing
    b = next(e for e in doc["models"] if e["id"] == "acme/model-b")
    assert b["quality"] == 68.4
    assert b["quality_match"] == "api"
    assert b["discount"] is None


def test_catalog_future_created_age_days_clamped():
    # created in the future must not produce a negative age_days
    created = time.time() + 86400
    args = make_args(min_context=0)
    models = [make_model(id="acme/model-a", created=created)]
    candidates, dropped = mc.build_candidates(models, args, {}, {"acme/model-a"}, [])
    mc.compute_scores(candidates, args, {})
    doc = mc.build_catalog(args, models, candidates, dropped, [], {}, {}, None)
    (entry,) = doc["models"]
    expected = datetime.fromtimestamp(created, tz=timezone.utc).date().isoformat()
    assert entry["listed_at"] == expected
    assert entry["age_days"] == 0


def test_catalog_family_null_without_prefix():
    # no model in the default pool exercises the None path; check directly
    assert mc.model_family("kimi/k2") is None


def test_catalog_overall_covers_all_priorities():
    args, _, candidates, _, _, _, quality_by_id = catalog_pool()
    doc = mc.build_catalog(args, [], candidates, {}, [], {}, quality_by_id, "AA API v2")
    weights = mc.catalog_weights(candidates, quality_by_id)
    for entry in doc["models"]:
        s = entry["scores"]
        for priority, w in weights.items():
            expected = round(
                w.get("quality", 0.0) * s["quality"]
                + w.get("price", 0.0) * s["price"]
                + w.get("context", 0.0) * s["context"]
                + w.get("age", 0.0) * s["age"],
                4,
            )
            assert s["overall"][priority] == expected


def test_catalog_overall_matches_compute_scores_for_current_priority():
    args, _, candidates, _, _, _, quality_by_id = catalog_pool(priority="price")
    doc = mc.build_catalog(args, [], candidates, {}, [], {}, quality_by_id, "AA API v2")
    by_id = {c["id"]: c for c in candidates}
    for entry in doc["models"]:
        assert entry["scores"]["overall"]["price"] == pytest.approx(
            round(by_id[entry["id"]]["score"], 4), abs=2e-4
        )


def test_catalog_scores_in_unit_range_and_no_nan():
    doc = build_doc()
    for entry in doc["models"]:
        for key in ("price", "quality", "context", "age"):
            value = entry["scores"][key]
            assert isinstance(value, (int, float)) and 0.0 <= value <= 1.0
        for value in entry["scores"]["overall"].values():
            assert 0.0 <= value <= 1.0


def test_catalog_filtered_entries_and_sorting():
    doc = build_doc()
    assert doc["filtered"] == [
        {"id": "acme/small", "name": "Small", "reasons": ["context"]}
    ]
    overalls = [e["scores"]["overall"]["balanced"] for e in doc["models"]]
    assert overalls == sorted(overalls, reverse=True)
    ids = [e["id"] for e in doc["models"]]
    assert ids == [
        e["id"]
        for e in sorted(
            doc["models"], key=lambda e: (-e["scores"]["overall"]["balanced"], e["id"])
        )
    ]


def test_catalog_deterministic_modulo_generated_at():
    doc1 = build_doc()
    doc2 = build_doc()
    doc1.pop("generated_at")
    doc2.pop("generated_at")
    assert doc1 == doc2


def test_catalog_document_passes_site_validator():
    # drift between the producer and the site validator must fail the
    # suite here, not the publish
    bsd.validate_catalog(build_doc())  # must not raise


def test_catalog_aa_mode_mapping():
    assert build_doc(aa_source="AA page scrape")["sources"]["aa"]["mode"] == "scrape"
    assert build_doc(aa_source=None)["sources"]["aa"]["mode"] == "none"
    doc = build_doc(aa_source=None)
    assert all(
        e["quality_match"] is None and e["quality"] is None for e in doc["models"]
    )


def test_catalog_unknown_aa_source_fails_loudly():
    # a new AA source must update the mode map -- never publish a document
    # claiming mode "none" while entries carry matched quality
    with pytest.raises(ValueError, match="unknown AA source"):
        build_doc(aa_source="AA carrier pigeon")


def test_catalog_no_zdr_marks_skipped(monkeypatch, capsys):
    monkeypatch.setattr(mc, "fetch_discount_map", lambda args: ({}, False))
    monkeypatch.setattr(mc, "fetch_zdr_set", lambda args: (set(), False))
    monkeypatch.setattr(mc, "fetch_aa_entries", lambda args: ([], None, False))
    models = [
        make_model(id="acme/model-a", created=1_700_000_000),
        make_model(
            id="acme/model-b", pricing={"prompt": "0.000002", "completion": "0.000004"}
        ),
    ]
    args = make_args(min_context=0, no_zdr=True, catalog=True)
    assert mc.run(args, models, False) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["sources"]["zdr"] == "skipped"
    assert doc["parameters"]["zdr_required"] is False
    assert all(e["zdr"] is None for e in doc["models"])


def test_catalog_fails_closed_without_zdr(monkeypatch, capsys):
    monkeypatch.setattr(mc, "fetch_discount_map", lambda args: ({}, False))
    monkeypatch.setattr(mc, "fetch_zdr_set", lambda args: (set(), False))
    monkeypatch.setattr(mc, "fetch_aa_entries", lambda args: ([], None, False))
    models = [make_model(id="acme/model-a")]
    args = make_args(min_context=0, catalog=True)
    assert mc.run(args, models, False) == 1
    captured = capsys.readouterr()
    assert captured.out == ""  # no document
    assert "ZDR" in captured.err


def test_catalog_discounts_unavailable_source():
    args, models, candidates, dropped, filtered, _discounts, quality_by_id = (
        catalog_pool()
    )
    doc = mc.build_catalog(
        args, models, candidates, dropped, filtered, {}, quality_by_id, None
    )
    assert doc["sources"]["discounts"] == "unavailable"


def test_parse_args_rejects_catalog_with_best(capsys):
    with pytest.raises(SystemExit) as exc:
        mc.parse_args(["--catalog", "--best"])
    assert exc.value.code == 2
    assert "cannot be combined" in capsys.readouterr().err


def test_parse_args_rejects_catalog_with_json(capsys):
    with pytest.raises(SystemExit) as exc:
        mc.parse_args(["--catalog", "--json"])
    assert exc.value.code == 2
    assert "cannot be combined" in capsys.readouterr().err


def test_parse_args_accepts_catalog_alone():
    args = mc.parse_args(["--catalog"])
    assert args.catalog is True
