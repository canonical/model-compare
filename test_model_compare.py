"""Unit tests for model_compare scoring, matching, and filtering logic.

These cover pure functions only -- no network access is performed. Run with:

    pytest -q
"""

from __future__ import annotations

import json
import time
from argparse import Namespace

import pytest

import model_compare as mc


def make_args(**overrides):
    """Build an args namespace with the defaults build_candidates/compute_scores expect."""
    base = dict(
        min_context=0,
        priority="balanced",
        top=5,
        best=False,
        json=False,
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


def test_match_quality_exact_by_id():
    entries = [{"key": "acme/model-a", "name": "Model A", "index": 60.0}]
    exact, fuzzy = mc.build_aa_lookup(entries)
    assert (
        mc.match_quality({"id": "acme/model-a", "name": "Acme: Model A"}, exact, fuzzy)
        == 60.0
    )


def test_match_quality_fuzzy_overlap():
    entries = [{"key": "gpt-4o-mini", "name": "GPT 4o mini", "index": 50.0}]
    exact, fuzzy = mc.build_aa_lookup(entries)
    result = mc.match_quality(
        {"id": "openai/gpt-4o-mini", "name": "OpenAI: GPT-4o mini"}, exact, fuzzy
    )
    assert result == 50.0


def test_match_quality_no_match_returns_none():
    entries = [{"key": "acme/model-a", "name": "Model A", "index": 60.0}]
    exact, fuzzy = mc.build_aa_lookup(entries)
    assert (
        mc.match_quality({"id": "other/unrelated-xyz", "name": "Zzz"}, exact, fuzzy)
        is None
    )


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
