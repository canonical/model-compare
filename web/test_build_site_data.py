"""Tests for build_site_data.py -- the data.json builder for the published site."""

import json
from datetime import datetime, timezone

import pytest

import build_site_data as bsd


def make_row(**overrides):
    row = {
        "model": "acme/model-a",
        "opencode_model": "openrouter/acme/model-a",
        "name": "Acme: Model A",
        "score": 0.7,
        "quality_index": 55.0,
        "input_usd_per_m": 1.0,
        "output_usd_per_m": 2.0,
        "blended_usd_per_m": 1.25,
        "discount": None,
        "discount_pct": "--",
        "context_tokens": 2_000_000,
        "age_days": 10.0,
        # data.json v2 row keys (Task 9): print_json always emits them, and
        # the nulls are the pre-endpoint degradation the producer prints.
        "default_precision": None,
        "risk_flag": None,
        "price_spread": None,
        "session_cost_usd": None,
        "quant_note": None,
        # data.json v2 additive provenance (quality source reconciliation):
        # print_json emits the record's label, as_of, drift verdict and OR
        # check pair; the nulls are the no-record degradation it prints.
        "quality_source": "openrouter",
        "quality_as_of": None,
        "quality_drift": None,
        "quality_or_index": None,
        "quality_or_as_of": None,
        # data.json v3 row keys (thinking depth): print_json always emits
        # the main score's effort and the model's effective default effort;
        # null is its unknown-effort degradation.
        "quality_effort": None,
        "default_effort": None,
    }
    row.update(overrides)
    return row


def make_priorities():
    return {
        "balanced": [make_row()],
        "price": [make_row()],
        "quality": [make_row()],
    }


def test_build_data_happy_path():
    data = bsd.build_data("z-ai/glm-5.3-flash", make_priorities())
    assert data["best"] == "z-ai/glm-5.3-flash"
    assert set(data["priorities"]) == {"balanced", "price", "quality"}
    assert data["priorities"]["balanced"][0]["model"] == "acme/model-a"
    assert "generated_at" in data


def test_build_data_carries_discount_pct_through():
    priorities = make_priorities()
    priorities["price"] = [make_row(discount=0.5, discount_pct="50%")]
    data = bsd.build_data("acme/model-a", priorities)
    assert data["priorities"]["price"][0]["discount_pct"] == "50%"
    assert data["priorities"]["balanced"][0]["discount_pct"] == "--"


def test_build_data_rejects_row_without_discount_pct():
    priorities = make_priorities()
    del priorities["quality"][0]["discount_pct"]
    with pytest.raises(ValueError, match="discount_pct"):
        bsd.build_data("acme/model-a", priorities)


def test_build_data_accepts_variant_suffix():
    data = bsd.build_data("nvidia/nemotron-3-ultra-550b-a95b:free", make_priorities())
    assert data["best"].endswith(":free")


def test_build_data_accepts_provider_qualified_best():
    data = bsd.build_data("openrouter/z-ai/glm-5.3-flash", make_priorities())
    assert data["best"] == "openrouter/z-ai/glm-5.3-flash"


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "no-slash",
        "openrouter/two/slashes/here",
        "acme/a b",
        "acme/",
        "/model",
        "openrouter//model",
    ],
)
def test_build_data_rejects_malformed_best(bad):
    with pytest.raises(ValueError):
        bsd.build_data(bad, make_priorities())


def test_build_data_rejects_missing_priority():
    priorities = make_priorities()
    del priorities["quality"]
    with pytest.raises(ValueError):
        bsd.build_data("acme/model-a", priorities)


def test_build_data_rejects_empty_rows():
    priorities = make_priorities()
    priorities["price"] = []
    with pytest.raises(ValueError):
        bsd.build_data("acme/model-a", priorities)


def test_build_data_rejects_row_missing_keys():
    priorities = make_priorities()
    del priorities["balanced"][0]["score"]
    with pytest.raises(ValueError, match="score"):
        bsd.build_data("acme/model-a", priorities)


def _rows(*ids):
    return [make_row(model=model_id) for model_id in ids]


def test_order_rows_reorders_to_the_ranking():
    rows = _rows("acme/b", "acme/a", "acme/c")
    ordered = bsd.order_rows(rows, ["acme/c", "acme/a", "acme/b", "acme/d"])
    assert [row["model"] for row in ordered] == ["acme/c", "acme/a", "acme/b"]
    assert [row["model"] for row in rows] == ["acme/b", "acme/a", "acme/c"]  # pure


def test_order_rows_rejects_drift_naming_missing_and_extra_ids():
    rows = _rows("acme/a", "acme/x")
    with pytest.raises(ValueError) as exc:
        bsd.order_rows(rows, ["acme/a", "acme/b", "acme/x"])
    message = str(exc.value)
    assert "acme/b" in message  # ranked in the top-2 but missing from rows
    assert "acme/x" in message  # in rows but not in the ranking's top-2


def test_order_rows_names_duplicated_ids():
    # [a, b, a] vs top-3 [a, b, c] used to report "missing: [c], extra: []",
    # hiding the real problem: the duplicate.
    rows = _rows("acme/b", "acme/a", "acme/b", "acme/a")
    with pytest.raises(ValueError) as exc:
        bsd.order_rows(rows, ["acme/a", "acme/b", "acme/c", "acme/d"])
    assert "rows contain duplicate model ids: ['acme/a', 'acme/b']" in str(exc.value)


@pytest.mark.parametrize(
    "rows,ranking",
    [
        pytest.param(_rows("acme/a", "acme/a"), ["acme/a", "acme/b"], id="dup-row"),
        pytest.param(
            _rows("acme/a", "acme/b", "acme/a"), ["acme/a", "acme/b"], id="len"
        ),
        pytest.param([make_row(model=["acme/a"])], ["acme/a"], id="unhashable"),
    ],
)
def test_order_rows_rejects_mismatched_rows(rows, ranking):
    with pytest.raises(ValueError):
        bsd.order_rows(rows, ranking)


def _ranked_catalog():
    doc = make_catalog()
    doc["rankings"] = {
        "balanced": ["acme/c", "acme/a", "acme/b"],
        "price": ["acme/b", "acme/c", "acme/a"],
        "quality": ["acme/a", "acme/b", "acme/c"],
    }
    return doc


def test_build_data_orders_rows_by_catalog_rankings():
    priorities = {p: _rows("acme/a", "acme/b", "acme/c") for p in bsd.PRIORITIES}
    catalog = _ranked_catalog()
    data = bsd.build_data("acme/a", priorities, catalog=catalog)
    for priority in bsd.PRIORITIES:
        table = [row["model"] for row in data["priorities"][priority]]
        assert table == catalog["rankings"][priority], priority


def test_build_data_without_catalog_keeps_rows_verbatim():
    priorities = {p: _rows("acme/b", "acme/a") for p in bsd.PRIORITIES}
    data = bsd.build_data("acme/a", priorities)
    for priority in bsd.PRIORITIES:
        assert [r["model"] for r in data["priorities"][priority]] == [
            "acme/b",
            "acme/a",
        ]


def test_build_data_rejects_rows_that_drift_from_catalog_rankings():
    priorities = {p: _rows("acme/a", "acme/b") for p in bsd.PRIORITIES}
    with pytest.raises(ValueError, match="acme/c"):
        # balanced ranks acme/c in its top 2; the rows do not have it
        bsd.build_data("acme/a", priorities, catalog=_ranked_catalog())


def _three_model_catalog():
    doc = make_catalog()
    doc["models"] = [
        make_catalog_entry(id=model_id) for model_id in ("acme/a", "acme/b", "acme/c")
    ]
    doc["pool"].update(listed=4, candidates=3)
    # All three entries carry the OR label: keep the disjoint buckets honest.
    doc["sources"]["aa"]["matched_openrouter"] = 3
    doc["rankings"] = _ranked_catalog()["rankings"]
    bsd.validate_catalog(doc)
    return doc


def test_main_orders_rows_from_catalog_file(tmp_path):
    raw = tmp_path / "catalog-raw.json"
    raw.write_text(json.dumps(_three_model_catalog()))
    argv, out = catalog_argv(tmp_path, raw)
    rows = tmp_path / "rows.json"
    rows.write_text(json.dumps(_rows("acme/a", "acme/b", "acme/c")))
    assert bsd.main(argv) == 0
    data = json.loads(out.read_text())
    for priority in bsd.PRIORITIES:
        table = [row["model"] for row in data["priorities"][priority]]
        assert table == _ranked_catalog()["rankings"][priority], priority


def test_main_fails_when_rows_drift_from_catalog_file(tmp_path, capsys):
    raw = tmp_path / "catalog-raw.json"
    raw.write_text(json.dumps(_three_model_catalog()))
    argv, out = catalog_argv(tmp_path, raw)
    rows = tmp_path / "rows.json"
    rows.write_text(json.dumps(_rows("acme/a", "acme/b")))
    assert bsd.main(argv) == 1
    assert "acme/c" in capsys.readouterr().err
    assert not out.exists()


def test_main_end_to_end(tmp_path):
    best_file = tmp_path / "best.txt"
    best_file.write_text("openrouter/z-ai/glm-5.3-flash\n")
    files = {}
    for name in ("balanced", "price", "quality"):
        f = tmp_path / f"{name}.json"
        f.write_text(json.dumps([make_row(model=f"acme/{name}")]))
        files[name] = f
    out = tmp_path / "data.json"
    argv = ["--best-file", str(best_file), "--output", str(out)]
    for name, f in files.items():
        argv += ["--priority", f"{name}={f}"]
    assert bsd.main(argv) == 0
    data = json.loads(out.read_text())
    assert data["best"] == "openrouter/z-ai/glm-5.3-flash"
    assert data["priorities"]["price"][0]["model"] == "acme/price"


def test_main_fails_loudly_on_bad_best(tmp_path, capsys):
    best_file = tmp_path / "best.txt"
    best_file.write_text("garbage")
    f = tmp_path / "balanced.json"
    f.write_text(json.dumps([make_row()]))
    argv = [
        "--best-file",
        str(best_file),
        "--output",
        str(tmp_path / "data.json"),
        "--priority",
        f"balanced={f}",
        "--priority",
        f"price={f}",
        "--priority",
        f"quality={f}",
    ]
    assert bsd.main(argv) == 1
    assert "error:" in capsys.readouterr().err
    assert not (tmp_path / "data.json").exists()


def test_main_rejects_duplicate_priority(tmp_path, capsys):
    best_file = tmp_path / "best.txt"
    best_file.write_text("acme/model-a\n")
    f = tmp_path / "rows.json"
    f.write_text(json.dumps([make_row()]))
    out = tmp_path / "data.json"
    argv = ["--best-file", str(best_file), "--output", str(out)]
    for name in ("balanced", "price", "quality", "balanced"):
        argv += ["--priority", f"{name}={f}"]
    assert bsd.main(argv) == 1
    err = capsys.readouterr().err
    assert "error:" in err
    assert "duplicate" in err
    assert "balanced" in err
    assert not out.exists()


# ---------------------------------------------------------------------------
# catalog.json publication
# ---------------------------------------------------------------------------


def with_tier_keys(pricing):
    """Expand a plain 3-scalar pricing dict into the v2 shape."""
    return {
        **pricing,
        "base": {
            "input_per_1m": pricing["input_per_1m"],
            "output_per_1m": pricing["output_per_1m"],
            "blended_per_1m": pricing["blended_per_1m"],
        },
        "tiers": [],
        "schedule": None,
    }


def make_catalog_entry(**overrides):
    entry = {
        "id": "acme/model-a",
        "name": "Model A",
        "provider": "acme",
        "family": "model",
        "pricing": {
            "input_per_1m": 1.0,
            "output_per_1m": 2.0,
            "blended_per_1m": 1.25,
            "base": {
                "input_per_1m": 1.0,
                "output_per_1m": 2.0,
                "blended_per_1m": 1.25,
            },
            "tiers": [],
            "schedule": None,
        },
        "context": 2_000_000,
        "listed_at": "2026-01-15",
        "age_days": 10,
        "tool_calling": True,
        "zdr": True,
        "discount": None,
        "expired": False,
        "quality": 55.0,
        "aa": {
            "intelligence_index": 55.0,
            "coding_index": 61.0,
            "agentic_index": 48.5,
        },
        "quality_match": "openrouter",
        # Schema v4 provenance: the record's own as_of and drift verdict,
        # null for this OR-matched row (no second source to disagree with).
        "quality_as_of": None,
        "quality_drift": None,
        # Schema v5 (thinking depth): the effort the main quality was
        # measured at, and the reasoning metadata plus measured depths. The
        # producer emits the block on every entry; a model with no metadata
        # and no AA depth rows degrades to this empty shape.
        "quality_effort": None,
        "reasoning": {
            "default_effort": None,
            "supported_efforts": [],
            "default_enabled": None,
            "mandatory": None,
            "supports_max_tokens": None,
            "depths": [],
        },
        "scores": {
            "price": 0.9,
            "quality": 0.78,
            "context": 1.0,
            "age": 0.94,
            "overall": {"balanced": 0.85, "price": 0.82, "quality": 0.86},
        },
        # Catalog v3 additive blocks (Task 9): null when the candidate
        # skipped the endpoint stage -- the producer's degradation shape.
        "endpoints": None,
        "default_precision": None,
        "advertised_pricing": None,
        "annotations": None,
    }
    entry.update(overrides)
    return entry


def make_catalog():
    return {
        "schema_version": 6,
        "tool": "model-compare",
        "generated_at": "2026-09-02T09:15:00+00:00",
        "parameters": {
            "input_share": 0.75,
            "quality_ref": 70.0,
            "min_context": 1000000,
            "recency_half_life": 120.0,
            "max_age_days": 0.0,
            "zdr_required": True,
            "require_tools": True,
            "exclude_free": False,
            "include_batch": False,
            "weights": {
                "balanced": {"quality": 0.4, "price": 0.4, "context": 0.1, "age": 0.1},
                "price": {"quality": 0.2, "price": 0.6, "context": 0.1, "age": 0.1},
                "quality": {"quality": 0.6, "price": 0.2, "context": 0.1, "age": 0.1},
            },
        },
        "sources": {
            "openrouter": "ok",
            # Catalog v6: the keyed CI producer's normal chain, its three
            # disjoint label buckets (the one matched entry is OR-labeled)
            # and the provenance meta (no hydration, AA index v4.3).
            "aa": {
                "mode": "aa",
                "matched_aa": 0,
                "matched_catalog": 0,
                "matched_openrouter": 1,
                "catalog_generated_at": None,
                "intelligence_index_version": 4.3,
            },
            "zdr": "ok",
            "discounts": "ok",
        },
        "pool": {"listed": 2, "candidates": 1, "dropped": {"context": 1}},
        "models": [make_catalog_entry()],
        "rankings": {p: ["acme/model-a"] for p in ("balanced", "price", "quality")},
        "schedules": [],
        "filtered": [{"id": "acme/small", "name": "Small", "reasons": ["context"]}],
    }


def rank_models(doc):
    """Give doc a valid rankings object: every priority in models order."""
    ids = [e["id"] for e in doc["models"]]
    doc["rankings"] = {p: list(ids) for p in ("balanced", "price", "quality")}
    return doc


def catalog_argv(tmp_path, catalog_file=None):
    best_file = tmp_path / "best.txt"
    best_file.write_text("openrouter/acme/model-a\n")
    rows = tmp_path / "rows.json"
    rows.write_text(json.dumps([make_row()]))
    out = tmp_path / "data.json"
    argv = ["--best-file", str(best_file), "--output", str(out)]
    for name in ("balanced", "price", "quality"):
        argv += ["--priority", f"{name}={rows}"]
    if catalog_file is not None:
        argv += ["--catalog-file", str(catalog_file)]
    return argv, out


def test_validate_catalog_happy_path():
    bsd.validate_catalog(make_catalog())  # must not raise


def test_build_site_data_accepts_new_row_keys():
    # The five provenance keys are ROW_KEYS members: build_data demands
    # them on every row (the fail-loudly missing-key contract) and passes
    # them through to data.json verbatim -- the schema stays 2, additive.
    for key in (
        "quality_source",
        "quality_as_of",
        "quality_drift",
        "quality_or_index",
        "quality_or_as_of",
    ):
        assert key in bsd.ROW_KEYS
    priorities = make_priorities()
    del priorities["balanced"][0]["quality_source"]
    with pytest.raises(ValueError, match="quality_source"):
        bsd.build_data("acme/model-a", priorities)
    data = bsd.build_data("acme/model-a", make_priorities())
    row = data["priorities"]["balanced"][0]
    assert row["quality_source"] == "openrouter"
    assert row["quality_as_of"] is None
    assert row["quality_drift"] is None
    assert row["quality_or_index"] is None
    assert row["quality_or_as_of"] is None


def test_build_site_data_rejects_v3_sources_aa_shape():
    # The v3 fallback/matched block predates the disjoint buckets: no v4
    # producer can emit it, so the validator must reject it fail-loudly
    # instead of silently validating a shape the site cannot trust.
    doc = make_catalog()
    doc["sources"]["aa"] = {
        "mode": "openrouter",
        "fallback": "aa",
        "matched": 1,
        "matched_openrouter": 1,
    }
    with pytest.raises(ValueError):
        bsd.validate_catalog(doc)


def test_build_site_data_enforces_disjoint_bucket_invariant():
    # The three counts are disjoint label buckets (strict rank counts each
    # model once): their sum must equal the number of entries with a
    # non-null quality_match. An inflated bucket would let the document
    # claim more matched models than it lists.
    doc = make_catalog()
    doc["sources"]["aa"]["matched_aa"] = 5
    with pytest.raises(ValueError, match="disjoint"):
        bsd.validate_catalog(doc)


@pytest.mark.parametrize(
    "mode,counts",
    [
        ("aa", {"matched_aa": 1, "matched_catalog": 0, "matched_openrouter": 0}),
        ("catalog", {"matched_aa": 0, "matched_catalog": 1, "matched_openrouter": 0}),
        ("none", {"matched_aa": 0, "matched_catalog": 0, "matched_openrouter": 0}),
    ],
)
def test_validate_catalog_accepts_aa_modes(mode, counts):
    doc = make_catalog()
    doc["sources"]["aa"].update(mode=mode, **counts)
    if mode == "none":
        doc["models"][0]["quality_match"] = None
        doc["models"][0]["quality_as_of"] = None
        doc["models"][0]["quality_drift"] = None
    else:
        # The bucket must agree with the entries' own labels: an aa-matched
        # entry carries the aa label, a catalog-hydrated one the catalog
        # label, each with the record's own as_of and drift verdict.
        doc["models"][0]["quality_match"] = mode
        doc["models"][0]["quality_as_of"] = "2026-10-23T13:15:07Z"
        doc["models"][0]["quality_drift"] = False
    bsd.validate_catalog(doc)  # must not raise


@pytest.mark.parametrize(
    "mutate",
    [
        lambda aa: aa.pop("matched_aa"),
        lambda aa: aa.update(matched_aa="three"),
        lambda aa: aa.update(matched_aa=True),
        lambda aa: aa.update(catalog_generated_at="Sept 2"),
        lambda aa: aa.update(intelligence_index_version="v4"),
        lambda aa: aa.update(intelligence_index_version=True),
    ],
)
def test_validate_catalog_rejects_bad_aa_sources_block(mutate):
    doc = make_catalog()
    mutate(doc["sources"]["aa"])
    with pytest.raises(ValueError, match="sources.aa"):
        bsd.validate_catalog(doc)


def test_validate_catalog_accepts_null_aa_fields_and_all_provenances():
    doc = make_catalog()
    doc["models"][0]["aa"] = {
        "intelligence_index": None,
        "coding_index": None,
        "agentic_index": None,
    }
    doc["models"][0]["quality_match"] = None
    doc["models"][0]["quality_as_of"] = None
    doc["models"][0]["quality_drift"] = None
    doc["sources"]["aa"] = {
        "mode": "none",
        "matched_aa": 0,
        "matched_catalog": 0,
        "matched_openrouter": 0,
        "catalog_generated_at": None,
        "intelligence_index_version": None,
    }
    bsd.validate_catalog(doc)  # must not raise
    doc["models"][0]["quality_match"] = "aa"
    doc["models"][0]["quality_as_of"] = "2026-10-23T13:15:07Z"
    doc["models"][0]["quality_drift"] = False
    doc["sources"]["aa"].update(
        mode="aa",
        matched_aa=1,
        catalog_generated_at="2026-10-01T00:00:00Z",
    )
    bsd.validate_catalog(doc)  # must not raise


@pytest.mark.parametrize(
    "mutate, field",
    [
        (lambda doc: doc["models"][0].update(quality_as_of=13.15), "quality_as_of"),
        (lambda doc: doc["models"][0].update(quality_as_of=True), "quality_as_of"),
        (lambda doc: doc["models"][0].update(quality_drift="no"), "quality_drift"),
        (lambda doc: doc["models"][0].update(quality_drift=1), "quality_drift"),
    ],
)
def test_validate_catalog_rejects_bad_quality_provenance(mutate, field):
    # Schema v4 provenance is presence-checked via CATALOG_ENTRY_KEYS; the
    # value types are pinned here so a malformed document fails loudly:
    # quality_as_of is a string stamp or null, quality_drift a boolean
    # verdict or null.
    doc = make_catalog()
    mutate(doc)
    with pytest.raises(ValueError, match=field):
        bsd.validate_catalog(doc)


@pytest.mark.parametrize("bad", [-50.0, 200.0, 1e308, float("inf"), "45.1"])
def test_validate_catalog_rejects_out_of_range_models_quality(bad):
    # models[].quality is an AA intelligence index and carries the same
    # 0..100 range the schedules rows enforce -- the validator used to
    # range-check schedules[].quality only, so a catalog with
    # models[0].quality = -50.0 passed.
    doc = make_catalog()
    doc["models"][0]["quality"] = bad
    with pytest.raises(ValueError, match="quality"):
        bsd.validate_catalog(doc)


@pytest.mark.parametrize(
    "mutate, field",
    [
        (lambda doc: doc["sources"]["aa"].update(mode="scrape"), "sources.aa.mode"),
        (lambda doc: doc["models"][0].update(quality_match="scrape"), "quality_match"),
    ],
)
def test_validate_catalog_rejects_removed_scrape_values(mutate, field):
    # the AA page scrape was removed: a catalog carrying "scrape" anywhere
    # could not have been produced by this tool, so it fails closed
    doc = make_catalog()
    mutate(doc)
    with pytest.raises(ValueError, match=field):
        bsd.validate_catalog(doc)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda doc: doc.update(schema_version=99),
        lambda doc: doc.update(tool="other-tool"),
        lambda doc: doc.pop("parameters"),
        lambda doc: doc["pool"].update(candidates=5),
        lambda doc: doc["pool"].update(dropped={"context": 9}),
        lambda doc: doc["pool"].update(dropped={"context": "one"}),
        lambda doc: doc["pool"].update(listed="two"),
        lambda doc: doc["models"][0].pop("family"),
        lambda doc: doc["models"][0].pop("scores"),
        lambda doc: doc["models"][0]["scores"].update(price=1.5),
        lambda doc: doc["models"][0]["scores"].update(overall={"balanced": 0.5}),
        lambda doc: doc["models"][0].update(zdr=False),
        lambda doc: doc["filtered"].append({"id": "x/y", "name": "X", "reasons": []}),
        lambda doc: doc["models"].append("not-a-dict"),
        # a consumer deduping on content must never see the same id twice
        lambda doc: doc["models"].append(dict(doc["models"][0])),
        # a model cannot be both a ranked candidate and filtered out
        lambda doc: doc["filtered"].append(
            {"id": "acme/model-a", "name": "A", "reasons": ["context"]}
        ),
        # scores.overall is documented as reproducible from parameters.weights
        lambda doc: doc["parameters"].pop("weights"),
        lambda doc: doc["parameters"]["weights"].pop("price"),
        lambda doc: doc["parameters"]["weights"]["balanced"].update(price=0.9),
        lambda doc: doc["parameters"]["weights"]["balanced"].update(age=0),
        lambda doc: doc["models"][0].pop("aa"),
        lambda doc: doc["models"][0]["aa"].pop("coding_index"),
        lambda doc: doc["models"][0]["aa"].update(intelligence_index=101),
        lambda doc: doc["models"][0]["aa"].update(intelligence_index=float("nan")),
        lambda doc: doc["models"][0]["aa"].update(intelligence_index=True),
        lambda doc: doc["models"][0].update(quality_match="psychic"),
        # Schema v5: the effort label and the reasoning block are
        # presence-required and fail closed on a malformed value.
        lambda doc: doc["models"][0].update(quality_effort=7),
        lambda doc: doc["models"][0].update(quality_effort=""),
        lambda doc: doc["models"][0].update(reasoning=None),
        lambda doc: doc["sources"]["aa"].update(mode="psychic"),
        lambda doc: doc["sources"]["aa"].update(matched_openrouter=5),
        lambda doc: doc["sources"]["aa"].update(matched_aa=-1),
        lambda doc: doc["sources"]["aa"].update(matched_openrouter=True),
        # pricing values must be usable in arithmetic downstream: the diff
        # builder and history snapshots consume them as numbers
        lambda doc: doc["models"][0]["pricing"].update(blended_per_1m="1.0"),
        lambda doc: doc["models"][0]["pricing"].update(blended_per_1m=-1.0),
        lambda doc: doc["models"][0]["pricing"].update(input_per_1m=True),
        lambda doc: doc["models"][0]["pricing"].pop("blended_per_1m"),
        lambda doc: doc["models"][0].update(discount="0.5"),
        lambda doc: doc["models"][0].update(context="2m"),
        # build_snapshot/merge_history slice and parse generated_at, so a
        # malformed stamp must fail validation, not crash as a traceback
        lambda doc: doc.pop("generated_at"),
        lambda doc: doc.update(generated_at=None),
        lambda doc: doc.update(generated_at=""),
        lambda doc: doc.update(generated_at="not-a-date"),
    ],
)
def test_validate_catalog_rejects(mutate):
    doc = make_catalog()
    mutate(doc)
    with pytest.raises(ValueError):
        bsd.validate_catalog(doc)


def _add_second_model(doc):
    doc["models"].append(make_catalog_entry(id="acme/model-b"))
    doc["pool"].update(listed=3, candidates=2)
    # Keep the disjoint-bucket invariant honest: one more OR-labeled entry,
    # one more openrouter count.
    doc["sources"]["aa"]["matched_openrouter"] += 1
    return doc


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda doc: doc.pop("rankings"), id="missing"),
        pytest.param(lambda doc: doc.update(rankings=["acme/model-a"]), id="not-dict"),
        pytest.param(lambda doc: doc["rankings"].pop("price"), id="missing-key"),
        pytest.param(
            lambda doc: doc["rankings"].update(extra=["acme/model-a"]), id="extra-key"
        ),
        pytest.param(
            lambda doc: doc["rankings"].update(balanced="acme/model-a"), id="not-list"
        ),
        pytest.param(
            lambda doc: doc["rankings"].update(
                balanced=["acme/model-a", "acme/model-a"]
            ),
            id="duplicate",
        ),
        pytest.param(
            lambda doc: doc["rankings"].update(balanced=["acme/unknown"]), id="unknown"
        ),
        pytest.param(
            lambda doc: _add_second_model(doc)["rankings"].update(
                balanced=["acme/model-a"]
            ),
            id="wrong-length",
        ),
        pytest.param(lambda doc: doc["rankings"].update(balanced=[7]), id="non-string"),
        pytest.param(lambda doc: doc["rankings"].update(balanced=[""]), id="empty-id"),
    ],
)
def test_validate_catalog_rejects_bad_rankings(mutate):
    doc = make_catalog()
    mutate(doc)
    with pytest.raises(ValueError, match="rankings"):
        bsd.validate_catalog(doc)


def test_validate_catalog_accepts_any_permutation_per_priority():
    doc = _add_second_model(make_catalog())
    doc["rankings"] = {
        "balanced": ["acme/model-a", "acme/model-b"],
        "price": ["acme/model-b", "acme/model-a"],
        "quality": ["acme/model-b", "acme/model-a"],
    }
    bsd.validate_catalog(doc)  # must not raise


def test_main_writes_catalog_next_to_data_json(tmp_path):
    raw = tmp_path / "catalog-raw.json"
    raw.write_text(json.dumps(make_catalog()))
    argv, out = catalog_argv(tmp_path, raw)
    assert bsd.main(argv) == 0
    catalog_path = out.parent / "catalog.json"
    written = json.loads(catalog_path.read_text())
    assert written["schema_version"] == 6
    assert written["models"][0]["id"] == "acme/model-a"
    assert out.exists()  # data.json still written


def test_main_rejects_malformed_catalog_before_writing_anything(tmp_path, capsys):
    doc = make_catalog()
    doc["models"][0].pop("zdr")
    raw = tmp_path / "catalog-raw.json"
    raw.write_text(json.dumps(doc))
    argv, out = catalog_argv(tmp_path, raw)
    assert bsd.main(argv) == 1
    err = capsys.readouterr().err
    assert "error:" in err and "catalog" in err
    assert not out.exists()  # validation runs before the data.json write
    assert not (out.parent / "catalog.json").exists()


def test_main_rejects_unparseable_catalog(tmp_path, capsys):
    raw = tmp_path / "catalog-raw.json"
    raw.write_text("{not json")
    argv, out = catalog_argv(tmp_path, raw)
    assert bsd.main(argv) == 1
    assert "error:" in capsys.readouterr().err
    assert not out.exists()


def test_main_without_catalog_file_writes_no_catalog(tmp_path):
    argv, out = catalog_argv(tmp_path)
    assert bsd.main(argv) == 0
    assert out.exists()
    assert not (out.parent / "catalog.json").exists()


# ---------------------------------------------------------------------------
# history.json publication
# ---------------------------------------------------------------------------


def make_history_snapshot(date, generated_at, **overrides):
    snap = {
        "generated_at": generated_at,
        "pool_ids": ["acme/model-a"],
        "tabs": {
            "balanced": [
                {"id": "acme/model-a", "rank": 1, "quality": 55.0, "blended": 1.25}
            ],
            "price": [
                {"id": "acme/model-a", "rank": 1, "quality": 55.0, "blended": 1.25}
            ],
            "quality": [
                {"id": "acme/model-a", "rank": 1, "quality": 55.0, "blended": 1.25}
            ],
        },
        "aa": {"acme/model-a": 55.0},
        "prices": {"acme/model-a": [1.0, 2.0, 1.25, None]},
    }
    snap.update(overrides)
    return snap


def make_history(**overrides):
    doc = {
        "schema_version": 5,
        "updated_at": "2026-08-26T09:15:00+00:00",
        "snapshots": {
            "2026-08-26": make_history_snapshot(
                "2026-08-26", "2026-08-26T09:15:00+00:00"
            )
        },
    }
    doc.update(overrides)
    return doc


def test_build_snapshot_projects_catalog():
    snap = bsd.build_snapshot(
        make_catalog(),
    )
    assert snap["generated_at"] == "2026-09-02T09:15:00+00:00"
    assert snap["pool_ids"] == ["acme/model-a", "acme/small"]
    assert snap["tabs"]["balanced"][0] == {
        "id": "acme/model-a",
        "rank": 1,
        "quality": 55.0,
        "blended": 1.25,
    }
    assert snap["aa"] == {"acme/model-a": 55.0}
    assert snap["prices"]["acme/model-a"] == [1.0, 2.0, 1.25, None]


def test_build_snapshot_ranks_per_priority_top20():
    doc = make_catalog()
    for i in range(12):
        doc["models"].append(
            make_catalog_entry(
                id=f"acme/m{i}",
                quality=60.0 + i,
                scores={
                    "price": 0.5,
                    "quality": 0.8,
                    "context": 0.5,
                    "age": 0.5,
                    "overall": {
                        "balanced": round(0.5 - i * 0.01, 4),
                        "price": round(0.9 - (i % 3) * 0.1, 4),
                        "quality": round(0.3 + i * 0.01, 4),
                    },
                },
            )
        )
    rank_models(doc)
    ids = [e["id"] for e in doc["models"]]
    doc["rankings"]["price"] = list(reversed(ids))
    doc["rankings"]["quality"] = ids[1:] + ids[:1]
    snap = bsd.build_snapshot(doc)
    for priority in ("balanced", "price", "quality"):
        assert len(snap["tabs"][priority]) == 13
        assert [row["rank"] for row in snap["tabs"][priority]] == list(range(1, 14))
        assert [row["id"] for row in snap["tabs"][priority]] == (
            doc["rankings"][priority][:13]
        )


def test_build_snapshot_projects_rankings_not_a_sort():
    # Authority fence: tabs follow catalog rankings even when a naive sort of
    # models (on scores.overall or any tiebreak) would order them otherwise.
    def entry(model_id, overall, quality, blended):
        return make_catalog_entry(
            id=model_id,
            quality=quality,
            pricing=with_tier_keys(
                {
                    "input_per_1m": 1.0,
                    "output_per_1m": 2.0,
                    "blended_per_1m": blended,
                }
            ),
            scores={
                "price": 0.5,
                "quality": 0.5,
                "context": 0.5,
                "age": 0.5,
                "overall": {"balanced": overall, "price": overall, "quality": overall},
            },
        )

    doc = make_catalog()
    doc["models"] = [
        entry("acme/a-top", 0.9, 70.0, 1.0),
        entry("acme/b-mid", 0.5, 50.0, 2.0),
        entry("acme/c-low", 0.1, 30.0, 3.0),
    ]
    doc["pool"].update(listed=4, candidates=3)
    # All three entries carry the OR label, so the disjoint buckets must
    # count them (the base fixture counts only its single entry).
    doc["sources"]["aa"]["matched_openrouter"] = 3
    doc["rankings"] = {
        "balanced": ["acme/c-low", "acme/a-top", "acme/b-mid"],
        "price": ["acme/b-mid", "acme/c-low", "acme/a-top"],
        "quality": ["acme/c-low", "acme/b-mid", "acme/a-top"],
    }
    bsd.validate_catalog(doc)
    snap = bsd.build_snapshot(doc)
    by_id = {e["id"]: e for e in doc["models"]}
    for priority in ("balanced", "price", "quality"):
        rows = snap["tabs"][priority]
        assert [row["id"] for row in rows] == doc["rankings"][priority]
        assert [row["rank"] for row in rows] == [1, 2, 3]
        for row in rows:  # attributes merged from the models entry
            assert row["quality"] == by_id[row["id"]]["quality"]
            assert row["blended"] == by_id[row["id"]]["pricing"]["blended_per_1m"]


def test_build_snapshot_dedupes_pool_ids():
    doc = make_catalog()
    doc["filtered"].append(
        {"id": "acme/model-a", "name": "Model A", "reasons": ["context"]}
    )
    snap = bsd.build_snapshot(doc)
    assert snap["pool_ids"] == ["acme/model-a", "acme/small"]


def test_merge_history_upserts_and_prunes():
    prev = make_history()
    snaps = prev["snapshots"]
    for d in range(1, 12):
        snaps[f"2026-08-{d:02d}"] = make_history_snapshot(
            f"2026-08-{d:02d}", f"2026-08-{d:02d}T09:15:00+00:00"
        )
    today = "2026-09-02"
    snap = make_history_snapshot(today, "2026-09-02T09:15:00+00:00")
    merged = bsd.merge_history(prev, snap)
    assert list(merged["snapshots"]) == sorted(merged["snapshots"])[-10:]
    assert len(merged["snapshots"]) == 10
    assert merged["snapshots"][today] == snap
    assert merged["updated_at"] == "2026-09-02T09:15:00+00:00"
    assert merged["schema_version"] == 5


def test_merge_history_same_day_last_write_wins():
    prev = make_history()
    snap_old = make_history_snapshot("2026-09-02", "2026-09-02T03:15:00+00:00")
    merged1 = bsd.merge_history(prev, snap_old)
    snap_new = make_history_snapshot("2026-09-02", "2026-09-02T09:15:00+00:00")
    merged2 = bsd.merge_history(merged1, snap_new)
    assert merged2["snapshots"]["2026-09-02"] == snap_new
    assert merged2["updated_at"] == "2026-09-02T09:15:00+00:00"
    assert len(merged2["snapshots"]) == 2


def test_merge_history_malformed_prev_starts_fresh():
    merged = bsd.merge_history(
        {"snapshots": "garbage"},
        make_history_snapshot("2026-09-02", "2026-09-02T09:15:00+00:00"),
    )
    assert list(merged["snapshots"]) == ["2026-09-02"]
    assert bsd.merge_history(
        None, make_history_snapshot("2026-09-02", "2026-09-02T09:15:00+00:00")
    )["snapshots"]["2026-09-02"]["pool_ids"] == ["acme/model-a"]


def test_merge_history_drops_prev_snapshot_missing_generated_at():
    prev = make_history()
    garbage = make_history_snapshot("2026-08-25", "2026-08-25T09:15:00+00:00")
    del garbage["generated_at"]
    prev["snapshots"]["2026-08-25"] = garbage
    merged = bsd.merge_history(
        prev, make_history_snapshot("2026-09-02", "2026-09-02T09:15:00+00:00")
    )
    assert "2026-08-25" not in merged["snapshots"]
    assert "2026-08-26" in merged["snapshots"]
    assert merged["snapshots"]["2026-09-02"]["pool_ids"] == ["acme/model-a"]


def test_merge_history_drops_prev_snapshot_with_non_dict_tabs():
    prev = make_history()
    prev["snapshots"]["2026-08-25"] = make_history_snapshot(
        "2026-08-25", "2026-08-25T09:15:00+00:00", tabs="garbage"
    )
    merged = bsd.merge_history(
        prev, make_history_snapshot("2026-09-02", "2026-09-02T09:15:00+00:00")
    )
    assert "2026-08-25" not in merged["snapshots"]
    assert "2026-08-26" in merged["snapshots"]
    assert merged["snapshots"]["2026-09-02"]["pool_ids"] == ["acme/model-a"]


def test_merge_history_drops_prev_snapshot_with_corrupt_ranks():
    prev = make_history()
    prev["snapshots"]["2026-08-25"] = make_history_snapshot(
        "2026-08-25", "2026-08-25T09:15:00+00:00"
    )
    prev["snapshots"]["2026-08-25"]["tabs"]["balanced"][0]["rank"] = 5
    merged = bsd.merge_history(
        prev, make_history_snapshot("2026-09-02", "2026-09-02T09:15:00+00:00")
    )
    assert "2026-08-25" not in merged["snapshots"]
    assert "2026-08-26" in merged["snapshots"]
    assert merged["snapshots"]["2026-09-02"]["pool_ids"] == ["acme/model-a"]


def test_merge_history_rejects_unknown_schema_version():
    prev = make_history()
    prev["schema_version"] = 6
    merged = bsd.merge_history(
        prev, make_history_snapshot("2026-09-02", "2026-09-02T09:15:00+00:00")
    )
    assert list(merged["snapshots"]) == ["2026-09-02"]
    assert merged["schema_version"] == 5


def test_merge_history_tolerates_prev_without_schema_version():
    prev = make_history()
    del prev["schema_version"]
    merged = bsd.merge_history(
        prev, make_history_snapshot("2026-09-02", "2026-09-02T09:15:00+00:00")
    )
    assert "2026-08-26" in merged["snapshots"]
    assert "2026-09-02" in merged["snapshots"]


def test_merge_history_drops_future_dated_snapshot():
    prev = make_history()
    prev["snapshots"]["2027-01-01"] = make_history_snapshot(
        "2027-01-01", "2027-01-01T09:15:00+00:00"
    )
    merged = bsd.merge_history(
        prev, make_history_snapshot("2026-09-02", "2026-09-02T09:15:00+00:00")
    )
    assert "2027-01-01" not in merged["snapshots"]
    assert merged["updated_at"] == "2026-09-02T09:15:00+00:00"


def test_merge_history_updated_at_is_the_written_snapshot():
    # A previously skewed run can leave a snapshot dated today+1 in the live
    # history; the horizon deliberately retains it. updated_at must still be
    # this run's stamp, or publish's stamp gate (history.updated_at ==
    # catalog.generated_at) wedges every deploy until the date catches up.
    prev = make_history()
    prev["snapshots"]["2026-09-03"] = make_history_snapshot(
        "2026-09-03", "2026-09-03T01:00:00+00:00"
    )
    snap = make_history_snapshot("2026-09-02", "2026-09-02T09:15:00+00:00")
    merged = bsd.merge_history(prev, snap)
    assert "2026-09-03" in merged["snapshots"]  # horizon tolerance unchanged
    assert max(merged["snapshots"]) == "2026-09-03"
    assert merged["snapshots"]["2026-09-02"] == snap
    assert merged["updated_at"] == "2026-09-02T09:15:00+00:00"


def test_merge_history_drops_basic_format_date_key():
    prev = make_history()
    prev["snapshots"]["20260825"] = make_history_snapshot(
        "2026-08-25", "2026-08-25T09:15:00+00:00"
    )
    merged = bsd.merge_history(
        prev, make_history_snapshot("2026-09-02", "2026-09-02T09:15:00+00:00")
    )
    assert "20260825" not in merged["snapshots"]
    assert "2026-08-26" in merged["snapshots"]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda snap: snap["tabs"]["balanced"].__setitem__(0, {"rank": 1}),
        lambda snap: snap["tabs"]["balanced"].__setitem__(0, {"id": 5, "rank": 1}),
        lambda snap: snap["prices"].__setitem__("acme/model-a", [1.0, 2.0]),
        lambda snap: snap["prices"].__setitem__("acme/model-a", "garbage"),
        lambda snap: snap["prices"].__setitem__(
            "acme/model-a", [1.0, 2.0, "1.25", None]
        ),
        lambda snap: snap["aa"].__setitem__("acme/model-a", "55"),
        lambda snap: snap.update(aa="garbage"),
        lambda snap: snap.update(pool_ids=[["unhashable"]]),
        lambda snap: snap.update(pool_ids="garbage"),
    ],
)
def test_has_snapshot_shape_rejects_malformed_fields(mutate):
    snap = make_history_snapshot("2026-08-25", "2026-08-25T09:15:00+00:00")
    mutate(snap)
    assert bsd._has_snapshot_shape(snap) is False


def test_merge_history_drops_snapshot_with_idless_rows():
    prev = make_history()
    prev["snapshots"]["2026-08-25"] = make_history_snapshot(
        "2026-08-25", "2026-08-25T09:15:00+00:00"
    )
    prev["snapshots"]["2026-08-25"]["tabs"]["balanced"][0] = {"rank": 1}
    merged = bsd.merge_history(
        prev, make_history_snapshot("2026-09-02", "2026-09-02T09:15:00+00:00")
    )
    assert "2026-08-25" not in merged["snapshots"]


def test_validate_history_happy_and_rejections():
    doc = make_history()
    doc["snapshots"]["2026-09-02"] = make_history_snapshot(
        "2026-09-02", "2026-09-02T09:15:00+00:00"
    )
    doc["updated_at"] = "2026-09-02T09:15:00+00:00"
    bsd.validate_history(doc)  # must not raise
    with pytest.raises(ValueError):
        bsd.validate_history({"schema_version": 4})
    bad = make_history()
    bad["snapshots"]["2026-08-26"]["tabs"]["balanced"][0]["rank"] = 5
    with pytest.raises(ValueError):
        bsd.validate_history(bad)


def test_main_writes_history_from_malformed_prev(tmp_path):
    prev_file = tmp_path / "history-prev.json"
    prev_file.write_text("{not json")
    catalog_file = tmp_path / "catalog-raw.json"
    catalog_file.write_text(json.dumps(make_catalog()))
    argv, out = catalog_argv(tmp_path, catalog_file)
    argv += ["--history-prev-file", str(prev_file)]
    assert bsd.main(argv) == 0
    history_path = out.parent / "history.json"
    written = json.loads(history_path.read_text())
    assert list(written["snapshots"]) == ["2026-09-02"]
    assert written["snapshots"]["2026-09-02"]["pool_ids"] == [
        "acme/model-a",
        "acme/small",
    ]
    assert out.exists()  # data.json still written


def test_main_history_prev_requires_catalog(tmp_path, capsys):
    prev_file = tmp_path / "history-prev.json"
    prev_file.write_text(json.dumps(make_history()))
    argv, out = catalog_argv(tmp_path, None)
    argv += ["--history-prev-file", str(prev_file)]
    assert bsd.main(argv) == 1
    err = capsys.readouterr().err
    assert "error:" in err and "--catalog-file" in err
    assert not out.exists()  # nothing written: the check precedes all writes
    assert not (out.parent / "history.json").exists()


def test_main_drops_rank_corrupted_prev_snapshot(tmp_path):
    prev = make_history()
    prev["snapshots"]["2026-08-25"] = make_history_snapshot(
        "2026-08-25", "2026-08-25T09:15:00+00:00"
    )
    prev["snapshots"]["2026-08-25"]["tabs"]["quality"][0]["rank"] = 7
    prev_file = tmp_path / "history-prev.json"
    prev_file.write_text(json.dumps(prev))
    catalog_file = tmp_path / "catalog-raw.json"
    catalog_file.write_text(json.dumps(make_catalog()))
    argv, out = catalog_argv(tmp_path, catalog_file)
    argv += ["--history-prev-file", str(prev_file)]
    assert bsd.main(argv) == 0
    written = json.loads((out.parent / "history.json").read_text())
    assert "2026-08-25" not in written["snapshots"]
    assert "2026-08-26" in written["snapshots"]
    assert "2026-09-02" in written["snapshots"]


# ---------------------------------------------------------------------------
# highlights.json publication
# ---------------------------------------------------------------------------


def make_highlights(**overrides):
    doc = {
        "schema_version": 1,
        "generated_at": "2026-09-02T09:15:00+00:00",
        "source": "openrouter",
        "sections": {
            "week": "`acme/model-b` climbed 1 spot(s).",
            "intelligence": "AA intelligence moves: `acme/model-b` +8.4.",
            "prices": "Blended price moves: `acme/model-a` 1.25 -> 1.0.",
        },
    }
    doc.update(overrides)
    return doc


def test_validate_highlights_happy_and_rejections():
    bsd.validate_highlights(make_highlights())  # must not raise
    with pytest.raises(ValueError):
        bsd.validate_highlights({"schema_version": 2})
    bad = make_highlights(source="psychic")
    with pytest.raises(ValueError):
        bsd.validate_highlights(bad)
    bad = make_highlights(sections={"week": "only one"})
    with pytest.raises(ValueError):
        bsd.validate_highlights(bad)
    bad = make_highlights(sections={"week": "", "intelligence": "i", "prices": "p"})
    with pytest.raises(ValueError):
        bsd.validate_highlights(bad)
    bad = make_highlights(generated_at="yesterday")
    with pytest.raises(ValueError):
        bsd.validate_highlights(bad)
    bad = make_highlights(generated_at=None)
    with pytest.raises(ValueError):
        bsd.validate_highlights(bad)
    bad = make_highlights()
    del bad["generated_at"]
    with pytest.raises(ValueError):
        bsd.validate_highlights(bad)


def test_main_writes_highlights_next_to_data_json(tmp_path):
    raw = tmp_path / "highlights-new.json"
    raw.write_text(json.dumps(make_highlights()))
    argv, out = catalog_argv(tmp_path)
    argv += ["--highlights-file", str(raw)]
    assert bsd.main(argv) == 0
    written = json.loads((out.parent / "highlights.json").read_text())
    assert written["source"] == "openrouter"
    assert set(written["sections"]) == {"week", "intelligence", "prices"}
    assert out.exists()


def test_main_rejects_malformed_highlights(tmp_path, capsys):
    raw = tmp_path / "highlights-new.json"
    raw.write_text(json.dumps(make_highlights(source="psychic")))
    argv, out = catalog_argv(tmp_path, None)
    argv += ["--highlights-file", str(raw)]
    assert bsd.main(argv) == 1
    assert "error:" in capsys.readouterr().err
    assert not out.exists()


def test_main_without_highlights_file_writes_no_highlights(tmp_path):
    argv, out = catalog_argv(tmp_path)
    assert bsd.main(argv) == 0
    assert out.exists()
    assert not (out.parent / "highlights.json").exists()


def test_main_writes_discount_pct_into_data_json(tmp_path):
    argv, out = catalog_argv(tmp_path)
    assert bsd.main(argv) == 0
    written = json.loads(out.read_text())
    for name in bsd.PRIORITIES:
        assert written["priorities"][name][0]["discount_pct"] == "--"


def test_build_data_uses_explicit_generated_at_verbatim():
    stamp = "2026-09-15T23:59:50+00:00"
    data = bsd.build_data("acme/model-a", make_priorities(), generated_at=stamp)
    assert data["generated_at"] == stamp


@pytest.mark.parametrize(
    "bad", ["", "yesterday", 20260915, ["2026-09-15T00:00:00+00:00"]]
)
def test_build_data_rejects_malformed_generated_at(bad):
    with pytest.raises(ValueError, match="generated_at"):
        bsd.build_data("acme/model-a", make_priorities(), generated_at=bad)


def test_main_rejects_catalog_with_malformed_generated_at(tmp_path, capsys):
    catalog = make_catalog()
    catalog["generated_at"] = "not-a-date"
    catalog_file = tmp_path / "catalog.json"
    catalog_file.write_text(json.dumps(catalog))
    argv, out = catalog_argv(tmp_path, catalog_file)
    assert bsd.main(argv) == 1
    assert "generated_at" in capsys.readouterr().err
    assert not out.exists()


def test_build_data_defaults_generated_at_to_now():
    now = datetime(2026, 9, 16, 0, 0, 5, tzinfo=timezone.utc)
    data = bsd.build_data("acme/model-a", make_priorities(), now=now)
    assert data["generated_at"] == "2026-09-16T00:00:05+00:00"


# ---------------------------------------------------------------------------
# source_revision: the web's traceability stamp (no web versioning)
# ---------------------------------------------------------------------------


def test_build_data_stamps_source_revision():
    data = bsd.build_data("acme/model-a", make_priorities(), revision="abc1234")
    assert data["source_revision"] == "abc1234"


def test_build_data_strips_source_revision():
    data = bsd.build_data("acme/model-a", make_priorities(), revision="  abc1234  ")
    assert data["source_revision"] == "abc1234"


def test_build_data_omits_source_revision_when_absent():
    data = bsd.build_data("acme/model-a", make_priorities())
    assert "source_revision" not in data


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "   ",
        42,
        ["abc1234"],
        # The stamp is what the site footer links to: only lowercase hex
        # commit hashes (the form git emits) may reach data.json -- any
        # other text would ship a dead footer link.
        "not-a-hash",
        "ABC1234",
        "abc xyz",
        "../../x",
        "<script>",
    ],
)
def test_build_data_rejects_malformed_revision(bad):
    with pytest.raises(ValueError, match="revision"):
        bsd.build_data("acme/model-a", make_priorities(), revision=bad)


def test_main_stamps_source_revision(tmp_path):
    argv, out = catalog_argv(tmp_path)
    assert bsd.main([*argv, "--revision", "abc1234"]) == 0
    written = json.loads(out.read_text())
    assert written["source_revision"] == "abc1234"


def test_main_omits_source_revision_without_flag(tmp_path):
    argv, out = catalog_argv(tmp_path)
    assert bsd.main(argv) == 0
    written = json.loads(out.read_text())
    assert "source_revision" not in written


def test_main_stamps_data_json_with_catalog_generated_at(tmp_path):
    # Catalog made just before UTC midnight, data.json built just after: the
    # site's D-7 baseline must use the catalog date that keys the history.
    catalog = make_catalog()
    catalog["generated_at"] = "2026-09-15T23:59:50+00:00"
    catalog_file = tmp_path / "catalog.json"
    catalog_file.write_text(json.dumps(catalog))
    argv, out = catalog_argv(tmp_path, catalog_file)
    assert bsd.main(argv) == 0
    written = json.loads(out.read_text())
    assert written["generated_at"] == "2026-09-15T23:59:50+00:00"


# ---------------------------------------------------------------------------
# cross-artifact invariant: one catalog drives table, history and stamps
# ---------------------------------------------------------------------------


def _invariant_catalog():
    """Five models carrying the producer's rankings.

    The rankings deliberately differ from a naive re-sort of the rounded
    scores.overall in one place: acme/d and acme/e tie after rounding, and
    the (unrounded) producer order puts acme/e first for balanced. Anything
    that re-sorts instead of projecting rankings fails the fence below.
    """
    spec = [
        # id, (balanced, price, quality) overall, quality, blended
        ("acme/a", (0.80, 0.70, 0.90), 60.0, 2.0),
        ("acme/b", (0.80, 0.90, 0.60), 50.0, 1.0),
        ("acme/c", (0.70, 0.90, 0.60), 50.0, 0.5),
        ("acme/d", (0.70, 0.60, 0.90), None, 1.0),
        ("acme/e", (0.70, 0.60, 0.90), None, 1.0),
    ]
    models = []
    for model_id, (bal, price, qual), quality, blended in spec:
        entry = make_catalog_entry(id=model_id, quality=quality)
        entry["pricing"] = with_tier_keys(
            {
                "input_per_1m": blended,
                "output_per_1m": blended,
                "blended_per_1m": blended,
            }
        )
        entry["scores"]["overall"] = {"balanced": bal, "price": price, "quality": qual}
        if quality is None:
            entry["aa"] = {
                "intelligence_index": None,
                "coding_index": None,
                "agentic_index": None,
            }
            entry["quality_match"] = None
        models.append(entry)
    doc = make_catalog()
    doc["models"] = models
    doc["pool"] = {"listed": 6, "candidates": 5, "dropped": {"context": 1}}
    doc["sources"]["aa"] = {
        "mode": "aa",
        "matched_aa": 0,
        "matched_catalog": 0,
        "matched_openrouter": 3,
        "catalog_generated_at": None,
        "intelligence_index_version": 4.3,
    }
    doc["rankings"] = {
        "balanced": ["acme/a", "acme/b", "acme/c", "acme/e", "acme/d"],
        "price": ["acme/c", "acme/b", "acme/a", "acme/d", "acme/e"],
        "quality": ["acme/a", "acme/d", "acme/e", "acme/c", "acme/b"],
    }
    bsd.validate_catalog(doc)
    return doc


def test_table_and_history_rank_from_one_catalog():
    """Projection fence: data.json rows == rankings[p][:10] == history tabs.

    No sort key lives in this test: the catalog's rankings are the only
    ranking, and every artifact must be a projection of it. The rows arrive
    in a scrambled order to prove build_data reorders rather than trusts.
    """
    catalog = _invariant_catalog()
    by_id = {e["id"]: e for e in catalog["models"]}
    rows = {}
    for priority in ("balanced", "price", "quality"):
        # same set as the CLI's top N, deliberately out of order
        scrambled = sorted(catalog["rankings"][priority][:10], reverse=True)
        rows[priority] = [
            make_row(
                model=model_id,
                opencode_model=f"openrouter/{model_id}",
                score=by_id[model_id]["scores"]["overall"][priority],
                quality_index=by_id[model_id]["quality"],
                blended_usd_per_m=by_id[model_id]["pricing"]["blended_per_1m"],
            )
            for model_id in scrambled
        ]

    # Mirror main: data.json stamped from and ordered by the catalog.
    data = bsd.build_data(
        "openrouter/acme/a",
        rows,
        generated_at=catalog["generated_at"],
        catalog=catalog,
    )
    history = bsd.merge_history(None, bsd.build_snapshot(catalog))
    bsd.validate_history(history)
    (tabs,) = [snap["tabs"] for snap in history["snapshots"].values()]

    for priority in ("balanced", "price", "quality"):
        table = [row["model"] for row in data["priorities"][priority]]
        tab = [row["id"] for row in tabs[priority]]
        assert table == tab == catalog["rankings"][priority][:10], priority
    assert data["generated_at"] == history["updated_at"] == catalog["generated_at"]


# ---------------------------------------------------------------------------
# Catalog v2: tiered-pricing validation and the schedules projection
# ---------------------------------------------------------------------------


def _tiered_pricing():
    return {
        "input_per_1m": 1.32,
        "output_per_1m": 3.96,
        "blended_per_1m": 1.98,
        "base": {
            "input_per_1m": 1.32,
            "output_per_1m": 3.96,
            "blended_per_1m": 1.98,
        },
        "tiers": [],
        "schedule": [
            {
                "utc_days": None,
                "utc_start": 0,
                "utc_end": 1200,
                "coverage": 0.5,
                "input_per_1m": 1.32,
                "output_per_1m": 3.96,
                "blended_per_1m": 1.98,
            },
            {
                "utc_days": None,
                "utc_start": 1200,
                "utc_end": 0,
                "coverage": 0.5,
                "input_per_1m": 0.66,
                "output_per_1m": 1.98,
                "blended_per_1m": 0.99,
            },
        ],
    }


def _schedule_entry(model_id, discount, off_in=0.66, off_out=1.98):
    return {
        "id": model_id,
        "name": model_id.split("/")[1],
        "context": 262144,
        "quality": None,
        # Provenance for the deals QUAL tooltip: null like a row whose
        # quality came without any (the producer emits the keys always).
        "quality_source": None,
        "quality_as_of": None,
        "quality_effort": None,
        "score": None,
        "max_discount": discount,
        "sched_note": f"-{discount:.0%}",
        "sched_detail": "daily 16:00-00:00 UTC",
        "peak": {
            "input_per_1m": 1.32,
            "output_per_1m": 3.96,
            "blended_per_1m": 1.98,
        },
        "offpeak": {
            "input_per_1m": off_in,
            "output_per_1m": off_out,
            "blended_per_1m": 0.99,
        },
        "schedule": [
            {
                "utc_days": None,
                "utc_start": 0,
                "utc_end": 1600,
                "coverage": 0.6667,
                "input_per_1m": 1.32,
                "output_per_1m": 3.96,
                "blended_per_1m": 1.98,
            },
            {
                "utc_days": None,
                "utc_start": 1600,
                "utc_end": 0,
                "coverage": 0.3333,
                "input_per_1m": off_in,
                "output_per_1m": off_out,
                "blended_per_1m": 0.99,
            },
        ],
    }


def make_schedules():
    return [
        _schedule_entry("acme/big-deal", 0.5),
        _schedule_entry("acme/small-deal", 0.1, off_in=0.7506, off_out=2.2509),
        _schedule_entry("acme/negligible", 0.004),  # formats 0% -> excluded
    ]


def test_validate_catalog_accepts_v2_with_subkeys():
    doc = make_catalog()
    pricing = _tiered_pricing()
    pricing["tiers"] = [
        {
            "min_prompt_tokens": 100000,
            "input_per_1m": 2.0,
            "output_per_1m": 4.0,
            "blended_per_1m": 2.5,
        },
        {
            "min_prompt_tokens": 200000,
            "input_per_1m": 3.0,
            "output_per_1m": 6.0,
            "blended_per_1m": 3.75,
        },
    ]
    doc["models"][0]["pricing"] = pricing
    doc["schedules"] = make_schedules()
    bsd.validate_catalog(doc)  # must not raise


def test_validate_catalog_rejects_v1():
    doc = make_catalog()
    doc["schema_version"] = 1
    with pytest.raises(ValueError, match="schema_version"):
        bsd.validate_catalog(doc)


def test_validate_catalog_requires_schedules_key():
    doc = make_catalog()
    del doc["schedules"]
    with pytest.raises(ValueError, match="schedules"):
        bsd.validate_catalog(doc)


def _pricing_doc(**pricing_overrides):
    doc = make_catalog()
    pricing = dict(_tiered_pricing())
    pricing.update(pricing_overrides)
    doc["models"][0]["pricing"] = pricing
    return doc


def test_validate_catalog_pricing_subkeys():
    with pytest.raises(ValueError):
        bsd.validate_catalog(_pricing_doc(base={"input_per_1m": 1.0}))
    with pytest.raises(ValueError):
        bsd.validate_catalog(
            _pricing_doc(
                base={
                    "input_per_1m": -1.0,
                    "output_per_1m": 2.0,
                    "blended_per_1m": 1.0,
                }
            )
        )
    with pytest.raises(ValueError):
        bsd.validate_catalog(
            _pricing_doc(
                tiers=[
                    {
                        "min_prompt_tokens": 200000,
                        "input_per_1m": 1.0,
                        "output_per_1m": 2.0,
                        "blended_per_1m": 1.5,
                    },
                    {
                        "min_prompt_tokens": 100000,
                        "input_per_1m": 1.0,
                        "output_per_1m": 2.0,
                        "blended_per_1m": 1.2,
                    },
                ]
            )
        )
    with pytest.raises(ValueError):
        bsd.validate_catalog(
            _pricing_doc(
                schedule=[dict(_tiered_pricing()["schedule"][0], utc_end=2400)]
            )
        )
    with pytest.raises(ValueError):
        bsd.validate_catalog(
            _pricing_doc(
                schedule=[dict(_tiered_pricing()["schedule"][0], utc_start=1075)]
            )
        )
    with pytest.raises(ValueError):
        bsd.validate_catalog(
            _pricing_doc(
                schedule=[dict(_tiered_pricing()["schedule"][0], utc_days=["funday"])]
            )
        )
    with pytest.raises(ValueError):
        bsd.validate_catalog(
            _pricing_doc(
                schedule=[dict(_tiered_pricing()["schedule"][0], coverage=0.0)]
            )
        )
    # coverages far from 1 (outside the 1e-3 tolerance)
    with pytest.raises(ValueError):
        bsd.validate_catalog(
            _pricing_doc(
                schedule=[dict(_tiered_pricing()["schedule"][0], coverage=0.4)]
            )
        )


def test_validate_catalog_schedule_coverage_sum_tolerance():
    # rounded coverages may drift slightly; within 1e-3 of 1 must pass
    windows = _tiered_pricing()["schedule"]
    drifted = [dict(windows[0], coverage=0.5004), windows[1]]
    doc = _pricing_doc(schedule=drifted)
    bsd.validate_catalog(doc)  # 1.0004 - 1 = 4e-4 < 1e-3


def test_validate_catalog_accepts_48_window_schedule_drift():
    # Producer coverages are rounded per window to 4 decimals (up to 5e-5
    # each): 48 half-hour windows sum to 0.9984 yet tile the week exactly.
    coverage = round(7 * 30 / 10080, 4)
    windows = []
    for i in range(48):
        start = (i // 2) * 100 + (i % 2) * 30
        end = ((i + 1) // 2) * 100 + ((i + 1) % 2) * 30
        windows.append(
            {
                "utc_days": None,
                "utc_start": start,
                "utc_end": 0 if end == 2400 else end,
                "coverage": coverage,
                "input_per_1m": 1.0 if i % 2 else 2.0,
                "output_per_1m": 4.0,
                "blended_per_1m": 1.75 if i % 2 else 2.5,
            }
        )
    assert abs(sum(w["coverage"] for w in windows) - 1.0) > 1e-3
    doc = _pricing_doc(schedule=windows)
    entry = _schedule_entry("acme/half-hourly", 0.3)
    entry["schedule"] = windows
    doc["schedules"] = [entry]
    bsd.validate_catalog(doc)  # must not raise


def test_validate_catalog_schedules_entries():
    doc = make_catalog()
    schedules = make_schedules()
    schedules[0]["max_discount"] = 1.5
    doc["schedules"] = schedules
    with pytest.raises(ValueError):
        bsd.validate_catalog(doc)
    doc = make_catalog()
    schedules = make_schedules()
    schedules[0]["quality"] = 200.0
    doc["schedules"] = schedules
    with pytest.raises(ValueError):
        bsd.validate_catalog(doc)
    doc = make_catalog()
    schedules = make_schedules()
    schedules[0]["score"] = 1.5
    doc["schedules"] = schedules
    with pytest.raises(ValueError):
        bsd.validate_catalog(doc)
    doc = make_catalog()
    schedules = make_schedules()
    del schedules[0]["offpeak"]
    doc["schedules"] = schedules
    with pytest.raises(ValueError):
        bsd.validate_catalog(doc)
    # identity and display fields the site consumes (BUG-04): a null or
    # empty sched_note would throw in renderDeals; ids must be catalog ids.
    for key, bad in (
        ("sched_note", None),
        ("sched_note", ""),
        ("sched_note", 38),
        ("id", None),
        ("id", ""),
        ("id", ["acme/x"]),
        ("id", "no-slash"),
        ("id", "acme/bad id"),
        ("name", 42),
        ("context", -1),
        ("context", "262144"),
        ("context", True),
        ("context", float("nan")),
        ("quality_source", "scrape"),
        ("quality_source", 42),
        ("quality_as_of", 42),
        ("quality_as_of", ["20261017"]),
        ("quality_effort", 42),
        ("quality_effort", ""),
        ("quality_effort", ["high"]),
    ):
        doc = make_catalog()
        schedules = make_schedules()
        schedules[0][key] = bad
        doc["schedules"] = schedules
        with pytest.raises(ValueError):
            bsd.validate_catalog(doc)
    # name and context may be null
    doc = make_catalog()
    schedules = make_schedules()
    schedules[0]["name"] = None
    schedules[0]["context"] = None
    doc["schedules"] = schedules
    bsd.validate_catalog(doc)  # must not raise
    # Provenance (QUAL tooltip): null or the winner's label and stamp; the
    # keys are required like the rest (the producer always emits them).
    doc = make_catalog()
    schedules = make_schedules()
    schedules[0].update(
        quality=50.0,
        quality_source="catalog",
        quality_as_of="20261017",
        quality_effort="high",
    )
    doc["schedules"] = schedules
    bsd.validate_catalog(doc)  # must not raise
    del schedules[0]["quality_source"]
    with pytest.raises(ValueError, match="quality_source"):
        bsd.validate_catalog(doc)


def test_project_schedules():
    catalog = make_catalog()
    catalog["schedules"] = make_schedules()
    projected = bsd.project_schedules(catalog)
    assert [row["model"] for row in projected] == [
        "acme/big-deal",
        "acme/small-deal",
    ]
    big = projected[0]
    assert big["sched_note"] == "-50%"
    assert big["input_per_1m"] == 0.66  # cheapest off-peak prices
    assert big["output_per_1m"] == 1.98
    assert big["quality"] is None and big["score"] is None
    # Provenance rides through to data.json verbatim (null without a
    # winner), so the deals QUAL tooltip can render on the site.
    assert big["quality_source"] is None and big["quality_as_of"] is None
    assert big["quality_effort"] is None
    assert big["context"] == 262144
    schedules = make_schedules()
    schedules[0].update(
        quality=30.0,
        quality_source="catalog",
        quality_as_of="20261017",
        quality_effort="high",
    )
    catalog["schedules"] = schedules
    (projected,) = [
        row for row in bsd.project_schedules(catalog) if row["model"] == "acme/big-deal"
    ]
    assert projected["quality"] == 30.0
    assert projected["quality_source"] == "catalog"
    assert projected["quality_as_of"] == "20261017"
    assert projected["quality_effort"] == "high"


def test_project_schedules_top10_cap():
    catalog = make_catalog()
    catalog["schedules"] = [
        _schedule_entry(f"acme/d{i:02d}", 0.9 - i * 0.01) for i in range(12)
    ]
    projected = bsd.project_schedules(catalog)
    assert len(projected) == 10
    assert projected[0]["model"] == "acme/d00"  # highest discount first
    assert projected[-1]["model"] == "acme/d09"


def test_build_data_schedules_key():
    catalog = make_catalog()
    catalog["schedules"] = make_schedules()
    data = bsd.build_data("openrouter/acme/model-a", make_priorities(), catalog=catalog)
    assert [row["model"] for row in data["schedules"]] == [
        "acme/big-deal",
        "acme/small-deal",
    ]
    bare = bsd.build_data("openrouter/acme/model-a", make_priorities())
    assert bare["schedules"] == []


def test_build_data_emits_schema_version():
    data = bsd.build_data("openrouter/acme/model-a", make_priorities())
    assert data["schema_version"] == 3


def test_row_keys_optional_passthrough():
    rows = make_priorities()
    rows["balanced"][0]["tier_note"] = ">100k"
    rows["balanced"][0]["sched_note"] = None
    rows["balanced"][0]["sched_detail"] = None
    data = bsd.build_data("openrouter/acme/model-a", rows)
    assert data["priorities"]["balanced"][0]["tier_note"] == ">100k"
    # data.json v2: the missing-key check demands the new keys on every row;
    # make_priorities' default fixture carries them null-valued (the
    # producer's pre-endpoint degradation), so validation passes.
    bsd.build_data("openrouter/acme/model-a", make_priorities())


def test_merge_history_resets_v1():
    prev = make_history()
    prev["schema_version"] = 1
    merged = bsd.merge_history(
        prev, make_history_snapshot("2026-09-02", "2026-09-02T09:15:00+00:00")
    )
    assert list(merged["snapshots"]) == ["2026-09-02"]
    assert merged["schema_version"] == 5


# ---------------------------------------------------------------------------
# Catalog v3: provider-aware blocks, data.json v3 rows, history v5 reset
# ---------------------------------------------------------------------------


def _v3_entry_blocks():
    """Realistic schema-v3 additive blocks for one entry (producer shapes)."""
    return {
        "endpoints": [
            {
                "provider_name": "Z.AI",
                "tag": "z-ai/fp8",
                "quantization": "fp8",
                "pricing": {
                    "prompt": 0.0000002,
                    "completion": 0.0000004,
                    "overrides": None,
                },
                "context_length": 262144,
                "status": 0,
                "service_tier": None,
            }
        ],
        "default_precision": "fp8",
        "advertised_pricing": {
            "input_per_1m": 1.0,
            "output_per_1m": 2.0,
            "blended_per_1m": 1.25,
            "source": "models-api",
        },
        "annotations": {
            "price_spread": 0.5,
            "cheapest_quantized": {
                "provider": "DeepInfra",
                "quantization": "fp4",
                "price": 0.00000005,
            },
            "risk_flag": True,
            "baseline_context_below_catalog": False,
        },
    }


def _v3_session_cost():
    return {
        "attribution": "Data © OpenRouter, CC BY 4.0",
        "app_slug": "hermes-agent",
        "as_of": "2026-10-15T00:00:00Z",
        "window_days": 30,
        "window_end_date": "2026-10-12",
        "medians": {
            "acme/model-a": {
                "1-turn": 0.05,
                "2-9-turns": 0.10,
                "10-49-turns": 1.74,
                "50-plus-turns": 9.99,
            }
        },
    }


def test_validate_catalog_accepts_v3_blocks():
    doc = make_catalog()
    doc["models"][0].update(_v3_entry_blocks())
    doc["session_cost"] = _v3_session_cost()
    bsd.validate_catalog(doc)  # must not raise
    # Keyless degradation shapes the producer emits: blocks null, the
    # session-cost key absent entirely.
    bsd.validate_catalog(make_catalog())  # must not raise


def test_validate_catalog_rejects_v2_document():
    doc = make_catalog()
    doc["schema_version"] = 2
    with pytest.raises(ValueError, match="schema_version"):
        bsd.validate_catalog(doc)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda ep: ep.update(status="0"),  # not an int
        lambda ep: ep.update(quantization="int16"),  # no ladder label, not null
        lambda ep: ep.update(
            quantization={"a": 1}
        ),  # unhashable: ValueError, not TypeError
        lambda ep: ep["pricing"].update(prompt="0.2"),  # price must be numeric
        lambda ep: ep.update(provider_name=7),
        lambda ep: ep.update(tag=["z-ai/fp8"]),
        lambda ep: ep.update(context_length=262144.0),
        lambda ep: ep.pop("service_tier"),
        lambda ep: ep["pricing"].pop("completion"),
    ],
)
def test_validate_catalog_rejects_bad_endpoint_entry(mutate):
    doc = make_catalog()
    doc["models"][0].update(_v3_entry_blocks())
    mutate(doc["models"][0]["endpoints"][0])
    with pytest.raises(ValueError):
        bsd.validate_catalog(doc)


def test_validate_catalog_rejects_bad_default_precision():
    # "unknown" never leaves the derivation: default_precision is a ladder
    # key or null, never the sentinel the endpoints themselves may carry.
    doc = make_catalog()
    doc["models"][0].update(_v3_entry_blocks())
    doc["models"][0]["default_precision"] = "unknown"
    with pytest.raises(ValueError, match="default_precision"):
        bsd.validate_catalog(doc)
    # Unhashable values must hit the same ValueError contract (a TypeError
    # would escape the "error: invalid catalog" handler as a traceback).
    doc["models"][0]["default_precision"] = ["fp8"]
    with pytest.raises(ValueError, match="default_precision"):
        bsd.validate_catalog(doc)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda doc: doc["models"][0]["annotations"].update(price_spread="0.5"),
        lambda doc: doc["models"][0]["annotations"].update(risk_flag="yes"),
        lambda doc: doc["models"][0]["annotations"].update(
            baseline_context_below_catalog=None
        ),
        lambda doc: doc["models"][0]["annotations"].update(
            cheapest_quantized={"provider": "X", "quantization": "fp4"}
        ),  # no price
        lambda doc: doc["models"][0]["annotations"].update(
            cheapest_quantized={"provider": "X", "quantization": "fp4", "price": "5"}
        ),
        lambda doc: doc["models"][0]["annotations"].pop("price_spread"),
        lambda doc: doc["models"][0].update(annotations="garbage"),
        lambda doc: doc["models"][0]["advertised_pricing"].update(input_per_1m="1.0"),
        lambda doc: doc["models"][0]["advertised_pricing"].update(source="scrape"),
        lambda doc: doc["models"][0]["advertised_pricing"].pop("source"),
        lambda doc: doc["models"][0]["advertised_pricing"].pop("blended_per_1m"),
        lambda doc: doc["models"][0].update(advertised_pricing=[]),
    ],
)
def test_validate_catalog_rejects_bad_annotation_and_advertised_blocks(mutate):
    doc = make_catalog()
    doc["models"][0].update(_v3_entry_blocks())
    mutate(doc)
    with pytest.raises(ValueError):
        bsd.validate_catalog(doc)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda doc: doc["session_cost"].update(window_days=0),
        lambda doc: doc["session_cost"].update(window_days=1.5),
        lambda doc: doc["session_cost"].update(window_days="30"),
        lambda doc: doc["session_cost"].update(as_of=None),
        lambda doc: doc["session_cost"].update(attribution=""),
        lambda doc: doc["session_cost"].update(medians="garbage"),
        lambda doc: doc["session_cost"]["medians"].update({"": {}}),
        lambda doc: doc["session_cost"]["medians"]["acme/model-a"].update(
            {"10-49-turns": "1.74"}
        ),  # median must be numeric
        lambda doc: doc["session_cost"]["medians"]["acme/model-a"].update(
            {"100-plus-turns": 1.0}
        ),  # not one of the dataset's four turn ranges
    ],
)
def test_validate_catalog_rejects_bad_session_cost_block(mutate):
    doc = make_catalog()
    doc["models"][0].update(_v3_entry_blocks())
    doc["session_cost"] = _v3_session_cost()
    mutate(doc)
    with pytest.raises(ValueError):
        bsd.validate_catalog(doc)


def test_validate_catalog_accepts_subset_of_medians():
    # The OpenRouter session-cost dataset can cover a model for fewer than
    # all four turn ranges: a medians cell dict carrying only SOME of them
    # passes validation. Unknown range keys are still rejected (the reject
    # block above pins that case).
    doc = make_catalog()
    doc["models"][0].update(_v3_entry_blocks())
    doc["session_cost"] = _v3_session_cost()
    doc["session_cost"]["medians"] = {
        "acme/model-a": {"1-turn": 0.05, "50-plus-turns": 9.99}
    }
    bsd.validate_catalog(doc)  # must not raise


def _v3_steering_block():
    """A well-formed schema-v3 steering block (producer shape)."""
    return {
        "model": "acme/model-a",
        "provider": {
            "order": ["acme/fp8"],
            "ignore": ["acme/fp4"],
            "quantizations": ["fp8"],
            "allow_fallbacks": True,
        },
    }


def test_validate_catalog_accepts_steering():
    doc = make_catalog()
    doc["models"][0]["steering"] = _v3_steering_block()
    bsd.validate_catalog(doc)  # must not raise
    # The null-default degradation shape the producer emits for
    # pre-endpoint candidates.
    doc["models"][0]["steering"] = None
    bsd.validate_catalog(doc)  # must not raise
    # Old fixtures (pre-endpoint-shaped documents) lack the key entirely:
    # the missing-key-only check tolerates its absence.
    bsd.validate_catalog(make_catalog())  # must not raise


@pytest.mark.parametrize(
    "mutate",
    [
        lambda st: st["provider"].update(tag="acme/fp8"),  # key outside the allowed set
        lambda st: st["provider"]["order"].append(42),  # order entry not a string
        lambda st: st["provider"].update(allow_fallbacks="yes"),  # non-bool
        lambda st: st["provider"].update(quantizations=[{}]),  # unhashable: ValueError
        lambda st: st.update(
            extra="junk"
        ),  # unknown top-level key (docstring contract)
    ],
)
def test_validate_catalog_rejects_bad_steering(mutate):
    doc = make_catalog()
    doc["models"][0]["steering"] = _v3_steering_block()
    mutate(doc["models"][0]["steering"])
    with pytest.raises(ValueError):
        bsd.validate_catalog(doc)


def test_rows_carry_new_keys_ordered_by_ranking():
    # data.json v2 rows: the five additive keys survive the projection onto
    # the catalog rankings (same fence as the pre-existing keys).
    catalog = _ranked_catalog()
    by_id = {
        "acme/a": {
            "default_precision": "fp8",
            "risk_flag": False,
            "price_spread": 0.5,
            "session_cost_usd": 0.42,
            "quant_note": None,
        },
        "acme/b": {
            "default_precision": None,
            "risk_flag": True,
            "price_spread": None,
            "session_cost_usd": None,
            "quant_note": "cheapest non-baseline: fp4 @ DeepInfra",
        },
        "acme/c": {
            "default_precision": "int4",
            "risk_flag": True,
            "price_spread": 2.0,
            "session_cost_usd": 0.9,
            "quant_note": None,
        },
    }
    rows = {}
    for priority in bsd.PRIORITIES:
        scrambled = sorted(catalog["rankings"][priority], reverse=True)
        rows[priority] = [
            make_row(
                model=model_id,
                opencode_model=f"openrouter/{model_id}",
                **by_id[model_id],
            )
            for model_id in scrambled
        ]
    data = bsd.build_data("acme/a", rows, catalog=catalog)
    assert data["schema_version"] == 3
    for priority in bsd.PRIORITIES:
        for row, model_id in zip(
            data["priorities"][priority], catalog["rankings"][priority]
        ):
            for key, value in by_id[model_id].items():
                assert row[key] == value, (priority, model_id, key)


def test_merge_history_resets_on_v4_history():
    # The live site's previous history (v4) must not be merged into a v5
    # document: any explicit version but the current one starts fresh.
    prev = make_history(schema_version=4)
    merged = bsd.merge_history(
        prev, make_history_snapshot("2026-09-02", "2026-09-02T09:15:00+00:00")
    )
    assert list(merged["snapshots"]) == ["2026-09-02"]
    assert merged["schema_version"] == 5


# ---------------------------------------------------------------------------
# Catalog v6 / data.json v3 / history v5 (best selectable depth)
# ---------------------------------------------------------------------------


def _reasoning_block(**overrides):
    """A well-formed schema-v6 reasoning block (producer shape)."""
    block = {
        "default_effort": "high",
        "supported_efforts": ["max", "high", "low"],
        "default_enabled": True,
        "mandatory": False,
        "supports_max_tokens": None,
        "depths": [
            {
                "effort": "high",
                "index": 46.8,
                "cost_per_task_usd": 0.88,
                "source": "aa",
                "as_of": "2026-10-20",
            },
            {
                "effort": "max",
                "index": 56.0,
                "cost_per_task_usd": None,
                "source": "aa",
                "as_of": None,
            },
        ],
    }
    block.update(overrides)
    return block


def test_catalog_v6_is_accepted_and_old_schemas_rejected():
    bsd.validate_catalog(make_catalog())
    for schema_version in (4, 5):
        with pytest.raises(ValueError, match="unknown catalog schema_version"):
            bsd.validate_catalog({**make_catalog(), "schema_version": schema_version})


def test_data_v3_rows_require_depth_fields_and_drop_reasoning():
    row = make_row(
        quality_effort="high", default_effort="high", reasoning={"depths": []}
    )
    doc = bsd.build_data("acme/model-a", {p: [dict(row)] for p in bsd.PRIORITIES})
    assert doc["schema_version"] == 3
    assert "reasoning" not in doc["priorities"]["balanced"][0]
    assert doc["priorities"]["balanced"][0]["quality_effort"] == "high"
    assert doc["priorities"]["balanced"][0]["default_effort"] == "high"
    # Both new keys are required: the missing-key check fails loudly.
    for key in ("quality_effort", "default_effort"):
        priorities = {p: [dict(row)] for p in bsd.PRIORITIES}
        del priorities["price"][0][key]
        with pytest.raises(ValueError, match=key):
            bsd.build_data("acme/model-a", priorities)


def test_history_v5_discards_v4_and_keeps_highlights_v1():
    assert bsd.HISTORY_SCHEMA_VERSION == 5
    assert bsd.HIGHLIGHTS_SCHEMA_VERSION == 1
    prev = {
        "schema_version": 4,
        "snapshots": {
            "2026-10-20": make_history_snapshot(
                "2026-10-20", "2026-10-20T00:00:00+00:00"
            )
        },
    }
    merged = bsd.merge_history(
        prev, make_history_snapshot("2026-10-26", "2026-10-26T00:00:00+00:00")
    )
    assert merged["schema_version"] == 5
    assert list(merged["snapshots"]) == ["2026-10-26"]


def test_validate_catalog_accepts_reasoning_block():
    doc = make_catalog()
    doc["models"][0]["reasoning"] = _reasoning_block()
    doc["models"][0]["quality_effort"] = "high"
    bsd.validate_catalog(doc)  # must not raise
    # The keyless degradation shape: no metadata, no measured depths.
    bsd.validate_catalog(make_catalog())  # must not raise


@pytest.mark.parametrize(
    "mutate, field",
    [
        (lambda r: r.update(default_effort=7), "default_effort"),
        (lambda r: r.update(default_effort=""), "default_effort"),
        (lambda r: r.update(mandatory="yes"), "mandatory"),
        (lambda r: r.update(default_enabled=1), "default_enabled"),
        (lambda r: r.update(supports_max_tokens="true"), "supports_max_tokens"),
        (lambda r: r.update(supported_efforts="high"), "supported_efforts"),
        (lambda r: r.update(supported_efforts=[7]), "supported_efforts"),
        (lambda r: r.update(supported_efforts=[""]), "supported_efforts"),
        (lambda r: r.update(depths=None), "depths"),
        (lambda r: r.update(depths=[{"effort": "high"}]), "depths"),
        (lambda r: r.pop("depths"), "depths"),
        (lambda r: r["depths"][0].update(index=float("nan")), "index"),
        (lambda r: r["depths"][0].update(index="46.8"), "index"),
        (lambda r: r["depths"][0].update(index=True), "index"),
        (
            lambda r: r["depths"][0].update(cost_per_task_usd=float("inf")),
            "cost_per_task_usd",
        ),
        (
            lambda r: r["depths"][0].update(cost_per_task_usd="0.88"),
            "cost_per_task_usd",
        ),
        (lambda r: r["depths"][0].update(effort=""), "effort"),
        (lambda r: r["depths"].append(dict(r["depths"][0])), "duplicate"),
        (lambda r: r["depths"][0].update(source=7), "source"),
        (lambda r: r["depths"][0].update(as_of=7), "as_of"),
        (lambda r: r["depths"][0].pop("as_of"), "as_of"),
    ],
)
def test_validate_catalog_rejects_bad_reasoning_block(mutate, field):
    doc = make_catalog()
    block = _reasoning_block()
    mutate(block)
    doc["models"][0]["reasoning"] = block
    with pytest.raises(ValueError, match=field):
        bsd.validate_catalog(doc)


def test_validate_catalog_rejects_quality_index_in_depth_record():
    # Depth records reuse the quality record's shape: the score field is
    # `index`. `quality_index` is the pre-release draft key -- a record
    # carrying it is malformed, never silently accepted.
    doc = make_catalog()
    block = _reasoning_block()
    block["depths"][0]["quality_index"] = 46.8
    doc["models"][0]["reasoning"] = block
    with pytest.raises(ValueError, match="quality_index"):
        bsd.validate_catalog(doc)


def test_validate_catalog_accepts_unlabeled_aa_reference():
    doc = make_catalog()
    doc["models"][0]["reasoning"] = _reasoning_block(
        aa_unlabeled={
            "index": 27.8,
            "cost_per_task_usd": 0.58,
            "source": "aa",
            "as_of": "2026-11-04T12:27:03+00:00",
        }
    )
    bsd.validate_catalog(doc)  # must not raise
    # Absent (older catalogs) and explicit null both mean "no reference".
    bsd.validate_catalog(make_catalog())
    doc["models"][0]["reasoning"] = _reasoning_block(aa_unlabeled=None)
    bsd.validate_catalog(doc)


@pytest.mark.parametrize(
    "mutate, field",
    [
        (lambda r: r.update(index=float("nan")), "index"),
        (lambda r: r.update(index="27.8"), "index"),
        (lambda r: r.update(index=True), "index"),
        (lambda r: r.update(cost_per_task_usd=float("inf")), "cost_per_task_usd"),
        (lambda r: r.update(cost_per_task_usd="0.58"), "cost_per_task_usd"),
        (lambda r: r.update(source=7), "source"),
        (lambda r: r.update(source=""), "source"),
        (lambda r: r.update(as_of=7), "as_of"),
        (lambda r: r.pop("as_of"), "as_of"),
        (lambda r: r.pop("index"), "index"),
        (lambda r: r.pop("cost_per_task_usd"), "cost_per_task_usd"),
        (lambda r: r.update(effort="max"), "effort"),
    ],
)
def test_validate_catalog_rejects_bad_unlabeled_reference(mutate, field):
    doc = make_catalog()
    record = {
        "index": 27.8,
        "cost_per_task_usd": 0.58,
        "source": "aa",
        "as_of": "2026-11-04T12:27:03+00:00",
    }
    mutate(record)
    doc["models"][0]["reasoning"] = _reasoning_block(aa_unlabeled=record)
    with pytest.raises(ValueError, match=field):
        bsd.validate_catalog(doc)


def test_validate_catalog_requires_reasoning_and_quality_effort_keys():
    doc = make_catalog()
    doc["models"][0].pop("reasoning")
    with pytest.raises(ValueError, match="reasoning"):
        bsd.validate_catalog(doc)
    doc = make_catalog()
    doc["models"][0].pop("quality_effort")
    with pytest.raises(ValueError, match="quality_effort"):
        bsd.validate_catalog(doc)


def test_validate_catalog_requires_schedule_quality_effort():
    doc = make_catalog()
    schedules = make_schedules()
    del schedules[0]["quality_effort"]
    doc["schedules"] = schedules
    with pytest.raises(ValueError, match="quality_effort"):
        bsd.validate_catalog(doc)


def test_project_schedules_carries_quality_effort():
    catalog = make_catalog()
    schedules = make_schedules()
    schedules[0].update(
        quality=30.0,
        quality_source="catalog",
        quality_as_of="20261017",
        quality_effort="high",
    )
    catalog["schedules"] = schedules
    (projected,) = [
        row for row in bsd.project_schedules(catalog) if row["model"] == "acme/big-deal"
    ]
    assert projected["quality_effort"] == "high"
