"""Tests for build_site_data.py -- the data.json builder for the published site."""

import json

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
        "context_tokens": 2_000_000,
        "age_days": 10.0,
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


def make_catalog_entry(**overrides):
    entry = {
        "id": "acme/model-a",
        "name": "Model A",
        "provider": "acme",
        "family": "model",
        "pricing": {"input_per_1m": 1.0, "output_per_1m": 2.0, "blended_per_1m": 1.25},
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
        "scores": {
            "price": 0.9,
            "quality": 0.78,
            "context": 1.0,
            "age": 0.94,
            "overall": {"balanced": 0.85, "price": 0.82, "quality": 0.86},
        },
    }
    entry.update(overrides)
    return entry


def make_catalog():
    return {
        "schema_version": 1,
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
            "aa": {"mode": "openrouter", "matched": 1, "matched_openrouter": 1},
            "zdr": "ok",
            "discounts": "ok",
        },
        "pool": {"listed": 2, "candidates": 1, "dropped": {"context": 1}},
        "models": [make_catalog_entry()],
        "filtered": [{"id": "acme/small", "name": "Small", "reasons": ["context"]}],
    }


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


def test_validate_catalog_accepts_null_aa_fields_and_all_provenances():
    doc = make_catalog()
    doc["models"][0]["aa"] = {
        "intelligence_index": None,
        "coding_index": None,
        "agentic_index": None,
    }
    doc["models"][0]["quality_match"] = None
    doc["sources"]["aa"] = {"mode": "none", "matched": 0, "matched_openrouter": 0}
    bsd.validate_catalog(doc)  # must not raise
    for provenance, mode in (("api", "api"), ("scrape", "scrape")):
        doc["models"][0]["quality_match"] = provenance
        doc["sources"]["aa"].update(mode=mode, matched=1, matched_openrouter=0)
        bsd.validate_catalog(doc)  # must not raise


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
        lambda doc: doc["sources"]["aa"].update(mode="psychic"),
        lambda doc: doc["sources"]["aa"].update(matched_openrouter=5),
        lambda doc: doc["sources"]["aa"].update(matched=-1),
        lambda doc: doc["sources"]["aa"].update(matched_openrouter=True),
    ],
)
def test_validate_catalog_rejects(mutate):
    doc = make_catalog()
    mutate(doc)
    with pytest.raises(ValueError):
        bsd.validate_catalog(doc)


def test_main_writes_catalog_next_to_data_json(tmp_path):
    raw = tmp_path / "catalog-raw.json"
    raw.write_text(json.dumps(make_catalog()))
    argv, out = catalog_argv(tmp_path, raw)
    assert bsd.main(argv) == 0
    catalog_path = out.parent / "catalog.json"
    written = json.loads(catalog_path.read_text())
    assert written["schema_version"] == 1
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
