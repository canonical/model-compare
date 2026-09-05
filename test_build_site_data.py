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
        "schema_version": 1,
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


def test_build_snapshot_ranks_per_priority_top10():
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
    snap = bsd.build_snapshot(doc)
    for priority in ("balanced", "price", "quality"):
        assert len(snap["tabs"][priority]) == 10
        assert [row["rank"] for row in snap["tabs"][priority]] == list(range(1, 11))


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
    assert merged["schema_version"] == 1


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


def test_validate_history_happy_and_rejections():
    doc = make_history()
    doc["snapshots"]["2026-09-02"] = make_history_snapshot(
        "2026-09-02", "2026-09-02T09:15:00+00:00"
    )
    doc["updated_at"] = "2026-09-02T09:15:00+00:00"
    bsd.validate_history(doc)  # must not raise
    with pytest.raises(ValueError):
        bsd.validate_history({"schema_version": 2})
    bad = make_history()
    bad["snapshots"]["2026-08-26"]["tabs"]["balanced"][0]["rank"] = 5
    with pytest.raises(ValueError):
        bsd.validate_history(bad)


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
