# Full Catalog Output (`--catalog`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `model_compare.py --catalog` — one stable, deterministic JSON document covering every evaluated model (ranked candidates + filtered-out with reasons) — validated and published as `catalog.json` by the site build.

**Architecture:** Output-surface-only change. `build_candidates` gains an optional filtered-collection out-param and four additive candidate fields; two small pure helpers (`model_family`, `catalog_weights`) feed a new pure `build_catalog` assembled from post-`compute_scores` candidates; a `--catalog` branch in `run()` prints it. `build_site_data.py` gains `--catalog-file` validation writing `catalog.json` next to `data.json`; the publish workflow passes it through.

**Tech Stack:** Python 3.10+ stdlib only (argparse, json, datetime), pytest, GitHub Actions.

**Worktree:** all work happens in `.worktrees/catalog-output` on branch `catalog-output` (already created; spec committed there).

## Global Constraints

- Spec: `docs/superpowers/specs/2026-09-02-catalog-output-design.md` (in the worktree) — consult for any detail not repeated here.
- Stdlib only, Python 3.10+, no new dependencies.
- Existing behavior byte-identical: `--best`, `--json`, table, exit codes (0/1/2), cache behavior, filter and scoring semantics.
- Existing tests pass untouched (their assertions only unpack 2-tuples and compare exact dicts on empty lists — verified).
- `schema_version` starts at `1`; the document is a stable contract (additive changes only without a bump).
- Drop-reason keys are the exact strings from `build_candidates`' `drop()` calls, verbatim: `malformed id`, `context`, `pricing`, `free`, `batch`, `no discount`, `not ZDR`, `modality`, `tool calling`, `expired`, `age`.
- Catalog `age_days`/`listed_at` use **date precision** (UTC), not wall-clock floats — determinism within a UTC day.
- Scores rounded to 4 decimals, prices to 6, discount to 4 (mirrors `print_json`); `null` never NaN.
- `models` sorted by `(-overall.balanced, id)`; `filtered` by `id`.
- ZDR fail-closed is inherited from `run()` — no new logic.
- Run `rtk pytest -q` and `rtk ruff check .` after every task; both must be clean before committing.

---

### Task 1: `build_candidates` — filtered collection + additive candidate fields

**Files:**
- Modify: `model_compare.py` (function `build_candidates`, lines 491-570)
- Test: `test_model_compare.py` (new section after the `build_candidates` tests)

**Interfaces:**
- Consumes: existing `build_candidates(models, args, discounts, zdr_ids) -> (candidates, dropped)`; `parse_iso_datetime`, `has_discount` unchanged.
- Produces: `build_candidates(models, args, discounts, zdr_ids, filtered_out=None) -> (candidates, dropped)`; when `filtered_out` is a list, one `{"id": str, "name": str, "reasons": [drop-reason-str]}` is appended per dropped model (in catalog order). Every surviving candidate dict gains four keys: `created` (float unix seconds, 0.0 when unknown), `tool_calling` (bool), `zdr` (True, or None under `--no-zdr`), `expired` (bool). Task 3's `build_catalog` and `run()` rely on all five of these.

- [ ] **Step 1: Write the failing tests**

Append to `test_model_compare.py` (after the last `build_candidates` test):

```python
def test_build_candidates_collects_filtered_entries():
    models = [
        make_model(id="acme/model-a"),
        make_model(id="acme/model-b", name="B corp: Model B", context_length=100),
        make_model(id="no-slash", name="No Slash"),
    ]
    filtered = []
    candidates, dropped = mc.build_candidates(
        models, make_args(), {}, {"acme/model-a"}, filtered
    )
    assert len(candidates) == 1
    assert sorted(filtered, key=lambda e: e["id"]) == [
        {"id": "acme/model-b", "name": "B corp: Model B", "reasons": ["context"]},
        {"id": "no-slash", "name": "No Slash", "reasons": ["malformed id"]},
    ]


def test_build_candidates_without_collector_matches_old_signature():
    # 2-tuple unpacking stays valid; no filtered list, no behavior change.
    candidates, dropped = mc.build_candidates(
        [make_model(id="acme/model-b", context_length=100)],
        make_args(),
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
```

Note: `make_args()` defaults `no_require_tools=True`, so no candidate is dropped on tools in these tests; `make_model` has no `expiration_date`, so nothing drops on expiry.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest test_model_compare.py -q -k "collects_filtered or old_signature or additive_fields or zdr_null or tool_calling_false"`
Expected: FAIL — `build_candidates() takes from 4 to 4 positional arguments but 5 were given` (or TypeError on unpack).

- [ ] **Step 3: Implement**

In `build_candidates` (model_compare.py), change the signature and `drop`, and update every `drop(...)` call site to pass `(model_id, model.get("name"))`. The function becomes:

```python
def build_candidates(models, args, discounts, zdr_ids, filtered_out=None):
    now = time.time()
    require_tools = not args.no_require_tools
    dropped = {}
    candidates = []

    def drop(reason, model_id, name):
        dropped[reason] = dropped.get(reason, 0) + 1
        if filtered_out is not None:
            filtered_out.append(
                {"id": model_id, "name": name or model_id, "reasons": [reason]}
            )

    for model in models:
        model_id = model.get("id") or ""
        if "/" not in model_id:
            drop("malformed id", model_id, model.get("name"))
            continue
        context = coerce_int(model.get("context_length"), 0)
        if context < args.min_context:
            drop("context", model_id, model.get("name"))
            continue
        pricing = model.get("pricing") or {}
        price_in = parse_price(pricing.get("prompt"))
        price_out = parse_price(pricing.get("completion"))
        if price_in is None or price_out is None or price_in < 0 or price_out < 0:
            drop("pricing", model_id, model.get("name"))
            continue
        if args.exclude_free and price_in == 0 and price_out == 0:
            drop("free", model_id, model.get("name"))
            continue
        if not args.include_batch and model_id.endswith(":batch"):
            drop("batch", model_id, model.get("name"))
            continue
        discount = (discounts or {}).get(model_id)
        if args.discount and not has_discount(discount):
            drop("no discount", model_id, model.get("name"))
            continue
        if not args.no_zdr and model_id not in (zdr_ids or set()):
            drop("not ZDR", model_id, model.get("name"))
            continue
        modality = (model.get("architecture") or {}).get("modality") or ""
        output_modality = (
            modality.split("->")[-1].strip() if "->" in modality else "text"
        )
        if output_modality != "text":
            drop("modality", model_id, model.get("name"))
            continue
        params = model.get("supported_parameters") or []
        tool_calling = "tools" in params and "tool_choice" in params
        if require_tools and not tool_calling:
            drop("tool calling", model_id, model.get("name"))
            continue
        expiry = parse_iso_datetime(model.get("expiration_date"))
        expired = bool(expiry and expiry < datetime.now(timezone.utc))
        if expired:
            drop("expired", model_id, model.get("name"))
            continue
        created = model.get("created") or 0
        try:
            created = float(created)
        except (TypeError, ValueError):
            created = 0.0
        age_days = max(0.0, (now - created) / 86400.0) if created > 0 else None
        if args.max_age_days and age_days is not None and age_days > args.max_age_days:
            drop("age", model_id, model.get("name"))
            continue
        price_in_m = price_in * 1_000_000.0
        price_out_m = price_out * 1_000_000.0
        blended_m = (
            args.input_share * price_in_m + (1.0 - args.input_share) * price_out_m
        )
        candidates.append(
            {
                "id": model_id,
                "name": model.get("name") or model_id,
                "context": context,
                "price_in": price_in_m,
                "price_out": price_out_m,
                "blended": blended_m,
                "age_days": age_days,
                "discount": discount,
                "created": created,
                "tool_calling": tool_calling,
                "zdr": None if args.no_zdr else model_id in (zdr_ids or set()),
                "expired": expired,
            }
        )

    return candidates, dropped
```

- [ ] **Step 4: Run the new tests and the full suite**

Run: `rtk pytest -q`
Expected: all pass — the new 5 plus all 108 existing (existing tests unpack the 2-tuple and only compare `candidates == []` or individual fields, so the extra keys are invisible to them; `print_json` selects keys explicitly so `--json` output is unchanged).

- [ ] **Step 5: Lint and commit**

Run: `rtk ruff check .`
Expected: no issues.

```bash
git add model_compare.py test_model_compare.py
git commit -m "Collect filtered-out models and additive candidate fields in build_candidates"
```

---

### Task 2: `model_family` and `catalog_weights` helpers

**Files:**
- Modify: `model_compare.py` (new functions inserted after `match_quality`, before the `Filtering and scoring` section header)
- Test: `test_model_compare.py` (new section)

**Interfaces:**
- Consumes: `PRIORITY_WEIGHTS` (module constant), `norm_key` not needed here.
- Produces:
  - `model_family(model_id: str) -> str | None` — leading dash/underscore/digit-delimited token of the lowercased base slug; `None` when empty/non-alphabetic.
  - `catalog_weights(candidates: list, quality_by_id: dict) -> dict[str, dict[str, float]]` — per-priority effective weights mirroring `compute_scores`' quality-blind renormalization. Task 3 consumes both.

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest test_model_compare.py -q -k "model_family or catalog_weights"`
Expected: FAIL with `AttributeError: module 'model_compare' has no attribute 'model_family'`.

- [ ] **Step 3: Implement**

Insert after `match_quality` (just above the `# Filtering and scoring` comment banner):

```python
def model_family(model_id: str) -> str | None:
    """Best-effort model family from the base slug (documented heuristic).

    The leading token delimited by dash, underscore or digit of the
    lowercased base slug: glm-5.3 -> glm, gpt-5.2-mini -> gpt,
    deepseek-chat-v4 -> deepseek. Oddballs yield oddballs (o4-mini -> "o");
    no alphabetic prefix (k2) yields None.
    """
    base = model_id.split(":", 1)[0].partition("/")[2].lower()
    token = re.split(r"[-_\d]", base, maxsplit=1)[0]
    return token or None


def catalog_weights(candidates, quality_by_id):
    """Effective per-priority weights for the catalog document.

    Mirrors compute_scores' quality-blind rule: when no candidate has a
    quality score the quality weight is dropped and the rest renormalize,
    so scores.overall in the catalog is exactly reproducible downstream.
    """
    quality_blind = not any(c["id"] in quality_by_id for c in candidates)
    weights = {}
    for priority, base in PRIORITY_WEIGHTS.items():
        effective = dict(base)
        if quality_blind and "quality" in effective:
            effective.pop("quality")
            total = sum(effective.values())
            effective = {name: value / total for name, value in effective.items()}
        weights[priority] = effective
    return weights
```

- [ ] **Step 4: Run the new tests and the full suite**

Run: `rtk pytest -q`
Expected: all pass.

- [ ] **Step 5: Lint and commit**

Run: `rtk ruff check .`

```bash
git add model_compare.py test_model_compare.py
git commit -m "Add model_family and catalog_weights helpers for the catalog"
```

---

### Task 3: `build_catalog` + `print_catalog` + `--catalog` CLI

**Files:**
- Modify: `model_compare.py` (new constants + `build_catalog`/`print_catalog` in the Output section; `parse_args`; `run`)
- Test: `test_model_compare.py` (new catalog section)
- Modify: `test_model_compare.py` helper `make_args` — add `catalog=False` to the base dict (needed by `run()`-level tests; harmless to existing tests).

**Interfaces:**
- Consumes: Task 1 (`filtered_out` collection, additive candidate keys), Task 2 (`model_family`, `catalog_weights`), existing `PROVIDER_PREFIX_RE`, `has_discount`, `PRIORITY_WEIGHTS`, `opencode_model_id` (unused here), `compute_scores`-produced fields `price_score`/`quality_score`/`context_score`/`age_score`/`quality`/`score`.
- Produces:
  - `CATALOG_SCHEMA_VERSION = 1` (module constant).
  - `CATALOG_DROP_REASONS` tuple (the 11 verbatim keys, zero-filled into `pool.dropped`).
  - `build_catalog(args, models, candidates, dropped, filtered, discounts, quality_by_id, aa_source) -> dict` — pure; never mutates `candidates`.
  - `print_catalog(document) -> None` — `json.dumps(document, indent=2)` to stdout.
  - `parse_args` accepts `--catalog`, rejects `--catalog` with `--best`/`--json` (exit 2), `--top` accepted but ignored for catalogs.
  - `run()` prints the catalog and returns 0 when `args.catalog`.

- [ ] **Step 1: Add `catalog=False` to `make_args` and the datetime import**

In `test_model_compare.py`, add one line to the `base` dict in `make_args` (after `json=False,`) — needed by the `run()`-level tests below; harmless to existing tests:

```python
        catalog=False,
```

Also add `from datetime import datetime` to the test-file imports (the envelope test parses `generated_at` with it).

- [ ] **Step 2: Write the failing tests**

Append a new section to `test_model_compare.py`:

```python
# ---------------------------------------------------------------------------
# catalog document
# ---------------------------------------------------------------------------


CATALOG_ENTRY_KEYS = {
    "id", "name", "provider", "family", "pricing", "context", "listed_at",
    "age_days", "tool_calling", "zdr", "discount", "expired", "quality",
    "quality_match", "scores",
}


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
        make_model(
            id="acme/small",
            name="Small",
            context_length=100,
        ),
    ]
    filtered = []
    candidates, dropped = mc.build_candidates(
        models, args, {"acme/model-a": 0.5}, {"acme/model-a", "acme/model-b"}, filtered
    )
    quality_by_id = {"acme/model-b": 68.4}
    mc.compute_scores(candidates, args, quality_by_id)
    return args, models, candidates, dropped, filtered, {"acme/model-a": 0.5}, quality_by_id


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
        assert set(entry["pricing"]) == {"input_per_1m", "output_per_1m", "blended_per_1m"}
        assert set(entry["scores"]) == {"price", "quality", "context", "age", "overall"}
        assert set(entry["scores"]["overall"]) == {"balanced", "price", "quality"}
    a = next(e for e in doc["models"] if e["id"] == "acme/model-a")
    assert a["name"] == "Model A"  # vendor prefix stripped
    assert a["provider"] == "acme"
    assert a["family"] == "model"
    assert a["pricing"] == {"input_per_1m": 1.0, "output_per_1m": 2.0, "blended_per_1m": 1.25}
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


def test_catalog_family_null_without_prefix():
    doc = build_doc()
    # no model in the default pool exercises the None path; check directly
    assert mc.model_family("kimi/k2") is None


def test_catalog_overall_covers_all_priorities():
    args, _, candidates, _, _, _, quality_by_id = catalog_pool()
    doc = mc.build_catalog(args, [], candidates, {}, [], {}, quality_by_id, "AA API v2")
    by_id = {c["id"]: c for c in candidates}
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
        for e in sorted(doc["models"], key=lambda e: (-e["scores"]["overall"]["balanced"], e["id"]))
    ]


def test_catalog_deterministic_modulo_generated_at():
    doc1 = build_doc()
    doc2 = build_doc()
    doc1.pop("generated_at")
    doc2.pop("generated_at")
    assert doc1 == doc2


def test_catalog_aa_mode_mapping():
    assert build_doc(aa_source="AA page scrape")["sources"]["aa"]["mode"] == "scrape"
    assert build_doc(aa_source=None)["sources"]["aa"]["mode"] == "none"
    doc = build_doc(aa_source=None)
    assert all(e["quality_match"] is None and e["quality"] is None for e in doc["models"])


def test_catalog_no_zdr_marks_skipped(monkeypatch, capsys):
    monkeypatch.setattr(mc, "fetch_discount_map", lambda args: ({}, False))
    monkeypatch.setattr(mc, "fetch_zdr_set", lambda args: (set(), False))
    monkeypatch.setattr(mc, "fetch_aa_entries", lambda args: ([], None, False))
    models = [
        make_model(id="acme/model-a", created=1_700_000_000),
        make_model(id="acme/model-b", pricing={"prompt": "0.000002", "completion": "0.000004"}),
    ]
    args = make_args(min_context=0, no_zdr=True)
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
    args = make_args(min_context=0)
    assert mc.run(args, models, False) == 1
    captured = capsys.readouterr()
    assert captured.out == ""  # no document
    assert "ZDR" in captured.err


def test_catalog_discounts_unavailable_source():
    args, models, candidates, dropped, filtered, _discounts, quality_by_id = catalog_pool()
    doc = mc.build_catalog(args, models, candidates, dropped, filtered, {}, quality_by_id, None)
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
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest test_model_compare.py -q -k "catalog"`
Expected: FAIL with `AttributeError: module 'model_compare' has no attribute 'build_catalog'` (and `CATALOG_DROP_REASONS`).

- [ ] **Step 4: Implement the catalog builder and printer**

Add near the top of `model_compare.py`, right after `PRIORITY_WEIGHTS`:

```python
CATALOG_SCHEMA_VERSION = 1

# The drop-reason keys build_candidates counts, verbatim -- keep in sync with
# its drop() call sites. pool.dropped lists all of them zero-filled so the
# contract is self-describing.
CATALOG_DROP_REASONS = (
    "malformed id",
    "context",
    "pricing",
    "free",
    "batch",
    "no discount",
    "not ZDR",
    "modality",
    "tool calling",
    "expired",
    "age",
)
```

Add to the Output section (after `print_json`):

```python
def build_catalog(args, models, candidates, dropped, filtered, discounts,
                  quality_by_id, aa_source):
    """Assemble the full-catalog document (see README "Catalog output").

    Pure: reads candidates post-compute_scores (which already carry the
    component scores) and never mutates them. Deterministic apart from
    generated_at: models sort by (-overall.balanced, id), filtered by id,
    and age uses date precision so two runs in the same UTC day match.
    """
    now = datetime.now(timezone.utc)
    weights = catalog_weights(candidates, quality_by_id)
    aa_mode = {"AA API v2": "api", "AA page scrape": "scrape"}.get(aa_source, "none")
    quality_match = {"AA API v2": "api", "AA page scrape": "scrape"}.get(aa_source)

    entries = []
    for cand in candidates:
        provider, _, _base = cand["id"].partition("/")
        created = cand["created"] or 0.0
        listed_date = (
            datetime.fromtimestamp(created, tz=timezone.utc).date()
            if created > 0
            else None
        )
        overall = {}
        for priority in PRIORITY_WEIGHTS:
            w = weights[priority]
            overall[priority] = round(
                w.get("quality", 0.0) * cand["quality_score"]
                + w.get("price", 0.0) * cand["price_score"]
                + w.get("context", 0.0) * cand["context_score"]
                + w.get("age", 0.0) * cand["age_score"],
                4,
            )
        entries.append(
            {
                "id": cand["id"],
                "name": PROVIDER_PREFIX_RE.sub("", cand["name"]).strip() or cand["name"],
                "provider": provider,
                "family": model_family(cand["id"]),
                "pricing": {
                    "input_per_1m": round(cand["price_in"], 6),
                    "output_per_1m": round(cand["price_out"], 6),
                    "blended_per_1m": round(cand["blended"], 6),
                },
                "context": cand["context"],
                "listed_at": listed_date.isoformat() if listed_date else None,
                "age_days": (now.date() - listed_date).days if listed_date else None,
                "tool_calling": cand["tool_calling"],
                "zdr": cand["zdr"],
                "discount": round(cand["discount"], 4)
                if has_discount(cand["discount"])
                else None,
                "expired": cand["expired"],
                "quality": cand["quality"],
                "quality_match": quality_match if cand["id"] in quality_by_id else None,
                "scores": {
                    "price": round(cand["price_score"], 4),
                    "quality": round(cand["quality_score"], 4),
                    "context": round(cand["context_score"], 4),
                    "age": round(cand["age_score"], 4),
                    "overall": overall,
                },
            }
        )
    entries.sort(key=lambda e: (-e["scores"]["overall"]["balanced"], e["id"]))

    return {
        "schema_version": CATALOG_SCHEMA_VERSION,
        "tool": "model-compare",
        "generated_at": now.isoformat(timespec="seconds"),
        "parameters": {
            "input_share": args.input_share,
            "quality_ref": args.quality_ref,
            "min_context": args.min_context,
            "recency_half_life": args.recency_half_life,
            "max_age_days": args.max_age_days,
            "zdr_required": not args.no_zdr,
            "require_tools": not args.no_require_tools,
            "exclude_free": args.exclude_free,
            "include_batch": args.include_batch,
            "weights": weights,
        },
        "sources": {
            "openrouter": "ok",
            "aa": {"mode": aa_mode, "matched": len(quality_by_id)},
            "zdr": "skipped" if args.no_zdr else "ok",
            "discounts": "ok" if discounts else "unavailable",
        },
        "pool": {
            "listed": len(models),
            "candidates": len(candidates),
            "dropped": {
                reason: dropped.get(reason, 0) for reason in CATALOG_DROP_REASONS
            },
        },
        "models": entries,
        "filtered": sorted(filtered, key=lambda e: e["id"]),
    }


def print_catalog(document):
    print(json.dumps(document, indent=2))
```

- [ ] **Step 5: Wire the CLI**

In `parse_args`, after the `--json` argument, add:

```python
    parser.add_argument(
        "--catalog",
        action="store_true",
        help="print the full model catalog (ranked candidates and filtered-out models with reasons) as one JSON document",
    )
```

After the existing `args = parser.parse_args(argv)` validations (before `return args`), add:

```python
    if args.catalog and (args.best or args.json):
        parser.error("--catalog cannot be combined with --best or --json")
```

In `run()`, replace:

```python
    candidates, dropped = build_candidates(models, args, discounts, zdr_ids)
```

with:

```python
    filtered = []
    candidates, dropped = build_candidates(models, args, discounts, zdr_ids, filtered)
```

and insert after `weights = compute_scores(candidates, args, quality_by_id)` (before `limit = 1 if args.best else args.top`):

```python
    if args.catalog:
        print_catalog(
            build_catalog(
                args, models, candidates, dropped, filtered, discounts,
                quality_by_id, aa_source,
            )
        )
        return 0
```

(`--top` is ignored for catalogs because the catalog branch returns before `limit` is applied; the ZDR fail-closed and no-candidates paths sit upstream of this and are inherited unchanged.)

- [ ] **Step 6: Run the full suite**

Run: `rtk pytest -q`
Expected: all pass (new catalog tests + 113 prior).

- [ ] **Step 7: Lint and commit**

Run: `rtk ruff check .`

```bash
git add model_compare.py test_model_compare.py
git commit -m "Add --catalog full-catalog output with deterministic schema"
```

---

### Task 4: `build_site_data.py --catalog-file` validation

**Files:**
- Modify: `build_site_data.py` (new `validate_catalog` + `--catalog-file` in `main`; add `import os`)
- Test: `test_build_site_data.py`

**Interfaces:**
- Consumes: Task 3's document shape (fields, `schema_version` 1, `zdr_required`, `pool.candidates`, `scores` ranges).
- Produces: `validate_catalog(document) -> None` (raises `ValueError` on any breach); `main(argv)` gains `--catalog-file FILE` — validates the raw `--catalog` output **before** anything is written, then writes `catalog.json` as a sibling of `--output`; omitted flag → no catalog written (current invocation keeps working). Exit 0/1 semantics unchanged.

- [ ] **Step 1: Write the failing tests**

Append to `test_build_site_data.py`:

```python
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
        "quality_match": "api",
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
            "aa": {"mode": "api", "matched": 1},
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


@pytest.mark.parametrize(
    "mutate",
    [
        lambda doc: doc.update(schema_version=99),
        lambda doc: doc.update(tool="other-tool"),
        lambda doc: doc.pop("parameters"),
        lambda doc: doc["pool"].update(candidates=5),
        lambda doc: doc["models"][0].pop("family"),
        lambda doc: doc["models"][0].pop("scores"),
        lambda doc: doc["models"][0]["scores"].update(price=1.5),
        lambda doc: doc["models"][0]["scores"].update(overall={"balanced": 0.5}),
        lambda doc: doc["models"][0].update(zdr=False),
        lambda doc: doc["filtered"].append({"id": "x/y", "name": "X", "reasons": []}),
        lambda doc: doc["models"].append("not-a-dict"),
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest test_build_site_data.py -q -k "catalog"`
Expected: FAIL with `AttributeError: module 'build_site_data' has no attribute 'validate_catalog'`.

- [ ] **Step 3: Implement**

In `build_site_data.py`, add `import os` to the imports and insert after the `MODEL_ID_RE` block:

```python
CATALOG_SCHEMA_VERSION = 1

CATALOG_ENTRY_KEYS = (
    "id", "name", "provider", "family", "pricing", "context", "listed_at",
    "age_days", "tool_calling", "zdr", "discount", "expired", "quality",
    "quality_match", "scores",
)
CATALOG_PRICING_KEYS = ("input_per_1m", "output_per_1m", "blended_per_1m")
CATALOG_SCORE_KEYS = ("price", "quality", "context", "age")
CATALOG_OVERALL_KEYS = ("balanced", "price", "quality")


def _is_score(value) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and 0.0 <= value <= 1.0
    )


def validate_catalog(document) -> None:
    """Validate a raw model_compare.py --catalog document.

    Raises ValueError on any breach so a broken run fails the build and
    never deploys. The contract is additive-only: unknown schema_version
    or missing fields fail here rather than downstream.
    """
    if not isinstance(document, dict):
        raise ValueError("catalog document is not an object")
    if document.get("schema_version") != CATALOG_SCHEMA_VERSION:
        raise ValueError(
            f"unknown catalog schema_version: {document.get('schema_version')!r}"
        )
    if document.get("tool") != "model-compare":
        raise ValueError(f"unexpected catalog tool: {document.get('tool')!r}")
    for key in ("generated_at", "parameters", "sources", "pool", "models", "filtered"):
        if key not in document:
            raise ValueError(f"catalog missing key: {key}")
    parameters = document["parameters"]
    if not isinstance(parameters, dict) or "zdr_required" not in parameters:
        raise ValueError("catalog parameters must set zdr_required")
    pool = document["pool"]
    if not isinstance(pool, dict) or not isinstance(pool.get("candidates"), int):
        raise ValueError("catalog pool.candidates must be an integer")
    models = document["models"]
    if not isinstance(models, list):
        raise ValueError("catalog models must be a list")
    if len(models) != pool["candidates"]:
        raise ValueError(
            f"catalog has {len(models)} models but pool.candidates is {pool['candidates']}"
        )
    for i, entry in enumerate(models):
        if not isinstance(entry, dict):
            raise ValueError(f"models[{i}] is not an object")
        missing = [key for key in CATALOG_ENTRY_KEYS if key not in entry]
        if missing:
            raise ValueError(f"models[{i}] is missing keys: {', '.join(missing)}")
        pricing = entry["pricing"]
        if not isinstance(pricing, dict) or any(
            key not in pricing for key in CATALOG_PRICING_KEYS
        ):
            raise ValueError(f"models[{i}] pricing is incomplete")
        scores = entry["scores"]
        if not isinstance(scores, dict):
            raise ValueError(f"models[{i}] scores is not an object")
        bad = [key for key in CATALOG_SCORE_KEYS if not _is_score(scores.get(key))]
        if bad:
            raise ValueError(f"models[{i}] scores out of range: {', '.join(bad)}")
        overall = scores.get("overall")
        if not isinstance(overall, dict) or any(
            key not in overall for key in CATALOG_OVERALL_KEYS
        ):
            raise ValueError(f"models[{i}] scores.overall is incomplete")
        bad_overall = [
            key for key in CATALOG_OVERALL_KEYS if not _is_score(overall[key])
        ]
        if bad_overall:
            raise ValueError(
                f"models[{i}] scores.overall out of range: {', '.join(bad_overall)}"
            )
        if parameters["zdr_required"] and entry["zdr"] is not True:
            raise ValueError(f"models[{i}] is not zdr=true under zdr_required")
    for i, entry in enumerate(document["filtered"]):
        if not isinstance(entry, dict):
            raise ValueError(f"filtered[{i}] is not an object")
        if not isinstance(entry.get("id"), str) or not entry["id"]:
            raise ValueError(f"filtered[{i}] has no id")
        if not isinstance(entry.get("reasons"), list) or not entry["reasons"]:
            raise ValueError(f"filtered[{i}] has no reasons")
```

Add to `main()`'s argument parser (after `--output`):

```python
    parser.add_argument(
        "--catalog-file",
        help="raw model_compare.py --catalog output; validated and written as catalog.json next to --output",
    )
```

In `main()`, immediately after `args = parser.parse_args(argv)` (before the existing `try:`), insert validation-first:

```python
    catalog = None
    if args.catalog_file:
        try:
            with open(args.catalog_file) as fh:
                catalog = json.load(fh)
            validate_catalog(catalog)
        except (OSError, ValueError) as exc:
            print(f"error: invalid catalog: {exc}", file=sys.stderr)
            return 1
```

(`json.JSONDecodeError` subclasses `ValueError`, so unparseable JSON lands here too.)

Then, after the existing `json.dump(data, ...)` / `fh.write("\n")` block and before `return 0`, add:

```python
    if catalog is not None:
        catalog_path = os.path.join(
            os.path.dirname(os.path.abspath(args.output)), "catalog.json"
        )
        try:
            with open(catalog_path, "w") as fh:
                json.dump(catalog, fh, indent=2)
                fh.write("\n")
        except OSError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
```

- [ ] **Step 4: Run the full suite**

Run: `rtk pytest -q`
Expected: all pass.

- [ ] **Step 5: Lint and commit**

Run: `rtk ruff check .`

```bash
git add build_site_data.py test_build_site_data.py
git commit -m "Validate and publish catalog.json in the site data builder"
```

---

### Task 5: Publish workflow, README, and docstring

**Files:**
- Modify: `.github/workflows/publish.yml` (Build site step)
- Modify: `README.md` (Options table + new "Catalog output" section + published-artifacts mention)
- Modify: `model_compare.py` (module docstring Examples block)

**Interfaces:**
- Consumes: `model_compare.py --catalog` (Task 3), `build_site_data.py --catalog-file` (Task 4).
- Produces: deployed `catalog.json` next to `data.json`; README documents the contract.

- [ ] **Step 1: Update the publish workflow**

In `.github/workflows/publish.yml`, the build step currently reads:

```yaml
          python build_site_data.py \
            --best-file best.txt \
            --output _site/data.json \
            --priority balanced=balanced.json \
            --priority price=price.json \
            --priority quality=quality.json
```

Change it to:

```yaml
          python model_compare.py --catalog > catalog.json
          python build_site_data.py \
            --best-file best.txt \
            --output _site/data.json \
            --catalog-file catalog.json \
            --priority balanced=balanced.json \
            --priority price=price.json \
            --priority quality=quality.json
```

`catalog.json` (validated, post-`build_site_data`) lands next to `_site/data.json` and deploys with the Pages artifact; the raw file stays in the runner's workspace. No `AA_API_KEY` → scrape fallback applies and `sources.aa.mode` records it.

- [ ] **Step 2: Update the module docstring**

In `model_compare.py`, in the docstring's `Examples:` block, add one line after the `--discount` example:

```python
  model_compare.py --catalog                full catalog (ranked + filtered) as one JSON document
```

- [ ] **Step 3: Update README.md**

Three edits:

1. In the **Team usage: published picks** section, after the sentence about `best.txt`, add:

```markdown
The same workflow publishes a [`catalog.json`](https://canonical.github.io/model-compare/catalog.json) artifact — see [Catalog output](#catalog-output).
```

2. Add a new `## Catalog output` section between **Zero data retention** and **Options**:

```markdown
## Catalog output

`--catalog` prints the full evaluation as one machine-readable JSON document:
every surviving candidate, ranked, plus every filtered-out model with its drop
reasons. The published site serves it as
[`catalog.json`](https://canonical.github.io/model-compare/catalog.json),
refreshed on the same 6-hour schedule as the picks.

```console
$ ./model_compare.py --catalog | python3 -m json.tool
```

The document is a **stable contract** consumed by internal Canonical tooling
(`tokens.canonical.com`), which deduplicates on content — same inputs produce
byte-identical output apart from `generated_at` (`age_days`/`listed_at` use
UTC date precision, so runs within a day match exactly). `schema_version`
starts at `1`: fields may be added without notice, but renaming or removing
one bumps the version.

Top level: `schema_version`, `tool`, `generated_at`, `parameters` (all knobs
plus the **effective** per-priority `weights` — reproducing `scores.overall`
needs nothing else), `sources` (`openrouter`, `aa` with `mode` `api`/`scrape`/`none`
and the match count, `zdr` `ok`/`skipped`, `discounts`), `pool` (`listed`,
`candidates`, `dropped`), `models`, `filtered`.

Each `models` entry carries: `id` (bare `provider/model`), `name`,
`provider`, `family` (heuristic: leading token of the slug, e.g. `glm-5.3`
→ `glm`; `null` when there is none), `pricing` (`input_per_1m`,
`output_per_1m`, `blended_per_1m` in USD per 1M tokens), `context`,
`listed_at`, `age_days`, `tool_calling`, `zdr`, `discount`, `expired`,
`quality` (AA intelligence index or `null`), `quality_match`
(`api`/`scrape`/`null`) and `scores` — the four component scores plus
`overall` for all three priorities, so downstream consumers never re-run
the scorer.

`filtered` entries are `{"id", "name", "reasons"}`; the reason keys are the
same strings the tool counts internally:

```
malformed id, context, pricing, free, batch, no discount, not ZDR,
modality, tool calling, expired, age
```

`pool.dropped` lists all of them zero-filled. `--top` is ignored with
`--catalog` (the document always covers the full pool); `--catalog`
cannot be combined with `--best` or `--json`.
```

3. Add one row to the **Options** table (after the `--json` row):

```markdown
| `--catalog` | off | print the full model catalog (ranked candidates + filtered, with reasons) as one JSON document |
```

- [ ] **Step 4: Verify docs render sensibly and run the suite**

Run: `rtk pytest -q && rtk ruff check .`
Expected: all pass, no lint issues.

- [ ] **Step 5: Commit**

```bash
git add .github/workflows/publish.yml README.md model_compare.py
git commit -m "Publish and document the catalog.json artifact"
```

---

### Task 6: Acceptance verification

**Files:**
- None (verification only; live network used, warm cache expected).

**Interfaces:**
- Consumes: everything built in Tasks 1-5.

- [ ] **Step 1: Schema and internal consistency**

Run:

```bash
./model_compare.py --catalog | python3 -m json.tool > /tmp/opencode/catalog-pretty.json && echo PARSED
python3 - <<'PY'
import json
doc = json.load(open("/tmp/opencode/catalog-pretty.json"))
assert doc["schema_version"] == 1 and doc["tool"] == "model-compare"
assert len(doc["models"]) == doc["pool"]["candidates"]
assert sum(doc["pool"]["dropped"].values()) + doc["pool"]["candidates"] == doc["pool"]["listed"]
assert all(0.0 <= e["scores"]["overall"][p] <= 1.0
           for e in doc["models"] for p in ("balanced", "price", "quality"))
assert doc["parameters"]["zdr_required"] and all(e["zdr"] is True for e in doc["models"])
print("CONSISTENT:", len(doc["models"]), "models,", doc["pool"]["listed"], "listed")
PY
```

Expected: `PARSED` then `CONSISTENT: …` (values vary with the live catalog).

- [ ] **Step 2: Determinism with a warm cache**

```bash
./model_compare.py --catalog > /tmp/opencode/catalog-run1.json
./model_compare.py --catalog > /tmp/opencode/catalog-run2.json
python3 - <<'PY'
import json
a = json.load(open("/tmp/opencode/catalog-run1.json"))
b = json.load(open("/tmp/opencode/catalog-run2.json"))
ga, gb = a.pop("generated_at"), b.pop("generated_at")
assert a == b, "documents differ beyond generated_at"
print("DETERMINISTIC apart from generated_at:", ga, "vs", gb)
PY
```

- [ ] **Step 3: Full suite and lint**

Run: `rtk pytest -q && rtk ruff check .`
Expected: all tests pass (existing + new), no lint issues.

- [ ] **Step 4: Report**

Summarize: commands run, outputs observed, any live-catalog surprises (e.g. new drop-reason keys appearing in `pool.dropped` beyond the 11 known ones would indicate drift — investigate before proceeding).

---

## Self-Review notes

- Spec coverage: CLI (Task 3), document shape (Task 3), determinism (Tasks 3+6), publication (Tasks 4+5), tests (Tasks 1-4, 6), acceptance (Task 6), README/docstring (Task 5). ZDR fail-closed test in Task 3.
- Type consistency: `build_catalog(args, models, candidates, dropped, filtered, discounts, quality_by_id, aa_source)` used identically in Task 3 wiring and tests; `catalog_weights(candidates, quality_by_id)` and `model_family(model_id)` from Task 2 consumed in Task 3; `validate_catalog(document)` from Task 4 consumed by Task 5's workflow only through the CLI flag.
- The one deliberate spec extension (`parameters.max_age_days`) is implemented in Task 3's envelope.
