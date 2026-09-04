# OpenRouter AA Benchmarks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consume OpenRouter's republished Artificial Analysis values (`data.benchmarks` in the frontend endpoint we already fetch) as the primary quality source — exact per-slug keys ending the proven fuzzy-mispairing bug — expose all three indices in the catalog document, and consolidate the frontend fetches to two loads.

**Architecture:** A pure `build_aa_benchmarks` map builder (dated-permaslug stripping, latest-date-wins) feeds a new `fetch_openrouter_frontend` that replaces `fetch_discount_map`/`fetch_zdr_set` with per-payload caches and decoupled failure semantics. Quality resolution becomes OR-first with exact-only AA fallback (`match_quality(allow_fuzzy=False)` default) via a pure `resolve_quality` helper. `build_catalog` gains the `aa` trio block and provenance fields; `validate_catalog` learns them.

**Tech Stack:** Python 3.10+ stdlib only (argparse, json, math, re, datetime), pytest.

**Worktree:** all work happens in `.worktrees/openrouter-aa-benchmarks` on branch `openrouter-aa-benchmarks`.

**Spec:** `docs/superpowers/specs/2026-09-02-openrouter-aa-benchmarks-design.md` (in this worktree) — the binding contract; consult for any detail not repeated here.

## Global Constraints

- Stdlib only, Python 3.10+, no new dependencies.
- `match_quality` gains `allow_fuzzy` **defaulting to `False`** — the fail-safe default; no fuzzy pairing may enter published output. Existing tests pinning fuzzy behavior pass `allow_fuzzy=True` explicitly.
- Failure semantics per source, decoupled as today: a base-URL (discounts/benchmarks) outage never triggers ZDR fail-closed; fail-closed exit 1 only when ZDR filtering is on and the ZDR data itself is unavailable; `--no-zdr` skips the ZDR fetch and touches no ZDR cache.
- Per-payload caches: `openrouter-frontend-discounts`, `openrouter-frontend-aa`, `openrouter-zdr` (reused unchanged); each written only when its payload is non-empty (never-cache-empty, self-healing); same TTL semantics (`--cache-ttl`, `--no-cache`).
- Warning texts preserved verbatim: `could not fetch discount data: {exc}`, `no discount entries found; treating discounts as unavailable`, `could not fetch ZDR data: {exc}`, `no ZDR entries found; treating ZDR data as unavailable`. New AA-benchmark warnings follow the same pattern: `could not fetch AA benchmark data: {exc}`, `no AA benchmark entries found; treating OpenRouter benchmarks as unavailable`.
- Catalog changes are additive: `schema_version` stays `1`; `quality_match` ∈ `"openrouter" | "api" | "scrape" | null`; `sources.aa` = `{"mode", "matched", "matched_openrouter"}` with `mode` ∈ `"openrouter" | "api" | "scrape" | "none"`.
- Non-finite guard everywhere: numbers accepted only when `isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)`.
- Existing behavior byte-identical except the intended quality corrections (spec §7): table/`--best`/`--json`/exit codes/cache-hit stderr notes all keep their shapes.
- Run `rtk pytest -q` (165 tests today) and `rtk ruff check .` after every task; both must be clean before committing. (`rtk pytest` can misreport counts — verify with `rtk proxy python -m pytest -q` when in doubt.)

---

### Task 1: `match_quality(allow_fuzzy=False)`

**Files:**
- Modify: `model_compare.py` (`match_quality`, lines ~477-506)
- Test: `test_model_compare.py` (match_quality section, lines ~868-891)

**Interfaces:**
- Consumes: existing `match_quality(model, exact, fuzzy)`; `build_aa_lookup(entries) -> (exact, fuzzy)` unchanged.
- Produces: `match_quality(model, exact, fuzzy, allow_fuzzy=False) -> float | None` — identical exact-tier behavior; fuzzy tier runs only when `allow_fuzzy=True`. Task 4's `resolve_quality` calls it without the flag.

- [ ] **Step 1: Write the failing tests**

In `test_model_compare.py`, update `test_match_quality_fuzzy_overlap` (line ~877) to pass the flag explicitly, and add a regression test after `test_match_quality_no_match_returns_none`:

```python
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


def test_match_quality_fuzzy_disabled_by_default():
    # Regression: the scrape+fuzzy path paired z-ai/glm-5.3-flash with the
    # single AA entry glm-5-3 (Jaccard 0.6 >= 0.5). Exact-only must refuse.
    entries = [{"key": "glm-5-3", "name": "GLM-5.3 (max)", "index": 59.5}]
    exact, fuzzy = mc.build_aa_lookup(entries)
    model = {"id": "z-ai/glm-5.3-flash", "name": "Z.AI: GLM 5.3 Flash"}
    assert mc.match_quality(model, exact, fuzzy) is None
    assert mc.match_quality(model, exact, fuzzy, allow_fuzzy=True) == 59.5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest test_model_compare.py -q -k "match_quality"`
Expected: FAIL — `test_match_quality_fuzzy_disabled_by_default` returns 59.5 on the default call (TypeError-free: the current signature ignores extra kwargs? No — it will raise `TypeError: match_quality() got an unexpected keyword argument 'allow_fuzzy'` in the `allow_fuzzy=True` call of the updated overlap test).

- [ ] **Step 3: Implement**

In `match_quality`, change the signature and gate the fuzzy block:

```python
def match_quality(model, exact, fuzzy, allow_fuzzy=False):
    model_id = model["id"]
    name = model.get("name") or ""
    display = PROVIDER_PREFIX_RE.sub("", name).strip()
    # Assumes model_id contains a "/" (provider/base). build_candidates drops
    # malformed ids upstream, so base is never empty here.
    _, _, base = model_id.split(":", 1)[0].partition("/")

    for tier, raw in (
        (0, model_id),
        (1, base),
        (2, display),
        (3, PAREN_RE.sub(" ", display)),
    ):
        normalized = norm_key(raw)
        if normalized and (tier, normalized) in exact:
            return exact[(tier, normalized)]

    if not allow_fuzzy:
        return None
    own = set(norm_key(base).split()) | set(norm_key(display).split())
    if own:
        best = 0.0
        best_index = None
        for tokens, index in fuzzy:
            overlap = len(own & tokens) / len(own | tokens)
            if overlap > best:
                best = overlap
                best_index = index
        if best >= 0.5 and best_index is not None:
            return best_index
    return None
```

- [ ] **Step 4: Run the full suite**

Run: `rtk pytest -q`
Expected: all 165 pass (the other match_quality tests use exact tiers and are unaffected by the new default).

- [ ] **Step 5: Lint and commit**

Run: `rtk ruff check .`

```bash
git add model_compare.py test_model_compare.py
git commit -m "Gate match_quality fuzzy tier behind allow_fuzzy, default off"
```

---

### Task 2: `base_model_id` + `build_aa_benchmarks` helpers

**Files:**
- Modify: `model_compare.py` (new functions after `match_quality`, before `model_family`)
- Test: `test_model_compare.py` (new section after the match_quality tests)

**Interfaces:**
- Consumes: nothing new (`re`, `math` already imported).
- Produces:
  - `base_model_id(model_id: str) -> str` — candidate id without its `:variant` suffix.
  - `build_aa_benchmarks(benchmarks) -> dict` — `{bare_openrouter_id: {"intelligence_index": f, "coding_index": f, "agentic_index": f}}`; entries with at least one valid field are kept. Task 3 (fetch) and Task 5 (catalog) consume both.

- [ ] **Step 1: Write the failing tests**

```python
# ---------------------------------------------------------------------------
# OR-published AA benchmarks (build_aa_benchmarks)
# ---------------------------------------------------------------------------


def test_build_aa_benchmaps_strips_dated_permaslugs():
    benchmarks = {
        "z-ai/glm-5.3-flash-20260826": {
            "aa": {"intelligence_index": 57.5, "coding_index": 71.5, "agentic_index": 58.2}
        },
        "acme/plain": {"aa": {"intelligence_index": 40}},
    }
    aa = mc.build_aa_benchmarks(benchmarks)
    assert aa["z-ai/glm-5.3-flash"] == {
        "intelligence_index": 57.5, "coding_index": 71.5, "agentic_index": 58.2
    }
    assert aa["acme/plain"] == {"intelligence_index": 40.0}


def test_build_aa_benchmarks_latest_date_wins():
    benchmarks = {
        "acme/m-20260801": {"aa": {"intelligence_index": 10}},
        "acme/m-20260820": {"aa": {"intelligence_index": 20}},
        "acme/m": {"aa": {"intelligence_index": 30}},
    }
    assert mc.build_aa_benchmarks(benchmarks)["acme/m"] == {"intelligence_index": 20.0}


def test_build_aa_benchmarks_skips_invalid_values_but_keeps_valid_ones():
    benchmarks = {
        "acme/partial": {
            "aa": {"intelligence_index": float("nan"), "coding_index": 70, "agentic_index": True}
        },
        "acme/no-aa": {"da": {"default_elo": 1300}},
        "acme/not-a-node": "garbage",
    }
    aa = mc.build_aa_benchmarks(benchmarks)
    assert aa == {"acme/partial": {"coding_index": 70.0}}


def test_build_aa_benchmarks_empty_inputs():
    assert mc.build_aa_benchmarks({}) == {}
    assert mc.build_aa_benchmarks(None) == {}


def test_base_model_id_strips_variant():
    assert mc.base_model_id("z-ai/glm-5.3-flash:free") == "z-ai/glm-5.3-flash"
    assert mc.base_model_id("z-ai/glm-5.3") == "z-ai/glm-5.3"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest test_model_compare.py -q -k "aa_benchmarks or base_model_id"`
Expected: FAIL with `AttributeError: module 'model_compare' has no attribute 'build_aa_benchmarks'`.

- [ ] **Step 3: Implement**

Insert after `match_quality`, before `model_family`:

```python
def base_model_id(model_id: str) -> str:
    """Candidate id without its :variant suffix (OR benchmarks are per base model)."""
    return model_id.split(":", 1)[0]


AA_BENCHMARK_FIELDS = ("intelligence_index", "coding_index", "agentic_index")


def build_aa_benchmarks(benchmarks) -> dict:
    """Map bare OpenRouter id -> OR-published AA trio from data.benchmarks.

    Keys are dated permaslugs (z-ai/glm-5.3-flash-20260826): the trailing
    -YYYYMMDD is stripped and undated keys map to themselves; when two keys
    strip to the same id the latest date wins. Values failing the
    finite-number guard are treated as absent; entries with at least one
    valid field are kept.
    """
    aa_by_id = {}
    seen_date = {}

    def accept(bare, date, trio):
        if bare in seen_date and seen_date[bare] >= date:
            return
        seen_date[bare] = date
        aa_by_id[bare] = trio

    for key, node in (benchmarks or {}).items():
        aa = node.get("aa") if isinstance(node, dict) else None
        if not isinstance(aa, dict):
            continue
        trio = {}
        for field in AA_BENCHMARK_FIELDS:
            value = aa.get(field)
            if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
            ):
                trio[field] = float(value)
        if not trio:
            continue
        match = re.match(r"^(.*)-(\d{8})$", key)
        if match:
            accept(match.group(1), match.group(2), trio)
        else:
            accept(key, "", trio)
    return aa_by_id
```

- [ ] **Step 4: Run the full suite**

Run: `rtk pytest -q`
Expected: all pass.

- [ ] **Step 5: Lint and commit**

Run: `rtk ruff check .`

```bash
git add model_compare.py test_model_compare.py
git commit -m "Add OR AA-benchmarks map builder and base id helper"
```

---

### Task 3: `fetch_openrouter_frontend` + run() fetch rewiring + test migration

**Files:**
- Modify: `model_compare.py` (remove `fetch_discount_map` lines 202-241 and `fetch_zdr_set` lines 244-279; add `fetch_openrouter_frontend` in their place; rewire `run()` lines 1079-1080 and the source_note block lines 1146-1159)
- Test: `test_model_compare.py` (rewrite the 13 fetch tests + 3 `run()` seams)

**Interfaces:**
- Consumes: Task 2 (`build_aa_benchmarks`), existing `load_cache`/`save_cache`/`fetch_json`, constants `OPENROUTER_DISCOUNTS_URL`, `OPENROUTER_ZDR_URL`.
- Produces: `fetch_openrouter_frontend(args) -> (discounts: dict, zdr_ids: set, aa_by_id: dict, cache_hits: set)` where `cache_hits ⊆ {"discounts", "zdr", "aa"}`. Task 4 consumes `aa_by_id`; Task 5's catalog consumes both via `run()`.

- [ ] **Step 1: Rewrite the fetch tests (replace the 6 `test_fetch_discount_map_*` bodies, the 7 `test_fetch_zdr_set_*` bodies, and the 3 `run()` seams)**

Replace `test_fetch_discount_map_builds_variant_aware_keys` with:

```python
def frontend_payload():
    """Base-URL payload with discounts and benchmarks in one response."""
    return {
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
                    "endpoint": {"variant": "batch", "pricing": {"discount": 0.75}},
                },
                {
                    "slug": "~acme/private",
                    "endpoint": {"variant": "standard", "pricing": {"discount": 0.9}},
                },
                {"slug": "acme/no-endpoint", "endpoint": None},
                {
                    "slug": "acme/no-discount-field",
                    "endpoint": {"variant": "standard", "pricing": {}},
                },
            ],
            "benchmarks": {
                "acme/a-20260826": {
                    "aa": {"intelligence_index": 57.5, "coding_index": 71.5, "agentic_index": 58.2}
                },
            },
        }
    }


def zdr_payload():
    return {
        "data": {
            "models": [
                {"slug": "acme/a", "endpoint": {"variant": "standard", "pricing": {"prompt": "0.1"}}},
                {"slug": "acme/b", "endpoint": {"variant": "batch", "pricing": {}}},
                {"slug": "~acme/private", "endpoint": {"variant": "standard"}},
                {"slug": "acme/no-endpoint", "endpoint": None},
            ]
        }
    }


def stub_frontend(monkeypatch, calls, base_payload, zdr_payload=None):
    """URL-dispatching fetch_json stub; zdr_payload None means the ZDR URL raises."""

    def fake_fetch(url, *a, **k):
        calls.append(url)
        if url == mc.OPENROUTER_ZDR_URL:
            if zdr_payload is None:
                raise RuntimeError("zdr down")
            return zdr_payload
        if isinstance(base_payload, Exception):
            raise base_payload
        return base_payload

    monkeypatch.setattr(mc, "fetch_json", fake_fetch)


def test_fetch_openrouter_frontend_derives_all_payloads(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    calls = []
    stub_frontend(monkeypatch, calls, frontend_payload(), zdr_payload())
    discounts, zdr_ids, aa_by_id, cache_hits = mc.fetch_openrouter_frontend(
        make_args(no_cache=True)
    )
    assert discounts == {"acme/a": 0.5, "acme/b:free": 0.0, "acme/c:batch": 0.75}
    assert zdr_ids == {"acme/a", "acme/b:batch", "acme/no-endpoint"}
    assert aa_by_id == {
        "acme/a": {"intelligence_index": 57.5, "coding_index": 71.5, "agentic_index": 58.2}
    }
    assert cache_hits == set()
    assert mc.OPENROUTER_DISCOUNTS_URL in calls
    assert mc.OPENROUTER_ZDR_URL in calls
```

Replace `test_fetch_discount_map_caches_result` + `test_fetch_zdr_set_caches_result` with:

```python
def test_fetch_openrouter_frontend_caches_per_payload(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    calls = []
    stub_frontend(monkeypatch, calls, frontend_payload(), zdr_payload())
    args = make_args(no_cache=False, cache_ttl=3600)
    first = mc.fetch_openrouter_frontend(args)
    second = mc.fetch_openrouter_frontend(args)
    assert len(calls) == 2  # base + zdr once each; second run fully cached
    assert first[3] == set()
    assert second[3] == {"discounts", "aa", "zdr"}
    assert first[:3] == second[:3]
```

Replace `test_fetch_discount_map_fetch_failure_returns_empty` + `test_fetch_zdr_set_fetch_failure_returns_empty` with:

```python
def test_fetch_openrouter_frontend_base_failure_keeps_zdr_decoupled(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    calls = []
    stub_frontend(monkeypatch, calls, RuntimeError("network down"), zdr_payload())
    discounts, zdr_ids, aa_by_id, cache_hits = mc.fetch_openrouter_frontend(
        make_args(no_cache=True)
    )
    assert discounts == {}
    assert aa_by_id == {}
    assert zdr_ids == {"acme/a", "acme/b:batch", "acme/no-endpoint"}
    assert cache_hits == set()
```

Replace `test_fetch_discount_map_does_not_cache_empty` + `test_fetch_zdr_set_does_not_cache_empty` with:

```python
def test_fetch_openrouter_frontend_does_not_cache_empty(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    calls = []
    stub_frontend(monkeypatch, calls, {"data": {"models": []}}, {"data": {"models": []}})
    args = make_args(no_cache=False, cache_ttl=3600)
    first = mc.fetch_openrouter_frontend(args)
    second = mc.fetch_openrouter_frontend(args)
    assert first[:3] == ({}, set(), {})
    assert second[:3] == ({}, set(), {})
    assert len(calls) == 4  # 2 per run (base + zdr); nothing cacheable
    cache_dir = tmp_path / "model-compare"
    assert not (cache_dir / "openrouter-frontend-discounts.json").exists()
    assert not (cache_dir / "openrouter-frontend-aa.json").exists()
    assert not (cache_dir / "openrouter-zdr.json").exists()
```

Replace `test_fetch_discount_map_treats_empty_cached_map_as_miss` with:

```python
def test_fetch_openrouter_frontend_treats_empty_cached_payloads_as_miss(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    calls = []
    stub_frontend(monkeypatch, calls, frontend_payload(), zdr_payload())
    cache_dir = tmp_path / "model-compare"
    cache_dir.mkdir(parents=True)
    now = time.time()
    (cache_dir / "openrouter-frontend-discounts.json").write_text(
        json.dumps({"fetched_at": now, "payload": {}})
    )
    (cache_dir / "openrouter-frontend-aa.json").write_text(
        json.dumps({"fetched_at": now, "payload": {}})
    )
    (cache_dir / "openrouter-zdr.json").write_text(
        json.dumps({"fetched_at": now, "payload": []})
    )
    discounts, zdr_ids, aa_by_id, cache_hits = mc.fetch_openrouter_frontend(
        make_args(no_cache=False, cache_ttl=3600)
    )
    assert len(calls) == 2  # all three payloads empty in cache -> both fetches
    assert discounts == {"acme/a": 0.5, "acme/b:free": 0.0, "acme/c:batch": 0.75}
    assert zdr_ids == {"acme/a", "acme/b:batch", "acme/no-endpoint"}
    assert cache_hits == set()
```

Replace `test_fetch_zdr_set_skips_fetch_when_disabled` with:

```python
def test_fetch_openrouter_frontend_no_zdr_skips_zdr_fetch_but_still_loads_base(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    calls = []
    stub_frontend(monkeypatch, calls, frontend_payload(), zdr_payload())
    discounts, zdr_ids, aa_by_id, cache_hits = mc.fetch_openrouter_frontend(
        make_args(no_cache=True, no_zdr=True)
    )
    assert zdr_ids == set()
    assert discounts != {}
    assert aa_by_id != {}
    assert calls == [mc.OPENROUTER_DISCOUNTS_URL]
```

Add a new independent-cache test after it:

```python
def test_fetch_openrouter_frontend_aa_cache_hit_skips_base_fetch(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    calls = []
    stub_frontend(monkeypatch, calls, frontend_payload(), zdr_payload())
    cache_dir = tmp_path / "model-compare"
    cache_dir.mkdir(parents=True)
    (cache_dir / "openrouter-frontend-discounts.json").write_text(
        json.dumps({"fetched_at": time.time(), "payload": {"acme/a": 0.5}})
    )
    (cache_dir / "openrouter-frontend-aa.json").write_text(
        json.dumps({"fetched_at": time.time(), "payload": {"acme/a": {"intelligence_index": 57.5}}})
    )
    discounts, zdr_ids, aa_by_id, cache_hits = mc.fetch_openrouter_frontend(
        make_args(no_cache=False, cache_ttl=3600)
    )
    assert len(calls) == 1  # only the ZDR URL
    assert discounts == {"acme/a": 0.5}
    assert aa_by_id == {"acme/a": {"intelligence_index": 57.5}}
    assert cache_hits == {"discounts", "aa"}
```

Replace `test_fetch_discount_map_realistic_slugs` + `test_fetch_zdr_set_realistic_slugs` with one test (keep the existing `_realistic_openrouter_payload()` helper; note it has no `benchmarks` key — `build_aa_benchmarks(None)` yields `{}`):

```python
def test_fetch_openrouter_frontend_realistic_slugs(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr(
        mc, "fetch_json", lambda *a, **k: _realistic_openrouter_payload()
    )
    discounts, zdr_ids, aa_by_id, cache_hits = mc.fetch_openrouter_frontend(
        make_args(no_cache=True)
    )
    assert discounts == {
        "openai/gpt-4o": 0.5,
        "openai/gpt-4o:free": 0.0,
        "anthropic/claude-sonnet-4-20250514:batch": 0.25,
    }
    assert zdr_ids == {
        "openai/gpt-4o",
        "openai/gpt-4o:free",
        "anthropic/claude-sonnet-4-20250514:batch",
    }
    assert aa_by_id == {}
```

Update the 3 `run()` seams — `test_best_prints_provider_qualified_id` (line ~733), `test_catalog_no_zdr_marks_skipped`, `test_catalog_fails_closed_without_zdr` — replacing the two fetcher patches:

```python
    monkeypatch.setattr(
        mc, "fetch_discount_map", lambda a: ({}, False)
    )
    monkeypatch.setattr(mc, "fetch_zdr_set", lambda a: ({"acme/model-a"}, False))
```

becomes:

```python
    monkeypatch.setattr(
        mc, "fetch_openrouter_frontend", lambda a: ({}, {"acme/model-a"}, {}, set())
    )
```

(In the two catalog tests use `({}, set(), {}, set())`; the `fetch_aa_entries` patches stay unchanged. `test_catalog_no_zdr_marks_skipped` additionally needs `no_zdr=True` args so the ZDR gate is skipped — it already has that.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest test_model_compare.py -q -k "frontend or best_prints or catalog_no_zdr or fails_closed"`
Expected: FAIL with `AttributeError: module 'model_compare' has no attribute 'fetch_openrouter_frontend'`.

- [ ] **Step 3: Implement**

Delete `fetch_discount_map` and `fetch_zdr_set` and add constants + the new function in their place:

```python
FRONTEND_DISCOUNTS_CACHE = "openrouter-frontend-discounts"
FRONTEND_AA_CACHE = "openrouter-frontend-aa"


def fetch_openrouter_frontend(args):
    """Derive discounts, ZDR ids and OR-published AA benchmarks.

    Two loads: the base frontend URL serves discounts and benchmarks in one
    response; the ?zdr=true URL serves ZDR and is skipped entirely under
    --no-zdr. Each payload caches independently under its own key and is
    cached only when non-empty, so a degraded payload self-heals on the
    next run. A base-URL outage never blocks the ZDR fetch, and vice
    versa. Returns (discounts, zdr_ids, aa_by_id, cache_hits) where
    cache_hits names the payloads served from cache ("discounts", "zdr",
    "aa").
    """
    discounts = {}
    aa_by_id = {}
    zdr_ids = set()
    cache_hits = set()

    aa_from_cache = False
    if not args.no_cache:
        cached = load_cache(FRONTEND_DISCOUNTS_CACHE, args.cache_ttl)
        if cached:
            discounts = cached
            cache_hits.add("discounts")
        cached = load_cache(FRONTEND_AA_CACHE, args.cache_ttl)
        if cached:
            aa_by_id = cached
            aa_from_cache = True
            cache_hits.add("aa")

    if not discounts or not aa_from_cache:
        try:
            payload = fetch_json(OPENROUTER_DISCOUNTS_URL, timeout=30)
        except Exception as exc:
            if not discounts:
                warn(f"could not fetch discount data: {exc}")
            if not aa_from_cache:
                warn(f"could not fetch AA benchmark data: {exc}")
        else:
            data = payload.get("data") if isinstance(payload, dict) else None
            entries = data.get("models") if isinstance(data, dict) else None
            fresh_discounts = {}
            for entry in entries or []:
                if not isinstance(entry, dict):
                    continue
                slug = entry.get("slug") or ""
                if not slug or slug.startswith("~"):
                    continue
                endpoint = entry.get("endpoint") or {}
                raw = (endpoint.get("pricing") or {}).get("discount")
                if not isinstance(raw, (int, float)):
                    continue
                variant = endpoint.get("variant") or ""
                key = slug if variant in ("", "standard") else f"{slug}:{variant}"
                fresh_discounts.setdefault(key, float(raw))
            fresh_aa = build_aa_benchmarks(
                data.get("benchmarks") if isinstance(data, dict) else None
            )
            if fresh_discounts:
                discounts = fresh_discounts
                save_cache(FRONTEND_DISCOUNTS_CACHE, discounts)
                cache_hits.discard("discounts")
            elif not discounts:
                # An intact endpoint always yields hundreds of entries
                # (discount: 0 is still an entry); an empty map means the
                # response shape changed -- never cache that, so the next
                # run recovers on its own.
                warn("no discount entries found; treating discounts as unavailable")
            if not aa_from_cache:
                if fresh_aa:
                    aa_by_id = fresh_aa
                    save_cache(FRONTEND_AA_CACHE, aa_by_id)
                else:
                    warn(
                        "no AA benchmark entries found; treating OpenRouter "
                        "benchmarks as unavailable"
                    )

    if args.no_zdr:
        return discounts, zdr_ids, aa_by_id, cache_hits
    if not args.no_cache:
        cached = load_cache("openrouter-zdr", args.cache_ttl)
        if cached:
            return discounts, set(cached), aa_by_id, cache_hits | {"zdr"}
    try:
        payload = fetch_json(OPENROUTER_ZDR_URL, timeout=30)
    except Exception as exc:
        warn(f"could not fetch ZDR data: {exc}")
        return discounts, zdr_ids, aa_by_id, cache_hits
    data = payload.get("data") if isinstance(payload, dict) else None
    entries = data.get("models") if isinstance(data, dict) else None
    ids = set()
    for entry in entries or []:
        if not isinstance(entry, dict):
            continue
        slug = entry.get("slug") or ""
        if not slug or slug.startswith("~"):
            continue
        endpoint = entry.get("endpoint") or {}
        variant = endpoint.get("variant") or ""
        ids.add(slug if variant in ("", "standard") else f"{slug}:{variant}")
    if not ids:
        warn("no ZDR entries found; treating ZDR data as unavailable")
        return discounts, zdr_ids, aa_by_id, cache_hits
    save_cache("openrouter-zdr", sorted(ids))
    return discounts, ids, aa_by_id, cache_hits
```

In `run()`, replace:

```python
    discounts, disc_cached = fetch_discount_map(args)
    zdr_ids, zdr_cached = fetch_zdr_set(args)
```

with:

```python
    discounts, zdr_ids, aa_by_id, frontend_cache_hits = fetch_openrouter_frontend(args)
```

and the source_note block:

```python
        if or_cached:
            source_note.append("catalog cached")
        if disc_cached:
            source_note.append("discounts cached")
        if zdr_cached:
            source_note.append("ZDR cached")
        if aa_cached and aa_source:
            source_note.append("quality cached")
```

becomes:

```python
        if or_cached:
            source_note.append("catalog cached")
        if "discounts" in frontend_cache_hits:
            source_note.append("discounts cached")
        if "zdr" in frontend_cache_hits:
            source_note.append("ZDR cached")
        if "aa" in frontend_cache_hits:
            source_note.append("AA benchmarks cached")
        if aa_cached and aa_source:
            source_note.append("quality cached")
```

(`aa_by_id` is fetched but not yet consumed for quality — Task 4 wires it in. Python will flag it unused only stylistically; assign to keep the tuple unpack.)

- [ ] **Step 4: Run the full suite**

Run: `rtk pytest -q`
Expected: all pass (rewritten fetch tests + untouched others).

- [ ] **Step 5: Lint and commit**

Run: `rtk ruff check .`

```bash
git add model_compare.py test_model_compare.py
git commit -m "Consolidate frontend fetches into fetch_openrouter_frontend"
```

---

### Task 4: OR-primary quality resolution

**Files:**
- Modify: `model_compare.py` (new `resolve_quality` after `build_aa_benchmarks`; `run()` quality loop lines ~1099-1103 and `quality_note` lines ~1136-1145)
- Test: `test_model_compare.py` (new section)

**Interfaces:**
- Consumes: Task 1 (`match_quality(allow_fuzzy=False)` default), Task 2 (`base_model_id`), Task 3 (`aa_by_id` from `fetch_openrouter_frontend`).
- Produces: `resolve_quality(candidates, aa_by_id, exact, fuzzy, aa_source) -> (quality_by_id: dict, source_by_id: dict)` with source values `"openrouter" | "api" | "scrape"`; `run()` consumes both (catalog does in Task 5).

- [ ] **Step 1: Write the failing tests**

```python
# ---------------------------------------------------------------------------
# resolve_quality (OR benchmarks first, exact AA fallback)
# ---------------------------------------------------------------------------


def test_resolve_quality_prefers_openrouter_benchmarks():
    candidates = [{"id": "acme/model-a", "name": "Acme: Model A"}]
    aa_by_id = {"acme/model-a": {"intelligence_index": 57.5}}
    exact, fuzzy = mc.build_aa_lookup(
        [{"key": "model-a", "name": "Model A", "index": 55.0}]
    )
    quality, source = mc.resolve_quality(
        candidates, aa_by_id, exact, fuzzy, "AA API v2"
    )
    assert quality == {"acme/model-a": 57.5}
    assert source == {"acme/model-a": "openrouter"}


def test_resolve_quality_falls_back_to_exact_aa_and_records_provenance():
    candidates = [{"id": "acme/model-b", "name": "Acme: Model B"}]
    exact, fuzzy = mc.build_aa_lookup(
        [{"key": "openai/model-b", "name": "Model B", "index": 51.2}]
    )
    quality, source = mc.resolve_quality(
        candidates, {}, exact, fuzzy, "AA API v2"
    )
    assert quality == {"acme/model-b": 51.2}
    assert source == {"acme/model-b": "api"}


def test_resolve_quality_variant_inherits_base_trio():
    candidates = [{"id": "acme/model-a:free", "name": "Acme: Model A (free)"}]
    aa_by_id = {"acme/model-a": {"intelligence_index": 57.5}}
    quality, source = mc.resolve_quality(candidates, aa_by_id, {}, {}, None)
    assert quality == {"acme/model-a:free": 57.5}
    assert source == {"acme/model-a:free": "openrouter"}


def test_resolve_quality_trio_without_intelligence_falls_through():
    candidates = [{"id": "acme/model-a", "name": "Acme: Model A"}]
    aa_by_id = {"acme/model-a": {"coding_index": 70.0}}
    exact, fuzzy = mc.build_aa_lookup(
        [{"key": "acme/model-a", "name": "Model A", "index": 44.0}]
    )
    quality, source = mc.resolve_quality(
        candidates, aa_by_id, exact, fuzzy, "AA page scrape"
    )
    assert quality == {"acme/model-a": 44.0}
    assert source == {"acme/model-a": "scrape"}


def test_resolve_quality_unmatched_is_none():
    candidates = [{"id": "acme/model-a", "name": "Acme: Model A"}]
    quality, source = mc.resolve_quality(candidates, {}, {}, {}, None)
    assert quality == {}
    assert source == {}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest test_model_compare.py -q -k "resolve_quality"`
Expected: FAIL with `AttributeError: ... no attribute 'resolve_quality'`.

- [ ] **Step 3: Implement**

Insert after `build_aa_benchmarks`:

```python
def resolve_quality(candidates, aa_by_id, exact, fuzzy, aa_source):
    """Per-candidate quality: OR benchmarks first, AA exact fallback, else None.

    OR's per-slug values cannot mispair; the AA fallback is exact-tier only
    (match_quality's fuzzy tier is the proven variant-conflation bug and
    stays off in production). Source values match the catalog contract:
    "openrouter", "api", "scrape".
    """
    fallback_source = {"AA API v2": "api", "AA page scrape": "scrape"}.get(aa_source)
    quality_by_id = {}
    source_by_id = {}
    for cand in candidates:
        bench = aa_by_id.get(base_model_id(cand["id"]))
        intelligence = bench.get("intelligence_index") if bench else None
        if intelligence is not None:
            quality_by_id[cand["id"]] = intelligence
            source_by_id[cand["id"]] = "openrouter"
            continue
        index = match_quality({"id": cand["id"], "name": cand["name"]}, exact, fuzzy)
        if index is not None:
            quality_by_id[cand["id"]] = index
            source_by_id[cand["id"]] = fallback_source
    return quality_by_id, source_by_id
```

In `run()`, replace:

```python
    quality_by_id = {}
    for cand in candidates:
        index = match_quality({"id": cand["id"], "name": cand["name"]}, exact, fuzzy)
        if index is not None:
            quality_by_id[cand["id"]] = index
    weights = compute_scores(candidates, args, quality_by_id)
```

with:

```python
    quality_by_id, quality_source_by_id = resolve_quality(
        candidates, aa_by_id, exact, fuzzy, aa_source
    )
    weights = compute_scores(candidates, args, quality_by_id)
```

and the quality_note block:

```python
        unmatched = sum(1 for c in candidates if c["id"] not in quality_by_id)
        if aa_source:
            matched = len(candidates) - unmatched
            quality_note = (
                f"quality via {aa_source}{' (cached)' if aa_cached else ''}: "
                f"{len(aa_entries)} scores, matched {matched}/{len(candidates)} candidates"
            )
            if unmatched:
                quality_note += "; unmatched candidates score 0 on quality"
        else:
            quality_note = "quality data unavailable, ranked on price/context/age"
```

becomes:

```python
        matched_or = sum(
            1 for s in quality_source_by_id.values() if s == "openrouter"
        )
        matched_aa = len(quality_by_id) - matched_or
        unmatched = len(candidates) - len(quality_by_id)
        bits = []
        if matched_or:
            bits.append(f"OpenRouter benchmarks ({matched_or})")
        if matched_aa:
            bits.append(f"{aa_source} exact ({matched_aa})")
        if bits:
            quality_note = (
                f"quality via {' + '.join(bits)}: matched "
                f"{len(quality_by_id)}/{len(candidates)} candidates"
            )
            if unmatched:
                quality_note += "; unmatched candidates score 0 on quality"
        else:
            quality_note = "quality data unavailable, ranked on price/context/age"
```

- [ ] **Step 4: Run the full suite**

Run: `rtk pytest -q`
Expected: all pass (existing run()-level tests stub `fetch_aa_entries` to `([], None, False)` and `aa_by_id` to `{}`, so notes degrade to the unchanged "quality data unavailable" path).

- [ ] **Step 5: Lint and commit**

Run: `rtk ruff check .`

```bash
git add model_compare.py test_model_compare.py
git commit -m "Resolve quality from OR benchmarks first with exact-only AA fallback"
```

---

### Task 5: Catalog `aa` block + provenance

**Files:**
- Modify: `model_compare.py` (`build_catalog` signature lines 831-832, `aa_modes`/gate lines 843-849, entry dict lines 879-905, `sources.aa` line 927; `run()` catalog call lines 1106-1118)
- Test: `test_model_compare.py` (catalog_pool/build_doc helpers lines 1138-1190; catalog tests)

**Interfaces:**
- Consumes: Task 4 (`resolve_quality` outputs), Task 2 (`base_model_id`).
- Produces: `build_catalog(args, models, candidates, dropped, filtered, discounts, quality_by_id, aa_source, aa_by_id, quality_source_by_id) -> dict` — entries gain `"aa"` (`{"intelligence_index", "coding_index", "agentic_index"}`, each `null` or finite number), `quality` published unconditionally, `quality_match` from `quality_source_by_id`; `sources.aa` gains `matched_openrouter`. Task 6's validator consumes the shape.

- [ ] **Step 1: Update the catalog test helpers and add failing tests**

In `catalog_pool` (line ~1138), add an AA map to the returned tuple and build quality through `resolve_quality`. The current helper ends with manual `quality_by_id = {"acme/model-b": 68.4}` + `mc.compute_scores(...)`. Change the tail to:

```python
    aa_by_id = {
        "acme/model-b": {
            "intelligence_index": 68.4,
            "coding_index": 74.8,
            "agentic_index": 59.1,
        }
    }
    quality_by_id, quality_source_by_id = mc.resolve_quality(
        candidates, aa_by_id, {}, {}, overrides.get("aa_source", "AA API v2")
    )
    mc.compute_scores(candidates, args, quality_by_id)
    return (
        args,
        models,
        candidates,
        dropped,
        filtered,
        {"acme/model-a": 0.5},
        quality_by_id,
        quality_source_by_id,
        aa_by_id,
    )
```

In `build_doc` (line ~1182), pass the new tuple through:

```python
def build_doc(**overrides):
    aa_source = overrides.pop("aa_source", "AA API v2")
    (
        args,
        models,
        candidates,
        dropped,
        filtered,
        discounts,
        quality_by_id,
        quality_source_by_id,
        aa_by_id,
    ) = catalog_pool(**overrides)
    return mc.build_catalog(
        args,
        models,
        candidates,
        dropped,
        filtered,
        discounts,
        quality_by_id,
        aa_source,
        aa_by_id,
        quality_source_by_id,
    )
```

Add failing tests (after `test_catalog_aa_mode_mapping`):

```python
def test_catalog_aa_block_carries_or_trio():
    doc = build_doc()
    b = next(e for e in doc["models"] if e["id"] == "acme/model-b")
    assert b["aa"] == {
        "intelligence_index": 68.4,
        "coding_index": 74.8,
        "agentic_index": 59.1,
    }
    a = next(e for e in doc["models"] if e["id"] == "acme/model-a")
    assert a["aa"] == {"intelligence_index": None, "coding_index": None, "agentic_index": None}


def test_catalog_quality_match_and_sources_counts():
    doc = build_doc()
    b = next(e for e in doc["models"] if e["id"] == "acme/model-b")
    assert b["quality"] == 68.4
    assert b["quality_match"] == "openrouter"
    a = next(e for e in doc["models"] if e["id"] == "acme/model-a")
    assert a["quality"] is None
    assert a["quality_match"] is None
    assert doc["sources"]["aa"] == {
        "mode": "openrouter",
        "matched": 1,
        "matched_openrouter": 1,
    }


def test_catalog_quality_published_even_without_any_aa_source():
    doc = build_doc(aa_source=None)
    b = next(e for e in doc["models"] if e["id"] == "acme/model-b")
    assert b["quality"] == 68.4
    assert b["quality_match"] == "openrouter"
    assert doc["sources"]["aa"]["mode"] == "openrouter"


def test_catalog_aa_mode_reflects_fallback_when_or_empty():
    doc = build_doc(aa_source="AA page scrape", aa_by_id={})
    assert doc["sources"]["aa"] == {"mode": "scrape", "matched": 0, "matched_openrouter": 0}
```

(`build_doc`/`catalog_pool` must honor an `aa_by_id` override: in `catalog_pool`, `aa_by_id = overrides.pop("aa_by_id", {...default...})` — adjust the helper to pop it before `resolve_quality`.)

Update the existing expectations that shift:
- `test_catalog_entry_shape`: `CATALOG_ENTRY_KEYS` set gains `"aa"`; model-b assertions stay valid (quality 68.4 via openrouter now); model-a `quality_match` stays `None`.
- `test_catalog_aa_mode_mapping` (line ~1343): `build_doc(aa_source="AA page scrape")` now yields mode `"openrouter"` (model-b still matched via the trio) — rewrite this test to drop the OR map as in the new fallback test above, and `build_doc(aa_source=None)` keeps mode `"openrouter"`; fold into the new tests and delete the old one.
- `test_catalog_overall_matches_compute_scores_for_current_priority` and `test_catalog_deterministic_modulo_generated_at` keep passing via the updated helpers.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest test_model_compare.py -q -k "catalog"`
Expected: FAIL — `build_catalog() takes 8 positional arguments but 10 were given`.

- [ ] **Step 3: Implement**

Change the signature and provenance logic in `build_catalog`:

```python
def build_catalog(
    args,
    models,
    candidates,
    dropped,
    filtered,
    discounts,
    quality_by_id,
    aa_source,
    aa_by_id,
    quality_source_by_id,
):
```

Replace the `aa_modes`/gate block (lines 843-849) with:

```python
    aa_modes = {"AA API v2": "api", "AA page scrape": "scrape"}
    if aa_source is not None and aa_source not in aa_modes:
        # Never claim mode "none" for a source we do not know: the document
        # would contradict itself (mode none with matched quality scores).
        raise ValueError(f"unknown AA source: {aa_source!r}")
    matched_openrouter = sum(
        1 for s in quality_source_by_id.values() if s == "openrouter"
    )
    aa_mode = (
        "openrouter"
        if matched_openrouter
        else aa_modes.get(aa_source, "none")
    )
```

Inside the entry loop, before `entries.append(...)`, add:

```python
        bench = aa_by_id.get(base_model_id(cand["id"])) or {}
```

and replace the two entry fields:

```python
                "quality": cand["quality"] if aa_source else None,
                "quality_match": quality_match if cand["id"] in quality_by_id else None,
```

with:

```python
                "aa": {
                    "intelligence_index": bench.get("intelligence_index"),
                    "coding_index": bench.get("coding_index"),
                    "agentic_index": bench.get("agentic_index"),
                },
                "quality": cand["quality"],
                "quality_match": quality_source_by_id.get(cand["id"]),
```

(Delete the now-unused `quality_match = aa_modes.get(aa_source)` line.)

Update `sources`:

```python
        "sources": {
            "openrouter": "ok",
            "aa": {
                "mode": aa_mode,
                "matched": len(quality_by_id),
                "matched_openrouter": matched_openrouter,
            },
            "zdr": "skipped" if args.no_zdr else "ok",
            "discounts": "ok" if discounts else "unavailable",
        },
```

In `run()`'s catalog branch, pass the two new arguments:

```python
    if args.catalog:
        print_catalog(
            build_catalog(
                args,
                models,
                candidates,
                dropped,
                filtered,
                discounts,
                quality_by_id,
                aa_source,
                aa_by_id,
                quality_source_by_id,
            )
        )
        return 0
```

- [ ] **Step 4: Run the full suite**

Run: `rtk pytest -q`
Expected: all pass.

- [ ] **Step 5: Lint and commit**

Run: `rtk ruff check .`

```bash
git add model_compare.py test_model_compare.py
git commit -m "Publish OR AA trio and provenance in the catalog document"
```

---

### Task 6: Validator `aa` block + provenance checks

**Files:**
- Modify: `build_site_data.py` (`CATALOG_ENTRY_KEYS`, new constants + checks in `validate_catalog`; add `import math` if absent)
- Test: `test_build_site_data.py` (fixtures `make_catalog_entry`/`make_catalog`; validate tests)

**Interfaces:**
- Consumes: Task 5's document shape.
- Produces: `validate_catalog` accepting/rejecting the evolved contract.

- [ ] **Step 1: Update fixtures and write failing tests**

In `make_catalog_entry` (line ~154), add after `"quality": 55.0,`:

```python
        "aa": {
            "intelligence_index": 55.0,
            "coding_index": 61.0,
            "agentic_index": 48.5,
        },
```

In `make_catalog` (line ~182), change the sources line to:

```python
        "sources": {
            "openrouter": "ok",
            "aa": {"mode": "openrouter", "matched": 1, "matched_openrouter": 1},
            "zdr": "ok",
            "discounts": "ok",
        },
```

Extend the `test_validate_catalog_rejects` parametrize list with:

```python
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
```

(`float("nan")` requires the JSON round-trip in `main`-level tests to preserve NaN — `json.dumps` writes `NaN` and `json.load` parses it back, so the value-level test works through `validate_catalog` directly and through the file path.)

Add acceptance tests after `test_validate_catalog_happy_path`:

```python
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
        doc["sources"]["aa"]["mode"] = mode
        bsd.validate_catalog(doc)  # must not raise
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest test_build_site_data.py -q -k "validate_catalog"`
Expected: FAIL — happy-path fixture now carries `aa`/`matched_openrouter` that the validator does not know (`models[0] is missing keys: aa`) and the acceptance test raises on mode `"none"`.

- [ ] **Step 3: Implement**

In `build_site_data.py`: add `import math` to the imports; extend the constants:

```python
CATALOG_AA_KEYS = ("intelligence_index", "coding_index", "agentic_index")
CATALOG_QUALITY_MATCH_VALUES = ("openrouter", "api", "scrape")
CATALOG_AA_MODES = ("openrouter", "api", "scrape", "none")
```

(`CATALOG_ENTRY_KEYS` gains `"aa"` after `"quality"`, keeping `"quality_match"`, `"scores"` after it.)

Add the helper next to `_is_score`:

```python
def _is_aa_value(value) -> bool:
    if value is None:
        return True
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and 0.0 <= value <= 100.0
    )
```

In `validate_catalog`'s per-entry loop, after the `scores`/`overall` checks:

```python
        aa = entry["aa"]
        if not isinstance(aa, dict) or any(key not in aa for key in CATALOG_AA_KEYS):
            raise ValueError(f"models[{i}] aa is incomplete")
        bad_aa = [key for key in CATALOG_AA_KEYS if not _is_aa_value(aa[key])]
        if bad_aa:
            raise ValueError(f"models[{i}] aa out of range: {', '.join(bad_aa)}")
        if (
            entry["quality_match"] is not None
            and entry["quality_match"] not in CATALOG_QUALITY_MATCH_VALUES
        ):
            raise ValueError(f"models[{i}] has unknown quality_match")
```

And after the pool count check (before the per-entry loop):

```python
    sources = document["sources"]
    aa_sources = sources.get("aa") if isinstance(sources, dict) else None
    if not isinstance(aa_sources, dict) or aa_sources.get("mode") not in CATALOG_AA_MODES:
        raise ValueError("catalog sources.aa.mode is unknown")
    for key in ("matched", "matched_openrouter"):
        value = aa_sources.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"catalog sources.aa.{key} must be a non-negative integer")
    if aa_sources["matched_openrouter"] > aa_sources["matched"]:
        raise ValueError("catalog sources.aa.matched_openrouter exceeds matched")
```

- [ ] **Step 4: Run the full suite**

Run: `rtk pytest -q`
Expected: all pass (the producer→validator integration test already built in Task 5's fixture flow now exercises the evolved contract end-to-end).

- [ ] **Step 5: Lint and commit**

Run: `rtk ruff check .`

```bash
git add build_site_data.py test_build_site_data.py
git commit -m "Validate catalog aa trio and provenance fields"
```

---

### Task 7: README + module docstring + live acceptance

**Files:**
- Modify: `README.md` (data-sources section, Artificial Analysis section, catalog section)
- Modify: `model_compare.py` (module docstring, lines 9-13)

**Interfaces:**
- Consumes: everything above.

- [ ] **Step 1: Update the module docstring**

Replace the AA paragraph (lines 9-13):

```
  * Artificial Analysis intelligence index - model quality, obtained via
    the AA API v2 when an API key is available (--aa-api-key or the
    AA_API_KEY environment variable; free key at artificialanalysis.ai),
    falling back to a best-effort scrape of artificialanalysis.ai/models.
    If neither works, ranking continues on price/context/age alone.
```

with:

```
  * Artificial Analysis intelligence index - model quality. Primary
    source: OpenRouter's republished AA benchmarks (data.benchmarks in
    the frontend models API, exact per-model keys). When a model is not
    covered there: the AA API v2 (--aa-api-key or the AA_API_KEY
    environment variable; free key at artificialanalysis.ai), then a
    best-effort scrape of artificialanalysis.ai/models -- both matched
    by exact slug/name only. Unmatched models rank on price/context/age.
```

- [ ] **Step 2: Update README.md**

1. In **Data sources** (§ How it works), rewrite item 2's bullets: OpenRouter's own `?discount=true`-style frontend API now also publishes the AA intelligence/coding/agentic indices per model (`benchmarks.aa`), consumed first; the AA API v2 and the JSON-LD scrape remain as fallbacks matched by **exact slug/name only** (fuzzy token matching removed — it conflated variants like `glm-5.3-flash` with `glm-5.3`). Mention the incident briefly as rationale, in the README's existing tone.

2. In **Catalog output**, extend the models-entry field list: after "`quality` (AA intelligence index or `null`)", add "`aa` (the OpenRouter-published trio `intelligence_index`/`coding_index`/`agentic_index`, each possibly `null`)"; change "`quality_match` (`api`/`scrape`/`null`)" to "`quality_match` (`openrouter`/`api`/`scrape`/`null`)"; and in the top-level list change the `sources` description to mention `mode` ∈ `openrouter`/`api`/`scrape`/`none` plus `matched_openrouter`.

- [ ] **Step 3: Run the suite**

Run: `rtk pytest -q && rtk ruff check .`
Expected: all pass, clean.

- [ ] **Step 4: Commit**

```bash
git add README.md model_compare.py
git commit -m "Document OR-benchmark-sourced quality and catalog aa fields"
```

---

### Task 8: Live acceptance verification

**Files:**
- None (verification only; live network, warm cache).

- [ ] **Step 1: Correctness spot-checks**

```bash
./model_compare.py --catalog > /tmp/opencode/aa-catalog-1.json
python3 - <<'PY'
import json
doc = json.load(open("/tmp/opencode/aa-catalog-1.json"))
src = doc["sources"]["aa"]
print("sources.aa:", src)
models = {e["id"]: e for e in doc["models"]}
glm = models.get("z-ai/glm-5.3-flash")
assert glm, "glm-5.3-flash not in pool?"
print("glm-5.3-flash: quality =", glm["quality"], "| match =", glm["quality_match"], "| aa =", glm["aa"])
assert glm["quality_match"] == "openrouter", "expected OR-sourced quality"
assert abs(glm["aa"]["intelligence_index"] - glm["quality"]) < 1e-9
assert all(e["quality_match"] in ("openrouter", "api", "scrape", None) for e in doc["models"])
assert all(set(e["aa"]) == {"intelligence_index", "coding_index", "agentic_index"} for e in doc["models"])
assert len(doc["models"]) == doc["pool"]["candidates"]
print("OK:", src["matched_openrouter"], "OR-sourced of", src["matched"], "matched")
PY
```

- [ ] **Step 2: Determinism with a warm cache**

```bash
./model_compare.py --catalog > /tmp/opencode/aa-catalog-2.json
python3 - <<'PY'
import json
a = json.load(open("/tmp/opencode/aa-catalog-1.json"))
b = json.load(open("/tmp/opencode/aa-catalog-2.json"))
a.pop("generated_at"); b.pop("generated_at")
assert a == b, "documents differ beyond generated_at"
print("DETERMINISTIC apart from generated_at")
PY
```

- [ ] **Step 3: Table path unchanged in shape**

```bash
./model_compare.py --top 3 2>&1 | tail -4
```
Expected: normal table + pool/weights footer with the new quality note (e.g. "quality via OpenRouter benchmarks (N): matched N/M candidates").

- [ ] **Step 4: Full suite and lint, then report**

Run: `rtk pytest -q && rtk ruff check .`
Report: commands, outputs, the OR-sourced match count, any catalog surprises (e.g. models losing quality vs the old fuzzy output — expected per spec §7).

---

## Self-Review notes

- Spec coverage: fetch consolidation + per-payload caches + cache_hits (Task 3), latest-date-wins/undated/non-finite map building (Task 2), allow_fuzzy fail-safe default + regression (Task 1), OR-primary resolution with provenance (Task 4), catalog `aa`/`quality`-gate rewrite/`sources.aa` (Task 5), validator + fixtures (Task 6), README/docstring (Task 7), acceptance incl. the glm-5.3-flash correction (Task 8). Test-migration scope (13 direct + 3 seams) covered in Task 3.
- Type consistency: `fetch_openrouter_frontend` 4-tuple used identically in Tasks 3→4; `resolve_quality` outputs consumed by Task 5's catalog; `build_catalog`'s 10-parameter signature consistent between Task 5 and its tests; validator constant names match Task 6 usage.
- Deliberate sequencing: Task 3 keeps quality behavior unchanged (OR map fetched, unused) so each task lands green independently.
