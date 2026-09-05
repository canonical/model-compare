# Weekly History, Movement Column, and Highlights Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish a rolling 10-day `history.json`, a client-computed `7-day` movement column on the site tables, and three grounded LLM-generated highlight sections (`highlights.json`, daily cadence, template fallback).

**Architecture:** `build_site_data.py` projects each publish's catalog document into a daily snapshot, merges/prunes the previous `history.json` (carry-forward via the workflow fetching the live site), and validates/places both new artifacts. A new stdlib `generate_highlights.py` computes a deterministic numeric diff against the exact-7-days-ago snapshot, reuses LLM output younger than 24 h, otherwise makes one OpenRouter chat call (model fallback chain, `temperature 0`, grounding rules) with deterministic templated sentences as the never-fail fallback. `site/index.html` renders the `7-day` column and the three sections client-side (escape-then-backticks for monospace model ids).

**Tech Stack:** Python 3.10+ stdlib (json, urllib), GitHub Actions curl step, vanilla JS. `model_compare.py` untouched.

**Worktree:** `.worktrees/history-highlights`, branch `history-highlights`.

**Spec:** `docs/superpowers/specs/2026-09-02-history-highlights-design.md` (in this worktree) — the binding contract.

## Global Constraints

- Retention: **10** newest UTC dates; comparison baseline = snapshot dated **exactly 7 days before today**; same-day upsert is last-write-wins; `updated_at` = newest snapshot's `generated_at`.
- Snapshot scope: `pool_ids` = catalog `models` + `filtered` ids; `aa`/`prices`/`tabs` cover **candidates (`models`) only**; `prices` arrays are `[input_per_1m, output_per_1m, blended_per_1m, discount]`.
- `history.json` validation is strict on identity (schema_version, required keys, ≤ 10 snapshots, ISO dates) and runs AFTER merge+prune; a malformed previous file means "start fresh", never a build failure.
- `highlights.json`: `schema_version` 1; `source` ∈ {`openrouter`, `fallback`}; section keys `week`/`intelligence`/`prices` required with non-empty strings; extra keys tolerated; strict on identity, tolerant on extensions.
- Highlights reuse: previous output copied through only when `source == "openrouter"` AND younger than 24 h; `source == "fallback"` output is always regenerated.
- LLM: one OpenRouter chat-completions call per generation, model fallback chain, `temperature: 0`, ~30 s timeout, `max_tokens` 300, grounding rules verbatim from Task 3; any failure → templated fallback (backtick-wrapped ids); a broken LLM never fails the deploy.
- Site: `7-day` column header, placed immediately after `RANK`; cells `↑N` green / `↓N` red / `•` blue with no zero / `new` green / blank when no baseline; plain UTF text, no emoji; all three JSON fetches use `cache: "no-cache"`; model ids in highlights rendered as `<code>` via escape-then-backticks.
- Workflow: previous artifacts fetched with `curl -fsSL … || true`; `OPENROUTER_API_KEY` optional secret on the build step.
- Every artifact write happens only after its validation passes (fail closed, like `catalog.json`).
- Run `rtk pytest -q` (195 tests today) and `rtk ruff check .` after every task; both must be clean before committing.

---

### Task 1: History snapshot + merge/prune + validation in `build_site_data.py`

**Files:**
- Modify: `build_site_data.py` (new constants + `build_snapshot`/`merge_history`/`_is_iso_date`/`validate_history` after `validate_catalog`; `main()` gains `--history-prev-file`)
- Test: `test_build_site_data.py` (new section; extend `make_catalog()` with an `aa` block — already present from the aa task — and confirm `make_catalog()` is usable as snapshot input)

**Interfaces:**
- Consumes: existing `validate_catalog` output shape (`models` entries with `id`, `pricing`, `discount`, `quality`, `aa`, `scores.overall`; `filtered` with `id`); existing sibling-write pattern (`os.path.dirname(os.path.abspath(args.output))`).
- Produces:
  - `build_snapshot(catalog: dict) -> dict` — `{"generated_at", "pool_ids", "tabs", "aa", "prices"}` (tabs derived by sorting `models` on `scores.overall[priority]`, top 10, rank 1-based).
  - `merge_history(prev, snapshot) -> dict` — upsert under the snapshot date (from `snapshot["generated_at"]`), prune to 10, `updated_at` = newest snapshot's `generated_at`.
  - `validate_history(document) -> None` (raises `ValueError`).
  - `main()` accepts `--history-prev-file FILE` (requires `--catalog-file`; parser.error otherwise) and writes `history.json` next to `data.json` after validation. Task 4 adds `--highlights-file` alongside.

- [ ] **Step 1: Write the failing tests**

Append to `test_build_site_data.py`:

```python
# ---------------------------------------------------------------------------
# history.json publication
# ---------------------------------------------------------------------------


def make_history_snapshot(date, generated_at, **overrides):
    snap = {
        "generated_at": generated_at,
        "pool_ids": ["acme/model-a"],
        "tabs": {
            "balanced": [{"id": "acme/model-a", "rank": 1, "quality": 55.0, "blended": 1.25}],
            "price": [{"id": "acme/model-a", "rank": 1, "quality": 55.0, "blended": 1.25}],
            "quality": [{"id": "acme/model-a", "rank": 1, "quality": 55.0, "blended": 1.25}],
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
    snap = bsd.build_snapshot(make_catalog(), )
    assert snap["generated_at"] == "2026-09-02T09:15:00+00:00"
    assert snap["pool_ids"] == ["acme/model-a", "acme/small"]
    assert snap["tabs"]["balanced"][0] == {
        "id": "acme/model-a", "rank": 1, "quality": 55.0, "blended": 1.25
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
    merged = bsd.merge_history({"snapshots": "garbage"}, make_history_snapshot(
        "2026-09-02", "2026-09-02T09:15:00+00:00"
    ))
    assert list(merged["snapshots"]) == ["2026-09-02"]
    assert bsd.merge_history(None, make_history_snapshot(
        "2026-09-02", "2026-09-02T09:15:00+00:00"
    ))["snapshots"]["2026-09-02"]["pool_ids"] == ["acme/model-a"]


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest test_build_site_data.py -q -k "snapshot or merge_history or validate_history"`
Expected: FAIL with `AttributeError: module 'build_site_data' has no attribute 'build_snapshot'`.

- [ ] **Step 3: Implement**

In `build_site_data.py`, after `validate_catalog` add:

```python
HISTORY_SCHEMA_VERSION = 1
HISTORY_RETENTION = 10
HISTORY_TOP_N = 10
HISTORY_TAB_KEYS = ("balanced", "price", "quality")


def _is_iso_date(value) -> bool:
    try:
        date.fromisoformat(str(value))
    except (TypeError, ValueError):
        return False
    return True


def build_snapshot(catalog) -> dict:
    """Project a validated catalog document into one daily history snapshot.

    tabs derive from the catalog itself: per priority, models sorted by
    scores.overall descending (id tiebreak), top 10, rank 1-based. aa and
    prices cover candidates only -- filtered entries carry neither.
    """
    models = catalog["models"]
    tabs = {}
    for priority in HISTORY_TAB_KEYS:
        ranked = sorted(
            models, key=lambda e: (-e["scores"]["overall"][priority], e["id"])
        )
        tabs[priority] = [
            {
                "id": entry["id"],
                "rank": i + 1,
                "quality": entry["quality"],
                "blended": entry["pricing"]["blended_per_1m"],
            }
            for i, entry in enumerate(ranked[:HISTORY_TOP_N])
        ]
    return {
        "generated_at": catalog["generated_at"],
        "pool_ids": sorted(
            [e["id"] for e in models] + [e["id"] for e in catalog["filtered"]]
        ),
        "tabs": tabs,
        "aa": {
            e["id"]: e["aa"]["intelligence_index"]
            for e in models
            if e["aa"]["intelligence_index"] is not None
        },
        "prices": {
            e["id"]: [
                e["pricing"]["input_per_1m"],
                e["pricing"]["output_per_1m"],
                e["pricing"]["blended_per_1m"],
                e["discount"],
            ]
            for e in models
        },
    }


def merge_history(prev, snapshot) -> dict:
    """Upsert the snapshot under its UTC date and prune to the newest 10.

    A malformed previous document starts fresh; garbage entries inside a
    well-formed one are dropped per date. Same-day upserts are
    last-write-wins.
    """
    snapshots = {}
    if isinstance(prev, dict) and isinstance(prev.get("snapshots"), dict):
        for date_key, snap in prev["snapshots"].items():
            if _is_iso_date(date_key) and isinstance(snap, dict):
                snapshots[date_key] = snap
    today = snapshot["generated_at"][:10]
    snapshots[today] = snapshot
    kept = sorted(snapshots)[-HISTORY_RETENTION:]
    return {
        "schema_version": HISTORY_SCHEMA_VERSION,
        "updated_at": snapshots[kept[-1]]["generated_at"],
        "snapshots": {date_key: snapshots[date_key] for date_key in kept},
    }


def validate_history(document) -> None:
    """Validate the merged history document (strict identity, <=10 dates)."""
    if not isinstance(document, dict):
        raise ValueError("history document is not an object")
    if document.get("schema_version") != HISTORY_SCHEMA_VERSION:
        raise ValueError(
            f"unknown history schema_version: {document.get('schema_version')!r}"
        )
    snapshots = document.get("snapshots")
    if not isinstance(snapshots, dict) or not snapshots:
        raise ValueError("history snapshots must be a non-empty object")
    if not all(_is_iso_date(d) for d in snapshots):
        raise ValueError("history snapshot keys must be ISO dates")
    if len(snapshots) > HISTORY_RETENTION:
        raise ValueError(
            f"history holds {len(snapshots)} snapshots (max {HISTORY_RETENTION})"
        )
    for date_key, snap in snapshots.items():
        if not isinstance(snap, dict):
            raise ValueError(f"history snapshot {date_key} is not an object")
        for key in ("generated_at", "pool_ids", "tabs", "aa", "prices"):
            if key not in snap:
                raise ValueError(f"history snapshot {date_key} missing key: {key}")
        for priority in HISTORY_TAB_KEYS:
            rows = snap["tabs"].get(priority)
            if not isinstance(rows, list):
                raise ValueError(f"history snapshot {date_key} tabs.{priority} missing")
            for i, row in enumerate(rows):
                if not isinstance(row, dict) or row.get("rank") != i + 1:
                    raise ValueError(
                        f"history snapshot {date_key} tabs.{priority} rank mismatch"
                    )
```

(`from datetime import date, datetime, timezone` — extend the existing datetime import if needed.)

In `main()`, add the argument after `--catalog-file`:

```python
    parser.add_argument(
        "--history-prev-file",
        help="previous history.json (fetched from the live site); merged, pruned and rewritten as history.json next to --output",
    )
```

and after the `catalog.json` write block, before `return 0`:

```python
    if args.history_prev_file is not None:
        if catalog is None:
            print(
                "error: --history-prev-file requires --catalog-file",
                file=sys.stderr,
            )
            return 1
        prev = None
        if args.history_prev_file:
            try:
                with open(args.history_prev_file) as fh:
                    prev = json.load(fh)
            except (OSError, ValueError):
                prev = None  # malformed/missing previous: start fresh
        history = merge_history(prev, build_snapshot(catalog))
        try:
            validate_history(history)
        except ValueError as exc:
            print(f"error: invalid history: {exc}", file=sys.stderr)
            return 1
        history_path = os.path.join(
            os.path.dirname(os.path.abspath(args.output)), "history.json"
        )
        try:
            with open(history_path, "w") as fh:
                json.dump(history, fh, indent=2)
                fh.write("\n")
        except OSError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
```

(The `catalog` local already holds the parsed, validated document from the `--catalog-file` block.)

- [ ] **Step 4: Run the full suite**

Run: `rtk pytest -q`
Expected: all pass.

- [ ] **Step 5: Lint and commit**

Run: `rtk ruff check .`

```bash
git add build_site_data.py test_build_site_data.py
git commit -m "Build and publish history.json from the catalog document"
```

---

### Task 2: `generate_highlights.py` — diff builder + fallback templates

**Files:**
- Create: `generate_highlights.py`
- Test: `test_generate_highlights.py` (new file)

**Interfaces:**
- Consumes: the deployed `history.json` shape (Task 1) and the catalog document.
- Produces (Task 3 and 4 consume):
  - `seven_days_before(today: str) -> str`
  - `build_diff(catalog: dict, history: dict) -> dict` — deterministic numeric-only diff.
  - `fallback_texts(diff: dict) -> dict` — `{"week", "intelligence", "prices"}` with backtick-wrapped ids.

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for generate_highlights.py -- diff builder and fallback templates."""

import datetime

import pytest

import generate_highlights as gh


def make_diff_history(**snap_overrides):
    base = {
        "generated_at": "2026-08-26T09:15:00+00:00",
        "pool_ids": ["acme/model-a", "acme/model-b", "acme/old"],
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
                "aa": {"intelligence_index": 68.4, "coding_index": 74.8, "agentic_index": 59.1},
                "pricing": {"input_per_1m": 1.6, "output_per_1m": 3.2, "blended_per_1m": 2.0},
                "discount": None,
            },
            {
                "id": "acme/model-a",
                "quality": 55.0,
                "aa": {"intelligence_index": 55.0, "coding_index": None, "agentic_index": None},
                "pricing": {"input_per_1m": 0.8, "output_per_1m": 1.6, "blended_per_1m": 1.0},
                "discount": 0.5,
            },
            {
                "id": "acme/fresh",
                "quality": 40.0,
                "aa": {"intelligence_index": 40.0, "coding_index": None, "agentic_index": None},
                "pricing": {"input_per_1m": 0.5, "output_per_1m": 1.0, "blended_per_1m": 0.62},
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest test_generate_highlights.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'generate_highlights'`.

- [ ] **Step 3: Implement**

Create `generate_highlights.py`:

```python
#!/usr/bin/env python3
"""Generate the site's three highlight sections from weekly history data.

Computes a deterministic numeric diff of today's catalog against the
snapshot dated exactly 7 days back, then either reuses recent LLM output,
asks OpenRouter for three grounded sentences, or falls back to templates.
A broken LLM never fails the deploy: the fallback is always available.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timedelta, timezone

HIGHLIGHTS_SCHEMA_VERSION = 1
SECTION_KEYS = ("week", "intelligence", "prices")


def seven_days_before(today: str) -> str:
    return (date.fromisoformat(today) - timedelta(days=7)).isoformat()


def build_diff(catalog, history) -> dict:
    """Numeric-only diff of today's catalog vs the snapshot 7 days ago.

    Deterministic ordering everywhere; all mover lists are bounded by
    top-10 history membership (see the design spec).
    """
    today = catalog["generated_at"][:10]
    snapshots = history.get("snapshots") if isinstance(history, dict) else None
    baseline = snapshots.get(seven_days_before(today)) if isinstance(snapshots, dict) else None
    diff = {"baseline_present": baseline is not None}
    if baseline is None:
        diff["tabs"] = {
            key: {"entries": [], "new_ids": []} for key in ("balanced", "price", "quality")
        }
        diff["new_pool_ids_count"] = 0
        diff["new_pool_id_sample"] = []
        diff["aa_movers"] = {"up": [], "down": []}
        diff["price_moves"] = {"down": [], "up": []}
        diff["discounts"] = {"appeared": [], "vanished": []}
        return diff

    prev_pool = set(baseline.get("pool_ids") or [])
    now_pool = set(
        [e["id"] for e in catalog["models"]]
        + [e["id"] for e in catalog["filtered"]]
    )
    new_pool = sorted(now_pool - prev_pool)

    tabs = {}
    for key in ("balanced", "price", "quality"):
        prev_ranks = {
            row["id"]: row["rank"] for row in baseline["tabs"].get(key) or []
        }
        ranked = sorted(
            catalog["models"],
            key=lambda e: (-e["scores"]["overall"][key], e["id"]),
        )[:10]
        entries = []
        new_ids = []
        for i, entry in enumerate(ranked, start=1):
            prev = prev_ranks.get(entry["id"])
            entries.append(
                {
                    "id": entry["id"],
                    "rank": i,
                    "prev_rank": prev,
                    "delta": None if prev is None else prev - i,
                }
            )
            if prev is None:
                new_ids.append(entry["id"])
        tabs[key] = {"entries": entries, "new_ids": new_ids}
    diff["tabs"] = tabs

    diff["new_pool_ids_count"] = len(new_pool)
    diff["new_pool_id_sample"] = new_pool[:5]

    prev_aa = baseline.get("aa") or {}
    movers = []
    for entry in catalog["models"]:
        prev = prev_aa.get(entry["id"])
        now = entry["aa"]["intelligence_index"]
        if prev is None or now is None:
            continue
        movers.append({"id": entry["id"], "delta": round(now - prev, 2), "value": now})
    movers.sort(key=lambda m: (-m["delta"], m["id"]))
    diff["aa_movers"] = {"up": movers[:3], "down": sorted(movers, key=lambda m: (m["delta"], m["id"]))[:3]}

    prev_prices = baseline.get("prices") or {}
    price_moves = []
    appeared, vanished = [], []
    prev_discounts = {
        model_id: row[3] for model_id, row in (baseline.get("prices") or {}).items()
    }
    for entry in catalog["models"]:
        old = prev_prices.get(entry["id"])
        if old is None:
            continue
        new_blended = entry["pricing"]["blended_per_1m"]
        price_moves.append(
            {"id": entry["id"], "old": old[2], "new": new_blended, "delta": round(new_blended - old[2], 4)}
        )
        old_disc, new_disc = prev_discounts.get(entry["id"]), entry["discount"]
        if old_disc is None and new_disc is not None:
            appeared.append(entry["id"])
        elif old_disc is not None and new_disc is None:
            vanished.append(entry["id"])
    price_moves.sort(key=lambda m: (m["delta"], m["id"]))
    diff["price_moves"] = {
        "down": price_moves[:3],
        "up": sorted(price_moves, key=lambda m: (-m["delta"], m["id"]))[:3],
    }
    diff["discounts"] = {"appeared": sorted(appeared), "vanished": sorted(vanished)}
    return diff


def fallback_texts(diff) -> dict:
    """Deterministic template sentences (same backtick convention as the LLM)."""
    if not diff["baseline_present"]:
        return {
            "week": (
                "Weekly comparison data is still building up; movement tracking "
                "starts once seven days of snapshots exist."
            ),
            "intelligence": (
                "No weekly intelligence baseline yet; AA movements will be "
                "summarised once history covers seven days."
            ),
            "prices": (
                "No weekly price baseline yet; price and discount movements "
                "will be summarised once history covers seven days."
            ),
        }
    week_bits = []
    if diff["new_pool_ids_count"]:
        week_bits.append(
            f"{diff['new_pool_ids_count']} new model(s) on OpenRouter"
        )
    balanced = diff["tabs"]["balanced"]
    movers = [e for e in balanced["entries"] if e["delta"]]
    for entry in movers[:2]:
        arrow = "climbed" if entry["delta"] > 0 else "fell"
        week_bits.append(
            f"`{entry['id']}` {arrow} {abs(entry['delta'])} spot(s)"
        )
    if balanced["new_ids"]:
        week_bits.append(
            f"`{balanced['new_ids'][0]}` entered the balanced top ten"
        )
    week = "This week: " + "; ".join(week_bits) + "." if week_bits else "This week: no top-ten changes."

    intel_bits = [
        f"`{m['id']}` {'+' if m['delta'] > 0 else ''}{m['delta']}"
        for m in diff["aa_movers"]["up"][:2] + diff["aa_movers"]["down"][:2]
    ]
    intelligence = (
        "AA intelligence moves: " + "; ".join(intel_bits) + "."
        if intel_bits
        else "No AA intelligence changes this week."
    )

    price_bits = [
        f"`{m['id']}` {m['old']} -> {m['new']}" for m in diff["price_moves"]["down"][:2]
    ]
    if diff["discounts"]["appeared"]:
        price_bits.append(
            "discount appeared for " + ", ".join(f"`{i}`" for i in diff["discounts"]["appeared"][:2])
        )
    prices = (
        "Blended price moves: " + "; ".join(price_bits) + "."
        if price_bits
        else "No notable price moves this week."
    )
    return {"week": week, "intelligence": intelligence, "prices": prices}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate highlight sections from weekly history data"
    )
    parser.add_argument("--catalog", required=True, help="today's catalog.json")
    parser.add_argument("--history", required=True, help="previous history.json (may be absent/empty)")
    parser.add_argument("--prev-highlights", required=True, help="previous highlights.json")
    parser.add_argument("--output", required=True, help="highlights.json destination")
    args = parser.parse_args(argv)
    try:
        with open(args.catalog) as fh:
            catalog = json.load(fh)
    except (OSError, ValueError) as exc:
        print(f"error: could not read catalog: {exc}", file=sys.stderr)
        return 1
    try:
        with open(args.history) as fh:
            history = json.load(fh)
    except (OSError, ValueError):
        history = {}
    diff = build_diff(catalog, history)
    # Task 3 inserts the reuse rule + LLM call here; Task 2 ships templates only.
    texts = fallback_texts(diff)
    document = {
        "schema_version": HIGHLIGHTS_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "fallback",
        "sections": texts,
    }
    with open(args.output, "w") as fh:
        json.dump(document, fh, indent=2)
        fh.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the full suite**

Run: `rtk pytest -q`
Expected: all pass.

- [ ] **Step 5: Lint and commit**

Run: `rtk ruff check .`

```bash
git add generate_highlights.py test_generate_highlights.py
git commit -m "Add weekly diff builder and fallback templates for highlights"
```

---

### Task 3: `generate_highlights.py` — reuse rule + LLM client + CLI completion

**Files:**
- Modify: `generate_highlights.py` (constants + `_post_chat`, `generate_with_llm`, reuse logic in `main`)
- Test: `test_generate_highlights.py`

**Interfaces:**
- Consumes: Task 2 (`build_diff`, `fallback_texts`).
- Produces: `generate_with_llm(diff, api_key) -> dict | None` (texts or None); `main()` implements the full reuse/generate/fallback decision and writes the final document.

- [ ] **Step 1: Write the failing tests**

```python
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
    prev = make_prev_highlights("openrouter", datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"))
    prev_file = tmp_path / "prev.json"
    prev_file.write_text(json.dumps(prev))
    monkeypatch.setattr(gh, "generate_with_llm", lambda *a, **k: pytest.fail("must not call LLM"))
    out = tmp_path / "out.json"
    assert gh.main([
        "--catalog", str(_write_catalog(tmp_path)),
        "--history", str(_write_empty_history(tmp_path)),
        "--prev-highlights", str(prev_file),
        "--output", str(out),
    ]) == 0
    written = json.loads(out.read_text())
    assert written["sections"] == prev["sections"]
    assert written["source"] == "openrouter"
    assert written["generated_at"] == prev["generated_at"]


def test_regenerates_fallback_output_regardless_of_age(monkeypatch, tmp_path):
    prev = make_prev_highlights("fallback", "2026-08-01T00:00:00+00:00")
    prev_file = tmp_path / "prev.json"
    prev_file.write_text(json.dumps(prev))
    monkeypatch.setattr(
        gh, "generate_with_llm",
        lambda diff, key: {"week": "w2", "intelligence": "i2", "prices": "p2"},
    )
    out = tmp_path / "out.json"
    assert gh.main([
        "--catalog", str(_write_catalog(tmp_path)),
        "--history", str(_write_empty_history(tmp_path)),
        "--prev-highlights", str(prev_file),
        "--output", str(out),
    ]) == 0
    written = json.loads(out.read_text())
    assert written["source"] == "openrouter"
    assert written["sections"]["week"] == "w2"


def test_llm_failure_falls_back_to_templates(monkeypatch, tmp_path):
    prev = make_prev_highlights("fallback", "2026-08-01T00:00:00+00:00")
    prev_file = tmp_path / "prev.json"
    prev_file.write_text(json.dumps(prev))
    monkeypatch.setattr(gh, "generate_with_llm", lambda diff, key: None)
    out = tmp_path / "out.json"
    assert gh.main([
        "--catalog", str(_write_catalog(tmp_path)),
        "--history", str(_write_empty_history(tmp_path)),
        "--prev-highlights", str(prev_file),
        "--output", str(out),
    ]) == 0
    written = json.loads(out.read_text())
    assert written["source"] == "fallback"
    assert set(written["sections"]) == {"week", "intelligence", "prices"}


def test_generate_with_llm_model_fallback_chain(monkeypatch):
    responses = {gh.LLM_MODELS[0]: RuntimeError("boom")}
    captured = []

    def fake_post(model, body, api_key):
        captured.append(model)
        if model in responses and isinstance(responses[model], Exception):
            raise responses[model]
        return '{"week": "w", "intelligence": "i", "prices": "p"}'

    monkeypatch.setattr(gh, "_post_chat", fake_post)
    texts = gh.generate_with_llm({"baseline_present": False}, "key")
    assert texts == {"week": "w", "intelligence": "i", "prices": "p"}
    assert captured == list(gh.LLM_MODELS)


def test_generate_with_llm_rejects_unparseable(monkeypatch):
    monkeypatch.setattr(gh, "_post_chat", lambda model, body, key: "not json at all")
    assert gh.generate_with_llm({"baseline_present": False}, "key") is None


def test_generate_with_llm_no_key_returns_none():
    assert gh.generate_with_llm({"baseline_present": False}, None) is None
```

Add the two file helpers used above:

```python
def _write_catalog(tmp_path):
    f = tmp_path / "catalog.json"
    f.write_text(json.dumps(make_diff_catalog()))
    return f


def _write_empty_history(tmp_path):
    f = tmp_path / "history.json"
    f.write_text(json.dumps({"snapshots": {}}))
    return f
```

(`import json`, `import datetime` at the top of the test file.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest test_generate_highlights.py -q -k "reuse or regenerates or llm"`
Expected: FAIL with `AttributeError: ... no attribute 'generate_with_llm'`.

- [ ] **Step 3: Implement**

In `generate_highlights.py`, add constants and the client:

```python
OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
LLM_MODELS = (
    "z-ai/glm-5.3-flash:free",
    "deepseek/deepseek-chat-v3.1:free",
    "meta-llama/llama-3.3-70b-instruct:free",
)
PROMPT_RULES = (
    "You write deployment notes for a model-ranking site. The user message is"
    " a JSON diff of the last 7 days. Reply with ONE JSON object with keys"
    ' "week", "intelligence", "prices" -- each value 1-2 short declarative'
    " sentences (max ~30 words each). State facts only from the diff; invent"
    " nothing; no hype or filler adjectives; wrap every model id in"
    " backticks; no markdown besides those backticks. If baseline_present is"
    " false, say in one sentence that weekly data collection is still"
    " building up."
)
HIGHLIGHTS_MAX_AGE_HOURS = 24


def _post_chat(model, body, api_key):
    import urllib.request

    request = urllib.request.Request(
        OPENROUTER_CHAT_URL,
        data=json.dumps(body).encode(),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as resp:
        return json.load(resp)


def generate_with_llm(diff, api_key):
    """One grounded call; first model in the chain that answers wins.

    Returns the three texts, or None on no key / all-models failure /
    unparseable output -- the caller then falls back to templates.
    """
    if not api_key:
        return None
    body = {
        "messages": [
            {"role": "system", "content": PROMPT_RULES},
            {"role": "user", "content": json.dumps(diff)},
        ],
        "temperature": 0,
        "max_tokens": 300,
    }
    for model in LLM_MODELS:
        try:
            payload = _post_chat(model, body, api_key)
            content = payload["choices"][0]["message"]["content"]
            stripped = content.strip().removeprefix("```json").removeprefix("```")
            stripped = stripped.removesuffix("```").strip()
            parsed = json.loads(stripped)
        except Exception:
            continue
        if (
            isinstance(parsed, dict)
            and all(isinstance(parsed.get(key), str) and parsed[key].strip() for key in SECTION_KEYS)
        ):
            return {key: parsed[key] for key in SECTION_KEYS}
    return None
```

In `main()`, replace the `texts = fallback_texts(diff)` block with the full decision:

```python
    prev = None
    try:
        with open(args.prev_highlights) as fh:
            prev = json.load(fh)
    except (OSError, ValueError):
        prev = None

    def _prev_is_recent_llm():
        if not isinstance(prev, dict) or prev.get("source") != "openrouter":
            return False
        try:
            generated = datetime.fromisoformat(str(prev.get("generated_at")))
            if generated.tzinfo is None:
                generated = generated.replace(tzinfo=timezone.utc)
        except ValueError:
            return False
        age = datetime.now(timezone.utc) - generated
        return age < timedelta(hours=HIGHLIGHTS_MAX_AGE_HOURS)

    if _prev_is_recent_llm():
        document = prev
    else:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        texts = generate_with_llm(diff, api_key)
        source = "openrouter" if texts is not None else "fallback"
        if texts is None:
            texts = fallback_texts(diff)
        document = {
            "schema_version": HIGHLIGHTS_SCHEMA_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source": source,
            "sections": texts,
        }
    with open(args.output, "w") as fh:
        json.dump(document, fh, indent=2)
        fh.write("\n")
    return 0
```

(Add `import os` to the imports. Remove the Task 2 "Task 3 inserts" comment.)

- [ ] **Step 4: Run the full suite**

Run: `rtk pytest -q`
Expected: all pass.

- [ ] **Step 5: Lint and commit**

Run: `rtk ruff check .`

```bash
git add generate_highlights.py test_generate_highlights.py
git commit -m "Add LLM generation with reuse rule and template fallback to highlights"
```

---

### Task 4: `--highlights-file` validation + placement

**Files:**
- Modify: `build_site_data.py` (constants + `validate_highlights`; `main()` gains `--highlights-file`)
- Test: `test_build_site_data.py`

**Interfaces:**
- Consumes: Task 3's document shape.
- Produces: `validate_highlights(document) -> None`; `main()` accepts `--highlights-file FILE` (optional; requires nothing else) and writes `highlights.json` next to `data.json` after validation.

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest test_build_site_data.py -q -k "highlights"`
Expected: FAIL with `AttributeError: ... no attribute 'validate_highlights'`.

- [ ] **Step 3: Implement**

In `build_site_data.py`:

```python
HIGHLIGHTS_SCHEMA_VERSION = 1
HIGHLIGHTS_SECTION_KEYS = ("week", "intelligence", "prices")
HIGHLIGHTS_SOURCES = ("openrouter", "fallback")


def validate_highlights(document) -> None:
    """Validate the highlights document (strict identity, tolerant extensions)."""
    if not isinstance(document, dict):
        raise ValueError("highlights document is not an object")
    if document.get("schema_version") != HIGHLIGHTS_SCHEMA_VERSION:
        raise ValueError(
            f"unknown highlights schema_version: {document.get('schema_version')!r}"
        )
    if document.get("source") not in HIGHLIGHTS_SOURCES:
        raise ValueError(f"unknown highlights source: {document.get('source')!r}")
    sections = document.get("sections")
    if not isinstance(sections, dict):
        raise ValueError("highlights sections must be an object")
    for key in HIGHLIGHTS_SECTION_KEYS:
        text = sections.get(key)
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"highlights section {key} must be a non-empty string")
```

In `main()`, add after `--history-prev-file`:

```python
    parser.add_argument(
        "--highlights-file",
        help="generate_highlights.py output; validated and written as highlights.json next to --output",
    )
```

and the write block (same pattern as history — validate before ANY write, so put the parse+validate with the other validations, right after the history block's parse but before the data.json write; simplest correct ordering: parse+validate the highlights file at the top alongside the catalog validation, then write `highlights.json` after the `history.json` write):

```python
    highlights = None
    if args.highlights_file:
        try:
            with open(args.highlights_file) as fh:
                highlights = json.load(fh)
            validate_highlights(highlights)
        except (OSError, ValueError) as exc:
            print(f"error: invalid highlights: {exc}", file=sys.stderr)
            return 1
```

and before `return 0`:

```python
    if highlights is not None:
        highlights_path = os.path.join(
            os.path.dirname(os.path.abspath(args.output)), "highlights.json"
        )
        try:
            with open(highlights_path, "w") as fh:
                json.dump(highlights, fh, indent=2)
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
git commit -m "Validate and publish highlights.json in the site data builder"
```

---

### Task 5: Site — `7-day` column + highlights sections

**Files:**
- Modify: `site/index.html` (fetches, movement logic, table header/cells, highlights markup + CSS)

**Interfaces:**
- Consumes: `history.json` (Task 1), `highlights.json` (Task 4), existing `data.json` render flow.

- [ ] **Step 1: Add the fetches**

In `init()`, alongside the existing `fetch("data.json", { cache: "no-cache" })`, add two same-shaped fetches (404-tolerant — a first deploy may lack them):

```javascript
      fetch("history.json", { cache: "no-cache" })
        .then(function (r) {
          return r.ok ? r.json() : null;
        })
        .then(function (payload) {
          historyData = payload && payload.snapshots ? payload : null;
          render();
        })
        .catch(function () {
          historyData = null;
        });

      fetch("highlights.json", { cache: "no-cache" })
        .then(function (r) {
          return r.ok ? r.json() : null;
        })
        .then(function (payload) {
          if (payload && payload.sections) renderHighlights(payload.sections);
        })
        .catch(function () {});
```

(`historyData` is a new `var` next to `var data = null;`. Note the data.json `.then` currently calls `render()` — history may arrive after it; `render()` is idempotent, so the history `.then` re-renders.)

- [ ] **Step 2: Add the `7-day` column**

Add `<th>7-day</th>` immediately after `<th>RANK</th>`; add `class="movement"` styling:

```css
      #table .movement {
        text-align: center;
      }
      .movement .up {
        color: #1a7f37;
      }
      .movement .down {
        color: #c62828;
      }
      .movement .flat {
        color: #1565c0;
      }
      .movement .new {
        color: #1a7f37;
      }
```

In `render()`, compute the baseline once:

```javascript
      var baseline = null;
      if (historyData && historyData.snapshots) {
        var today = new Date().toISOString().slice(0, 10);
        var d = new Date(today + "T00:00:00Z");
        d.setUTCDate(d.getUTCDate() - 7);
        baseline = historyData.snapshots[d.toISOString().slice(0, 10)] || null;
      }
```

and per row, insert the movement cell after the rank cell:

```javascript
      var movement = { cls: "", text: "" };
      if (baseline) {
        var prevRank = null;
        (baseline.tabs[current] || []).forEach(function (r) {
          if (r.id === row.id) prevRank = r.rank;
        });
        if (prevRank === null) movement = { cls: "new", text: "new" };
        else if (prevRank > i + 1) movement = { cls: "up", text: "↑" + (prevRank - i - 1) };
        else if (prevRank < i + 1) movement = { cls: "down", text: "↓" + (i + 1 - prevRank) };
        else movement = { cls: "flat", text: "•" };
      }
      // build the movement <td> (className = movement.cls, text = movement.text)
      // and tr.appendChild it between the rank cell and the model cell
      // (the rank/model cells come from the existing cells.forEach builder,
      // so insert explicitly: create td, set className/textContent, and
      // splice it in before the model cell)
```

(Concretely: keep the existing `cells.forEach` builder for the other columns but insert the movement cell first — e.g. build `tr` with the rank td, then the movement td (`td.className = "movement " + movement.cls; td.textContent = movement.text;`), then let the remaining cells append. The `cells` array starts with `String(i + 1)` for rank and `row.model` for model; simplest is to remove both from the generic loop and append rank, movement, then the rest.)

- [ ] **Step 3: Add the highlights sections**

Markup between the tip hints (`<p class="hint">` block ending `best.txt`) and `<div class="tabs">`:

```html
    <div id="highlights" hidden>
      <h2>News from OpenRouter</h2>
      <p id="hl-week"></p>
      <h2>Quality moves</h2>
      <p id="hl-intelligence"></p>
      <h2>Price movements &amp; deals</h2>
      <p id="hl-prices"></p>
    </div>
```

CSS:

```css
      #highlights h2 {
        font-size: 1.05rem;
        margin: 1rem 0 0.25rem;
      }
      #highlights p {
        margin: 0 0 0.5rem;
        font-size: 0.9rem;
      }
```

Renderer (escape first, then backticks):

```javascript
      function renderHighlights(sections) {
        document.getElementById("highlights").hidden = false;
        Object.keys(sections).forEach(function (key) {
          var el = document.getElementById("hl-" + key);
          if (!el) return;
          var safe = sections[key]
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;");
          el.innerHTML = safe.replace(
            /`([^`]+)`/g,
            '<code>$1</code>'
          );
        });
      }
```

- [ ] **Step 4: Verify by eye and run the suite**

Run: `rtk pytest -q && rtk ruff check .`
(Site JS is untested per codebase convention; verify the page locally — see Task 6.)

- [ ] **Step 5: Commit**

```bash
git add site/index.html
git commit -m "Render 7-day movement column and highlight sections on the site"
```

---

### Task 6: Workflow wiring, README, live acceptance

**Files:**
- Modify: `.github/workflows/publish.yml` (Build site step)
- Modify: `README.md` (new section + published-artifacts list)

**Interfaces:**
- Consumes: Tasks 1-5.

- [ ] **Step 1: Wire the workflow**

The Build site step env gains `OPENROUTER_API_KEY: ${{ secrets.OPENROUTER_API_KEY }}`; the run block gains, after `python model_compare.py --catalog > catalog.json`:

```yaml
          curl -fsSL https://canonical.github.io/model-compare/history.json -o history-prev.json || true
          curl -fsSL https://canonical.github.io/model-compare/highlights.json -o highlights-prev.json || true
          python generate_highlights.py \
            --catalog catalog.json \
            --history history-prev.json \
            --prev-highlights highlights-prev.json \
            --output highlights-new.json
```

and `build_site_data.py` gains the two arguments:

```yaml
            --history-prev-file history-prev.json \
            --highlights-file highlights-new.json \
```

- [ ] **Step 2: Update README.md**

In **Team usage: published picks**, extend the artifact sentence to mention `history.json` and `highlights.json`. Add a new section **"Weekly history and highlights"** (after "Catalog output"): the `7-day` column (`↑N` green / `↓N` red / `•` blue / `new`; weekly baseline, blank the first week); `history.json` (10 daily snapshots, what each holds); the three highlight sections and headings; regeneration cadence (LLM output reused <24 h, one grounded OpenRouter call, deterministic templates on any failure); `OPENROUTER_API_KEY` as the optional secret enabling LLM generation.

- [ ] **Step 3: Run the suite**

Run: `rtk pytest -q && rtk ruff check .`
Expected: all pass, clean.

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/publish.yml README.md
git commit -m "Wire history and highlights generation into the publish workflow"
```

---

### Task 7: Live acceptance verification

**Files:**
- None (verification; live network for fetch-side, LLM likely falls back locally without `OPENROUTER_API_KEY`).

- [ ] **Step 1: Full local build**

```bash
./model_compare.py --catalog > /tmp/opencode/hh-catalog.json
curl -fsSL https://canonical.github.io/model-compare/history.json -o /tmp/opencode/hh-history-prev.json || echo '{}' > /tmp/opencode/hh-history-prev.json
curl -fsSL https://canonical.github.io/model-compare/highlights.json -o /tmp/opencode/hh-highlights-prev.json || echo '{}' > /tmp/opencode/hh-highlights-prev.json
mkdir -p /tmp/opencode/hh-site
python generate_highlights.py --catalog /tmp/opencode/hh-catalog.json --history /tmp/opencode/hh-history-prev.json --prev-highlights /tmp/opencode/hh-highlights-prev.json --output /tmp/opencode/hh-highlights-new.json
python build_site_data.py \
  --best-file <(./model_compare.py --best) \
  --output /tmp/opencode/hh-site/data.json \
  --catalog-file /tmp/opencode/hh-catalog.json \
  --history-prev-file /tmp/opencode/hh-history-prev.json \
  --highlights-file /tmp/opencode/hh-highlights-new.json \
  --priority balanced=<(./model_compare.py --priority balanced --json --top 10) \
  --priority price=<(./model_compare.py --priority price --json --top 10) \
  --priority quality=<(./model_compare.py --priority quality --json --top 10)
python3 - <<'PY'
import json
h = json.load(open("/tmp/opencode/hh-site/history.json"))
assert len(h["snapshots"]) <= 10
hl = json.load(open("/tmp/opencode/hh-site/highlights.json"))
assert hl["source"] in ("openrouter", "fallback")
assert set(hl["sections"]) == {"week", "intelligence", "prices"}
print("history dates:", sorted(h["snapshots"]), "| highlights source:", hl["source"])
PY
```

- [ ] **Step 2: Second-run merge check**

Run Step 1's build again (same prev file fetched) and assert `history.json` still holds ≤ 10 dates with today upserted (no duplicates).

- [ ] **Step 3: Full suite and lint, then report**

Run: `rtk pytest -q && rtk ruff check .`
Report: snapshot dates, highlights source (openrouter vs fallback), any surprises.

---

## Self-Review notes

- Spec coverage: carry-forward + 10-date retention + malformed-prev (Task 1), exact-7-day baseline + top-10-boundary `new` (Tasks 1-2, 5), numeric diff + first-week message (Task 2), reuse/LLM/fallback with grounding rules (Task 3), validators + fail-closed placement (Tasks 1, 4), site column + sections + `cache: "no-cache"` + escape-then-backticks (Task 5), workflow + `OPENROUTER_API_KEY` (Task 6), README (Task 6), live acceptance incl. merge check (Task 7).
- Type consistency: `build_snapshot(catalog)` / `merge_history(prev, snapshot)` / `validate_history(doc)` used identically in Task 1 code and tests; `build_diff(catalog, history)` / `fallback_texts(diff)` / `generate_with_llm(diff, api_key)` consistent across Tasks 2-3; `validate_highlights` consistent between Tasks 4 and 6.
- Known accepted nuance: snapshot tabs derive from the catalog's per-priority ordering (4-dp rounded overalls); a rounding-tie could diverge from `data.json`'s full-precision order — spec-sanctioned (catalog M-8 note).
- Deliberate sequencing: Task 2 ships `main()` with templates only; Task 3 completes the decision logic — each task lands green.
