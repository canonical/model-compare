# model-compare — full catalog output: design

- Date: 2026-09-02
- Status: approved design, pre-implementation
- Source brief: `upstream-catalog-brief.md`
- Scope: `model_compare.py`, `build_site_data.py`, `.github/workflows/publish.yml`, `README.md`, tests

## 1. Objective and contract

Add `model_compare.py --catalog`: one machine-readable JSON document on stdout
covering every model the tool evaluates — ranked candidates *and* filtered-out
models with drop reasons. The internal Canonical platform
(`tokens.canonical.com`) ingests this artifact on a schedule, so the document is
a **stable contract**: additive changes are fine; renaming or removing fields
requires bumping `schema_version` (initial value: `1`).

Stdlib only, Python 3.10+. Existing behavior (`--best`, `--json`, table, exit
codes, cache, filters, scoring) stays byte-identical; existing tests pass
untouched. This is an output-surface change only.

## 2. CLI

- New flag `--catalog` prints the document to stdout.
- `--catalog` is mutually exclusive with `--best` and `--json`
  (argparse mutually exclusive group → clear usage error, exit 2).
- `--top` is accepted but ignored: the catalog is always the full pool.
- Exit codes unchanged: `0` success, `1` fetch failure, `2` no candidates
  survive the filters (warn on stderr, no document).
- ZDR stays fail-closed: with the default ZDR filter, no ZDR data → exit 1,
  exactly as ranking does today (inherited from `run()`, no new logic).
  Under `--no-zdr`, per-model `zdr` is `null` and `sources.zdr` is
  `"skipped"` — never a silent `zdr: false`.
- Module docstring examples gain one `--catalog` line (epilog is generated
  from the docstring).

## 3. Document shape

Envelope (field order is the serialization order; `json.dumps(indent=2)`):

```json
{
  "schema_version": 1,
  "tool": "model-compare",
  "generated_at": "2026-09-02T09:15:00+00:00",
  "parameters": {
    "input_share": 0.75,
    "quality_ref": 70,
    "min_context": 1000000,
    "recency_half_life": 120,
    "max_age_days": 0.0,
    "zdr_required": true,
    "require_tools": true,
    "exclude_free": false,
    "include_batch": false,
    "weights": {
      "balanced": {"quality": 0.4, "price": 0.4, "context": 0.1, "age": 0.1},
      "price":    {"quality": 0.2, "price": 0.6, "context": 0.1, "age": 0.1},
      "quality":  {"quality": 0.6, "price": 0.2, "context": 0.1, "age": 0.1}
    }
  },
  "sources": {
    "openrouter": "ok",
    "aa": {"mode": "api", "matched": 214},
    "zdr": "ok",
    "discounts": "ok"
  },
  "pool": {
    "listed": 312,
    "candidates": 298,
    "dropped": {"malformed id": 0, "context": 4, "pricing": 0, "free": 0,
                 "batch": 2, "no discount": 0, "not ZDR": 8, "modality": 0,
                 "tool calling": 0, "expired": 0, "age": 0}
  },
  "models": [],
  "filtered": []
}
```

### Candidate entry (`models`, ranked)

```json
{
  "id": "z-ai/glm-5.3",
  "name": "GLM 5.3",
  "provider": "z-ai",
  "family": "glm",
  "pricing": {"input_per_1m": 0.6, "output_per_1m": 2.2, "blended_per_1m": 1.0},
  "context": 200000,
  "listed_at": "2026-01-15",
  "age_days": 231,
  "tool_calling": true,
  "zdr": true,
  "discount": 0.75,
  "expired": false,
  "quality": 68.4,
  "quality_match": "api",
  "scores": {
    "price": 0.83, "quality": 0.91, "context": 0.5, "age": 0.4,
    "overall": {"balanced": 0.78, "price": 0.8, "quality": 0.76}
  }
}
```

### Filtered entry (`filtered`)

```json
{"id": "some/model:batch", "name": "…", "reasons": ["batch"]}
```

### Field decisions

| Field | Rule |
|---|---|
| `id` | Bare OpenRouter id (`provider/model`), never opencode-qualified. |
| `name` | OpenRouter name with vendor prefix stripped (`Z.AI: GLM 5.3` → `GLM 5.3`) via the existing `PROVIDER_PREFIX_RE`; falls back to raw name / id. |
| `provider` | `id` before the first `/`. |
| `family` | Slug-prefix heuristic (user decision): the leading dash/underscore/digit-delimited token of the base slug, lowercased — `glm-5.3` → `glm`, `gpt-5.2-mini` → `gpt`, `claude-opus-4.6` → `claude`, `qwen3-max` → `qwen`, `deepseek-chat-v4` → `deepseek`, `nova-pro-2` → `nova`; `null` when there is no alphabetic prefix. Oddballs yield oddballs (`o4-mini` → `o`). Documented in README as a heuristic. |
| `pricing.*` | USD per 1M tokens, rounded to 6 decimals (same rounding as `--json`). |
| `context` | Tokens, int. |
| `listed_at` | UTC calendar date (`YYYY-MM-DD`) of the OpenRouter `created` unix timestamp; `null` when unknown. |
| `age_days` | Whole days between `listed_at` and today (UTC), int; `null` when unknown. See determinism. |
| `tool_calling` | `tools` and `tool_choice` both in `supported_parameters`. |
| `zdr` | `true`/`false` when ZDR data was used; `null` under `--no-zdr`. |
| `discount` | Fractional discount rounded to 4 decimals when it counts as a discount (`has_discount`); else `null`. |
| `expired` | The same expiry test the filter applies; `false` for every entry in `models` in practice. |
| `quality` | Raw AA intelligence index float, or `null` (never NaN). |
| `quality_match` | `"api"` when the AA source was the API, `"scrape"` when the page scrape, `null` when unmatched or no AA data. |
| `scores.*` | Component scores and `overall` rounded to 4 decimals, floats in [0, 1]. |
| `scores.overall` | All three priorities; components are priority-independent so this costs nothing and saves downstream re-runs. |

### Envelope decisions

- `parameters.weights` carries the **effective** weights (after the existing
  quality-blind renormalization) per priority — equal to the base table
  (`PRIORITY_WEIGHTS`) whenever any candidate has quality data. This makes
  `scores.overall` exactly reproducible downstream. (Deliberate choice over
  the brief's base-weight example; identical output in all normal cases.)
- `parameters.max_age_days` is added beyond the brief's field list (additive,
  self-consistent: it shapes the pool when set; `0.0` = off).
- `pool.dropped` emits **all known drop-reason keys, zero-filled**:
  `malformed id`, `context`, `pricing`, `free`, `batch`, `no discount`,
  `not ZDR`, `modality`, `tool calling`, `expired`, `age` — verbatim, the
  same strings the tool already counts internally and prints in the table
  footer note. Zero-filled keys make the contract self-describing.
- `sources.openrouter` is always `"ok"` in an emitted document (fetch failure
  exits 1 with no document).
- `sources.aa` = `{"mode": "api"|"scrape"|"none", "matched": N}`: `mode`
  maps from the AA fetch source (`AA API v2` → `api`, `AA page scrape` →
  `scrape`, none → `none`); `matched` = candidates matched to a quality
  score (same count the table note reports).
- `sources.zdr` ∈ `"ok" | "unavailable" | "skipped"`: `"skipped"` under
  `--no-zdr`; `"ok"` whenever a document is emitted with ZDR filtering
  active (`"unavailable"` is enumerated for contract completeness but can
  never appear — the run fails closed with exit 1 instead).
- `sources.discounts`: `"ok"` when the discount map is populated,
  `"unavailable"` when empty (the tool already treats an empty map as
  unavailable).

## 4. Determinism

`models` sorts by `scores.overall.balanced` descending, then `id` ascending;
`filtered` sorts by `id`. Same inputs must produce byte-identical output
apart from `generated_at` — downstream deduplicates on content.

Catch this design closes: `age_days` is currently wall-clock-derived
(`(now − created) / 86400`, float), so two runs minutes apart would differ
and break content dedup. The catalog therefore uses **date precision**:
`age_days` = whole days between `listed_at` and today (UTC). Determinism
holds within a UTC day; cross-day runs legitimately differ as models age.
The existing float `age_days` used by the table and `--json` is untouched.
The one other time-dependent evaluation is the expiry test, evaluated at
run time like the filter itself: a listing crossing its expiry mid-day
changes the pool — a legitimate data change, not a determinism violation.

## 5. Implementation mapping (model_compare.py)

| Change | Where | Why this way |
|---|---|---|
| `build_candidates(models, args, discounts, zdr_ids, filtered_out=None)` | filter section | Optional out-param appends `{"id", "name", "reasons": [...]}` when a list is passed; signature and 2-tuple return unchanged → existing tests untouched (they unpack the 2-tuple and only assert exact-dict equality on empty lists). |
| Four additive candidate keys: `created`, `tool_calling`, `zdr`, `expired` | `build_candidates` | Existing tests only assert exact-dict equality on empty lists (verified); `print_json` selects keys explicitly so `--json` stays byte-identical. |
| `effective_weights(args, quality_by_id)` helper | scoring section | Mirrors the quality-blind renormalization for any priority without touching `compute_scores`. |
| `build_catalog(...)` + `print_catalog(...)` | new Output subsection | Pure assembly + serialization from post-`compute_scores` candidates (which already carry `price_score`, `quality_score`, `context_score`, `age_score`, `quality`). |
| `run()` wiring: `if args.catalog: print_catalog(...); return 0` after `compute_scores`, before the `--top`/`limit` logic | CLI section | Catalog ignores `--top`; full pool. |
| Mutually exclusive `--best` / `--json` / `--catalog` group | `parse_args` | argparse usage error covers rejection. |

`family` derivation is a tiny pure function next to the catalog builder so it
is unit-testable in isolation.

## 6. Publication

- `build_site_data.py` gains `--catalog-file FILE`: validates the raw
  `model_compare.py --catalog` output loudly — `schema_version` known;
  `len(models) == pool.candidates`; every entry has all fields;
  scores within [0, 1]; `zdr` true on every model when
  `parameters.zdr_required` — and writes `catalog.json` as a sibling of the
  `--output` path ("next to data.json"). Omitted flag → no catalog written,
  so the current workflow invocation keeps working. A broken run fails the
  build, never deploys.
- `.github/workflows/publish.yml` build step gains
  `python model_compare.py --catalog > catalog.json` and passes
  `--catalog-file catalog.json`; `catalog.json` deploys alongside `data.json`
  and `best.txt`. The site does not render it.
- No `AA_API_KEY` secret → scrape fallback applies and `sources.aa.mode`
  records it — graceful, matching current behavior.

## 7. Tests

Catalog builder (stubbed fetches, no network):

- Envelope: `schema_version`, `tool`, `generated_at` ISO-with-offset format,
  `parameters`, `sources`, `pool` consistency (`listed ≥ candidates`,
  `dropped` sums + candidates = listed).
- Per-entry shape: every documented field present, correct types, `null`
  where unknown; prices/scores within documented rounding.
- `scores.overall` carries all three priorities and matches a manual
  weighted sum of components.
- Filtered entries: reasons reuse the internal drop-reason keys verbatim;
  a dropped model never appears in `models`.
- Ordering: `models` by `(-overall.balanced, id)`; `filtered` by `id`;
  two builds byte-identical modulo `generated_at`.
- `--no-zdr`: `zdr` is `null` per model, `sources.zdr == "skipped"`.
- ZDR fail-closed: no ZDR data + default flags → exit 1, no document.

`build_site_data.py` validation:

- Malformed envelope (each: bad `schema_version`, `models`/`candidates`
  count mismatch, missing entry field, out-of-range score, non-ZDR model
  under `zdr_required`) → non-zero exit, no `catalog.json` written.
- Valid envelope → `catalog.json` written next to the output path.

## 8. Acceptance criteria

1. `./model_compare.py --catalog | python3 -m json.tool` parses and matches
   the documented schema; counts internally consistent.
2. Two runs with a warm cache are byte-identical apart from `generated_at`.
3. Existing tests pass unchanged; new tests cover the contract above.
4. README documents `--catalog`, the published `catalog.json` artifact, the
   schema, the drop-reason keys, and the `schema_version` stability promise.

## 9. Out of scope

Site rendering of the catalog, historical snapshots, any API/server, changes
to ranking or filtering behavior.
