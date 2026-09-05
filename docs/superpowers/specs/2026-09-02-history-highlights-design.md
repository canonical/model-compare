# model-compare — weekly history, movement column, LLM highlights: design

- Date: 2026-09-02
- Status: approved design, pre-implementation
- Scope: `build_site_data.py`, new `generate_highlights.py`, `.github/workflows/publish.yml`,
  `site/index.html`, `README.md`, tests. `model_compare.py` itself is untouched (the
  catalog document already carries everything history needs).
- Builds on: the `--catalog` document (branch point: main @ 4096914).

## 1. Objective

Three user-facing capabilities driven by week-over-week change: (a) a rolling
one-week history of published snapshots enabling cross-date comparison, (b) a
per-row movement column on the site's top-ten tables, (c) three short, grounded
LLM-generated highlight sections on the page. Two new deployed artifacts
(`history.json`, `highlights.json`), both carry-forward files maintained by the
publish workflow.

## 2. History — `history.json`

### Mechanism (carry-forward, no repo writes)

Each publish: the workflow curls the currently-deployed `history.json`
(404-tolerant; `|| true`), passes it as `--history-prev-file` to
`build_site_data.py`, which upserts today's UTC-date snapshot, prunes to the
newest **10** dates, validates loudly, and writes `history.json` next to
`data.json`. Retention is 10, not 8: the 7-day lookback consumes the
8th-oldest slot, so two full-day outages of slack are needed before the
movement column can blank out. No new workflow permissions; the configured
concurrency group prevents overlapping runs (cancelling, not queueing — a
cancelled run never uploads its artifact, so the deployed `history.json` is
untouched); an artifact fetched one run stale self-heals next run. A malformed
or absent previous file yields a fresh one-document history — never a build
failure.

### Snapshot shape (per UTC date)

```json
{
  "schema_version": 1,
  "updated_at": "2026-09-02T09:15:00+00:00",
  "snapshots": {
    "2026-09-02": {
      "generated_at": "2026-09-02T09:15:00+00:00",
      "pool_ids": ["z-ai/glm-5.3", "..."],
      "tabs": {
        "balanced": [{"id": "z-ai/glm-5.3", "rank": 1, "quality": 59.5, "blended": 1.0}],
        "price": [],
        "quality": []
      },
      "aa": {"z-ai/glm-5.3": 59.5},
      "prices": {"z-ai/glm-5.3": [0.6, 2.2, 1.0, null]}
    }
  }
}
```

- `pool_ids`: every id in the catalog document (`models` + `filtered`) — makes
  "new on OpenRouter" derivable. Sorted, deduplicated.
- `tabs`: the day's top 10 per priority in rank order (`rank` is 1-based).
  Because history keeps only top-10 membership, a model entering today's top
  ten from rank 11+ can only ever render as `new`, never `↑N`; the same bound
  applies to "largest movers" in the highlight diff (top-10 membership on
  both ends).
- `aa`: intelligence index per model id, from the catalog's
  `aa.intelligence_index`.
- `prices`: `[input_per_1m, output_per_1m, blended_per_1m, discount]` arrays
  per model id, from the catalog's pricing/discount (blended retained — it is
  the price signal the tables rank on).
- **Scope asymmetry (deliberate):** `aa` and `prices` cover **candidates
  (`models`) only**; filtered entries carry no AA or pricing data, so they
  appear in `pool_ids` but never in `aa`/`prices`. A model moving
  `filtered → models` is therefore in `pool_ids` across weeks while its
  `aa`/`prices` history starts the day it becomes a candidate — consumers of
  `aa`/`prices` must null-check against missing prior values.
- Upsert rule: within a UTC day the snapshot is replaced by the latest run's
  data (last-write-wins per date). Prune keeps the newest 10 dates.
- Top-level `updated_at` is defined as the `generated_at` of the most recent
  snapshot (max over snapshots) — after a same-day last-write-wins upsert it
  is the day's latest run time, not the file-write time.

### Comparison rule

Deltas always compare against the snapshot dated **exactly 7 days before
today** (same weekday). If that snapshot is absent (the first week of
collection), the movement column renders empty for every row and the
highlights generator works from whatever snapshots exist (it says so when the
baseline is missing). After the first week the 7-day snapshot is always
present.

## 3. Movement column — site table

- New column headed **`7-day`** placed immediately after `RANK` in all three
  priority tabs.
- Computed client-side in `site/index.html` from `history.json` + the current
  `data.json` (all three fetched same-origin with `cache: "no-cache"`,
  mirroring the existing `data.json` fetch, so browsers never serve a stale
  history or highlights file).
- Cell values (plain UTF text, no emoji):
  - `↑N` green — climbed N positions,
  - `↓N` red — fell N positions,
  - `•` blue, no number — unchanged,
  - `new` green — in today's top ten but absent from the 7-day-ago snapshot,
  - blank — no 7-day baseline (first week).
- CSS: four classes (`.up`, `.down`, `.flat`, `.new`) with hex colors that hold
  on both light and dark schemes (page already sets `color-scheme: light dark`).

## 4. Highlights — `highlights.json`

### Shape

```json
{
  "schema_version": 1,
  "generated_at": "2026-09-02T09:15:00+00:00",
  "source": "openrouter",
  "sections": {
    "week": {"text": "…"},
    "intelligence": {"text": "…"},
    "prices": {"text": "…"}
  }
}
```

Static headings live in the site, not the artifact: **"News from OpenRouter"**
(week), **"Quality moves"** (intelligence), **"Price movements & deals"**
(prices). Model ids inside texts are wrapped in **backticks** and rendered by
the site as `<code>` (monospace) after HTML-escaping — never raw HTML.

### Producer — new `generate_highlights.py` (stdlib)

Inputs: current `catalog.json`, `history.json`, previous `highlights.json`
(all files staged by the workflow; no network fetching inside the script).

1. Compute a deterministic **numeric diff** against the exact-7-days-ago
   snapshot: per-tab rank deltas; ids new to the pool (count + any that entered
   a top ten); biggest AA intelligence movers (up and down, top 3 each, with
   values); biggest price moves and discount appearances/disappearances. All
   mover rankings are bounded by top-10 history membership (see §2). If a
   7-day baseline is missing, the diff says so explicitly (the LLM is told to
   say data collection is still building up, in one sentence).
2. **Reuse rule:** if the previous `highlights.json` has
   `source: "openrouter"` and is younger than 24 hours, copy it through
   unchanged (output = previous). Previous output with
   `source: "fallback"` is **always regenerated** on the next publish
   regardless of age, so a string of LLM failures cannot freeze fallback text
   for days — every publish retries until an LLM attempt succeeds.
3. **Generation:** one OpenRouter chat-completions call (env
   `OPENROUTER_API_KEY`; primary `:free` model id + ordered fallback model ids,
   first that responds wins; `temperature 0`; ~30 s timeout; `max_tokens` tight
   ~300) with the numeric diff and **grounding rules**: exactly one JSON object
   back with keys `week`/`intelligence`/`prices`; each value 1-2 short
   declarative sentences; no hype or filler adjectives; mention only ids and
   values present in the diff; wrap every model id in backticks; no markdown
   beyond those backticks.
4. **Fallback:** any failure (no key, all models fail, timeout, unparseable
   output) → deterministic templated sentences built from the same diff data
   (same backtick convention), `"source": "fallback"`. A broken LLM never
   fails the deploy.
5. Output passes through `build_site_data.py --highlights-file` for
   validation and placement next to `data.json`.

### Site rendering

Three sections between the tip hints and the tabs: static `<h2>`-style heading
(page uses `h1` only today; use a small heading consistent with existing type
scale) + one paragraph per section, text rendered with escape-then-backticks.
If `highlights.json` is missing or fails validation client-side, the sections
render empty — cosmetic, never blocking the table.

## 5. Workflow wiring (publish.yml)

Build site step gains, before `build_site_data.py`:

```
curl -fsSL https://canonical.github.io/model-compare/history.json -o history-prev.json || true
curl -fsSL https://canonical.github.io/model-compare/highlights.json -o highlights-prev.json || true
python generate_highlights.py \
  --catalog catalog.json --history history-prev.json \
  --prev-highlights highlights-prev.json --output highlights-new.json
```

and `build_site_data.py` gains `--history-prev-file history-prev.json` and
`--highlights-file highlights-new.json`, writing `history.json` and
`highlights.json` next to `data.json`. `OPENROUTER_API_KEY` becomes a new
optional secret env on the build step.

## 6. Validation (build_site_data.py)

- `history.json`: schema_version known; dates ISO; ≤ 8 snapshots; snapshot
  shape (tabs/pool_ids/aa/prices present, ranks consistent 1..10); validate
  AFTER merge+prune (a malformed previous file means "start fresh", not "fail").
- `highlights.json`: schema_version known; `source` in
  {`openrouter`, `fallback`}; the three section keys present, each text a
  non-empty string; `generated_at` ISO. Validation is strict on identity
  (schema_version, required keys) and tolerant on extensions (unexpected
  section keys are not rejected — forward compatibility). Text content is the
  LLM's — validated for shape, not prose.

## 7. Tests

- Snapshot upsert/prune: same-day last-write-wins; 10-date retention; fresh
  start on malformed previous; empty/absent prev file; `updated_at` = newest
  snapshot's `generated_at`.
- Comparison helper: exact-7-day baseline hit and miss; `new` case; unchanged
  case; rank arithmetic; top-10-boundary entrants render as `new`.
- Diff builder: deterministic ordering; numeric-only output; handles empty
  history (first-week message).
- `generate_highlights.py` with stubbed HTTP: reuse path (<24 h, LLM-sourced
  only); **fallback regeneration regardless of age**; LLM success (parsed
  JSON → texts); LLM failure → fallback templates; fallback carries
  backticks; model-fallback chain (primary fails → second id used).
- `build_site_data.py`: `--history-prev-file`/`--highlights-file` validation
  and sibling placement; malformed prev → fresh history, exit 0.
- Site: JS stays untested per codebase convention (inline script).
- README: history/movement-column/highlights sections; new artifacts listed;
  `OPENROUTER_API_KEY` documented; `7-day` column semantics.

## 8. Out of scope

Longer retention or intra-day history; rendering history as a chart; scoring
changes; highlights about models outside the diff data; any use of
`coding_index`/`agentic_index` in highlights beyond what the diff includes
later (YAGNI: intelligence + prices first).
