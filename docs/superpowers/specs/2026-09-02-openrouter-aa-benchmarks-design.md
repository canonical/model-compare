# model-compare — OpenRouter-published AA benchmarks: design

- Date: 2026-09-02
- Status: approved design, pre-implementation
- Builds on: `docs/superpowers/specs/2026-09-02-catalog-output-design.md` (branch `openrouter-aa-benchmarks` is stacked on `catalog-output`)
- Scope: `model_compare.py`, `build_site_data.py`, `README.md`, tests. No workflow change (the artifact contract evolves additively; CI needs no new flags).

## 1. Objective

Consume the Artificial Analysis values OpenRouter already publishes — the same
data shown on https://openrouter.ai/models — instead of relying on the AA API
(needs a key) and the JSON-LD page scrape (11-entry floor, fuzzy-matched). The
immediate driver is a proven accuracy bug: the scrape + token-Jaccard matcher
conflates variants and generations. Live evidence (2026-09-02):

- AA's page carries one GLM entry (`glm-5-3`, 59.5); the matcher pairs
  `z-ai/glm-5.3-flash` with it (Jaccard 0.6 ≥ 0.5), so the tool publishes
  59.5 where AA/OR say 57.5.
- `openai/gpt-5.6-sol` pairs with `gpt-5-6-luna` (52.3 vs OR's 60.9);
  `meta/muse-spark-1.1` with `muse-spark-1-3-xhigh`; `google/gemini-3.6-flash`
  with `gemini-3-8-flash`. Every large disagreement is a mispairing, not
  staleness: the one exact-slug comparison available (`nvidia-nemotron-3-ultra…`)
  shows AA=38.318… vs OR=38.3.

Goals: (a) OR's per-slug `aa` values become the primary quality source —
exact keys make conflation impossible; (b) all three OR-published indices
(`intelligence_index`, `coding_index`, `agentic_index`) are exposed in the
catalog document (intelligence remains the only scoring input); (c) the
frontend endpoint is fetched the minimum number of times (two, same as today).

## 2. Verified data facts

Endpoint: `https://openrouter.ai/api/frontend/v1/models/find?output_modalities=text`
— the same URL `fetch_discount_map` already uses. Its `data` object carries,
besides `models`:

- `benchmarks`: 207 keys (dated permaslugs like `z-ai/glm-5.3-flash-20260826`);
  144 with an `aa` object, 118 with a numeric `intelligence_index`; `aa` holds
  `intelligence_index`, `coding_index`, `agentic_index` (0–100 floats).
- `zdr=true` is a **separate URL** (`…&zdr=true`, 309 entries). The base
  response has no explicit ZDR flag (only `dataPolicy.retentionDays` hints) —
  deriving ZDR from those would be a heuristic against an undocumented API.
  **Two loads therefore remain**: base (discounts + benchmarks) and `zdr=true`.
- Freshness: OR's copy tracks AA's live values (nemotron d=0.02;
  `glm-5.3-flash` 57.5 ≈ AA's current 57).
- Coverage vs today: 143 of 427 catalog ids match after normalization;
  62 of 149 context ≥ 1M models (the tool's default pool) vs 17 via scrape.
  Frontier models can be absent (`openai/gpt-5.2`, `anthropic/claude-opus-4.6`
  had no `aa` block) — hence a fallback chain.

## 3. Fetch consolidation (model_compare.py)

- New `fetch_openrouter_frontend(args) -> (discounts, zdr_ids, aa_by_id,
  cache_hits)`: reads the per-payload caches first; fetches
  `OPENROUTER_DISCOUNTS_URL` once when either base-derived payload is
  missing from cache, deriving the discount map and the AA map from its
  `data`; fetches `OPENROUTER_ZDR_URL` **only when** `not args.no_zdr` and
  the ZDR cache misses, deriving `zdr_ids` exactly as `fetch_zdr_set` does
  today (empty ⇒ unavailable ⇒ fail closed upstream, unchanged).
  `cache_hits` is the set of payload names served from cache
  (`"discounts"`, `"zdr"`, `"aa"`), letting `run()` reproduce today's
  stderr cache notes verbatim.
- Cache: one entry per payload — `openrouter-frontend-discounts`,
  `openrouter-frontend-aa`, and `openrouter-zdr` — storing the **derived**
  payloads (never the ~1 MB raw response; the analytics/endpoint_perf blobs
  are never read). Each entry keeps today's never-cache-empty rule: it is
  written only when its payload is non-empty, so any degraded sub-fetch
  self-heals on the next run and healthy payloads keep caching
  independently. A `--no-zdr` run touches no ZDR cache (as today), so it can
  never leave an empty-ZDR entry that a later normal run would read as
  "unavailable". Same TTL semantics (`--cache-ttl`, `--no-cache` bypass).
  The old `openrouter-discounts` file is simply abandoned (6 h TTL makes
  migration moot); `openrouter-zdr` is reused unchanged.
  `fetch_discount_map` and `fetch_zdr_set` are removed; their warning text
  and never-cache-empty behavior move into the new function per payload.
- Failure semantics (per source, decoupled as today): the base fetch failing
  ⇒ discounts `{}`, aa map empty, neither cached, warning emitted — and the
  ZDR fetch is still attempted, so a discounts outage alone never triggers
  fail-closed; ranking proceeds with discounts shown as `--`, exactly as
  when only the discounts fetch fails today. Fail-closed exit 1 happens only
  when ZDR filtering is on and the ZDR data itself is unavailable.
  `--no-zdr` skips the ZDR fetch entirely, as today.
- AA map: `{bare_openrouter_id: {"intelligence_index": f, "coding_index": f,
  "agentic_index": f}}` built from each `benchmarks` key by stripping the
  trailing `-YYYYMMDD` permaslug suffix (a key without a date suffix maps to
  itself unchanged — undated keys must not be silently dropped); when two
  dated keys strip to the same id the **latest date wins** (its values are
  the current ones).
  Lookup for a candidate normalizes the candidate id by dropping any
  `:variant` suffix (`z-ai/glm-5.3-flash:free` → `z-ai/glm-5.3-flash`) and
  hits the map directly — variants inherit their base model's trio.
- Non-finite guard: benchmark values are accepted only when
  `isinstance(x, (int, float)) and not isinstance(x, bool) and
  math.isfinite(x)`; anything else is treated as absent.

## 4. Quality resolution (model_compare.py)

Per candidate, first match wins:

1. `aa_by_id[cand_id].intelligence_index`, when present and finite →
   `quality_match = "openrouter"`. An `aa_by_id` entry whose
   `intelligence_index` is absent (even if `coding_index`/`agentic_index`
   are present) does **not** match here; resolution falls through to step 2.
2. AA API / page-scrape fallback via `match_quality(..., allow_fuzzy=False)` —
   **exact tiers only** (full id, base slug, display name, paren-stripped
   display). `match_quality` gains a keyword `allow_fuzzy` defaulting to
   `False` — the fail-safe default, since fuzzy pairing is the proven
   mispairing bug this design removes; the existing unit tests pinning
   fuzzy-tier behavior are updated to pass `allow_fuzzy=True` explicitly,
   and production keeps the default. No fuzzy pairing can enter the
   published output.
3. Nothing → `quality = None` (scores 0 on the quality axis, as unmatched
   models do today).

`aa_api_entries` / `aa_scrape_entries` / `build_aa_lookup` are unchanged data
providers; `build_aa_lookup` still returns `(exact, fuzzy)` (fuzzy is simply
unused when `allow_fuzzy=False`).

## 5. Catalog document (additive; `schema_version` stays 1)

- Each `models` entry gains `"aa"`: `{"intelligence_index": f|null,
  "coding_index": f|null, "agentic_index": f|null}` — always the
  **OR-published trio** (the only source carrying all three), independent of
  which source drove `quality`. `null` per field where OR lacks it.
- `quality` keeps its meaning (the scoring input); `quality_match` gains the
  value `"openrouter"` (`"openrouter" | "api" | "scrape" | null`).
  Implementation note: `build_catalog` currently gates an entry's `quality`
  on the run-wide `aa_source` (`"quality": cand["quality"] if aa_source else
  None`) — that gate must be rewritten: `quality` is now source-independent
  and must be published even when the AA fallback is entirely unavailable
  (`aa_source is None`).
- `sources.aa` becomes `{"mode": "openrouter"|"api"|"scrape"|"none",
  "matched": N, "matched_openrouter": N}`: `matched` = candidates with any
  quality; `matched_openrouter` = candidates whose quality came from OR;
  `mode` = `"openrouter"` when `matched_openrouter > 0`, else the AA fallback
  source (`api`/`scrape`), else `"none"`.
- Rounding/determinism: `aa` values emitted raw (OR publishes 1-decimal
  values); ordering of map iteration irrelevant (per-entry fields); the
  byte-identical-apart-from-`generated_at` promise is unaffected.- Rounding/determinism: `aa` values emitted raw (OR publishes 1-decimal
  values); ordering of map iteration irrelevant (per-entry fields); the
  byte-identical-apart-from-`generated_at` promise is unaffected.

## 6. Validator (build_site_data.py)

Additive to `validate_catalog`: `CATALOG_ENTRY_KEYS` gains `"aa"`; the `aa`
block must be an object with exactly the three keys, each `null` or a finite
number in [0, 100]; `quality_match`, when not null, must be one of the three
provenance values (`openrouter`, `api`, `scrape`); and `sources.aa` gains a
small check — `mode` must be one of the four values,
`matched`/`matched_openrouter` non-negative ints with
`matched_openrouter <= matched`. The producer→validator integration test
(`build_catalog` output through `validate_catalog`) is extended to a pool
that exercises all three provenances (openrouter / api / none).

## 7. Intended behavior changes (flagged)

- Published `quality` values change for models the fuzzy matcher mispaired
  (`glm-5.3-flash` 59.5→57.5, `gpt-5.6-sol` 52.3→60.9, `muse-spark-1.1`
  60.8→53.2, …); some non-OR-covered models lose an incorrect score and rank
  on price/context/age. Rankings (`best.txt`, catalog order, table) may
  shift — this is the correctness fix, not a regression.
- `catalog.json` gains fields (consumers must tolerate unknown keys, per the
  documented stability promise).

## 8. Tests

- `fetch_openrouter_frontend`: stubbed payload → correct discounts/zdr/aa
  derivation; base fetch only when a base-derived payload misses cache; ZDR
  fetch only when `not no_zdr` and the ZDR cache misses; `cache_hits`
  reports which payloads came from cache; base failure → discounts `{}`,
  aa `{}`, nothing cached, ZDR fetch still attempted (fail-closed only when
  ZDR itself is unavailable); per-source warning texts preserved.
- Cache invariants: an empty/degraded payload is never cached (next run
  self-heals); a `--no-zdr` run writes no ZDR cache — a `--no-zdr` run
  followed by a normal run on a warm cache must **not** fail closed.
- Test migration: the ~15 existing tests that call or `monkeypatch`
  `fetch_discount_map` / `fetch_zdr_set` (12 direct unit tests plus 3
  `run()`-level seams) are rewritten against `fetch_openrouter_frontend`;
  `make_catalog_entry` / `make_catalog` fixtures in `test_build_site_data.py`
  gain the `aa` key and the new `sources.aa` shape.
- AA map building: `-YYYYMMDD` stripping, undated keys mapping to
  themselves, latest-date-wins on duplicate keys, variant inheritance,
  non-finite/bool values skipped.
- `match_quality`: exact tiers work under the `allow_fuzzy=False` default;
  the fuzzy tier runs only when `allow_fuzzy=True` is passed explicitly
  (regression: `glm-5.3-flash` must NOT receive `glm-5-3`'s index, and the
  default call must not produce a fuzzy pairing).
- Catalog: `aa` block shape/values; `quality_match` = `"openrouter"` for
  OR-sourced entries; `sources.aa` mode/matched/matched_openrouter across
  mixed provenance; determinism re-verified modulo `generated_at`.
- Validator: `aa` block acceptance/rejection (missing key, non-finite,
  out-of-range), new `quality_match`/mode values, producer→validator
  integration across provenances.
- README: data-sources section rewritten (OR benchmarks primary; AA API then
  scrape as exact-only fallback; the mispairing incident as rationale);
  catalog section documents the `aa` block and new `quality_match` value.

## 9. Out of scope

Scoring on `coding_index`/`agentic_index` (fields are exposed, unused);
deriving ZDR from `retentionDays`; reducing to a single HTTP load (no safe
ZDR flag in the base response); caching the raw frontend payload; changes to
weights, filters, or exit codes.
