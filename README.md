# model-compare

A single-file Python CLI that picks the best value-for-money LLM on
[OpenRouter](https://openrouter.ai/) before you start a coding agent. It queries
the OpenRouter catalog, scores every candidate on quality, price, context
window and freshness, and prints the winners — either as a ranked table or as a
single parseable `provider/model` id.

Built by an AI coding agent (opencode, powered mostly by GLM 5.3 Flash).

## Quick start

```console
$ ./model_compare.py                          # top 5, balanced
$ ./model_compare.py --best                   # only "#1 model" as an opencode id
$ MODEL=$(./model_compare.py --best)          # shell integration
$ opencode --model $MODEL
$ ./model_compare.py --priority price --top 3
```

`--best` prints the pick ready to use: provider-qualified as opencode
expects, e.g. `openrouter/z-ai/glm-5.3-flash` (`openrouter/` comes from a
single constant in the script, so other providers can be supported later).
The `--json` output carries both forms: `model` (catalog id) and
`opencode_model` (qualified id).

No dependencies beyond Python 3.10+ (stdlib only). Exit codes: `0` success,
`1` fetch failure, `2` no candidates survive the filters.

## Team usage: published picks

A [GitHub Actions workflow](.github/workflows/publish.yml) publishes
[the picks](https://canonical.github.io/model-compare/) — a top-10 table per
priority (`balanced` / `price` / `quality`) with copy buttons for the current
#1 model, the ready-made `opencode --model …` command, and the curl variant —
plus a machine-readable `best.txt` for teammates who don't want to run the
script:

```console
$ opencode --model "$(curl -fsSL https://canonical.github.io/model-compare/best.txt)"
```

The same workflow publishes a [`catalog.json`](https://canonical.github.io/model-compare/catalog.json) artifact — see [Catalog output](#catalog-output).

Tip: save it as an alias so every launch picks up the fresh value:

```console
$ alias oc-best='opencode --model "$(curl -fsSL https://canonical.github.io/model-compare/best.txt)"'
```

One-time setup: enable **Settings → Pages → Source: GitHub Actions**. The
build (`build_site_data.py`) validates every payload and fails loudly, so a
broken run never deploys a broken site. `best.txt` always serves the
*balanced* #1, regardless of which tab the page shows. Published picks
consider ZDR models only, matching the tool's default.

## How it works

### Data sources

1. **OpenRouter** — `https://openrouter.ai/api/v1/models` (public, no key),
   the same endpoint used by
   [openrouterlist](https://github.com/jvrck/openrouterlist). Provides the
   catalog: per-token input/output prices, context window, listing date,
   supported parameters (used for the tool-calling check) and modality info.
2. **Artificial Analysis** — the [AA intelligence index](https://artificialanalysis.ai/models),
   a 0–100 composite of reasoning/coding/knowledge benchmarks. Obtained via:
   - **OpenRouter benchmarks (primary)**: the same undocumented frontend API
     behind the [`?discount=true`](https://openrouter.ai/models?discount=true)
     and ZDR filters also republishes the AA intelligence/coding/agentic
     indices per model (`benchmarks.aa`), keyed by exact OpenRouter id.
     Consumed first — no separate fetch, and exact keys make conflation
     impossible.
   - **AA API v2** (`artificialanalysis.ai/api/v2/data/llms/models`) for
     models OpenRouter does not cover, when an API key is supplied through
     `--aa-api-key` or the `AA_API_KEY` env var
     ([free key](https://artificialanalysis.ai)); full model coverage.
   - **Page scrape fallback**: the leaderboard embeds a JSON-LD benchmark
     dataset; the scraper extracts every entry carrying an
     `artificialAnalysisIntelligenceIndex` (a few dozen top models). Both
     fallbacks match to OpenRouter ids by exact slug/name only. The previous
     token-overlap fuzzy pass is gone: it paired `z-ai/glm-5.3-flash` with
     AA's `glm-5-3` entry and published the wrong model's index. Unmatched
     models simply score 0 on quality — the tool degrades gracefully rather
     than failing.

If no quality data is obtainable at all, the quality weight is dropped and the
remaining weights renormalize.

### Scoring

Each criterion is normalized to [0, 1] across the candidate pool and combined
with priority-dependent weights:

| priority  | quality | price | context | age |
|-----------|--------:|------:|--------:|----:|
| balanced  | 0.40    | 0.40  | 0.10    | 0.10 |
| price     | 0.20    | 0.60  | 0.10    | 0.10 |
| quality   | 0.60    | 0.20  | 0.10    | 0.10 |

- **price score** — prices are converted to USD per 1M tokens and blended:
  `blended = input_share × $in/M + (1 − input_share) × $out/M` (default
  `--input-share 0.75`, i.e. a 3:1 input:output mix, the convention used by
  Artificial Analysis and typical of coding-agent traffic). The score is the
  model's position on a **log-scaled** price axis within the current candidate
  pool: `1.0 − (log10(price+0.01) − log10(min+0.01)) / span`, clamped to
  [0, 1]; free listings score 1.0. Log-scaling keeps the pool's three-orders-
  -of-magnitude price spread from crushing everything above the cheapest
  listing into one undifferentiated bucket.
- **quality score** — AA intelligence index ÷ `--quality-ref` (default 70),
  clamped to [0, 1].
- **context score** — logarithmic ramp from `--min-context` up to 4× that
  threshold: headroom helps, but with diminishing returns.
- **age score** — exponential decay with a `--recency-half-life` of 120 days:
  a listing half the age of another scores ~0.5× higher on this axis.
- **context filter** — hard minimum (`--min-context`, default 1M tokens),
  so a cheap small-window model can never win a big-context job.

### What gets filtered out

Below-minimum context, non-text outputs (image/audio), listings without
tool-calling support (`--no-require-tools` to relax), unparseable/negative
prices, expired listings, `:batch` variants (asynchronous completion — no
good for interactive agents; `--include-batch` to keep them), models without
a zero-data-retention (ZDR) endpoint by default (`--no-zdr` to consider
everything), and (with `--exclude-free`) rate-limited `:free` variants.

## Artificial Analysis data, in short

The intelligence index no longer needs a separate source: OpenRouter
republishes the AA indices alongside its own benchmark data, keyed by exact
OpenRouter id, and the script reads them from the frontend API it already
fetches. Only models missing there fall back to the AA API (key required;
free tier is enough) and then the JSON-LD scrape embedded in
`artificialanalysis.ai/models` — both matched by exact slug/name only. All
paths are cached identically; if none yields a value you get a warning on
stderr and a price/context/age-only ranking. `--quality-ref` controls how
generous the quality normalization is.

## Caching

Responses are cached under `~/.cache/model-compare/` (`XDG_CACHE_HOME` is
honored) with a 6-hour TTL (`--cache-ttl`); `--no-cache` forces a refetch. The
cache keeps repeated invocations (e.g. in shell prompts or wrappers) fast and
polite.

## Discounts

The `DISC` column and the `--discount` filter use the same data as the
website's [`?discount=true` model filter](https://openrouter.ai/models?discount=true):
OpenRouter's frontend models API reports, per model variant, the fraction by
which the listed price is currently discounted (e.g. `75%`). The endpoint is
undocumented, so the tool degrades gracefully — if it breaks, every model
shows `--` and `--discount` returns nothing.

Ranking always uses the listed (undiscounted) prices; the discount is shown
as information. Variants are matched individually, so a `:batch` twin of a
discounted model only shows a discount when that variant itself is
discounted.

## Zero data retention

By default, only models with zero-data-retention (ZDR) endpoints are ranked —
providers that do not retain prompts or outputs. The ZDR set comes from the
same OpenRouter frontend API as the discounts (the website's `?zdr=true`
filter). If that data cannot be fetched, the tool refuses to rank rather than
silently considering non-ZDR models; pass `--no-zdr` to explicitly consider
everything.

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
needs nothing else), `sources` (`openrouter`, `aa` with `mode`
`openrouter`/`api`/`scrape`/`none` plus the `matched` and
`matched_openrouter` counts, `zdr` `ok`/`skipped`, `discounts`
`ok`/`unavailable` — where `unavailable` covers both a failed discount fetch
and a live pool with zero discounts), `pool` (`listed`, `candidates`,
`dropped`), `models`, `filtered`.

Each `models` entry carries: `id` (bare `provider/model`), `name`,
`provider`, `family` (heuristic: leading token of the slug, e.g. `glm-5.3`
→ `glm`; `null` when there is none), `pricing` (`input_per_1m`,
`output_per_1m`, `blended_per_1m` in USD per 1M tokens), `context`,
`listed_at`, `age_days`, `tool_calling`, `zdr`, `discount`, `expired`,
`quality` (AA intelligence index or `null`), `aa` (the OpenRouter-published
trio `intelligence_index`/`coding_index`/`agentic_index`, each possibly
`null`), `quality_match` (`openrouter`/`api`/`scrape`/`null`) and `scores` —
the four component scores plus
`overall` for all three priorities, so downstream consumers never re-run
the scorer.

`filtered` entries are `{"id", "name", "reasons"}`; the reason keys are the
same strings the tool counts internally:

```
malformed id, context, pricing, free, batch, no discount, not ZDR,
modality, tool calling, expired, age
```

`pool.dropped` lists all of them zero-filled. `--top` and `--priority` are
ignored with `--catalog` (the document always covers the full pool, sorted by
the balanced overall score); `--catalog` cannot be combined with `--best` or
`--json`.

## Options

| flag | default | meaning |
|------|---------|---------|
| `--priority` | `balanced` | `price`, `quality` or `balanced` |
| `--top N` | 5 | how many models to list |
| `--best` | off | print only the #1 model id (for scripting) |
| `--json` | off | machine-readable output |
| `--catalog` | off | print the full model catalog (ranked candidates + filtered, with reasons) as one JSON document |
| `--min-context N` | 1000000 | hard context-window floor (tokens) |
| `--input-share F` | 0.75 | input share of the blended price (0–1) |
| `--recency-half-life D` | 120 | age decay half-life (days) |
| `--max-age-days D` | off | drop models listed more than N days ago |
| `--quality-ref N` | 70 | index counting as full quality score |
| `--aa-api-key KEY` | `$AA_API_KEY` | Artificial Analysis API key |
| `--exclude-free` | off | drop `:free` variants |
| `--include-batch` | off | keep `:batch` (async completion) variants |
| `--discount` | off | only models with an active discount |
| `--no-zdr` | off | rank all models, not just zero-data-retention ones |
| `--no-require-tools` | off | allow models without tool calling |
| `--no-cache` / `--cache-ttl S` | 6h | cache control |

## Tests

A `pytest` suite in `test_model_compare.py` covers the pure logic — input
coercion, candidate filtering, scoring math, discount parsing, and
Artificial Analysis matching — with no network access (external calls are
stubbed). Run it with:

```console
$ pytest
```

## License

GPL-3.0 — see [LICENSE](LICENSE).
