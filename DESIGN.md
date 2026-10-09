# Design

How the tool works under the hood. End-user docs live in the
[README](README.md), the site's docs in [web/README.md](web/README.md).

## Data sources

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
   - **AA API v2** (`artificialanalysis.ai/api/v2/language/models/free`) for
     models OpenRouter does not cover, when an API key is supplied through
     `--aa-api-key` or the `AA_API_KEY` env var
     ([free key](https://artificialanalysis.ai)); full model coverage. It
     matches to OpenRouter ids by exact slug/name only. The previous
     token-overlap fuzzy pass is gone: it paired `z-ai/glm-5.3-flash` with
     AA's `glm-5-3` entry and published the wrong model's index. Without a
     key there is no AA fallback: keyless runs get AA data only through the
     OpenRouter benchmarks, and do so silently (the catalog records the
     absence in `sources.aa.fallback`). Unmatched models simply score 0 on
     quality — the tool degrades gracefully rather than failing.

If no quality data is obtainable at all, the quality weight is dropped and the
remaining weights renormalize.

## Scoring

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
  [0, 1]; free listings score 1.0. Log-scaling keeps the pool's
  three-orders-of-magnitude price spread from crushing everything above the
  cheapest listing into one undifferentiated bucket.
- **quality score** — AA intelligence index ÷ `--quality-ref` (default 70),
  clamped to [0, 1].
- **context score** — logarithmic ramp from `--min-context` up to 4× that
  threshold: headroom helps, but with diminishing returns.
- **age score** — exponential decay with a `--recency-half-life` of 120 days:
  a listing half the age of another scores ~0.5× higher on this axis.
- **context filter** — hard minimum (`--min-context`, default 1M tokens),
  so a cheap small-window model can never win a big-context job.

## What gets filtered out

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
fetches. Only models missing there fall back to the AA API — the supported
V2 free-tier endpoint
[`/api/v2/language/models/free`](https://artificialanalysis.ai/data-api/docs)
(key required; any tier key works, free is enough; responses paginate at 200
models per page and are followed), matched by exact slug/name only. Without
a key there is no fallback and no warning: OpenRouter is the primary source.
With a key, an AA API failure or an empty result prints a warning on stderr.
Both paths are cached identically; models without a value rank on
price/context/age only. `--quality-ref` controls how
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
providers that do not retain prompts or outputs. The ZDR set comes from
OpenRouter's public per-endpoint ZDR list
(`https://openrouter.ai/api/v1/endpoints/zdr`, public; its OpenAPI entry gives
the response shape but little semantics): one entry per ZDR endpoint, keyed by
the exact model id. A model id counts as ZDR if and only if it has at least
one entry in that list whose `status` is the integer `0`. OpenRouter's API
schema (`EndpointStatus` in `https://openrouter.ai/openapi.json`) lists the
values `0`, `-1`, `-2`, `-3`, `-5` and `-10` without describing them, so the
meaning of the non-zero values is unverified; any status other than `0`,
including a missing one, does not count (fail closed). The website's
`zdr=true` model filter matches this at the model level, but the list is per
variant, so non-ZDR variants of a ZDR model are excluded — NVIDIA's `:free`
endpoints, for example, train on prompts and are not listed, while their paid
twins are. The default filter therefore excludes `:free` variants without a
ZDR endpoint of their own, and `:batch` variants as well: the list currently
has no `:batch` ids, so `--include-batch` keeps nothing unless `--no-zdr` is
also given (by default `:batch` ids are dropped as `batch`; with
`--include-batch` they are dropped as `not ZDR` instead). The list and the
previous source (the `models/find?zdr=true` frontend filter) also differ in
other ids, in both directions; in a comparison on 2026-10-08, none of the ids
only the list had was a text-output model, and almost all of the text-output
ids it lacked were `:batch` or `:free` variants. If the list cannot be fetched
or yields no entries, the tool refuses to rank rather than silently
considering non-ZDR models; pass `--no-zdr` to explicitly consider everything.

## Catalog output contract

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
the same output apart from `generated_at` and the age score (`age_days`/
`listed_at` use UTC date precision, but `scores.age` decays continuously, so
two runs on the same day differ slightly in `scores.age` and the deduplicator
sees new content). `schema_version`
starts at `1`: fields may be added without notice, but renaming or removing
one bumps the version.

Top level: `schema_version`, `tool`, `generated_at`, `parameters` (all knobs
plus the **effective** per-priority `weights` — reproducing `scores.overall`
needs nothing else), `sources` (`openrouter`, `aa` with `mode`
`openrouter`/`api`/`none`, `fallback` `api`/`none`, plus
the `matched` and `matched_openrouter` counts, `zdr` `ok`/`skipped`, `discounts`
`ok`/`unavailable` — where `unavailable` covers both a failed discount fetch
and a live pool with zero discounts), `pool` (`listed`, `candidates`,
`dropped`), `models`, `rankings`, `filtered`.

`sources.aa.mode` is `openrouter` whenever at least one candidate's AA data
came from OpenRouter, and otherwise names the fallback. It says nothing about
whether the AA API worked. `sources.aa.fallback` records that on its own:
`api` when the AA API yielded entries, `none` when no key was set or the API
failed or returned nothing. It is an additive field, so it arrived without a
`schema_version` bump. Earlier releases could also report `scrape` (a JSON-LD
page-scrape fallback, now removed); that value can no longer be produced and
`web/build_site_data.py` rejects it, which needed no `schema_version` bump
because no emitted document changes. With `AA_API_KEY` set,
`web/publish.py` fails the run unless `fallback` is `api` — a rejected key
yields `none` — and
checks this right after the `--catalog` step, before any artifact is written.

`rankings` maps each priority (`balanced`, `price`, `quality`) to the full
ordered list of candidate ids for that priority (every `models` id exactly
once; rank = index + 1). The order is exactly what a `--priority P` run
prints: the unrounded weighted score, then quality descending, blended price
ascending, id. `rankings` is an additive field, so it arrived without a
`schema_version` bump; content-deduplicating consumers saw a one-time content
change on the first run that emitted it. `rankings` adds no new source of
drift; the only intra-day variance remains the decaying age score.

Each `models` entry carries: `id` (bare `provider/model`), `name`,
`provider`, `family` (heuristic: leading token of the slug, e.g. `glm-5.3`
→ `glm`; `null` when there is none), `pricing` (`input_per_1m`,
`output_per_1m`, `blended_per_1m` in USD per 1M tokens), `context`,
`listed_at`, `age_days`, `tool_calling`, `zdr`, `discount`, `expired`,
`quality` (AA intelligence index or `null`), `aa` (the OpenRouter-published
trio `intelligence_index`/`coding_index`/`agentic_index`, each possibly
`null`), `quality_match` (`openrouter`/`api`/`null`) and `scores` —
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

On the published site the catalog is the ranking and timestamping authority.
It carries the ranking itself in `rankings`, and everything else projects it
or is checked against it: the `data.json` table rows are reordered to it (and
the build fails if the `--json` runs ranked different models), the
`history.json` tabs and the weekly highlights diff slice it, and
`web/publish.py` fails the run unless `best.txt` is `rankings.balanced[0]`.
The CLI and the catalog share one score function and one sort key in
`model_compare.py` rather than copies of either. `web/publish.py` runs
`--catalog` first to prime the shared caches, which makes the later
invocations very likely to rank the same inputs. It is not a guarantee:
degraded fetches (an empty discount map, no AA benchmark entries) are
deliberately never cached, so a later run may fetch fresh, different data.
The publish contract is fail-loud: such a mismatch makes `order_rows` or
`web/publish.py` fail the run, and the next run recovers on its own.
`data.json` is stamped with the catalog's `generated_at` (as is the
newest `history.json` snapshot), and `web/publish.py` fails the run when the
deployed artifacts' stamps disagree.

## Tests

A `pytest` suite covers the ranking logic (`test_model_compare.py` — input
coercion, candidate filtering, scoring math, discount parsing, Artificial
Analysis matching) and the site tooling (`web/` — the data builder, the
highlights generator, the preview script and the publish orchestrator), with
no network access (external calls are stubbed). Run it from the repo root:

```console
$ pytest
```

## Release checklist

1. Bump `VERSION` in `model_compare.py`, `web/publish.py` and
   `web/generate_highlights.py` so `--version` and the user agents match,
   then update the pinned literal in
   `test_model_compare.py::test_version_flag` (it exists precisely so a
   forgotten bump fails the suite).
2. Add a `CHANGELOG.md` entry dated today.
3. Run `pytest` (collects the root and `web/` suites).
4. `git tag -a vX.Y.Z -m "model-compare X.Y.Z"`, push main + tag.
5. `gh release create vX.Y.Z` with notes linking the pinned standalone
   download:
   `raw.githubusercontent.com/canonical/model-compare/vX.Y.Z/model_compare.py`.
6. Never rename the catalog `tool` literal or remove schema fields without
   bumping `schema_version`.
