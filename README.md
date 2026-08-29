# model-compare

A single-file Python CLI that picks the best value-for-money LLM on
[OpenRouter](https://openrouter.ai/) before you start a coding agent. It queries
the OpenRouter catalog, scores every candidate on quality, price, context
window and freshness, and prints the winners — either as a ranked table or as a
single parseable `provider/model` id.

Built by an AI coding agent (opencode, powered by GLM) in a single session,
from a one-paragraph brief.

## Quick start

```console
$ ./model_compare.py                          # top 5, balanced
$ ./model_compare.py --best                   # only "#1 model id" (parseable)
$ MODEL=$(./model_compare.py --best)          # shell integration
$ ./model_compare.py --priority price --top 3
```

No dependencies beyond Python 3.10+ (stdlib only). Exit codes: `0` success,
`1` fetch failure, `2` no candidates survive the filters.

## How it works

### Data sources

1. **OpenRouter** — `https://openrouter.ai/api/v1/models` (public, no key),
   the same endpoint used by
   [openrouterlist](https://github.com/jvrck/openrouterlist). Provides the
   catalog: per-token input/output prices, context window, listing date,
   supported parameters (used for the tool-calling check) and modality info.
2. **Artificial Analysis** — the [AA intelligence index](https://artificialanalysis.ai/models),
   a 0–100 composite of reasoning/coding/knowledge benchmarks. Obtained via:
   - **AA API v2** (`artificialanalysis.ai/api/v2/data/llms/models`) when an
     API key is supplied through `--aa-api-key` or the `AA_API_KEY` env var
     ([free key](https://artificialanalysis.ai)); full model coverage.
   - **Page scrape fallback**: the leaderboard embeds a JSON-LD benchmark
     dataset; the scraper extracts every entry carrying an
     `artificialAnalysisIntelligenceIndex` (a few dozen top models). Matching
     to OpenRouter ids is done by tiered slug/name normalization, then a
     token-overlap (Jaccard ≥ 0.5) fuzzy pass. Unmatched models simply score
     0 on quality — the tool degrades gracefully rather than failing.

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
prices, expired listings, and (with `--exclude-free`) rate-limited `:free`
variants.

## Artificial Analysis data, in short

The intelligence index is the only signal OpenRouter does not expose. The
script tries the AA API first (key required; free tier is enough), then falls
back to scraping the JSON-LD embedded in `artificialanalysis.ai/models`. Both
paths are cached identically; if both fail you get a warning on stderr and a
price/context/age-only ranking. `--quality-ref` controls how generous the
quality normalization is.

## Caching

Responses are cached under `~/.cache/model-compare/` (`XDG_CACHE_HOME` is
honored) with a 6-hour TTL (`--cache-ttl`); `--no-cache` forces a refetch. The
cache keeps repeated invocations (e.g. in shell prompts or wrappers) fast and
polite.

## Options

| flag | default | meaning |
|------|---------|---------|
| `--priority` | `balanced` | `price`, `quality` or `balanced` |
| `--top N` | 5 | how many models to list |
| `--best` | off | print only the #1 model id (for scripting) |
| `--json` | off | machine-readable output |
| `--min-context N` | 1000000 | hard context-window floor (tokens) |
| `--input-share F` | 0.75 | input share of the blended price (0–1) |
| `--recency-half-life D` | 120 | age decay half-life (days) |
| `--max-age-days D` | off | drop models listed more than N days ago |
| `--quality-ref N` | 70 | index counting as full quality score |
| `--aa-api-key KEY` | `$AA_API_KEY` | Artificial Analysis API key |
| `--exclude-free` | off | drop `:free` variants |
| `--no-require-tools` | off | allow models without tool calling |
| `--no-cache` / `--cache-ttl S` | 6h | cache control |

## Tests

A `pytest` suite in `test_model_compare.py` covers the pure logic — input
coercion, candidate filtering, scoring math, and Artificial Analysis
matching — with no network access (external calls are stubbed). Run it with:

```console
$ pytest
```

## License

GPL-3.0 — see [LICENSE](LICENSE).
