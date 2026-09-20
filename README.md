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
| `--version` | off | print `model-compare <VERSION>` and exit |

## Published picks

No need to run anything: a [GitHub Actions workflow](web/README.md) refreshes
[the picks](https://canonical.github.io/model-compare/) every 6 hours, and
`best.txt` always serves the current *balanced* #1 as a plain model id:

```console
$ opencode --model "$(curl -fsSL https://canonical.github.io/model-compare/best.txt)"
$ alias oc-best='opencode --model "$(curl -fsSL https://canonical.github.io/model-compare/best.txt)"'
```

The site also serves the machine-readable `catalog.json` (`--catalog` output)
plus weekly `history.json` and `highlights.json`. How that pipeline works:
[web/README.md](web/README.md).

## How it works

OpenRouter's public catalog provides prices, context windows and listing
dates; model quality comes from the Artificial Analysis intelligence index
(read via OpenRouter's own published benchmarks first, with graceful
fallbacks). Each candidate is scored on quality, blended price, context
headroom and listing age with weights depending on `--priority`, after hard
filters (context floor, text-only, tool-calling, ZDR by default). The full
scoring formulas, filter list, the `--catalog` JSON contract and the test
setup are in [DESIGN.md](DESIGN.md).

## License

GPL-3.0 — see [LICENSE](LICENSE).
