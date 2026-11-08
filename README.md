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
$ ./model_compare.py --reasoning-effort max  # re-rank at max reasoning effort
```

`--best` prints the pick ready to use: provider-qualified as opencode
expects, e.g. `openrouter/z-ai/glm-5.3-flash` (`openrouter/` comes from a
single constant in the script, so other providers can be supported later).
The `--json` output carries both forms: `model` (catalog id) and
`opencode_model` (qualified id), plus `discount_pct`, the `DISC` column's
display string (e.g. `25%` or `--`).

No dependencies beyond Python 3.10+ (stdlib only). Exit codes: `0` success,
`1` fetch failure, `2` no candidates survive the filters, `130` interrupted
(Ctrl-C while paging).

## Sample output

The table goes to stdout; the summary goes to stderr. The numbers below are
illustrative, not current data. The stderr lines come first:

```console
$ ./model_compare.py --top 4
74 of 468 listed models ranked
(dropped: 275 context, 61 not ZDR, 47 batch, 6 pricing, 5 tool calling)

RANK  MODEL               QUAL  $IN/M  $OUT/M  DISC  PREC  TIER   CTX    AGE  SCORE
----  ------------------  ----  -----  ------  ----  ----  -----  -----  ---  -----
   1  z-ai/glm-5.3-flash  58.0  0.120   0.400    --   fp8     --  1.00M  42d  0.812
   2  acme/coder-mini     51.0  0.250    1.00   25%  bf16  >200k  1.05M  3mo  0.744
   3  moonshot/kimi-lite  49.0  0.300    1.20    --   fp8     --  1.00M  5mo  0.701
   4  deepseek/v3-turbo      -  0.180   0.550    --   fp8   -38%  1.00M  20d  0.655

Weights: quality 40%, price 40%, context 10%, age 10%
Quality source: AA index v4.3 - live AA (2 models, checked 13:15Z) |
                our published catalog (1 model, as of 2026-10-23 13:15Z);
                1 model unmatched (no quality data; quality score 0)

QUALITY   AA intelligence index at the model's best measured selectable depth
          (default; --reasoning-effort overrides); "-" = unknown, scores 0
$IN/$OUT  USD per 1M tokens at the cheapest eligible endpoint, blended
          75% input / 25% output (--input-share); differs from the
          openrouter.ai default-endpoint price by design. A higher rate
          may apply once the prompt outgrows a model's short-context tier.
DISC      active discount
PREC      quoted precision ("!" = default routing favors cheaper
          sub-default endpoints)
TIER      how prices vary: >Nk = pricier above that prompt size; -N% =
          scheduled off-peak discount (price shown: the standard rate);
          SCHED = schedule with a discount under 1% (--off-peak lists them)
CTX       context window; AGE = days since listed on OpenRouter
SCORE     weighted blend of the four criteria above
```

On a terminal, the table header (except SCORE) and rank numbers are bold, the
#1 row is green (its rank stays bold), and the dropped line and the `-`
quality placeholders are dim. A `Quality source:` block that has unmatched
models turns yellow. Piped output is plain under `auto`. On a terminal the
table runs through a pager (`$PAGER`, default `less`); `--no-pager` prints
directly. The `less` fallback exits immediately when the output fits one
screen, and any pager is spawned with `LESS=FRX` when `$LESS` is unset — a
convention only `less` honors — so a bare `PAGER=less` renders colors
instead of showing as escape codes. Color stays on through the pager under
`auto`; `--color off` turns it off everywhere.

`--detail simple` shows RANK, MODEL, QUALITY, IN $/M, OUT $/M, DISCOUNT and
AGE; `--bare` drops the summary, the weights line, the quality block and the
legend, leaving only the table(s).

## Options

| flag | default | meaning |
|------|---------|---------|
| `--priority` | `balanced` | `price`, `quality` or `balanced` |
| `--top N` | 5 | how many models to list |
| `--best` | off | print only the #1 model id (for scripting) |
| `--json` | off | machine-readable output |
| `--catalog` | off | print the full model catalog (ranked candidates + filtered, with reasons, plus the per-priority model ranking) as one JSON document |
| `--detail {simple,expert}` | `expert` | table density; `simple` hides PREC, TIER, CTX and SCORE and uses the web page's labels (QUALITY, IN $/M, OUT $/M, DISCOUNT) |
| `--off-peak` | off | also list scheduled off-peak deals, ranked by discount size (capped at 10) |
| `--bare` | off | print just the table(s), no summary or legend |
| `--color {auto,on,off}` | `auto` | colorize the table; `auto` colors only on a terminal (including through the built-in pager) and `NO_COLOR` disables it, `on` forces color (e.g. `less -R`, ignoring `NO_COLOR`), `off` never colors |
| `--no-pager` | off | never page the table output even on a terminal |
| `--min-context N` | 1000000 | hard context-window floor (tokens) |
| `--input-share F` | 0.75 | input share of the blended price (0–1) |
| `--recency-half-life D` | 120 | age decay half-life (days) |
| `--max-age-days D` | off | drop models listed more than N days ago |
| `--quality-ref N` | 70 | index counting as full quality score |
| `--reasoning-effort E` | `default` | choices: `default`, `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`; `default` scores each model at its best measured selectable effort; an explicit effort selects exactly that effort and drops models that cannot run it (it never substitutes another effort) |
| `--depths` | off | also print the per-model depth detail below the table (the `DEPTH` and `QUAL` columns are always shown) |
| `--aa-api-key KEY` | `$AA_API_KEY` | Artificial Analysis API key |
| `--exclude-free` | off | drop `:free` variants |
| `--include-batch` | off | keep `:batch` (async completion) variants; no effect without `--no-zdr`, as the ZDR list has no `:batch` ids today |
| `--discount` | off | only models with an active discount |
| `--no-zdr` | off | rank all models, not just zero-data-retention ones |
| `--no-require-tools` | off | allow models without tool calling |
| `--no-cache` / `--cache-ttl S` | 6h | cache control |
| `--version` | off | print `model-compare <VERSION>` and exit |

## Published picks

No need to run anything: a [GitHub Actions workflow](web/README.md) refreshes
[the picks](https://canonical.github.io/model-compare/) every 3 hours, and
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
at each model's best measured selectable reasoning effort by default, resolved
through a three-source hierarchy with visible provenance (see
[Model quality](#model-quality) below). An explicit `--reasoning-effort`
selects that exact effort and filters models that cannot run it. Each
candidate is scored on quality, blended price, context headroom and listing
age with weights depending on `--priority`, after hard filters (context floor,
text-only, tool-calling, ZDR by default). The full scoring formulas, filter list, the `--catalog`
JSON contract and the test setup are in [DESIGN.md](DESIGN.md).

## Model quality

Quality is the [Artificial Analysis](https://artificialanalysis.ai/models)
intelligence index (0–100). Each model's value is resolved through three
sources in a strict order — rank, not freshness: a higher-ranked source
always wins, even when a lower-ranked one is newer.

1. **Live AA API** — with `--aa-api-key` (or `AA_API_KEY`) set, the AA v2
   API is consulted first; this is the live index.
2. **Our published catalog** — without a key, quality hydrates from
   [model-compare.dev/catalog.json](https://model-compare.dev/catalog.json),
   the artifact this repo's keyed CI runs publish, joined by exact model id.
3. **OpenRouter snapshot** — the AA indices OpenRouter republishes per
   listing, used only as a fallback for known non-reasoning models at effort
   `none` when neither AA path covers them.

Every quality value carries its provenance end to end: `--json` rows carry
`quality_source`, `quality_as_of` and `quality_effort`, `catalog.json`
entries and the site's QUAL tooltips show them, and the CLI's footer note
lists how many models came from each source and when each was checked.

AA measures a model at several reasoning efforts, and the scores differ. By
default, headline `QUAL` is the highest AA intelligence index among the
model's measured efforts that OpenRouter can run (`supported_efforts`). If
scores tie, the deeper effort wins. When `supported_efforts` is missing or
empty, all measured efforts are considered. The OpenRouter-default marker
(`*`) is informational; it does not determine the default score. An explicit
`--reasoning-effort <effort>` (`none`, `minimal`, `low`, `medium`, `high`,
`xhigh` or `max`) selects exactly that effort and drops models that cannot
run it. It never substitutes another effort. If AA has no measurement at
the selected effort, `QUAL` is `-` unless the known non-reasoning `none`
OpenRouter fallback supplies a score. Truly unknown quality (no usable AA
data) scores 0.

The `DEPTH` column names the effort behind `QUAL` (`high*` marks OpenRouter's
default for information only). In Simple view the effort appears after the
quality as `· <effort>`. Below the table, a depth block per top-N model lists
every supported and measured effort with its AA index, its AA cost per
benchmark task, the cost ratio against the default depth, and the OpenCode
variant suffix that selects it (`#max`). The block is off by default;
`--depths` turns it on, and the `DEPTH` and `QUAL` columns are always
shown. AA `$/task` is AA's measured benchmark cost
at AA's reference price, not the OpenRouter per-token price, and it never
enters the score. An AA row with no effort label (some models AA ranks only
once) cannot fill `QUAL`: the depth block shows that index as a reference
and says it may not be the quality you get at a selectable effort. In
OpenCode, append `#<effort>` to the model id
(`openrouter/anthropic/claude-sonnet-5.5#max`) to select a depth; the
model-compare OpenCode plugin pins each alias to its scored effort using
`quality_effort` in the schema-6 catalog.

**Why our QUAL can differ from openrouter.ai:** OpenRouter republishes a
**frozen copy** of the intelligence index per dated listing, while the AA
API is live — a model whose index has moved since OpenRouter's snapshot
will legitimately show different numbers here. The comparison is only
like-for-like when both sides are non-reasoning: OpenRouter's snapshot
reports no effort, so a reasoning model keeps the snapshot as reference
(`quality_or_index` and `quality_or_as_of` name it) but is not flagged as
drift. For a non-reasoning model, when the winning value and OpenRouter's
snapshot differ by 3 or more index points, the row is flagged as drift
(`quality_drift`) — visible as a tooltip note on the site and counted in the
CLI footer. Prices can differ from the website too, for a different reason:
this tool ranks at the
cheapest eligible endpoint, not the default one (see
[Tiered pricing](#tiered-pricing)). Provenance — the source plus its as-of —
makes every such discrepancy explainable instead of mysterious; the
background is in issues
[#17](https://github.com/canonical/model-compare/issues/17) and
[#18](https://github.com/canonical/model-compare/issues/18).

## Tiered pricing

OpenRouter publishes conditional price overrides per model
(`pricing.overrides`): long-context tiers that charge more above a prompt
threshold, and off-peak schedules that charge less during certain UTC
windows. This tool ranks every model at **the tier in effect at a prompt
size of `min_context` tokens** — with the default 1M context that is the
worst case, which is what a long coding session actually pays. Time-windowed
models are ranked at their peak window (deterministic) and annotated; models
whose windows do not tile the week exactly are dropped (`schedule` drop
reason). The table's TIER column marks a token threshold (`>100k`), an
off-peak discount (`-38%`), or a sub-1% schedule (`SCHED`);
`--json` carries `pricing_tier_prompt_tokens`, `base_input_usd_per_m`,
`base_output_usd_per_m`, `tier_note`, `sched_note`, `sched_detail` and
`time_schedule`. `catalog.json` publishes the full picture: `pricing.base`,
`pricing.tiers`, `pricing.schedule` (with per-window `coverage` fractions
for time-weighted pricing) and the top-level `schedules` deals index.

## Precision and routing

`PREC` is the quantization the ranked endpoint quotes, not a property of the
model: one id can be served by several endpoints at different precisions and
prices. The tool ranks at the cheapest eligible endpoint, so `PREC` names that
endpoint's precision; a trailing `!` warns that OpenRouter's default routing
would send more than half the traffic to cheaper sub-default endpoints. Hover
the cell on the site, or read `quant_note` in `--json`, for the cheapest
non-baseline endpoint and its price spread.

That is why the price you pay can differ from the ranked price. The ranking
uses the cheapest eligible endpoint, blended `--input-share` input/output;
openrouter.ai shows the default endpoint. Tiered pricing and off-peak
schedules move the bill too (see [Tiered pricing](#tiered-pricing)).

To reproduce the ranked choice, pass the `steering` block that every `--json`
row and every `catalog.json` model entry carries:

```json
{
  "model": "deepseek/deepseek-v4.1-flash-1017",
  "provider": {
    "order": ["DeepInfra/fp8", "Baidu/bf16"],
    "ignore": ["Morph/int4"],
    "quantizations": ["bf16", "fp8"],
    "allow_fallbacks": true
  }
}
```

`order` lists the baseline-eligible endpoints, cheapest first; `ignore` names
every live endpoint that is not eligible; `quantizations` keeps precisions at
or above the quoted one (`null` default precision omits the key);
`allow_fallbacks` lets OpenRouter fall back when the ordered endpoints are
unavailable.

With the OpenAI SDK, pass the `provider` object through `extra_body` --
unknown top-level parameters are dropped before the request is sent:

```python
client.chat.completions.create(
    model="deepseek/deepseek-v4.1-flash-1017",
    messages=messages,
    extra_body={"provider": steering["provider"]},
)
```

In OpenCode, the model-compare plugin applies the block for you, and its
`body` option forwards a custom request body to the provider request.
Reasoning depth is part of the id (`#max`, `#high`, ...), and `:free` and
`:batch` ids are separate listings with their own prices; don't pair one
row's steering block with another variant's id. The
[site FAQ](https://canonical.github.io/model-compare/faq.html) covers the
same ground question by question.

## License

GPL-3.0 — see [LICENSE](LICENSE).
