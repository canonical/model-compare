# model-compare

A single-file Python CLI that finds the best value-for-money LLM on [OpenRouter](https://openrouter.ai/) before you start a coding session. It pulls OpenRouter's live catalog, pairs it with benchmark scores from [Artificial Analysis](https://artificialanalysis.ai/models), filters out impractical listings, and ranks the survivors on quality, real endpoint price, context window, and freshness.

Don't want to run anything locally? Browse the live tables at **[model-compare.dev](https://model-compare.dev/)** — rebuilt every 3 hours by GitHub Actions — or install [`model-compare-opencode-plugin`](https://github.com/canonical/model-compare-opencode-plugin) in OpenCode v2.


## Use the published picks

You don't need to clone this repo to use the rankings. Every 3 hours, the site workflow publishes ready-to-consume files over HTTPS:

- **Web tables, plugin docs, and FAQ:** [model-compare.dev](https://model-compare.dev/)
- **OpenCode v2 plugin:** [`model-compare-opencode-plugin`](https://model-compare.dev/plugin.html) adds live `balanced`, `price`, and `quality` aliases straight into OpenCode's model picker, steers requests to the ranked provider, and pins reasoning depth automatically.
- **OpenCode v1 shell alias:** [`best.txt`](https://model-compare.dev/best.txt) always holds the current *balanced* #1 as a single `openrouter/<provider>/<model>` line:

  ```sh
  alias oc-best='opencode --model "$(curl -fsSL https://model-compare.dev/best.txt)"'
  ```

- **JSON feeds for scripts:** [`data.json`](https://model-compare.dev/data.json) (top-20 tables and off-peak deals) and [`catalog.json`](https://model-compare.dev/catalog.json) (the complete scored dataset, drop reasons, per-endpoint pricing, reasoning depths, and OpenRouter `steering` blocks).
- **LiteLLM cost map and routing:** point `LITELLM_MODEL_COST_MAP_URL` at [`https://model-compare.dev/litellm.json`](https://model-compare.dev/litellm.json) before starting your proxy, or consume [`litellm_rate_card.json`](https://model-compare.dev/litellm_rate_card.json) and [`litellm_steering.json`](https://model-compare.dev/litellm_steering.json) directly.

Practical setup examples (including OpenAI SDK `extra_body` steering and LiteLLM usage) are available on the [How to use](https://model-compare.dev/how-to-use.html) page; architectural contracts and LiteLLM hook details are documented in [DESIGN.md](DESIGN.md).


## Run the CLI

`model_compare.py` needs only Python 3.12+ (standard library, zero third-party packages):

```console
$ ./model_compare.py                          # top 5, balanced
$ ./model_compare.py --best                   # print only the #1 model ID for scripting
$ ./model_compare.py --priority price --top 3 # three cheapest sensible picks
$ ./model_compare.py --reasoning-effort max   # rank strictly at max thinking effort
$ ./model_compare.py --off-peak --depths      # include off-peak deals and per-depth scores
$ ./model_compare.py --safety-retention google  # match the published site
```

Exit codes: `0` on success, `1` on network/fetch failure, `2` when no models survive the filters or argparse rejects the command line, and `130` on `Ctrl-C`. A closed stdout pipe (for example `| head`) is treated as success.

A typical terminal run (illustrative numbers, legend footer omitted; summary lines go to `stderr`, the table to `stdout`, paged automatically through `$PAGER` — default `less` — when output exceeds one screen):

```console
$ ./model_compare.py --top 4
74 of 468 listed models ranked
(dropped: 275 context, 61 not ZDR, 47 batch, 6 pricing, 5 tool calling)

RANK  MODEL               DEPTH  QUAL  $IN/M  $OUT/M  DISC  PREC  TIER   CTX    AGE  SCORE
----  ------------------  -----  ----  -----  ------  ----  ----  -----  -----  ---  -----
   1  z-ai/glm-5.3-flash  high*  58.0  0.120   0.400    --   fp8     --  1.00M  42d  0.812
   2  acme/coder-mini     max    51.0  0.250    1.00   25%  bf16  >200k  1.05M  3mo  0.744
   3  moonshot/kimi-lite   high  49.0  0.300    1.20    --   fp8     --  1.00M  5mo  0.701
   4  deepseek/v3-turbo       -     -  0.180   0.550    --   fp8   -38%  1.00M  20d  0.655
```


## CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--priority {balanced,price,quality}` | `balanced` | Weighting preset (`balanced` 40/40, `price` 20/60, or `quality` 60/20 for quality vs. price; context and age get 10% each) |
| `--top N` | `5` | Number of ranked models to display |
| `--best` | off | Print only the #1 `openrouter/<model>` ID |
| `--json` | off | Print the ranked candidates as JSON (includes per-model `steering` blocks) |
| `--catalog` | off | Print the full evaluation (ranked candidates, filtered models with reasons, schedules, and per-priority rankings) as one JSON document |
| `--detail {simple,expert}` | `expert` | Table density (`simple` hides `DEPTH`, `PREC`, `TIER`, `CTX`, and `SCORE`) |
| `--off-peak` | off | Append a second table of up to 10 scheduled off-peak deals, ranked by discount size; deals use the models-API schedule price, not the endpoint quote, and a ranked model's deal is hidden when it is pricier than its quote |
| `--depths` | off | Print per-effort benchmark scores, task costs, and `#<effort>` suffixes below the table |
| `--bare` | off | Print only the table(s), omitting summaries and legends |
| `--color {auto,on,off}` | `auto` | ANSI color mode (`auto` respects `NO_COLOR` and TTY detection) |
| `--no-pager` | off | Print straight to stdout without paging |
| `--reasoning-effort E` | `default` | `default` scores each model at its best measured selectable depth, or an unlabeled AA measurement as the last tier; `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, or `max` filters and ranks at that exact depth and never uses the unlabeled row |
| `--min-context N` | `1000000` | Minimum context window in tokens |
| `--input-share F` | `0.75` | Input-token weight in the blended price (`0`–`1`) |
| `--quality-ref N` | `70` | AA intelligence index treated as a full `1.0` quality score |
| `--recency-half-life D` | `120` | Listing-age decay half-life in days |
| `--max-age-days D` | off | Drop models listed more than `D` days ago |
| `--no-zdr` | off | Include providers without zero-data-retention (ZDR) guarantees, including endpoints whose provider trains on prompts |
| `--flex` | off | Quote and steer the Flex service tier (typically lower price, lower processing priority; can exceed another provider's standard endpoint) |
| `--safety-retention {none,google,all}` | `none` | Whether endpoints that only offer Safety Retention count as ZDR: never, only for Google's models, or always |
| `--byok` | off | Include BYOK-only endpoints, which need your own provider key and are excluded by default |
| `--no-moderation` | off | Exclude endpoints whose provider requires moderation |
| `--include-deranked` | off | Include endpoints OpenRouter deprioritises in default routing |
| `--no-require-tools` | off | Include models that don't support tool/function calling |
| `--exclude-free` | off | Drop zero-cost/free models (`:free` listings and other zero-priced candidates) |
| `--include-batch` | off | Keep asynchronous `:batch` variants (requires `--no-zdr` in practice) |
| `--discount` | off | List only models whose **quoted endpoint** has an active discount |
| `--require-quality` | off | Rank only models with an AA quality record (an unlabeled measurement counts in a default run; drops `QUAL=-` models) |
| `--aa-api-key KEY` | `$AA_API_KEY` | Artificial Analysis API key (free at [artificialanalysis.ai](https://artificialanalysis.ai)); without one, local runs hydrate quality from `model-compare.dev/catalog.json` |
| `--openrouter-api-key KEY` | `$OPENROUTER_API_KEY` | OpenRouter API key (optional; lets `--json` and `--catalog` include OpenRouter's session-cost medians) |
| `--no-cache` / `--cache-ttl S` | `21600` (`6h`) | Bypass or adjust the local HTTP response cache |
| `--version` | off | Print version and exit |


## How rankings work

Out of the box, `model_compare.py` filters for models suitable for day-to-day coding agents: text output, tool-calling support, at least a **1M-token context window**, no asynchronous `:batch` listings, and at least one live **zero-data-retention (ZDR)** endpoint. Every one of those checks can be relaxed with the flags above. The ZDR check is applied to the quoted endpoint too, not just the model: a model cannot borrow compliance from a sibling endpoint that OpenRouter would not route to by default.

OpenRouter's ZDR list is wider than zero data retention: it also carries **Safety Retention** endpoints, which may keep prompts and responses, flagged or not, solely to investigate abuse. They never train on the data but they do store it, so a guardrail that enforces ZDR can refuse them. They do not count by default; `--safety-retention google` admits them for Google's models only, which is what the published tables use because that is what the org's guardrail permits. The site badges those quotes `SR`, not `ZDR`.

Genuinely free models (all endpoints zero-priced) are not dropped by a free-model gate, and `--exclude-free` drops them. They still need a ZDR endpoint in the default ZDR view, so non-ZDR free models normally appear only in All providers or with `--no-zdr`. A `:free` model listing can rank when its endpoints are all zero-priced and are not themselves marked `:free`/`:batch`; a zero-priced endpoint of a paid model is a variant and never sets that model's quote. Batch (`:batch`) listings are opt-in with `--include-batch`.

The published site has a **ZDR view** (the default) and an **All providers** view. The latter ranks the cheapest eligible endpoints regardless of retention and badges the quotes that still happen to be ZDR (`ZDR`) or Safety Retention (`SR`); its provider details load on first use. `best.txt`, history, highlights, deals and the LiteLLM artifacts stay on the ZDR view.

Surviving models are scored across four normalized dimensions — **quality**, **price**, **context headroom**, and **listing age**:

- **Quality (`QUAL` and `DEPTH`):** sourced from the Artificial Analysis intelligence index (`0`–`100`). With `AA_API_KEY` set, scores come straight from the live AA v2 API; without a key, local runs hydrate from [`model-compare.dev/catalog.json`](https://model-compare.dev/catalog.json) so your terminal matches the website out of the box. Because reasoning models score differently depending on thinking budget, each model is scored at its **best measured depth that OpenRouter lets you select** (shown in `DEPTH`). If AA reports an index without an effort label, a default run uses it as the last tier: `QUAL` shows the number with `DEPTH=unknown` (and `· unknown` in the Simple view); an explicit `--reasoning-effort` never uses it. A model with no quality record at all keeps `QUAL=-`, takes a zero quality score, and still competes on price/context/age; `--require-quality` excludes those rows when you want a benchmarked shortlist.
- **Endpoint pricing and steering (`PREC` and `TIER`):** one model ID on OpenRouter is often hosted by multiple providers at different quantizations and service tiers. Rather than trusting OpenRouter's headline number, `model_compare.py` quotes the **cheapest baseline-eligible standard-tier endpoint** at the **1M-token prompt tier**, blended 75% input / 25% output. Eligibility is endpoint-level: the quote must itself be ZDR-admitted, advertise the required tools and prompt context, and — when OpenRouter's frontend reports the endpoint's policy — be callable without a BYOK key unless `--byok` is set; a deranked endpoint is skipped while a better default route exists; a provider that trains on prompts is not ZDR; and moderation-required endpoints can be excluded with `--no-moderation`. That quote is a standard-tier **floor**, not a bill prediction: a plain request never reaches Flex or ultrafast endpoints (nor a closed-weight model's priority endpoint), but OpenRouter may serve an unsteered open-weight request from a priority endpoint at a higher rate. Pass the per-model `steering` block from `--json` or `catalog.json` to pin routing — it ignores every tier endpoint outside `order`, removing that exposure. Flex is never quoted unless you pass `--flex`. A trailing `!` on `PREC` warns that OpenRouter's unsteered routing would send most traffic to cheaper endpoints excluded specifically for precision or schedule reasons; ZDR, policy and context exclusions are not counted as quantization risk. `DISC` describes the quoted endpoint's own discount, not whichever endpoint the OpenRouter frontend features.

For the full scoring formulas, quality-source reconciliation rules, `catalog.json` v6 schema contract, and LiteLLM integration internals, see [DESIGN.md](DESIGN.md) and the [site FAQ](https://model-compare.dev/faq.html). For the static site pipeline, see [web/README.md](web/README.md).


## License

GPL-3.0 — see [LICENSE](LICENSE).
