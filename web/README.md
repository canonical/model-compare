# model-compare — the picks site

[canonical.github.io/model-compare](https://canonical.github.io/model-compare/)
is a static GitHub Pages site, rebuilt from scratch every 6 hours by the
`publish` workflow (plus manual runs). Everything it needs lives in this
directory; the ranking itself always comes from the standalone
[`model_compare.py`](../model_compare.py) at the repo root, invoked as a
subprocess.

## What gets published

| file | what it is |
|------|------------|
| `index.html` | the picks page: top-20 table per priority tab (10 shown, "Show more" expands) with copy buttons, a 7-day movement column, the TIER column and the Off-peak deals table (dark theme by default, light when the system or browser reports light) |
| `data.json` | the rendered rows: per-priority top 20 plus the current best id and the off-peak `schedules` rows |
| `best.txt` | the current *balanced* #1 as a plain `opencode --model`-ready id |
| `catalog.json` | the full `--catalog` document (stable contract, see [DESIGN.md](../DESIGN.md)) |
| `history.json` | rolling 10-day snapshot history feeding the 7-day column |
| `highlights.json` | the three prose sections at the bottom of the page |

## Pipeline

The workflow runs the pytest suite, then a single command:

```console
$ python web/publish.py            # assembles ./_site/ and fails loudly
```

`web/publish.py` runs `model_compare.py --catalog` first (priming the shared
caches), then `--best` and the per-priority `--json` runs, fetches the
previously published `history.json` / `highlights.json` (missing files are
fine — first run), regenerates both via `web/generate_highlights.py` and
`web/build_site_data.py`, and assembles the deploy directory. The table rows,
history tabs and highlights diff are all projections of the catalog's
`rankings`. Every payload is validated; a broken run never deploys.

Repository secrets: `AA_API_KEY` widens quality coverage (AA API fallback),
`OPENROUTER_API_KEY` enables LLM-written highlights — without it the site
serves deterministic templates. Only LLM highlights younger than 24 h are
reused between runs.

## Local preview

```console
$ web/preview.sh                # serve with the live site's data
$ web/preview.sh --build        # run the full pipeline locally instead
```

`--build` fetches the deployed site's previous history/highlights and, if
`OPENROUTER_API_KEY` is set, may make one real OpenRouter call for the
highlights.

## Details

- **7-day column** — each row is compared against the history snapshot dated
  exactly one week back: `▴N` if the model climbed, `▾N` if it slipped, `•` if
  it held rank, `new` if it was not in the top 20 then. Blank during the first
  week of data collection — also deliberately true for one week after the
  catalog-v2 deploy, which reset history so base-tier baselines could not be
  diffed against effective-tier prices.
- **Off-peak deals** — the second table lists the top 10 models with
  scheduled discount windows (sub-1% discounts excluded), ranked by discount size; prices shown are the discounted
  off-peak rates. It is built from the catalog's `schedules` index, which is
  deliberately unfiltered (it may list non-ZDR or sub-floor-context models
  that the ranked pool excludes).
- **Published picks consider ZDR models only**, matching the tool's default;
  `best.txt` always serves the *balanced* #1 regardless of the visible tab.
- **Highlights generation** walks the currently listed `:free` OpenRouter
  models best-first by their published intelligence index (the lineup rotates,
  so nothing is hardcoded) and falls back to deterministic templates on any
  failure — a broken LLM never fails the deploy.
- **Theming** — palettes and the dark-default/light-follows-system mechanism
  are CSS variables at the top of `site/index.html`.
