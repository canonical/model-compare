# model-compare — the picks site

[canonical.github.io/model-compare](https://canonical.github.io/model-compare/)
is a static GitHub Pages site, rebuilt from scratch on every push to `main`
by the `publish` workflow (plus 3-hour cron refreshes and manual runs).
Everything it needs lives in this
directory; the ranking itself always comes from the standalone
[`model_compare.py`](../model_compare.py) at the repo root, invoked as a
subprocess.

## What gets published

| file | what it is |
|------|------------|
| `faq.html` | the FAQ page: precision and the `!` marker, price versus bill, pinning routing with the `steering` block or the plugin, verifying the serving endpoint, depths and variants |
| `index.html` | the picks page: top-20 table per priority tab (10 shown, "Show more" expands) with copy buttons, a 7-day movement column, best-selectable-depth QUAL, the DEPTH column with a per-model depth detail row, the TIER column and the Off-peak deals table (dark theme by default, light when the system or browser reports light) |
| `favicon.svg` + `favicon-16x16.png` / `favicon-32x32.png` / `apple-touch-icon.png` | the icon: the SVG adapts its tile background to the browser theme (`prefers-color-scheme` inside the file — cream tile on dark chrome, dark tile on light chrome, and engines that skip the media query fall back to the dark base rule, matching the site's dark-by-default pages); the static PNGs are Safari/iOS fallbacks with the dark tile (#181a1b) baked in and the rounded corners cut out as real transparency, re-renderable with `web/make_favicon.py` (run it with `--check` to fail on stale committed output) |
| `data.json` | the rendered rows: per-priority top 20 plus the current best id, the off-peak `schedules` rows, and `generated_at` (ISO timestamp shown as "last refresh: …" between the tab and layout toggles); schema-versioned (`schema_version`, currently 3) and stamped with the `source_revision` (short commit hash) that built it — the footer links it to the exact commit |
| `best.txt` | the current *balanced* #1 as a plain `opencode --model`-ready id |
| `catalog.json` | the full `--catalog` document (stable contract, see [DESIGN.md](../DESIGN.md)) |
| `history.json` | rolling 10-day snapshot history feeding the 7-day column |
| `highlights.json` | the three prose sections at the bottom of the page |
| `litellm_rate_card.json` | pure LiteLLM cost-map fragment, one entry per ranked candidate keyed `openrouter/<catalog id>` |
| `litellm_steering.json` | routing/effort companion keyed like the rate card, with `_meta` provenance and the rate-card hash |
| `litellm.json` | LiteLLM's upstream map merged with the fragment (optional when the merged map is skipped) |
| `litellm_merge_status.json` | merge provenance, collisions, upstream failure count and reporting state |

## Pipeline

The workflow runs the pytest suite, then a single command:

```console
$ python web/publish.py            # assembles ./_site/ and fails loudly
```

`web/publish.py` runs `model_compare.py --catalog` first (priming the shared
caches), then `--best` and the per-priority `--json` runs, fetches the
previously published `history.json` / `highlights.json` / LiteLLM merged map
/ LiteLLM status (missing files are fine — first run), and fetches LiteLLM's
upstream cost map. It regenerates highlights with
`web/generate_highlights.py`, writes the LiteLLM artifacts with
`web/build_litellm_artifacts.py`, regenerates `data.json` with
`web/build_site_data.py`, and assembles the deploy directory. The table rows,
history tabs and highlights diff are all projections of the catalog's
`rankings`. Every payload is validated; a broken run never deploys. A bad
upstream fetch or an `openrouter/` key collision freezes only
`litellm.json`; the rest of the site still deploys, and
`web/report_litellm_merge.py` reports the condition through one deduplicated
GitHub issue.

Repository secrets: `AA_API_KEY` enables the live AA API quality source and
its per-effort rows (without it, quality hydrates from the published
catalog), `OPENROUTER_API_KEY` enables LLM-written highlights — without it the site
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
  week of data collection — also deliberately true for one week after each
  history reset: the catalog-v2 deploy (old snapshots priced at base-tier
  baselines not diffable against effective-tier prices), the history-v3
  cutover (old snapshots priced at advertised models-API scalars not diffable
  against baseline-endpoint prices), the history-v4 cutover (old snapshots
  carry max-effort quality scores not comparable with default-effort ones) and
  the history-v5 cutover (old snapshots carry default-effort quality and ranks,
  which are not comparable with best-selectable-depth quality and ranks).
- **Depth detail** — QUAL is scored at the model's best measured selectable
  effort by default; ties go to the deeper effort, and missing or empty
  `supported_efforts` falls back to all measured efforts. `QUAL -` means no
  measured selectable effort and no non-reasoning `none` fallback score. An
  explicit CLI `--reasoning-effort` selects exactly that effort and filters
  models that
  cannot run it. The DEPTH column and each row's expanded Depth section read
  the model's `reasoning` block from
  `catalog.json`: the AA index and AA `$/task` at every measured effort, the
  cost ratio against the default depth, and copy buttons for
  `openrouter/<id>#<effort>`. `*` marks OpenRouter's effective default for
  information only; it does not choose QUAL. An AA row for an effort
  OpenRouter does not support says `not selectable` and offers no copy button.
  An AA row with no effort label cannot fill QUAL: the Depth section shows
  its index as a reference and says it may not be the quality the user gets
  at a selectable effort. AA `$/task` is AA's measured benchmark cost at AA's
  reference price, not the TASK$ session cost. Without a catalog fetch the
  section says the details are unavailable.
- **Simple / Expert layout toggle** — the two buttons on the right of the tab
  line switch the ranked table between the full expert layout (14 columns:
  rank, 7-day, model, depth, quality, in, out, discount, precision, tier,
  context, age, score, task cost) and the simple summary (rank, 7-day,
  model, quality, in, out, discount, age, with unabbreviated headers; the
  depth detail stays one row-click away). The legend under the
  table hides the entries that explain columns absent from the simple view.
  The choice persists via a consent-first preference cookie
  (`model-compare-view=simple`, one year, scoped to this page's
  directory): the cookie is only written after the dialog's OK — Reject
  (or Escape) keeps the choice for the current page load but stores
  nothing, so the dialog stays closed until the next page load, while a
  stored consent that Expert later removes re-opens the dialog the next
  time Simple is picked. Nothing else is ever stored.
- **Off-peak deals** — the second table lists the top 10 models with
  scheduled discount windows (sub-1% discounts excluded), ranked by discount size; prices shown are the discounted
  off-peak rates. It is built from the catalog's `schedules` index, which is
  deliberately unfiltered (it may list non-ZDR or sub-floor-context models
  that the ranked pool excludes). Each row's QUAL carries the schedule
  entry's best measured selectable effort and provenance, so the Deals
  tooltip shows the entry's effort tag and provenance. Schedule entries carry
  no OpenRouter reference, so the main-table "not like-for-like" drift note
  does not apply to them.
- **Published picks consider ZDR models only**, matching the tool's default;
  `best.txt` always serves the *balanced* #1 regardless of the visible tab.
- **Highlights generation** walks the currently listed `:free` OpenRouter
  models best-first by their published intelligence index (the lineup rotates,
  so nothing is hardcoded) and falls back to deterministic templates on any
  failure — a broken LLM never fails the deploy.
- **Theming** — palettes and the dark-default/light-follows-system mechanism
  are CSS variables at the top of `site/index.html`.
