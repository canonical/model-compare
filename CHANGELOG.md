# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.7.8] - 2026-11-08

### Fixed

- Tool version is `model-compare 0.7.8`.
- Session-cost medians no longer lose whole turn ranges when one slug's
  cells arrive through different pairing passes. The merge path unions
  ranges for an undated key that names an already-paired slug (pass 2)
  and for the single unambiguous dated strip onto an already-paired id
  (pass 3), not just across same-slug snapshots in pass 1. Ambiguous
  strips still drop, and the caller's payload dictionaries are never
  mutated.
- The simple-view legend quotes the default-effort marker (`"*"`) like
  the expert and depth legends.
- `--depths` with no ranked candidates no longer leaves a stray blank
  line where the depth section would have been.
- The deals legend runs through the same marker renderer as the other
  legends, and both it and the depth legend size their label column from
  the unmarked text, so an inline marker cannot leak literally or skew
  alignment.
- `web/publish.py`'s module docstring lists `faq.html` among the
  published files.

## [0.7.7] - 2026-11-08

### Fixed

- Tool version is `model-compare 0.7.7`.
- The `SCORE` column header is bold like every other header, in both the
  ranked and the off-peak deals tables.
- The `PREC` risk marker (`!`) is yellow in the table cell, matching the
  legend, including inside the green #1 row.
- Both legends quote the default-effort marker (`"*"`), consistent with
  the quoted `"-"` placeholder.

## [0.7.6] - 2026-11-08

### Added

- Tool version is `model-compare 0.7.6`.
- README: a "Precision and routing" section explains what `PREC` and the
  `!` marker mean, why the ranked price can differ from the bill, and how
  to pin routing with the per-model `steering` block -- including the
  OpenAI SDK `extra_body` gotcha and OpenCode's plugin `body` option
  (issue #20).
- Site: a `faq.html` page, linked from the index footer, answers the
  routing questions (precision and `!`, price versus bill, pinning with
  `steering` or the plugin, verifying the serving endpoint) and covers
  depths and variants. It ships with the other static site assets.
- `web/README.md` lists the new page among the published files.

## [0.7.5] - 2026-11-08

### Changed

- Tool version is `model-compare 0.7.5`.
- The column legend now prints directly under the main table and its
  `Weights`/`Quality source` footer instead of below every depth block, so
  the definitions sit next to the columns they describe (issue #28).
- Depth details are opt-in: the per-model blocks print only with
  `--depths`, replacing `--no-depths`. The `DEPTH` and `QUAL` columns stay
  in both modes.
- Legends highlight by element type, consistently across the main, depth
  and deals legends: term labels stay bold, CLI options and typed tokens
  (`--depths`, `#effort`) are cyan -- bold is reserved for labels, so an
  option never reads as another label -- literal values and the `-`
  placeholder are dim, and symbol markers (`*`, `!`) are yellow.
- The depth section is headed by a bold `Depth details`, and its legend is
  chunked one term per line with hanging indents instead of a single
  run-on sentence. The heading's "AA index / AA $ per task; OR $/M is
  constant across depths" parenthetical is gone; its one non-obvious fact
  lives in the legend's `OR $/M` entry.
- The depth tables gained the styling the other tables already had: the
  model id bold and the OR pricing plus `default:` label dim on each title
  line, a bold header row over the dim dash rule, yellow `*` markers, and
  dim `-` cells in the value columns. The `select` column stays plain.

### Fixed

- Under a pager the summary lines above the table no longer disappear:
  the whole human stream is paged, so the cache note, the ranked count and
  the dropped line are part of the pager's scrollback instead of being
  painted over on stderr.

## [0.7.4] - 2026-11-08

### Fixed

- Tool version is `model-compare 0.7.4`.
- Redirect handling never leaks the session-cost bearer token: the
  cross-origin redirect guard strips `Authorization` in addition to
  `X-api-key`, so a cross-host redirect or an https->http downgrade from
  the session-cost fetch (which authenticates with `Authorization: Bearer
  <key>`) no longer forwards the credential (issue #16). An explicit
  default port (`https://host:443/`) counts as the scheme's default, so
  credentials survive that same-origin redirect; a non-default port still
  strips.
- Session-cost medians resolve permaslug keys with the same three-pass
  catalog pairing as the AA benchmarks instead of blind stripping with
  latest-date-wins, which merged snapshots across listings of one model
  family (issue #22, the #17 conflation shape). The pairing consumes the
  raw models-API listing (`id`/`canonical_slug`) translated by the shared
  `catalog_pairing_entries` helper, so it engages in production exactly
  as in the reasoning-metadata path. When several dated permaslugs pair
  with one slug, the payloads are turn-range maps, so they merge per
  range (each range resolves to the latest date's cell) instead of the
  latest listing replacing the whole map. An ambiguous strip with no
  catalog pairing to disambiguate drops rather than guesses; the result
  cache key moved to `session-cost-v3` so cached shapes from earlier
  builds of this release are never read.

## [0.7.3] - 2026-11-04

### Added

- Tool version is `model-compare 0.7.3`.
- Depth detail: a model AA ranks without a reasoning-effort label (for
  example `z-ai/glm-5.3-flash`) now shows that AA index as a reference in
  the CLI and the site, with a note that it may not be the quality the user
  gets at a selectable effort. `QUAL` stays `-`, and the reference never
  enters the weighted score. Catalog `reasoning` blocks carry it as
  `aa_unlabeled` (`index`, `cost_per_task_usd`, `source`, `as_of`).

## [0.7.2] - 2026-10-30

### Added

- Tool version is `model-compare 0.7.2`.
- CLI: on a terminal the table output opens in a pager when it exceeds one
  screen, with colors intact — `$PAGER` verbatim when set, else `less
  --quit-if-one-screen --RAW-CONTROL-CHARS --no-init`; a pager that cannot
  start falls back to direct printing; Ctrl-C exits 130; `--no-pager` opts
  out; `--best`, `--json` and `--catalog` never page. A `less` reached
  through `$PAGER` is spawned with `LESS=FRX` when `$LESS` is unset, so a
  bare `PAGER=less` renders colors instead of raw escape codes.

## [0.7.1] - 2026-10-29

### Changed

- Tool version is `model-compare 0.7.1`.
- The site's help is tooltip-first: every column header in all four tables
  (ranked, depth, providers, off-peak deals) carries a tooltip, and the
  ranked table's legend paragraphs and the deals legend paragraph are gone.
  The facts that only the legends carried (the precision ladder, the SCHED
  marker, the deals clipping note) moved into the matching header tooltips;
  one primer sentence in the subtitle keeps the tooltips discoverable. The
  priority tabs, the view toggle and the refresh stamp gained tooltips too.
- Table styling: ranked-table rows highlight on hover and keep the hover
  background while their dropdown is open; the off-peak deals table hovers
  the same way; row borders and the line below the table headers use muted
  colors. The dropdown's Depth and Providers headings are left-aligned, and
  the dropdown tables use sentence-case, unabbreviated headers
  (Precision, Context, In $/M instead of PREC, CTX, $IN/M).

## [0.7.0] - 2026-10-28

### Changed

- Tool version is `model-compare 0.7.0`.
- Default quality selection now scores each model at its best measured
  selectable depth: the highest AA index at an effort OpenRouter supports,
  with ties going to the deeper effort. For example,
  `deepseek/deepseek-v4.1-flash-1017` was previously unscored because its
  OpenRouter default `high` had no AA measurement; it now scores at its
  selectable measured `max` depth. The Off-peak Deals catalog uses the same
  rule. Explicit `--reasoning-effort` still selects exactly the requested
  effort and drops models that cannot run it; it never borrows another depth.
- CLI and web QUAL descriptions now tell the same story. `*` continues to mark
  OpenRouter's default, for information only. The web site's muted selectable
  alternative under QUAL was removed because QUAL itself is now the best
  selectable depth. Detail rows are thin and do not wrap.
- Catalog `schema_version` 5 → 6: the producer, quality reader gate and web
  validator now all require schema 6. The OpenCode plugin reads
  `quality_effort` from the v6 catalog to pin aliases at the scored depth.
- `history.json` 4 → 5: default-effort history cannot be compared with
  best-selectable-depth scores and ranks, so history starts fresh. The 7-day
  movement and weekly highlights remain without a baseline until fresh
  snapshots cover a week. `data.json` stays schema 3 and `highlights.json`
  stays schema 1.

## [0.6.0] - 2026-10-27

### Added

- Thinking depths: the headline `quality` and every ranking now use the AA
  intelligence index at the model's **effective OpenRouter default reasoning
  effort** instead of AA's primary `max` row, and every ranked model carries
  a depth block: the measured efforts, their AA index, AA `$/task`, the cost
  ratio against the default, and the OpenCode variant suffix (`#max`) that
  selects it. The CLI gains a `DEPTH` column (`high*` marks the OpenRouter
  default) and a per-model depth detail below the table; `QUAL -` means AA
  has not measured the model at the selected effort, and the score is never
  borrowed from another depth. AA `$/task` is AA's reference-price benchmark
  cost: display-only, never in the weighted score.
- `--reasoning-effort <effort>` (`default`, `none`, `minimal`, `low`,
  `medium`, `high`, `xhigh`, `max`) selects the effort used for scoring and
  ranking; a specific effort also drops models that cannot run it (the
  `reasoning effort` drop reason). `--no-depths` hides the depth detail
  block; the `DEPTH`/`QUAL` columns stay.
- `--json` rows gain `quality_effort`, `default_effort` and a nested
  `reasoning` block (effective default, expanded `supported_efforts`, the
  raw flags and the depth records); catalog entries gain `quality_effort` and
  the same `reasoning` block, and `sources.aa` gains `depth_rows`,
  `matched_depth` and `matched_default`. Schedule entries gain
  `quality_effort`/`quality_as_of`/`quality_source`.
- Site: a `DEPTH` column in Expert view (hidden in Simple), the effort under
  `QUAL` (`46.8 · high` in Simple), the best selectable measured alternative
  as a muted number, and a Depth section in each expanded row with copy
  buttons for `openrouter/<id>#<effort>`. An AA row for an effort OpenRouter
  does not support says `not selectable` and is never advertised as the best
  alternative. Deals rows carry the same effort tag and "not like-for-like"
  drift wording.

- CLI: color support (`--color auto/on/off`, `NO_COLOR`), one-term-per-line
  legend, `--detail simple|expert`, `--off-peak` deals table, `--bare`, and
  grouped `--help`.

### Changed

- Catalog `schema_version` 4 → 5: `quality` now means the
  selected/default-effort score, and each entry carries `quality_effort` and
  a `reasoning` block. The hydration gate accepts v5 only; an older catalog
  warns and hydrates no quality rather than mixing semantics.
- `data.json` 2 → 3: `quality_effort` and `default_effort` join the required
  row keys; `quality_index` is retained and means the selected/default-effort
  score. `history.json` 3 → 4: max-based snapshots are not comparable, so
  history starts fresh; the 7-day arrows and weekly highlights degrade to
  their no-baseline state until fresh snapshots cover a week.
  `highlights.json` stays schema 1.
- The publish gates fail a keyed run when `sources.aa.depth_rows` or
  `matched_default` is zero, and warn when default-effort coverage is below
  25% of candidates. `web/publish.py`'s `data.json` stamp check pins version
  3.
- Drift is only flagged for non-reasoning models: OpenRouter's snapshot
  reports no effort, so a reasoning model keeps it as a reference but
  `quality_drift` is null. The AA entries cache key moved to
  `aa-intelligence-v4` (one record per measured effort, deduped by AA id).
- Downstream handoff: the catalog is now v5, which the published OpenCode
  plugin (npm 0.1.3) does not accept; the plugin release that accepts v5 and
  pins the effort on aliases is `canonical/model-compare-opencode-plugin#16`.
  The `data.json` v3 pin bump in `canonical/agent-provisioner` is
  `canonical/agent-provisioner#638`; the backend keeps its previous snapshot
  until it lands. This release is upstream-only: both consumer changes are
  separate follow-ups.

## [0.5.0] - 2026-10-25

### Added

- Site: a favicon. `web/site/favicon.svg` draws the mark — two congruent
  triangles (green #00BD00 pointing up and sitting higher, red #BD0000
  pointing down and sitting lower) on a rounded tile — and adapts the tile
  background to the browser theme via a `prefers-color-scheme` media query:
  cream (#f6f3ec) on dark chrome, dark (#181a1b) on light chrome (the
  dark tile is the base rule, so no-preference browsers match the site's
  dark-by-default pages). Static
  PNG fallbacks for SVG-intolerant browsers (Safari, iOS home screens) keep
  the dark tile (#181a1b) baked in with the rounded corners cut out as real
  transparency, and are rendered by `web/make_favicon.py`
  (deterministic, stdlib-only; `--check` fails on stale committed output);
  `publish.py` copies all four files into the deploy directory and
  `index.html` links them, with `img-src 'self'` added to the CSP so the
  icon loads under the existing policy.
- Site: Simple/Expert layout toggle for the ranked table. Expert stays the
  default full 13-column layout; Simple shows only rank, 7-day, model,
  quality, in, out, discount and age, spells its headers out
  (quality, discount; price columns keep the dollar units) and hides the
  legend entries that explain columns it does not show. The choice persists
  via a consent-first preference cookie (`model-compare-view=simple`, one
  year, scoped to the page's directory, `Secure` on secure contexts): the
  cookie is written only after the dialog's OK, Reject or Escape stores
  nothing (the choice lasts one page load), switching back to Expert removes
  the cookie, and the dialog only re-asks when no consent decision is
  stored. The dialog is a real modal to assistive tech
  (`role="dialog"`/`aria-modal`, focus moves in and back out), and the
  toggles expose their state via `aria-pressed`.
- Quality provenance end to end: every quality value is a record carrying
  its source (`aa` — the keyed AA API v2, the live index; `catalog` — the
  repo's published `catalog.json`; `openrouter` — OpenRouter's republished
  benchmarks) and its `as_of`, plus the OpenRouter comparison pair and the
  drift verdict. Surfaced in `--json` rows (`quality_source`,
  `quality_as_of`, `quality_drift`, `quality_or_index`, `quality_or_as_of`),
  catalog v4 entries, the CLI footer's quality note and the site's QUAL
  tooltip. The AA envelope's `intelligence_index_version` is captured too
  and shown in the footer when known; the AA entries cache key moved to
  `aa-intelligence-v3`, since entries now carry the fetch stamp and version.
- Unkeyed runs hydrate quality from the repo's published catalog
  (`https://model-compare.dev/catalog.json`; the github.io host 301s to
  plain http, so the direct https URL is used deliberately): an exact
  catalog-id join with no re-matching, a `schema_version >= 3` gate (v3
  rows normalized — `quality_match` "api" becomes "aa" and the as-of is the
  catalog's `generated_at`; v4 rows keep their recorded `quality_as_of`), a
  warn-but-use staleness check past 14 days, and full-artifact degradation
  on any parse surprise — the ranking never crashes, worst case it proceeds
  without catalog data. Every hydrated record is labeled `catalog` — the
  run-level provenance is the source chain (read from our published
  catalog), not the row's own label inside the artifact. Cached under
  `published-catalog`; only usable results are ever written.
- Quality drift flag: when the winning source (`aa` or `catalog`) and
  OpenRouter's snapshot for the same model differ by 3 or more index
  points, the record carries `quality_drift: true` plus the snapshot pair
  (`quality_or_index`/`quality_or_as_of`); the site tooltip appends
  "differs from OpenRouter snapshot (...)" and the CLI footer counts the
  rows.
- Site: QUAL cells carry a provenance tooltip ("AA intelligence index -
  source: aa, checked 13:15Z" / "source: openrouter, as of 2026-10-17",
  the drift note appended where flagged) and the
  legend explains it; `data.json` files without the new fields keep
  rendering.

### Changed

- Site: heading hierarchy fixed — the page title renders above the section
  headings (1.6rem vs 1.25rem); the ranked table, deals table and OpenCode
  section headings share one size instead of three different ones.
- Site: the tab line now centers "last refresh: date, time" between the
  priority tabs and the layout toggle.
- Site: the CSS and JS moved out of `index.html` into `site/style.css` and
  `site/app.js`; the page's Content-Security-Policy tightened accordingly
  (same-origin assets instead of inline code).
- Site: the OpenCode section headings keep their "OpenCode v1/v2" text
  without the badge styling.
- Quality precedence inverted: the keyed AA API is now the primary quality
  source and OpenRouter's republished benchmarks the last resort — used
  only for models neither AA path covers — instead of OpenRouter first
  (part of issue #18's clarity goals; issue #17 covers the conflation this
  removes). The source label `api` is renamed `aa`. Chains are mode-scoped:
  keyed runs consult the AA API then OpenRouter and never fetch the
  published catalog (a CI/site build hydrating from its own published
  output would be a self-reference loop); unkeyed runs hydrate the catalog
  then OpenRouter.
- The CLI footer's quality note lists per-source counts with each source's
  own as-of ("live AA (58, checked 13:15Z) | our catalog (12, as of
  2026-10-23 13:15Z) | OpenRouter snapshot (2, as of 2026-10-17)"), the AA
  index version when known, and footnotes for unmatched models ("unmatched
  score 0") and drift.
- Catalog `schema_version` 3 → 4 — per-model entries gain additive
  `quality_as_of` and `quality_drift`, `quality_match` carries the winning
  source's label, and `sources.aa` is reshaped to `{mode:
  "aa"|"catalog"|"none", matched_aa, matched_catalog, matched_openrouter,
  catalog_generated_at, intelligence_index_version}` (the `fallback` and
  `matched` keys are removed; the three counts are disjoint buckets whose
  sum equals the number of matched entries — the site validator enforces
  the invariant and rejects the v3 shape). `data.json` stays schema 2: the
  five provenance fields are additive row keys.
- The publish gate (`check_aa_source`) reads the new block: with
  `AA_API_KEY` set it fails closed unless the catalog shows live AA data
  (`mode` "aa" with `matched_aa` > 0) — a rejected key or an AA-less chain
  can no longer deploy silently; keyless publishes are never gated.
- Downstream note: the published catalog is now schema v4. The OpenCode
  plugin's only published release (npm 0.1.3) accepts v2 catalogs only and
  has therefore served remote-source users their last cached document since
  the v3 publish — the v3-accepting release the v3 cutover required never
  shipped. Its default vendored source (`auto`) is unaffected: it runs a
  pinned upstream v0.3.1 copy that emits v2 catalogs. A plugin release
  accepting the published schema is needed before remote-source users see
  fresh picks again.

### Fixed

- OpenRouter's republished AA benchmarks are keyed by exact catalog slug
  via the frontend payload's slug/permaslug pairs (three passes: permaslug
  pairs first, undated keys when unclaimed, unambiguous date-strips only),
  so each dated listing resolves to its own snapshot — no cross-listing
  borrowing (issue #17): `deepseek/deepseek-v4.1-flash` no longer borrows
  the `-1017` snapshot's score, and every OpenRouter-sourced value carries
  its listing's date as `as_of`. The benchmark cache key moved to
  `openrouter-frontend-aa-v2`, so shapes cached before the fix are never
  read.

## [0.4.0] - 2026-10-21

### Added

- Provider-aware ranking: models are now priced at their cheapest live
  baseline-eligible endpoint at default precision (the maker's own
  quantization when disclosed, else the best available; a pinned precision
  ladder plus a first-party/partner carve-out with live-verified aliases),
  instead of the models-API advertised scalar. The advertised scalars are
  retained as documentation-only `advertised_pricing`.
- Quantization annotations per model (`price_spread`, `cheapest_quantized`,
  `risk_flag`, `baseline_context_below_catalog`), a PREC column in the CLI
  and the site (with `!` risk marker and a `quant_note` tooltip), and a CTX
  caveat tooltip when the baseline endpoint's context is below the catalog
  context.
- OpenRouter `provider` steering block (`order`, `ignore`, `quantizations`,
  `allow_fallbacks`) in `--json` rows and catalog entries for callers such
  as the model-compare-opencode-plugin that pin routing explicitly.
- Per-task cost dimension from OpenRouter's `session-cost` dataset: catalog
  `session_cost` block (all four turn ranges, `as_of`/window metadata,
  "Data © OpenRouter, CC BY 4.0" attribution) and a TASK$ site column —
  outside the weighted score, "—" (never zero) when unavailable.
- Site: per-model expandable endpoint table rendered from a new
  `catalog.json` fetch.
- Two machine-readable drop reasons (`no_baseline_eligible_endpoint`,
  `endpoint_fetch_failed`; stale cached annotations ride dropped models'
  `filtered` entries) and a publish gate that fails the run when more than
  5% of the entered pool is dropped for an endpoint-derived reason.

### Changed

- Catalog `schema_version` 3 — the per-model `pricing` block now describes
  the baseline endpoint's price structure in full; `data.json` 2 (five new
  row keys); `history.json` 3.
- **History resets at the v3 cutover**: old snapshots describe old price
  semantics (advertised scalars) and are not mixed with baseline-endpoint
  prices. The 7-day movement arrows and the weekly highlights diff degrade
  to their no-baseline state until seven days of fresh snapshots accumulate.
- Ranked prices change value as well as meaning: 9 Google Gemini models are
  repriced onto newly-eligible maker endpoints (gemini-3.7-flash and
  gemini-3.8-flash prices halve, 0.75 → 0.375).
- The strict downstream consumer is the model-compare-opencode-plugin
  (npm 0.1.1 rejects `schema_version` != 2); a v3-accepting plugin release
  must land before or with the first v3 publish.

## [0.3.3] - 2026-10-14

### Added

- The site gains a bottom-of-page "Use it with OpenCode" section that
  distinguishes OpenCode v1 (`opencode --model`, shell alias, copy
  buttons) from OpenCode v2 (the model-compare-opencode-plugin, with a
  syntax-highlighted `opencode.jsonc` snippet, the
  `model-compare/balanced|price|quality` picker aliases and `/use-best`).

### Changed

- The published catalog refreshes every 3 hours instead of 6, halving the
  worst-case staleness window.
- The page subtitle states the 3-hour cadence.

## [0.3.2] - 2026-10-10

### Added

- The TIER tooltip shows the below-threshold (short-context) prices when a
  row carries them.
- Node-backed JS test harness pinning the site's inline formatters to the
  CLI's `fmt_*` helpers, including half-to-even tie rounding (the site no
  longer rounds exact ties half-away like `toFixed`).

### Changed

- The TIER and OFF-PEAK columns use the default text color and weight like
  the rest of the table.
- Clipped SCHEDULE values regained the dotted hover underline.

## [0.3.1] - 2026-10-10

### Added

- `data.json` is schema-versioned (`schema_version`, currently 1); the
  publish gate fails the deploy when the field is missing or unexpected.

## [0.3.0] - 2026-10-10

### Changed

- Models with OpenRouter `pricing.overrides` are now ranked and displayed at
  the tier in effect at a prompt size of `--min-context` tokens (default 1M:
  the worst case a long session pays) instead of their base-tier price — ~80
  of ~460 models carry price overrides; candidates among them were
  previously listed up to 6.7x too cheap. The
  table gains a TIER column (`>100k`, `-38%`, sub-1% `SCHED`), `--json`
  gains tier/schedule fields, and time-windowed models are ranked at their
  deterministic peak window with a fail-closed `schedule` drop reason.
- `catalog.json` `schema_version` 1 → 2: the pricing scalars now hold
  effective-tier values, with additive `pricing.base`/`pricing.tiers`/
  `pricing.schedule` (per-window `coverage` fractions) and a top-level
  `schedules` deals index. **Breaking for consumers that pin v1** —
  coordinate before upgrading.
- The site table shows 20 rows per priority (10 shown, "Show more" expands),
  history depth grows to match, and the page gains an Off-peak deals table.
  History restarts at the new schema version: one week without 7-DAY
  movement or price-move highlights.

### Fixed

- A malformed AA API page (a non-list `data` field, or an intelligence index
  value too large for a float) no longer discards the whole AA dataset
  (issue #12): the malformed page or item is skipped and pages already
  collected are kept.

## [0.2.7] - 2026-10-09

### Removed

- The dead JSON-LD page-scrape fallback for AA intelligence data
  (`artificialanalysis.ai/models`), which had returned no entries for weeks
  (issue #11). With `AA_API_KEY` set, a publish now requires the AA API
  itself (`sources.aa.fallback` `api`), so a rejected key can no longer hide
  behind the scrape. Keyless runs get AA data only through OpenRouter
  benchmarks and no longer print a warning. The `sources.aa.mode`,
  `sources.aa.fallback` and `quality_match` enums narrow: `scrape` can no
  longer occur and `build_site_data.py` rejects it. An AA cache entry left
  by the scrape is ignored.

### Fixed

- The test suite no longer writes fixture AA entries into the real user
  cache (`~/.cache/model-compare`): an un-isolated test wrote them there,
  and a publish on the same machine loaded them on a silent cache hit and
  published them as `sources.aa.fallback` `scrape`.

## [0.2.6] - 2026-10-09

### Changed

- The AA intelligence fallback now reads the supported V2 endpoint
  `/api/v2/language/models/free` (any tier key; responses paginate at 200
  models per page and are followed, capped at 25 pages) instead of the legacy
  `/api/v2/data/llms/models`, which Artificial Analysis retires on
  2026-11-04. Failure behavior is unchanged: page-1 errors still fall back to
  the page scrape, later-page errors keep the pages already fetched.
- `publish.py` fails when `AA_API_KEY` is set but neither the AA API nor the
  page scrape yielded data. The catalog gains an additive
  `sources.aa.fallback` field (`api`, `scrape` or `none`) that records what
  the AA fallback yielded, independently of `sources.aa.mode`; `mode` reads
  `openrouter` as soon as one candidate's AA data came from OpenRouter, so it
  hid a failed fallback. The gate reads `fallback`, fails closed when it
  is missing or unknown, and runs straight after the `--catalog` step, so a
  failed gate writes no publish artifacts to the output directory.
  `build_site_data.py` rejects a catalog without a valid `fallback`.

### Fixed

- The ZDR filter now reads OpenRouter's per-endpoint ZDR list
  (`/api/v1/endpoints/zdr`): a model id counts as ZDR only if it has at least
  one ZDR endpoint of its own. The previous source, the models `?zdr=true`
  filter, is model-level and also listed non-ZDR variants, so NVIDIA's
  `:free` endpoints — which retain prompts and train on them — were published
  with `zdr: true` (e.g. `nvidia/nemotron-3-ultra-550b-a55b:free`). The ZDR
  cache key moved to `openrouter-zdr-v3`, so cached id sets from the old
  source are never read. An endpoint counts only when its `status` is the
  integer `0`; OpenRouter's API schema lists `0`, `-1`, `-2`, `-3`, `-5` and
  `-10` without saying what they mean, so any other or missing status is
  treated as not ZDR (fail closed). The default ZDR filter also
  excludes `:free` variants without a ZDR endpoint of their own and `:batch`
  variants: the list currently has no `:batch` ids, so `--include-batch` has
  no effect unless `--no-zdr` is also given. Compared on 2026-10-08, the
  two sources also differ in other ids, in both directions: the new list has
  91 ids the old source did not list, none of them a text-output model, and
  lacks 74 text-output ids the old source listed, 72 of them `:batch` or
  `:free` variants.
- HTTP 4xx responses (such as 401, 403 or 429) are no longer retried; other
  network errors, timeouts and 5xx responses are still retried once.
- The AA intelligence cache key moved to `aa-intelligence-v2`, so entries
  cached before the V2 endpoint migration are never read.
- The AA API key is no longer forwarded when a redirect changes the scheme,
  host or port; urllib copied it onto the redirected request.

## [0.2.5] - 2026-09-28

### Added

- The catalog carries the ranking itself: a top-level `rankings` field maps
  each priority (`balanced`, `price`, `quality`) to the full ordered model id
  list, produced by the same shared score and sort key the CLI table uses.
  `data.json` rows, `history.json` tabs and the weekly highlights diff are now
  projections of that one ranking, and `publish.py` runs `--catalog` first so
  every invocation in a build reads the same cached data. A 4dp score tie can
  no longer make the site show movement arrows or climbed/fell prose that did
  not happen.

### Changed

- `publish.py` gates `best.txt` against `rankings.balanced[0]` before deploy,
  and fails loudly when the deployed artifacts' ranking-derived facts disagree.

## [0.2.4] - 2026-09-28

### Fixed

- The site's 7-DAY movement arrows no longer flicker on unchanged data:
  history tabs now rank with the same quality/blended/id tiebreak as the
  table, and `data.json` is stamped with the catalog's `generated_at`
  instead of a second clock. `history.json`'s `updated_at` pins to the
  snapshot being written, so a stale future-dated snapshot can no longer
  wedge deploys.
- The DISC column now uses the same formatter as the CLI (`discount_pct`
  ships through `--json`), so 0.025-style discounts display identically in
  both places.
- The weekly highlights' price section reports rises and ended discounts
  instead of only drops and appeared discounts.
- Catalog validation rejects a malformed `generated_at` (non-string or not
  ISO-8601) with a clean error instead of a traceback.

### Added

- `publish.py` fails loudly when the deployed artifacts' `generated_at`
  stamps disagree, and tests pin the table, history and highlights to one
  ranking and one clock.

## [0.2.3] - 2026-09-28

### Fixed

- Weekly highlights no longer report a model whose AA intelligence index or
  blended price is unchanged as a mover, and a model can no longer appear in
  both the `up` and `down` lists of a section.

## [0.2.2] - 2026-09-20

### Changed

- README slimmed to end-user content; site docs moved to `web/README.md`,
  design details and the release checklist to `DESIGN.md`. Internal process
  docs are no longer committed (gitignored under `docs/superpowers/`).

## [0.2.1] - 2026-09-20

### Changed

- The picks site ships an explicit palette: dark by default, light when the
  system or browser reports light; 7-day movement arrows now use ▴/▾.

### Fixed

- User-agent strings point at `canonical/model-compare` instead of the
  stale `rkratky` URL (cosmetic; request headers only).

## [0.2.0] - 2026-09-20

### Changed

- Website tooling moved into `web/`: `build_site_data.py`,
  `generate_highlights.py`, `preview.sh` and `site/`; the `publish` workflow
  now runs the new `web/publish.py` orchestrator instead of scripting the
  pipeline inline. `model_compare.py` remains a standalone one-file script at
  the repo root and its CLI output is unchanged.
- `generate_highlights.py` and `web/publish.py` user agents now identify
  their release version.

### Added

- `--version` prints the tool version (added late in 0.1.0; the user agent
  string, which claimed `1.0` from the beginning, now derives from it).

## [0.1.0] - 2026-09-20

Initial release.

### Added

- `model_compare.py` — standalone, dependency-free CLI ranking OpenRouter
  models by blended price, quality, context window and listing age, with
  table, `--json`, `--best` and `--catalog` (stable machine-readable
  contract) output modes.
- Published picks site on GitHub Pages: per-priority top-10 tables with copy
  buttons, `best.txt`, `catalog.json`, weekly `history.json` and
  `highlights.json`, refreshed every 6 hours by the `publish` workflow.
- `preview.sh` for local visual checks against live or freshly built data.
- pytest suite covering the ranking logic, the site data builder, the
  highlights generator and `preview.sh`.
