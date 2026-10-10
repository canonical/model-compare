# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
