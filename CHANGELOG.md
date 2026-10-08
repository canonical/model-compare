# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- The AA intelligence fallback now reads the supported V2 endpoint
  `/api/v2/language/models/free` (any tier key; responses paginate at 200
  models per page and are followed, capped at 25 pages) instead of the legacy
  `/api/v2/data/llms/models`, which Artificial Analysis retires on
  2026-11-04. Failure behavior is unchanged: page-1 errors still fall back to
  the page scrape, later-page errors keep the pages already fetched.

### Fixed

- The ZDR filter now reads OpenRouter's per-endpoint ZDR list
  (`/api/v1/endpoints/zdr`): a model id counts as ZDR only if it has at least
  one ZDR endpoint of its own. The previous source, the models `?zdr=true`
  filter, is model-level and also listed non-ZDR variants, so NVIDIA's
  `:free` endpoints — which retain prompts and train on them — were published
  with `zdr: true` (e.g. `nvidia/nemotron-3-ultra-550b-a55b:free`). The ZDR
  cache key moved to `openrouter-zdr-v3`, so cached id sets from the old
  source are never read.

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
