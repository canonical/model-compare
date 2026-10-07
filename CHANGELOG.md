# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- The ZDR filter now reads OpenRouter's per-endpoint ZDR list
  (`/api/v1/endpoints/zdr`): a model id counts as ZDR only if it has at least
  one ZDR endpoint of its own. The previous source, the models `?zdr=true`
  filter, is model-level and also listed non-ZDR variants, so NVIDIA's
  `:free` endpoints — which retain prompts and train on them — were published
  with `zdr: true` (e.g. `nvidia/nemotron-3-ultra-550b-a55b:free`). The ZDR
  cache key moved to `openrouter-zdr-v3`, so cached id sets from the old
  source are never read.

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
