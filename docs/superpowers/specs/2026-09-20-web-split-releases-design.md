# Web split + versioned releases — design

Date: 2026-09-20
Status: approved (plan presented and confirmed by the maintainer)

## Context

The repo currently mixes two concerns at the top level:

- `model_compare.py` — the standalone, dependency-free user CLI (`--json`,
  `--best`, `--catalog`). Its `--catalog` output is a stable contract consumed
  by internal Canonical tooling (`tokens.canonical.com`). It contains no web
  logic and stays freely downloadable as a single file.
- Website tooling — `build_site_data.py`, `generate_highlights.py`,
  `preview.sh`, `site/index.html` — plus the pipeline orchestration, which
  today lives inline in `.github/workflows/publish.yml` (5 script invocations,
  2 `curl` fetches of the site's own previous history/highlights).

There are no tags, no changelog, and no version self-identification. The
`USER_AGENT` strings claim `model-compare/1.0`, which does not match any real
release scheme.

## Goals

1. Keep `model_compare.py` a standalone, one-file, stdlib-only script at the
   repo root with unchanged CLI behavior (including `--catalog`).
2. Move everything website-related into `web/`, fronted by a single
   orchestrator (`web/build.py`) that calls the standalone script via its CLI
   (subprocess) when needed. The `publish` workflow shrinks to: test, run the
   orchestrator, deploy.
3. Start versioned releases: tag `v0.1.0` from the current codebase (plus a
   minimal `--version`/changelog commit), then ship the web split as
   `v0.2.0`.
4. Identify problematic parts and mitigate or explicitly accept them.

## Non-goals

- No Python packaging (no pyproject/entry points) — the repo stays
  flat-scripts; the one-file download URL must remain stable.
- No behavior changes to ranking, scoring, or any CLI output bytes.
- No new third-party dependencies; `web/build.py` is stdlib-only.
- No separate repository for the site tooling.

## Decisions

- **Layout:** `web/` directory (chosen over a Python package or a separate
  repo). `model_compare.py` stays at the root.
- **Coupling:** the web side invokes `model_compare.py` only through its
  subprocess CLI — never by import — so the standalone contract stays the
  single integration point.
- **Orchestration:** a new `web/build.py` replaces the inline steps in
  `publish.yml`; `preview.sh --build` delegates to it too, so the pipeline is
  defined once and runnable locally.
- **Releases:** annotated tags (`v0.1.0`, `v0.2.0`), a Keep a Changelog
  `CHANGELOG.md`, GitHub Releases created from the tags, and a `--version`
  flag + `VERSION` constant in `model_compare.py`.
- **User-agent fix timing:** `model_compare.py`'s `USER_AGENT` derives from
  `VERSION` already in the 0.1 prep commit (so 0.1 does not ship
  self-identifying as `1.0`). `generate_highlights.py` gets its own
  `VERSION`/derived UA in 0.2. (Deviation from the first presentation, where
  the UA fix was listed under the 0.2 refactor only.)

## Target layout after the 0.2 refactor

```
model_compare.py            # unchanged CLI; VERSION = "0.2.0"; stays at root
CHANGELOG.md                # new in 0.1
web/
  build.py                  # new orchestrator (subprocess pipeline)
  build_site_data.py        # moved
  generate_highlights.py    # moved (+ VERSION/UA)
  preview.sh                # moved; --build delegates to web/build.py
  site/index.html           # moved
  test_build.py             # new
  test_build_site_data.py   # moved with its module
  test_generate_highlights.py
  test_preview_sh.py
test_model_compare.py       # stays at root
README.md                   # path updates + Releases section
.github/workflows/publish.yml
```

pytest collects `web/test_*.py` from the repo root automatically (recursive
collection; flat modules, so `import build` inside `web/` tests resolves via
pytest's rootdir `sys.path` insertion).

## `web/build.py` contract

```
usage: build.py [--output-dir DIR]   (default: _site, relative to repo root)
```

Pipeline (mirrors today's publish.yml step exactly; fails loudly via
`check=True`):

1. `model_compare.py --best` → `<out>/best.txt`
2. `model_compare.py --priority P --json --top 10` → scratch `<P>.json` for
   each of `balanced`, `price`, `quality`
3. `model_compare.py --catalog` → scratch `catalog.json`
4. Fetch previous `history.json` / `highlights.json` from the live site with
   stdlib `urllib`; on any failure behave like today's `curl … || true`
   (write nothing; the downstream scripts already treat a missing/malformed
   file as "start fresh" — verified at `generate_highlights.py:437-448` and
   `build_site_data.py:572-580`).
5. `generate_highlights.py --catalog … --history … --prev-highlights …
   --output …`
6. `build_site_data.py --best-file … --output <out>/data.json --catalog-file …
   --history-prev-file … --highlights-file … --priority name=file ×3`
   (it writes the validated `catalog.json` next to `--output`)
7. Copy `web/site/index.html` → `<out>/index.html`

Internal seams for testing: `run_script(script, args, out, cwd)`,
`fetch_prev(url)`, `build_site(output_dir)`.

## Workflow change (`publish.yml`)

The "Build site" step body becomes `python web/build.py` (same `AA_API_KEY`/
`OPENROUTER_API_KEY` env). The Test step (`pytest`) and the upload/deploy
steps are unchanged. Workflow and moved code land in the same merge so the
6-hour cron never observes a mismatched layout.

## `model_compare.py` change (the only one)

```python
VERSION = "0.2.0"          # "0.1.0" in the 0.1 prep commit
USER_AGENT = f"model-compare/{VERSION} (https://github.com/rkratky/model-compare)"
```

plus `parser.add_argument("--version", action="version",
version="%(prog)s {VERSION}")`. Nothing else changes; the catalog `tool`
field stays the literal `"model-compare"`.

## Tests

- `test_model_compare.py` (root): new `test_version_flag` asserting
  `--version` prints `model-compare <VERSION>`.
- Web tests move with their modules; `test_preview_sh.py` needs no path
  changes (`Path(__file__).parent`).
- New `web/test_build.py`: pipeline order/assembly (subprocess monkeypatched),
  `--output-dir` CLI, loud failure on a failing subprocess, graceful
  `fetch_prev` fallback.
- Cross-file `tool` literal: already pinned on both sides
  (`test_model_compare.py:1376`, `web/test_build_site_data.py:190,258`) —
  verification-only, no new tests.

## Problematic parts → mitigations

1. **`tool` literal coupling** (`validate_catalog` hard-checks
   `tool == "model-compare"` in both files): keep the literal stable; both
   sides are already test-pinned. Documented in the release checklist: never
   rename it without a schema bump.
2. **Duplicated UA constants** (user script + `generate_highlights.py`):
   each derives from its own `VERSION`; the release checklist bumps both.
   Accepted duplication — importing across the standalone boundary is
   forbidden by design.
3. **Site history self-feeds from the live site** (missing file on first run
   / site outage): preserved `|| true` semantics in `fetch_prev`; the
   downstream scripts start fresh defensively (already implemented).
4. **Tag permanence:** v0.1.0 is tagged before any layout change, so its
   workflow and download URL keep working forever.
5. **Cron race:** code + workflow change in one atomic merge.
6. **UA drift:** `model-compare/1.0` → `model-compare/<VERSION>`; intentional
   header-string change, no functional impact.

## Release checklist (every release)

1. Bump `VERSION` in `model_compare.py` (and `generate_highlights.py`,
   `web/build.py` for 0.2+) so `--version` and both user agents match.
2. Add a CHANGELOG entry dated today.
3. Run `pytest` (root; collects `web/`).
4. `git tag -a vX.Y.Z -m "model-compare X.Y.Z"`, push main + tag.
5. `gh release create vX.Y.Z` with notes; the notes link the pinned
   standalone download: `raw.githubusercontent.com/rkratky/model-compare/vX.Y.Z/model_compare.py`.
6. Never rename the catalog `tool` literal or remove schema fields without
   bumping `schema_version`.
