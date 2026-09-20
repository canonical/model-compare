# Web split + versioned releases — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut `v0.1.0` from the current codebase, then ship a `v0.2.0` refactor that moves all website tooling into `web/` behind a single `build.py` orchestrator that calls the standalone `model_compare.py` via its CLI.

**Architecture:** `model_compare.py` stays a standalone one-file script at the repo root (subprocess-only coupling). `web/build.py` runs the exact pipeline that today lives inline in `.github/workflows/publish.yml`. Releases are annotated tags with a Keep a Changelog file.

**Tech Stack:** Python 3.10+ stdlib only (subprocess, urllib, tempfile, shutil, argparse), bash (`preview.sh`), GitHub Actions, pytest.

**Spec:** `docs/superpowers/specs/2026-09-20-web-split-releases-design.md`

## Global Constraints

- Python 3.10+, stdlib only — no new dependencies anywhere.
- CLI contract unchanged: `--best` / `--json` / `--catalog` output bytes identical; catalog `tool` stays the literal `"model-compare"`; `schema_version` stays `1`.
- All changes happen in git worktrees under `.worktrees/` (already gitignored); commits/pushes/tags happen only in the steps that explicitly say so (pre-approved by the maintainer).
- Tags are annotated: `git tag -a vX.Y.Z`.
- `preview.sh` keeps working from `web/` with `SCRIPT_DIR`-relative paths only.
- Release checklist (spec) applies to every release: bump `VERSION` in `model_compare.py` (+ `generate_highlights.py` and `web/build.py` from 0.2 on), CHANGELOG entry, pytest, annotated tag, GH release notes with pinned raw download link.

---

### Task 1: v0.1.0 prep — VERSION, `--version`, UA fix, CHANGELOG

**Files:**
- Modify: `model_compare.py` (line 67 area; parser creation at line 1083)
- Modify: `test_model_compare.py` (append test at end; imports at top as needed)
- Create: `CHANGELOG.md`

**Interfaces:**
- Consumes: nothing.
- Produces: `model_compare.VERSION` module constant (`"0.1.0"`), `--version` flag printing `model-compare 0.1.0`; `CHANGELOG.md` with a `[0.1.0]` section. Task 2 tags from this state.

**Worktree:** `git worktree add .worktrees/release-0.1-prep -b release-0.1-prep` (from the main checkout). All paths below are relative to that worktree.

- [ ] **Step 1: Write the failing test**

Append to `test_model_compare.py` (add `import subprocess`, `import sys`, and `from pathlib import Path` at the top if not already imported):

```python
MC_SCRIPT = Path(__file__).resolve().parent / "model_compare.py"


def test_version_flag():
    proc = subprocess.run(
        [sys.executable, str(MC_SCRIPT), "--version"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert proc.stdout.strip() == f"model-compare {model_compare.VERSION}"
```

(This requires `import model_compare` at the top — the file already imports from `model_compare`.)

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest test_model_compare.py::test_version_flag -q` (inside the worktree)
Expected: FAIL — `error: unrecognized arguments: --version` (non-zero from subprocess via check=True).

- [ ] **Step 3: Implement**

In `model_compare.py`, replace line 67:

```python
USER_AGENT = "model-compare/1.0 (https://github.com/rkratky/model-compare)"
```

with:

```python
VERSION = "0.1.0"
USER_AGENT = f"model-compare/{VERSION} (https://github.com/rkratky/model-compare)"
```

Immediately after the `parser = argparse.ArgumentParser(` block that starts at line 1083, add:

```python
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {VERSION}",
    )
```

Create `CHANGELOG.md`:

```markdown
# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
```

- [ ] **Step 4: Run the full suite**

Run: `pytest -q`
Expected: all PASS (the UA string change is cosmetic; no test pins the old UA — if any does, update it to the derived form).

- [ ] **Step 5: Commit**

```bash
git add model_compare.py test_model_compare.py CHANGELOG.md
git commit -m "feat: 0.1.0 release prep (VERSION, --version, changelog)"
```

---

### Task 2: Tag + release v0.1.0

**Files:** none changed; git/gh operations only.

**Interfaces:**
- Consumes: `release-0.1-prep` branch from Task 1.
- Produces: remote tag `v0.1.0` + GitHub Release; `main` contains Task 1.

- [ ] **Step 1: Merge the prep branch into main**

From the main checkout (repo root):

```bash
git merge --ff-only release-0.1-prep
```

Expected: fast-forward, no merge commit.

- [ ] **Step 2: Tag and push**

```bash
git tag -a v0.1.0 -m "model-compare 0.1.0"
git push origin main v0.1.0
```

- [ ] **Step 3: Create the GitHub Release**

First load the `github-commentary` skill (repo rule) before writing the release body. Then, with the base URL derived from the actual remote (no hardcoded owner):

```bash
BASE=$(gh repo view --json nameWithOwner -q .nameWithOwner)
gh release create v0.1.0 --title "model-compare 0.1.0" --notes "$(cat <<EOF
First tagged release of model-compare.

- Standalone, dependency-free CLI \`model_compare.py\`: ranks OpenRouter
  models by blended price + quality + context + age. Modes: table, \`--json\`,
  \`--best\`, and \`--catalog\` (stable machine contract, schema_version 1).
- Published picks site on GitHub Pages: per-priority top-10 tables,
  \`best.txt\`, \`catalog.json\`, weekly \`history.json\` + \`highlights.json\`,
  refreshed every 6 hours.
- \`preview.sh\` for local visual checks; pytest suite for the logic.

Download the pinned standalone script:
https://raw.githubusercontent.com/$BASE/v0.1.0/model_compare.py
EOF
)"
```

- [ ] **Step 4: Verify**

```bash
gh release view v0.1.0 --json tagName,url
```

Expected: tag `v0.1.0`, URL printed.

---

### Task 3: Move website tooling into `web/`

**Files:**
- Move: `build_site_data.py`, `generate_highlights.py`, `preview.sh`,
  `test_build_site_data.py`, `test_generate_highlights.py`,
  `test_preview_sh.py`, `site/` → `web/…`
- Modify: `preview.sh` (script paths only in this task)
- Modify: `README.md` (path references)

**Interfaces:**
- Consumes: main at v0.1.0.
- Produces: `web/` tree with green tests; `preview.sh` still self-contains the build pipeline (delegation lands in Task 5).

**Worktree:** `git worktree add .worktrees/web-split -b web-split` (from main).

- [ ] **Step 1: Move the files**

```bash
mkdir web
git mv build_site_data.py generate_highlights.py preview.sh \
  test_build_site_data.py test_generate_highlights.py test_preview_sh.py web/
git mv site web/site
```

- [ ] **Step 2: Fix preview.sh paths for the new depth**

In `web/preview.sh`, replace `SCRIPT_DIR/model_compare.py` occurrences with
`SCRIPT_DIR/../model_compare.py` (the `--build` block at lines 96–109; the
`build_site_data.py` reference stays `SCRIPT_DIR/build_site_data.py` since
both files move together), and update the user-visible text:
- header comment line 2: `serve web/site/index.html locally`
- usage line: `Serves web/site/index.html at http://127.0.0.1:PORT …`
- `--build` help text: `build fresh data locally with model_compare.py + build_site_data.py instead of using the live data` → unchanged semantics, still accurate in this task
- `--site-dir` default text: `(default: ./site)` → `(default: ./web/site)`

- [ ] **Step 3: Update README path references**

Find and update every reference to the moved files:

```bash
grep -n 'build_site_data\|generate_highlights\|preview.sh\|site/index.html' README.md
```

Point them at the `web/` locations (e.g. `The build (`web/build_site_data.py`)`).
Do not change any other README content in this task.

- [ ] **Step 4: Run the suite**

Run: `pytest -q`
Expected: all PASS — pytest collects `web/test_*.py` recursively;
`test_preview_sh.py` needs no edit (`Path(__file__).parent / "preview.sh"`).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor: move website tooling into web/"
```

---

### Task 4: `web/build.py` orchestrator (TDD)

**Files:**
- Create: `web/build.py`
- Create: `web/test_build.py`

**Interfaces:**
- Consumes: `model_compare.py` CLI (`--best`, `--priority P --json --top 10`,
  `--catalog`); `generate_highlights.py` CLI; `build_site_data.py` CLI — all
  via subprocess only.
- Produces (for Tasks 5): `build.build_site(output_dir)`,
  `build.main(argv) -> int`, CLI `python web/build.py [--output-dir DIR]`
  (default `_site`, resolved against the repo root). Internal seams:
  `run_script(script, args, out=None)`, `fetch_prev(url) -> bytes | None`.

- [ ] **Step 1: Write the failing tests**

Create `web/test_build.py`:

```python
"""Tests for the web/build.py orchestrator (all subprocesses stubbed)."""

import subprocess
import urllib.error
from pathlib import Path

import pytest

import build


def _fake_run(calls):
    def fake_run(cmd, cwd=None, stdout=None, check=True):
        calls.append(cmd)
        if "--best" in cmd:
            stdout.write("openrouter/z-ai/glm-5.3-flash\n")
        return subprocess.CompletedProcess(cmd, 0)

    return fake_run


def test_build_site_pipeline_order_and_assembly(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(build.subprocess, "run", _fake_run(calls))
    monkeypatch.setattr(build, "fetch_prev", lambda url: None)
    out = tmp_path / "site"

    build.build_site(out)

    scripts = [Path(cmd[1]).name for cmd in calls]
    assert scripts == [
        "model_compare.py",  # --best
        "model_compare.py",  # balanced --json --top 10
        "model_compare.py",  # price --json --top 10
        "model_compare.py",  # quality --json --top 10
        "model_compare.py",  # --catalog
        "generate_highlights.py",
        "build_site_data.py",
    ]
    assert "--priority" not in next(cmd for cmd in calls if "--best" in cmd)
    assert sum(1 for cmd in calls if "--catalog" in cmd) == 1
    json_calls = [cmd for cmd in calls if "--json" in cmd]
    assert [
        json_calls[i][json_calls[i].index("--priority") + 1]
        for i in range(len(json_calls))
    ] == ["balanced", "price", "quality"]
    assert (out / "best.txt").read_text().strip() == "openrouter/z-ai/glm-5.3-flash"
    assert (out / "index.html").is_file()


def test_build_site_fails_loudly(tmp_path, monkeypatch):
    def boom(cmd, cwd=None, stdout=None, check=True):
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(build.subprocess, "run", boom)
    with pytest.raises(subprocess.CalledProcessError):
        build.build_site(tmp_path / "site")


def test_fetch_prev_returns_none_on_error(monkeypatch):
    def boom(req, timeout=None):
        raise urllib.error.URLError("unreachable")

    monkeypatch.setattr(build.urllib.request, "urlopen", boom)
    assert build.fetch_prev("http://127.0.0.1:9/history.json") is None


def test_main_output_dir(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(build, "build_site", lambda out: seen.setdefault("out", out))
    rc = build.main(["--output-dir", str(tmp_path / "deploy")])
    assert rc == 0
    assert seen["out"] == tmp_path / "deploy"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest web/test_build.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'build'`.

- [ ] **Step 3: Implement `web/build.py`**

```python
#!/usr/bin/env python3
"""Build the model-compare site.

Runs the standalone model_compare.py, then the web-side generators
(generate_highlights.py, build_site_data.py), and assembles the deploy
directory: data.json, catalog.json, best.txt and index.html. Fails loudly on
any unexpected result so a broken run never deploys a broken site.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = Path(__file__).resolve().parent
VERSION = "0.2.0"
USER_AGENT = f"model-compare/{VERSION} (https://github.com/rkratky/model-compare)"
PRIORITIES = ("balanced", "price", "quality")
PREV_HISTORY_URL = "https://canonical.github.io/model-compare/history.json"
PREV_HIGHLIGHTS_URL = "https://canonical.github.io/model-compare/highlights.json"


def run_script(script: Path, args: list[str], out: Path | None = None) -> None:
    """Run a repo script, optionally redirecting stdout into `out`."""
    cmd = [sys.executable, str(script), *args]
    if out is None:
        subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    else:
        with open(out, "w") as fh:
            subprocess.run(cmd, cwd=REPO_ROOT, stdout=fh, check=True)


def fetch_prev(url: str, timeout: int = 60) -> bytes | None:
    """Fetch a previously published file; None on any failure (|| true)."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except (urllib.error.URLError, OSError):
        return None


def build_site(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mc-build-") as tmp:
        scratch = Path(tmp)
        model_compare = REPO_ROOT / "model_compare.py"

        run_script(model_compare, ["--best"], out=output_dir / "best.txt")
        for priority in PRIORITIES:
            run_script(
                model_compare,
                ["--priority", priority, "--json", "--top", "10"],
                out=scratch / f"{priority}.json",
            )
        run_script(model_compare, ["--catalog"], out=scratch / "catalog.json")

        for url, name in (
            (PREV_HISTORY_URL, "history-prev.json"),
            (PREV_HIGHLIGHTS_URL, "highlights-prev.json"),
        ):
            prev = fetch_prev(url)
            if prev is not None:
                (scratch / name).write_bytes(prev)

        run_script(
            WEB_DIR / "generate_highlights.py",
            [
                "--catalog",
                str(scratch / "catalog.json"),
                "--history",
                str(scratch / "history-prev.json"),
                "--prev-highlights",
                str(scratch / "highlights-prev.json"),
                "--output",
                str(scratch / "highlights-new.json"),
            ],
        )

        priority_args: list[str] = []
        for p in PRIORITIES:
            priority_args += ["--priority", f"{p}={scratch / f'{p}.json'}"]
        run_script(
            WEB_DIR / "build_site_data.py",
            [
                "--best-file",
                str(output_dir / "best.txt"),
                "--output",
                str(output_dir / "data.json"),
                "--catalog-file",
                str(scratch / "catalog.json"),
                "--history-prev-file",
                str(scratch / "history-prev.json"),
                "--highlights-file",
                str(scratch / "highlights-new.json"),
                *priority_args,
            ],
        )

        shutil.copyfile(WEB_DIR / "site" / "index.html", output_dir / "index.html")

    for name in ("data.json", "catalog.json", "best.txt", "index.html"):
        print(f"build: wrote {output_dir / name}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="build.py",
        description="Build the model-compare site deploy directory",
    )
    parser.add_argument(
        "--output-dir",
        default="_site",
        help="directory to assemble (default: _site relative to the repo root)",
    )
    args = parser.parse_args(argv)
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    build_site(output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest web/test_build.py -q`
Expected: 4 PASS.

- [ ] **Step 5: Full suite + commit**

Run: `pytest -q` — all PASS.

```bash
git add web/build.py web/test_build.py
git commit -m "feat(web): add build.py orchestrator"
```

---

### Task 5: Workflow + preview delegate to `build.py`

**Files:**
- Modify: `.github/workflows/publish.yml` ("Build site" step)
- Modify: `web/preview.sh` (delegation + cleanup of the now-dead build block)

**Interfaces:**
- Consumes: `web/build.py` CLI from Task 4.
- Produces: single pipeline definition (workflow ≡ `python web/build.py`).

- [ ] **Step 1: Shrink the workflow**

Replace the entire `Build site` step body with:

```yaml
      - name: Build site
        env:
          AA_API_KEY: ${{ secrets.AA_API_KEY }}
          OPENROUTER_API_KEY: ${{ secrets.OPENROUTER_API_KEY }}
        run: python web/build.py
```

Everything else in the workflow (permissions, concurrency, Test step, upload/deploy actions) stays byte-identical.

- [ ] **Step 2: Delegate preview.sh --build**

In `web/preview.sh`:

Replace the `--build` block (currently lines 96–109) with:

```bash
if [ "$BUILD" -eq 1 ]; then
	echo "building data locally with web/build.py..."
	python3 "$SCRIPT_DIR/build.py" --output-dir "$PREVIEW_DIR"
elif command -v curl >/dev/null &&
```

Then clean up what the delegation made dead:
- delete the `BUILD_DIR=""` initialization (line 20) and both `BUILD_DIR` blocks in the `trap` (lines 88–90)
- header comment lines 4–6: `1. --build    run the real pipeline via web/build.py locally; set AA_API_KEY for better quality scores`
- usage text: `--build         build fresh data locally with web/build.py instead of using the live data`

- [ ] **Step 3: Sanity-check the workflow YAML**

Run: `python -c "import yaml,sys; yaml.safe_load(open('.github/workflows/publish.yml')); print('ok')" 2>/dev/null || python3 -c "print('pyyaml unavailable; eyeball the diff')"`

- [ ] **Step 4: Run the suite**

Run: `pytest -q`
Expected: all PASS (`test_preview_sh.py` exercises the synthetic-fallback path, not `--build`).

- [ ] **Step 5: Commit**

```bash
git add .github/workflows/publish.yml web/preview.sh
git commit -m "refactor(web): publish workflow and preview delegate to build.py"
```

---

### Task 6: Version the web scripts; verify the `tool` literal pins

**Files:**
- Modify: `web/generate_highlights.py` (line 30)

**Interfaces:**
- Consumes: nothing.
- Produces: `generate_highlights.USER_AGENT = model-compare/0.2.0 …`; verified test pins for the catalog `tool` literal on both sides.

- [ ] **Step 1: Bump the web UA**

In `web/generate_highlights.py`, replace line 30:

```python
USER_AGENT = "model-compare/1.0 (https://github.com/rkratky/model-compare)"
```

with:

```python
VERSION = "0.2.0"
USER_AGENT = f"model-compare/{VERSION} (https://github.com/rkratky/model-compare)"
```

(If a test pins the old UA literal, update it to the derived form.)

- [ ] **Step 2: Verify the cross-file `tool` literal is test-pinned**

```bash
grep -n '"tool"' test_model_compare.py web/test_build_site_data.py
grep -n 'other-tool' web/test_build_site_data.py
```

Expected: `test_model_compare.py:1376` asserts `doc["tool"] == "model-compare"`;
`web/test_build_site_data.py:190` uses `"tool": "model-compare"` in the fixture
and `:258` has the negative case (`tool="other-tool"` must fail validation).
Both sides pinned → no new tests needed. If either is missing, add the
assertion before proceeding.

- [ ] **Step 3: Run the suite + commit**

Run: `pytest -q` — all PASS.

```bash
git add web/generate_highlights.py
git commit -m "chore(web): version generate_highlights user agent"
```

---

### Task 7: README + CHANGELOG 0.2.0

**Files:**
- Modify: `README.md` (paths, Releases section, `--version` mention)
- Modify: `CHANGELOG.md` (add `[0.2.0]`)

**Interfaces:**
- Consumes: layout from Tasks 3–6.
- Produces: docs ready for the v0.2.0 tag.

- [ ] **Step 1: README updates**

1. Confirm Task 3 already repointed `build_site_data.py` / `generate_highlights.py` / `preview.sh` / `site/index.html` references to `web/…`.
2. In the Options table area, add a line documenting `--version` (prints `model-compare <VERSION>`), e.g. under the flags table:
   `| `--version` | off | print `model-compare <VERSION>` and exit |`
3. Add a short `## Releases` section before `## Tests`:

```markdown
## Releases

Releases are annotated git tags (`v0.1.0`, `v0.2.0`, …) with GitHub Releases
described in `CHANGELOG.md`. To pin a copy of the standalone script, download
it from a tag, e.g.:

    curl -fsSLO https://raw.githubusercontent.com/rkratky/model-compare/v0.1.0/model_compare.py

and check `./model_compare.py --version`. The release checklist lives in
`docs/superpowers/specs/2026-09-20-web-split-releases-design.md`.
```

(Use the actual owner/repo from `git remote get-url origin`.)

- [ ] **Step 2: CHANGELOG 0.2.0 entry**

Insert below the `[0.1.0]` header block (date = actual merge date):

```markdown
## [0.2.0] - <merge date>

### Changed

- Website tooling moved into `web/`: `build_site_data.py`,
  `generate_highlights.py`, `preview.sh` and `site/`; the `publish` workflow
  now runs the new `web/build.py` orchestrator instead of scripting the
  pipeline inline. `model_compare.py` remains a standalone one-file script at
  the repo root and its CLI output is unchanged.
- `generate_highlights.py` and `web/build.py` user agents now identify their
  release version.

### Added

- `model_compare.py --version` prints the tool version (also fixed the
  user-agent string, which claimed `1.0` since the beginning).
```

- [ ] **Step 3: Run the suite + commit**

Run: `pytest -q` — all PASS.

```bash
git add README.md CHANGELOG.md
git commit -m "docs: README paths, Releases section, 0.2.0 changelog"
```

---

### Task 8: Merge, tag v0.2.0, release (GATED on maintainer confirmation)

**Files:** none changed; git/gh operations only.

**Interfaces:**
- Consumes: `web-split` branch (Tasks 3–7), green suite.
- Produces: remote tag `v0.2.0` + GitHub Release.

> **Gate:** the maintainer reviews the finished `web-split` branch and
> explicitly confirms before this task runs. Do not merge or tag without it.

- [ ] **Step 1: Present the branch for review**

From the main checkout: `git log --oneline main..web-split` and
`git diff main...web-split --stat` — summarize for the maintainer and wait
for explicit go-ahead.

- [ ] **Step 2: Merge, tag, push**

```bash
git merge --ff-only web-split
git tag -a v0.2.0 -m "model-compare 0.2.0"
git push origin main v0.2.0
```

- [ ] **Step 3: Create the GitHub Release**

Load the `github-commentary` skill first; then (BASE as in Task 2):

```bash
BASE=$(gh repo view --json nameWithOwner -q .nameWithOwner)
gh release create v0.2.0 --title "model-compare 0.2.0" --notes "$(cat <<EOF
Web split + release plumbing.

- All website tooling moved into \`web/\` behind a single \`web/build.py\`
  orchestrator; the \`publish\` workflow now just runs it. The standalone
  \`model_compare.py\` stays a one-file script at the repo root with an
  unchanged CLI (\`--json\` / \`--best\` / \`--catalog\` contract intact).
- Versioned releases start here: \`--version\`, CHANGELOG, annotated tags.
- No scoring or output changes; \`schema_version\` stays 1 and the catalog
  \`tool\` literal stays \`model-compare\`.

Download the pinned standalone script:
https://raw.githubusercontent.com/$BASE/v0.2.0/model_compare.py
EOF
)"
```

- [ ] **Step 4: Verify + clean up**

```bash
gh release view v0.2.0 --json tagName,url
git worktree remove .worktrees/release-0.1-prep
git worktree remove .worktrees/web-split
```

Expected: release URL printed; worktrees removed (both clean — no uncommitted changes policy respected).

---

## Self-review notes

- Spec coverage: Phase 0 → Tasks 1–2; Phase 1 → Tasks 3–7; Phase 2 → Task 8; all six flagged risks covered (risk 1 verified in Task 6 Step 2; risk 6 fixed in Tasks 1/6).
- Placeholder scan: no TBDs; every code step contains the actual code; owner/repo values are derived at execution time from the remote (deliberate, not a placeholder).
- Type consistency: `build_site(output_dir)` / `run_script(script, args, out=None)` / `fetch_prev(url) -> bytes | None` are used identically in `web/test_build.py` and `web/preview.sh`/workflow expectations.
