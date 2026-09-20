# Site theming (0.2.1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Apply the approved dark-default + light-follows-system palette and the four polish items to `web/site/index.html`, then prepare release 0.2.1.

**Architecture:** CSS custom properties on `:root` (dark base) with a light block inside `@media (prefers-color-scheme: light)`; every hardcoded color replaced. Single-file UI change; three `VERSION` bumps for the release.

**Tech Stack:** Plain CSS (custom properties + media query), vanilla JS glyph change, pytest for the suite, Playwright for visual verification.

**Spec:** `docs/superpowers/specs/2026-09-20-site-theming-design.md`

## Global Constraints

- No JS beyond the two glyph swaps; no layout/content changes; no toggle.
- `color-scheme: light dark` stays on `:root`.
- Existing suite must stay green; contract items (`tool`, `schema_version`) untouched.
- All changes in a worktree under `.worktrees/`; tag/release only after the maintainer gate.

---

### Task 1: Palette + polish on `web/site/index.html`

**Files:**
- Modify: `web/site/index.html` (CSS lines 11–149; JS lines 370–372)

**Interfaces:**
- Consumes: approved token table (spec).
- Produces: `--bg/--text/--muted/--btn-bg/--btn-border/--thead-border/--row-border/--status/--up/--down/--flat` variables; ▴/▾ movement glyphs; borderless `#copybuttons`; primary `#alias-hint`; hover rule.

**Worktree:** `git worktree add .worktrees/site-theming -b site-theming` (from main). The plan doc itself is committed as the first commit on this branch.

- [ ] **Step 1: CSS token block** — replace the `:root { color-scheme: light dark; }` rule (lines 12–14) with:

```css
      :root {
        color-scheme: light dark;
        /* dark is the base (default); light applies via prefers-color-scheme */
        --bg: #181A1B;
        --text: #E8E6E3;
        --muted: #988F81;
        --btn-bg: #262A2B;
        --btn-border: var(--muted);
        --thead-border: var(--text);
        --row-border: var(--muted);
        --status: #E06C60;
        --up: #7BB98C;
        --down: #DE8482;
        --flat: #7FA6D9;
      }
      @media (prefers-color-scheme: light) {
        :root {
          --bg: #F6F3EC;
          --text: #33302B;
          --muted: #8A8175;
          --btn-bg: #E9E4D8;
          --status: #B03A2A;
          --up: #2E7D4F;
          --down: #B0392B;
          --flat: #3A6EA5;
        }
      }
```

- [ ] **Step 2: Replace every hardcoded color**

- `body`: add `background: var(--bg); color: var(--text);`
- `.sub`, `.tabs .stamp`, `.hint`, `footer`: `color: gray` → `color: var(--muted)`
- `.tabs button`: `border: 1px solid #999` → `var(--btn-border)`; `background: transparent` → `var(--bg)`
- `.tabs button.active`: `background: #333` → `var(--btn-bg)`; `color: #fff` → `var(--text)`; `border-color: #333` → `var(--btn-border)`
- `.actions button`: `border: 1px solid #333` → `var(--btn-border)`; `background: #333` → `var(--btn-bg)`; `color: #fff` → `var(--text)`
- `.status`: `color: #b00` → `var(--status)`
- `#table th, #table td`: `border-bottom: 1px solid #ddd` → `var(--row-border)`
- `#table thead th`: `border-bottom: 2px solid #333` → `var(--thead-border)`
- `.movement.up` → `var(--up)`; `.movement.down` → `var(--down)`; `.movement.flat` → `var(--flat)`; `.movement.new` → `var(--up)`
- `#copybuttons { border: none; }` → `#copybuttons, #copybuttons td { border: none; }`
- After the `code` rule, add:

```css
      #alias-hint {
        color: var(--text);
      }
```

- Replace `a { color: inherit; }` with:

```css
      a {
        color: inherit;
      }
      a:hover {
        text-decoration: none;
        color: var(--text);
      }
```

- [ ] **Step 3: Arrows** — in `render()` (lines 370–372): `text: "↑"` → `"▴"`, `"↓"` → `"▾"`.

- [ ] **Step 4: Verify**

Run: `pytest -q` (expect 278 passing) and:

```bash
grep -n '▴\|▾\|--btn-border\|#alias-hint\|#copybuttons td' web/site/index.html
```

Expected: both glyphs in `render()`; token block present; alias + borderless rules present; no remaining `#999`/`#ddd`/`#b00`/` gray` hardcodes (`grep -n '#999\|#ddd\|#b00\|: gray' web/site/index.html` → no matches).

- [ ] **Step 5: Visual verification** — serve `web/preview.sh --port P`, navigate Playwright to it, screenshot with `colorScheme` emulated `dark` and `light`; compare against the approved mockup (dark: `#181A1B` bg, `#262A2B` filled buttons with muted borders, primary thead rule, muted row rules, borderless copy table; light: `#F6F3EC` bg equivalents). Hover a link: underline gone, primary color. Kill the server afterwards.

- [ ] **Step 6: Commit**

```bash
git add web/site/index.html
git commit -m "feat(site): dark-default + light palette, ▴/▾ arrows, polish"
```

---

### Task 2: 0.2.1 release prep

**Files:**
- Modify: `model_compare.py:67` (`VERSION = "0.2.0"` → `"0.2.1"`)
- Modify: `web/publish.py:23` (same bump)
- Modify: `web/generate_highlights.py:30` (same bump)
- Modify: `test_model_compare.py::test_version_flag` literal → `"model-compare 0.2.1"`
- Modify: `CHANGELOG.md` — rename `## [Unreleased]` → `## [0.2.1] - <merge date>` and add site-theming items

**Interfaces:** Consumes Task 1; Produces the tree the `v0.2.1` tag points at.

- [ ] **Step 1: Bump versions** — the three `VERSION` constants and the test literal.

- [ ] **Step 2: CHANGELOG** — turn the `[Unreleased]` section into:

```markdown
## [0.2.1] - <merge date>

### Changed

- The picks site ships an explicit palette: dark by default, light when the
  system or browser reports light; 7-day movement arrows now use ▴/▾.

### Fixed

- User-agent strings point at `canonical/model-compare` instead of the
  stale `rkratky` URL (cosmetic; request headers only).
```

- [ ] **Step 3: Verify + commit** — `pytest -q` (278 passing, `--version` prints `model-compare 0.2.1`).

```bash
git add model_compare.py web/publish.py web/generate_highlights.py test_model_compare.py CHANGELOG.md
git commit -m "chore: release 0.2.1 (version bumps, changelog)"
```

---

### Task 3 (GATED): merge, tag v0.2.1, release

> **Gate:** maintainer reviews the `site-theming` branch and explicitly confirms.

- [ ] ff-merge to main, `pytest -q` on main, `git tag -a v0.2.1 -m "model-compare 0.2.1"`, push main + tag, `gh release create v0.2.1` (github-commentary style notes + pinned raw link), verify, remove worktree.

---

## Self-review

- Spec coverage: tokens/polish → Task 1; release checklist → Task 2; gate → Task 3. ✓
- Placeholders: none (exact CSS/glyph code inline; `<merge date>` filled at execution). ✓
- Consistency: token names in Task 1 match the spec table and the mock. ✓
