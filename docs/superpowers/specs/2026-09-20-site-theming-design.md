# Site theming (dark default + light) — design

Date: 2026-09-20
Status: approved (palette locked via mockup review; this doc records it)
Target release: 0.2.1

## Context

`web/site/index.html` is styled by browser defaults plus a handful of hardcoded
values (`#333` button fills, `#999`/`#ddd` borders, `gray` secondary text,
GitHub-ish movement colors). The active-tab fill is hardcoded dark, so the
page never had a deliberate palette in either scheme.

## Goal

An explicit two-scheme palette — **dark by default, light when the system or
browser reports light** — plus four small polish items (movement arrows, alias
line color, copy-table borders, link hover). Single-file change to
`web/site/index.html`. No JS, no toggle, no new requests; CSP
(`style-src 'unsafe-inline'`) unaffected.

## Decisions

- **Mechanism:** CSS custom properties on `:root`; the dark palette is the
  base (default), the light palette sits inside
  `@media (prefers-color-scheme: light)`.
- **Default-dark nuance:** browsers with no setting report `light` (the
  `no-preference` value was removed from the spec), so "explicitly light" vs
  "default light" is not distinguishable in CSS. The truly undetectable case
  (browsers without the media query) falls through to base dark. A manual
  toggle is out of scope (YAGNI).

## Tokens

| token | dark (base/default) | light (`prefers-color-scheme: light`) |
|---|---|---|
| `--bg` | `#181A1B` | `#F6F3EC` |
| `--text` | `#E8E6E3` | `#33302B` |
| `--muted` | `#988F81` | `#8A8175` |
| `--btn-border` (all buttons) | `#988F81` | `#8A8175` |
| `--btn-bg` (active tab + copy buttons) | `#262A2B` | `#E9E4D8` |
| `--thead-border` (2px, above body) | `#E8E6E3` | `#33302B` |
| `--row-border` (1px, between rows) | `#988F81` | `#8A8175` |
| `--status` | `#E06C60` | `#B03A2A` |
| `--up` / new | `#7BB98C` | `#2E7D4F` |
| `--down` | `#DE8482` | `#B0392B` |
| `--flat` | `#7FA6D9` | `#3A6EA5` |

Applied mapping (replacing every hardcoded color):

- `body` → `--bg` / `--text`; `.sub`, `.stamp`, `.hint`, `footer` → `--muted`.
- Tabs: all buttons `1px solid var(--btn-border)`; non-selected bg `--bg`;
  active bg `--btn-bg`, text `--text`, border still `--btn-border`.
- Action (copy) buttons: bg `--btn-bg`, border `--btn-border`, text `--text`.
- Table: thead `2px solid var(--thead-border)` (primary), rows
  `1px solid var(--row-border)` (muted) — deliberate swap of the current
  hierarchy.
- `.status` → `--status`; movement classes → `--up`/`--down`/`--flat`.
- `color-scheme` stays `light dark` on `:root` (form controls/scrollbars
  follow the media query).

## Polish items

1. **Movement arrows:** `↑`/`↓` → `▴`/`▾` in `render()`
   (web/site/index.html:370-372); `•` (flat) and `new` unchanged.
2. **Alias line:** `#alias-hint { color: var(--text); }` (was muted via
   `.hint` inheritance).
3. **Copy-buttons table:** zero borders, table and cells:
   `#copybuttons, #copybuttons td { border: none; }`.
4. **Link hover:** `a:hover { text-decoration: none; color: var(--text); }`;
   resting state unchanged (underlined, inherit).

## Non-goals

- No manual theme toggle, no cookie/localStorage persistence.
- No layout, font, or content changes beyond the above.

## Testing

- Existing suite must stay green (preview smoke tests assert page content,
  not colors).
- Visual verification: `web/preview.sh` served locally, screenshotted via
  Playwright with `prefers-color-scheme` forced to `dark` and to `light`;
  both compared against the approved mockup. Regression pins: the arrows
  glyphs, `#alias-hint` color rule, and the no-border rule are checked in the
  rendered HTML/CSS (grep-level asserts in the implementation task).
- The synthetic-fallback path of `preview.sh` is unaffected (no JS/CSS
  contract changes beyond colors).

## Release (0.2.1)

Per the standing release checklist (web-split spec, "Release checklist"):
bump `VERSION` in `model_compare.py`, `web/publish.py`,
`web/generate_highlights.py` to `0.2.1`; update the pinned literal in
`test_model_compare.py::test_version_flag`; add the CHANGELOG `0.2.1`
entry; pytest; ff-merge; then (maintainer gate) `git tag -a v0.2.1` + GitHub
Release with the pinned raw download link. Site-only change otherwise: the
`--catalog` contract, `tool` literal and `schema_version` are untouched.
