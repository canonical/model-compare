#!/usr/bin/env python3
"""Publish the model-compare site.

Runs the standalone model_compare.py, then the web-side generators
(generate_highlights.py, build_site_data.py), and assembles the deploy
directory: data.json, catalog.json, history.json, highlights.json,
best.txt, index.html and its style.css/app.js/favicon assets. Fails
loudly on any unexpected result -- including
data.json/history.json stamps that disagree with catalog.json's
generated_at, or a best.txt that is not catalog.json's balanced rank 1 --
so a broken run never deploys a broken site.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = Path(__file__).resolve().parent
USER_AGENT = "model-compare (https://github.com/canonical/model-compare)"
PRIORITIES = ("balanced", "price", "quality")
PREV_HISTORY_URL = "https://canonical.github.io/model-compare/history.json"
PREV_HIGHLIGHTS_URL = "https://canonical.github.io/model-compare/highlights.json"
# Endpoint-stage failures are normal churn (provider outages, gating), but a
# mass exodus means the catalog is no longer comparable to previous runs.
# publish imports nothing from model_compare/build_site_data (subprocess-only
# by design), so this threshold is pinned here rather than imported.
MAX_ENDPOINT_DROP_RATE = 0.05
# The only pool.dropped reasons the endpoint stage itself produces; every
# other CATALOG_DROP_REASONS entry (malformed id, context, pricing, ...)
# drops a candidate before the endpoint stage is ever tried. publish
# imports nothing from model_compare (subprocess-only by design), so the
# reasons are named here rather than imported.
ENDPOINT_DROP_REASONS = ("no_baseline_eligible_endpoint", "endpoint_fetch_failed")
# A git commit hash in short (7) to full (40) lowercase hex form. Both
# revision sources (GITHUB_SHA, `git rev-parse` output) are validated
# against it: whatever lands in data.json is what the site footer links
# to, so garbage must fail the publish rather than ship a dead link.
REVISION_RE = re.compile(r"[0-9a-f]{7,40}")


def resolve_revision() -> str:
    """Return the revision that built the site (short commit hash).

    CI sets GITHUB_SHA; locally, fall back to the checkout's git HEAD.
    The revision is stamped into data.json so the site is always
    traceable to the code that produced it, without versioning the web
    itself. Fails loudly when unresolvable or not a hex commit hash: an
    unattributable deploy is treated like any other unexpected result.
    """
    sha = os.environ.get("GITHUB_SHA")
    if sha:
        sha = sha.strip()
        if not REVISION_RE.fullmatch(sha):
            raise RuntimeError(
                "publish: cannot resolve site revision: "
                f"GITHUB_SHA {sha!r} is not a hex commit hash"
            )
        return sha[:7]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short=7", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"publish: cannot resolve site revision: {exc}") from exc
    rev = result.stdout.strip()
    if not REVISION_RE.fullmatch(rev):
        raise RuntimeError(
            "publish: cannot resolve site revision: "
            f"git rev-parse output {rev!r} is not a hex commit hash"
        )
    return rev


def run_script(script: Path, args: list[str], out: Path | None = None) -> None:
    """Run a repo script, optionally redirecting stdout into `out`."""
    cmd = [sys.executable, str(script), *args]
    if out is None:
        subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    else:
        with open(out, "w") as fh:
            subprocess.run(cmd, cwd=REPO_ROOT, stdout=fh, check=True)


def fetch_prev(url: str, timeout: int = 60, attempts: int = 3) -> bytes | None:
    """Fetch a previously published file; None after all attempts fail.

    Mirrors the old `curl -fsSL --retry 2 ... || true`: retry transient
    failures (a single blip must never reset the accumulated history
    baseline) and degrade to None on any failure, including truncated
    bodies (http.client.HTTPException is outside URLError/OSError).
    """
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except (urllib.error.URLError, OSError, http.client.HTTPException):
            if attempt == attempts - 1:
                return None
            time.sleep(2**attempt)
    return None


def _load_artifact(path: Path, required: tuple[str, ...]) -> dict:
    """json.load an artifact, raising RuntimeError that names it on any defect."""
    try:
        with open(path) as fh:
            doc = json.load(fh)
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"publish: unreadable {path.name}: {exc}") from exc
    if not isinstance(doc, dict):
        raise RuntimeError(f"publish: {path.name} is not a JSON object")
    absent = [key for key in required if key not in doc]
    if absent:
        raise RuntimeError(f"publish: {path.name} lacks {', '.join(absent)}")
    return doc


def check_stamps(output_dir: Path) -> dict:
    """Fail unless data.json and history.json carry the catalog's generated_at.

    The catalog is the single clock: build_site_data stamps data.json from
    it and merge_history sets updated_at to the newest snapshot's stamp,
    which is today's catalog. highlights.json is deliberately excluded --
    its generated_at is its own writing time (the 24h LLM-reuse window).
    data.json must also carry source_revision: the site footer links to
    the commit that produced it, so an unattributable document is as
    broken as one with a wrong stamp. Returns the loaded catalog (which
    must carry rankings) for check_best.
    """
    catalog = _load_artifact(output_dir / "catalog.json", ("generated_at", "rankings"))
    data = _load_artifact(
        output_dir / "data.json", ("schema_version", "generated_at", "source_revision")
    )
    # .get: fail closed with the clean RuntimeError below even if the
    # required tuple above ever stops listing schema_version.
    version = data.get("schema_version")
    # Strict identity: True and 3.0 compare equal to 3 but are not v3.
    if type(version) is not int or version != 3:
        # publish imports nothing from build_site_data (subprocess-only by
        # design); the expected data.json version is pinned here, like the
        # opencode id prefix in check_best.
        raise RuntimeError(
            f"publish: data.json schema_version must be 3, got {version!r}"
        )
    history = _load_artifact(output_dir / "history.json", ("updated_at", "snapshots"))
    stamp = catalog["generated_at"]
    if data["generated_at"] != stamp or history["updated_at"] != stamp:
        raise RuntimeError(
            "publish: generated_at mismatch: "
            f"catalog={stamp!r} data={data['generated_at']!r} "
            f"history={history['updated_at']!r}"
        )
    return catalog


def check_best(output_dir: Path, catalog: dict) -> None:
    """Fail unless best.txt names the catalog's balanced rank-1 model.

    `--best` is the argmax of the same ranking the catalog carries, so any
    disagreement means the invocations saw different inputs.
    """
    rankings = catalog["rankings"]
    balanced = rankings.get("balanced") if isinstance(rankings, dict) else None
    if not isinstance(balanced, list) or not balanced:
        raise RuntimeError(
            "publish: catalog.json rankings.balanced is empty or missing"
        )
    top_id = balanced[0]
    if not isinstance(top_id, str) or not top_id:
        raise RuntimeError(
            f"publish: catalog.json rankings.balanced[0] is not an id: {top_id!r}"
        )
    try:
        best_clean = (output_dir / "best.txt").read_text().strip()
    except (OSError, UnicodeDecodeError) as exc:
        raise RuntimeError(f"publish: unreadable best.txt: {exc}") from exc
    # "openrouter/" duplicates model_compare.OPENCODE_PROVIDER; publish is
    # deliberately subprocess-only and imports nothing from it. Exact
    # membership, never prefix stripping (openrouter/openrouter/x must fail).
    if best_clean not in (top_id, "openrouter/" + top_id):
        raise RuntimeError(
            f"publish: best.txt {best_clean!r} disagrees with catalog "
            f"rankings.balanced[0] {top_id!r}"
        )


AA_LIVE_MODE = "aa"
# Quality-coverage regression threshold (spec 9.4): below this share of
# candidates with a quality record the publish warns but still deploys.
MIN_MATCHED_DEFAULT_COVERAGE = 0.25


def check_aa_source(catalog_path: Path) -> None:
    """With AA_API_KEY set, fail unless the AA API itself produced quality.

    sources.aa.mode records the run's quality chain: "aa" when the AA API
    returned usable entries, "catalog" for published-catalog hydration
    (keyed runs never hydrate), "none" when neither tier produced data. A
    keyed publish must have live AA data standing behind its quality:
    mode "aa" AND at least one AA-matched model (matched_aa > 0), at least
    one per-effort row (depth_rows > 0), and at least one candidate with a
    quality-coverage record (matched_default > 0) -- a rejected key, an
    empty result, a catalog-or-OR-only chain, an unknown mode or a
    malformed count all fail closed. Coverage below
    MIN_MATCHED_DEFAULT_COVERAGE warns without failing: a coverage
    regression is visible but not a deploy blocker. Without a key the
    catalog chain (or OR-only data) is expected, so the gate never raises.
    """
    if not os.environ.get("AA_API_KEY"):
        return
    catalog = _load_artifact(catalog_path, ())
    sources = catalog.get("sources")
    aa_sources = sources.get("aa") if isinstance(sources, dict) else None
    if not isinstance(aa_sources, dict):
        aa_sources = {}
    mode = aa_sources.get("mode")
    matched_aa = aa_sources.get("matched_aa")
    depth_rows = aa_sources.get("depth_rows")
    matched_default = aa_sources.get("matched_default")
    if mode != AA_LIVE_MODE:
        raise RuntimeError(
            "publish: AA_API_KEY is set but catalog sources.aa is "
            f"{{mode: {mode!r}, matched_aa: {matched_aa!r}}}: the AA API "
            "did not yield data. Check the key and artificialanalysis.ai, "
            "or unset AA_API_KEY, before deploying."
        )
    for label, value in (
        ("matched_aa", matched_aa),
        ("depth_rows", depth_rows),
        ("matched_default", matched_default),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise RuntimeError(
                f"publish: AA_API_KEY is set but catalog sources.aa.{label} "
                f"is {value!r}: the AA API did not yield usable data. Check "
                "the key and artificialanalysis.ai, or unset AA_API_KEY, "
                "before deploying."
            )
    pool = catalog.get("pool")
    candidates = pool.get("candidates") if isinstance(pool, dict) else None
    if (
        isinstance(candidates, int)
        and not isinstance(candidates, bool)
        and candidates > 0
        and matched_default / candidates < MIN_MATCHED_DEFAULT_COVERAGE
    ):
        print(
            "publish: warning: AA quality-coverage is "
            f"{matched_default} of {candidates} candidates "
            f"({matched_default / candidates:.0%} < "
            f"{MIN_MATCHED_DEFAULT_COVERAGE:.0%}): quality records cover few candidates",
            file=sys.stderr,
        )
    print(
        f"publish: AA source mode {mode}, matched_aa {matched_aa}, "
        f"depth_rows {depth_rows}, matched_default {matched_default}"
    )


def check_endpoint_drops(catalog_path: Path) -> None:
    """Fail unless the endpoint stage kept most of the pool it entered.

    `pool.candidates` counts the models that survived the endpoint stage, so
    the models that entered it are `candidates` plus the stage's own drop
    reasons: a model dropped before the endpoint stage never reached it, and
    one that dropped during it is one of these two reasons. A few endpoint
    failures are normal churn (provider outages, per-key gating), but a mass
    exodus means the ranking no longer reflects the same candidate pool as
    previous runs -- fail the publish rather than ship it silently.
    Publish imports nothing from model_compare (subprocess-only by design),
    so the reasons are listed here rather than imported.
    """
    catalog = _load_artifact(catalog_path, ("generated_at", "pool"))
    pool = catalog["pool"]
    if not isinstance(pool, dict):
        pool = {}
    dropped = pool.get("dropped")
    if not isinstance(dropped, dict):
        dropped = {}

    def _count(value: object, label: str) -> int:
        # Missing counts default to 0 (an older catalog never recorded
        # them); a present-but-malformed count is a producer bug, and the
        # module contract is to fail loudly on any unexpected result.
        if value is None:
            return 0
        if type(value) is not int or value < 0:
            raise RuntimeError(
                f"publish: {catalog_path.name} {label} is not a count: {value!r}"
            )
        return value

    no_baseline = _count(
        dropped.get(ENDPOINT_DROP_REASONS[0]),
        f"pool.dropped.{ENDPOINT_DROP_REASONS[0]}",
    )
    fetch_failed = _count(
        dropped.get(ENDPOINT_DROP_REASONS[1]),
        f"pool.dropped.{ENDPOINT_DROP_REASONS[1]}",
    )
    candidates = _count(pool.get("candidates"), "pool.candidates")
    dropped_total = no_baseline + fetch_failed
    entered = candidates + dropped_total
    if entered == 0:
        return
    if dropped_total / entered > MAX_ENDPOINT_DROP_RATE:
        raise RuntimeError(
            f"publish: {catalog_path.name} endpoint stage dropped "
            f"{dropped_total} of {entered} "
            f"({dropped_total / entered:.0%} > "
            f"{MAX_ENDPOINT_DROP_RATE:.0%} limit): "
            f"{ENDPOINT_DROP_REASONS[0]} {no_baseline}, "
            f"{ENDPOINT_DROP_REASONS[1]} {fetch_failed}"
        )


def build_site(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mc-build-") as tmp:
        scratch = Path(tmp)
        model_compare = REPO_ROOT / "model_compare.py"

        # --catalog first: it fetches OpenRouter/AA data and primes the shared
        # caches, so later invocations very likely rank identical inputs.
        # Not guaranteed: degraded fetches (empty discounts, no AA entries)
        # are never cached, so a later run may fetch different data. That
        # fails loudly (order_rows, check_best) and the next run self-heals.
        run_script(model_compare, ["--catalog"], out=scratch / "catalog.json")
        # Gates before anything else runs, so a failed AA source gate or an
        # endpoint-drop exodus writes no publish artifacts to output_dir.
        check_aa_source(scratch / "catalog.json")
        check_endpoint_drops(scratch / "catalog.json")
        run_script(model_compare, ["--best"], out=output_dir / "best.txt")
        for priority in PRIORITIES:
            run_script(
                model_compare,
                ["--priority", priority, "--json", "--top", "20"],
                out=scratch / f"{priority}.json",
            )

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
                "--revision",
                resolve_revision(),
                *priority_args,
            ],
        )

        for name in (
            "index.html",
            "faq.html",
            "style.css",
            "app.js",
            "favicon.svg",
            "favicon-16x16.png",
            "favicon-32x32.png",
            "apple-touch-icon.png",
        ):
            shutil.copyfile(WEB_DIR / "site" / name, output_dir / name)

    artifacts = (
        "data.json",
        "catalog.json",
        "history.json",  # self-feeds the next run via PREV_HISTORY_URL
        "highlights.json",  # ditto via PREV_HIGHLIGHTS_URL
        "best.txt",
        "index.html",
        "faq.html",
        "style.css",
        "app.js",
        "favicon.svg",
        "favicon-16x16.png",
        "favicon-32x32.png",
        "apple-touch-icon.png",
    )
    missing = [name for name in artifacts if not (output_dir / name).is_file()]
    if missing:
        raise RuntimeError(
            f"publish: expected artifacts not written: {', '.join(missing)}"
        )
    check_best(output_dir, check_stamps(output_dir))
    for name in artifacts:
        print(f"publish: wrote {output_dir / name}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="publish.py",
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
