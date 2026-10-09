#!/usr/bin/env python3
"""Publish the model-compare site.

Runs the standalone model_compare.py, then the web-side generators
(generate_highlights.py, build_site_data.py), and assembles the deploy
directory: data.json, catalog.json, history.json, highlights.json, best.txt
and index.html. Fails loudly on any unexpected result -- including
data.json/history.json stamps that disagree with catalog.json's
generated_at, or a best.txt that is not catalog.json's balanced rank 1 --
so a broken run never deploys a broken site.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
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
VERSION = "0.2.7"
USER_AGENT = f"model-compare/{VERSION} (https://github.com/canonical/model-compare)"
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
    Returns the loaded catalog (which must carry rankings) for check_best.
    """
    catalog = _load_artifact(output_dir / "catalog.json", ("generated_at", "rankings"))
    data = _load_artifact(output_dir / "data.json", ("generated_at",))
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


AA_FALLBACKS_LIVE = ("api",)


def check_aa_fallback(catalog_path: Path) -> None:
    """With AA_API_KEY set, fail unless the AA API itself worked.

    sources.aa.mode "openrouter" means at least one candidate's AA data came
    from OpenRouter; it says nothing about whether the AA API worked.
    sources.aa.fallback records that ("api" or "none"; the page scrape that
    could also yield "scrape" was removed), so the gate reads fallback and
    requires "api": a rejected key yields "none" and fails. A missing or
    unknown value fails closed. Without a key AA data comes only from
    OpenRouter, so the gate never raises.
    """
    if not os.environ.get("AA_API_KEY"):
        return
    catalog = _load_artifact(catalog_path, ())
    sources = catalog.get("sources")
    aa_sources = sources.get("aa") if isinstance(sources, dict) else None
    if not isinstance(aa_sources, dict):
        aa_sources = {}
    fallback = aa_sources.get("fallback")
    if fallback not in AA_FALLBACKS_LIVE:
        raise RuntimeError(
            "publish: AA_API_KEY is set but catalog sources.aa.fallback is "
            f"{fallback!r} (mode {aa_sources.get('mode')!r}): the AA API "
            "did not yield data. Check the key and artificialanalysis.ai, "
            "or unset AA_API_KEY, before deploying."
        )
    print(f"publish: AA source mode {aa_sources.get('mode')}, fallback {fallback}")


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
        # Gate before anything else runs, so a failed AA fallback writes no
        # publish artifacts to output_dir.
        check_aa_fallback(scratch / "catalog.json")
        run_script(model_compare, ["--best"], out=output_dir / "best.txt")
        for priority in PRIORITIES:
            run_script(
                model_compare,
                ["--priority", priority, "--json", "--top", "10"],
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
                *priority_args,
            ],
        )

        shutil.copyfile(WEB_DIR / "site" / "index.html", output_dir / "index.html")

    artifacts = (
        "data.json",
        "catalog.json",
        "history.json",  # self-feeds the next run via PREV_HISTORY_URL
        "highlights.json",  # ditto via PREV_HIGHLIGHTS_URL
        "best.txt",
        "index.html",
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
