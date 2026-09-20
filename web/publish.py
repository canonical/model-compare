#!/usr/bin/env python3
"""Publish the model-compare site.

Runs the standalone model_compare.py, then the web-side generators
(generate_highlights.py, build_site_data.py), and assembles the deploy
directory: data.json, catalog.json, best.txt and index.html. Fails loudly on
any unexpected result so a broken run never deploys a broken site.
"""

from __future__ import annotations

import argparse
import http.client
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
VERSION = "0.2.0"
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
