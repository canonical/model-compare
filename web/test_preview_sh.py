"""Smoke tests for preview.sh.

These run the script end-to-end without any network access: the live
fetch is pointed at an unreachable URL, so the script falls back to its
synthetic placeholder data and serves it locally.
"""

import json
import os
import signal
import socket
import subprocess
import time
import urllib.request
from pathlib import Path

import publish

SCRIPT = Path(__file__).parent / "preview.sh"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _get(url: str, timeout: float = 2.0) -> bytes:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url, timeout=timeout) as resp:
        return resp.read()


def test_preview_sh_syntax():
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)


def test_preview_sh_help():
    subprocess.run([str(SCRIPT), "--help"], check=True, capture_output=True, timeout=10)


def test_preview_sh_rejects_bad_port():
    proc = subprocess.run(
        [str(SCRIPT), "--port", "abc"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert proc.returncode == 2
    assert "--port wants a number" in proc.stderr


def test_preview_sh_serves_synthetic_fallback(tmp_path):
    port = _free_port()
    env = dict(os.environ, TMPDIR=str(tmp_path))
    proc = subprocess.Popen(
        [
            "bash",
            str(SCRIPT),
            "--port",
            str(port),
            "--live-url",
            "http://127.0.0.1:9",
        ],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        deadline = time.time() + 30
        data = None
        while time.time() < deadline:
            try:
                data = json.loads(_get(f"http://127.0.0.1:{port}/data.json", timeout=1))
                break
            except Exception:
                if proc.poll() is not None:
                    raise AssertionError("preview.sh exited before serving data.json")
                time.sleep(0.25)
        assert data is not None, "preview.sh never served data.json"

        assert set(data["priorities"]) == {"balanced", "price", "quality"}
        assert all(len(rows) == 20 for rows in data["priorities"].values())
        assert data["schema_version"] == 2  # data.json is schema-versioned too
        assert data["schedules"] == []  # every v2 data.json carries the key
        assert data["best"].startswith("openrouter/")
        assert (
            _get(f"http://127.0.0.1:{port}/best.txt").strip() == data["best"].encode()
        )
        served_html = _get(f"http://127.0.0.1:{port}/index.html")
        assert b"generated-top" in served_html
        assert b'id="deals"' in served_html and b"TIER" in served_html
        # The site's CSS/JS are external files now: preview must serve
        # them next to index.html (the schedule clip wrapper lives in
        # style.css, the table renderer in app.js).
        served_css = _get(f"http://127.0.0.1:{port}/style.css")
        assert b"schedule-text" in served_css
        served_js = _get(f"http://127.0.0.1:{port}/app.js")
        assert b"function render" in served_js
    finally:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=10)


def test_preview_serves_the_same_files_as_publish():
    # A page added to publish.SITE_FILES but not to preview.sh 404s in the
    # local preview with no test failing.
    line = next(
        line
        for line in SCRIPT.read_text().splitlines()
        if line.startswith("SITE_FILES=")
    )
    names = line.split("=", 1)[1].strip().strip('"').split()
    assert set(names) == set(publish.SITE_FILES)


def test_preview_fetches_companion_json_artifacts_when_live_succeeds(tmp_path):
    # Without highlights.json, history.json and catalog.json, local preview
    # hides Weekly highlights, blanks 7-DAY, and shows a red catalog error.
    text = SCRIPT.read_text()
    for artifact in ("highlights.json", "history.json", "catalog.json"):
        assert f"$LIVE_URL/{artifact}" in text, artifact
