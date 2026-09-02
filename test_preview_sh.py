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
        assert all(len(rows) == 10 for rows in data["priorities"].values())
        assert data["best"].startswith("openrouter/")
        assert (
            _get(f"http://127.0.0.1:{port}/best.txt").strip() == data["best"].encode()
        )
        assert b"generated-top" in _get(f"http://127.0.0.1:{port}/index.html")
    finally:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=10)
