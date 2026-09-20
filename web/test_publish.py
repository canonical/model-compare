"""Tests for the web/publish.py orchestrator (all subprocesses stubbed)."""

import http.client
import subprocess
import urllib.error
from pathlib import Path

import pytest

import publish


class _FakeResp:
    def __init__(self, payload: bytes):
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _TruncatedResp:
    def read(self):
        raise http.client.IncompleteRead(b"part", 10)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _fake_run(calls, simulate_build_site_data=True):
    def fake_run(cmd, cwd=None, stdout=None, check=True):
        calls.append(cmd)
        if "--best" in cmd:
            stdout.write("openrouter/z-ai/glm-5.3-flash\n")
        if simulate_build_site_data and Path(cmd[1]).name == "build_site_data.py":
            out_dir = Path(cmd[cmd.index("--output") + 1]).parent
            for name in (
                "data.json",
                "catalog.json",
                "history.json",
                "highlights.json",
            ):
                (out_dir / name).touch()
        return subprocess.CompletedProcess(cmd, 0)

    return fake_run


def test_build_site_pipeline_order_and_assembly(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(publish.subprocess, "run", _fake_run(calls))
    monkeypatch.setattr(publish, "fetch_prev", lambda url: None)
    out = tmp_path / "site"

    publish.build_site(out)

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
    best_cmd = next(cmd for cmd in calls if "--best" in cmd)
    assert not any(a == "--priority" for a in best_cmd)
    assert (
        sum(
            1
            for cmd in calls
            if Path(cmd[1]).name == "model_compare.py" and "--catalog" in cmd
        )
        == 1
    )
    json_calls = [cmd for cmd in calls if "--json" in cmd]
    assert [
        json_calls[i][json_calls[i].index("--priority") + 1]
        for i in range(len(json_calls))
    ] == ["balanced", "price", "quality"]
    assert (out / "best.txt").read_text().strip() == "openrouter/z-ai/glm-5.3-flash"
    assert (out / "index.html").is_file()

    # Downstream argument wiring is the riskiest surface: pin it exactly.
    gh_cmd = next(cmd for cmd in calls if Path(cmd[1]).name == "generate_highlights.py")
    assert gh_cmd[2::2] == ["--catalog", "--history", "--prev-highlights", "--output"]
    gh_paths = [Path(p) for p in gh_cmd[3::2]]
    assert gh_paths[0].name == "catalog.json"
    assert gh_paths[3].name == "highlights-new.json"
    assert all(p.parent == gh_paths[0].parent for p in gh_paths)

    bsd_cmd = next(cmd for cmd in calls if Path(cmd[1]).name == "build_site_data.py")
    assert bsd_cmd[2:12:2] == [
        "--best-file",
        "--output",
        "--catalog-file",
        "--history-prev-file",
        "--highlights-file",
    ]
    bsd_paths = [Path(p) for p in bsd_cmd[3:13:2]]
    assert bsd_paths[0] == out / "best.txt"
    assert bsd_paths[1] == out / "data.json"
    assert bsd_paths[2].parent == gh_paths[0].parent  # shared scratch dir
    priority_values = bsd_cmd[13::2]
    assert [p.split("=", 1)[0] for p in priority_values] == [
        "balanced",
        "price",
        "quality",
    ]
    assert all(
        Path(p.split("=", 1)[1]).parent == gh_paths[0].parent for p in priority_values
    )


def test_build_site_fails_loudly(tmp_path, monkeypatch):
    def boom(cmd, cwd=None, stdout=None, check=True):
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(publish.subprocess, "run", boom)
    with pytest.raises(subprocess.CalledProcessError):
        publish.build_site(tmp_path / "site")


def test_fetch_prev_retries_transient_failures(monkeypatch):
    attempts = []
    sleeps = []

    def flaky(req, timeout=None):
        attempts.append(1)
        if len(attempts) < 3:
            raise urllib.error.URLError("transient")
        return _FakeResp(b"cached")

    monkeypatch.setattr(publish.urllib.request, "urlopen", flaky)
    monkeypatch.setattr(publish.time, "sleep", sleeps.append)
    assert publish.fetch_prev("https://x/history.json") == b"cached"
    assert len(attempts) == 3
    assert sleeps == [1, 2]


def test_fetch_prev_gives_up_after_attempts(monkeypatch):
    attempts = []

    def down(req, timeout=None):
        attempts.append(1)
        raise urllib.error.URLError("down")

    monkeypatch.setattr(publish.urllib.request, "urlopen", down)
    monkeypatch.setattr(publish.time, "sleep", lambda s: None)
    assert publish.fetch_prev("https://x/history.json") is None
    assert len(attempts) == 3


def test_fetch_prev_handles_truncated_body(monkeypatch):
    monkeypatch.setattr(
        publish.urllib.request, "urlopen", lambda req, timeout=None: _TruncatedResp()
    )
    monkeypatch.setattr(publish.time, "sleep", lambda s: None)
    assert publish.fetch_prev("https://x/history.json") is None


def test_fetch_prev_returns_none_on_error(monkeypatch):
    def boom(req, timeout=None):
        raise urllib.error.URLError("unreachable")

    monkeypatch.setattr(publish.urllib.request, "urlopen", boom)
    monkeypatch.setattr(publish.time, "sleep", lambda s: None)
    assert publish.fetch_prev("http://127.0.0.1:9/history.json") is None


def test_fetch_success_feeds_prev_files_to_generators(tmp_path, monkeypatch):
    seen = {}

    def fake_run(cmd, cwd=None, stdout=None, check=True):
        if "--best" in cmd:
            stdout.write("openrouter/x/y\n")
        if Path(cmd[1]).name == "build_site_data.py":
            prev_arg = Path(cmd[cmd.index("--history-prev-file") + 1])
            # read during the call: the scratch dir is removed afterwards
            seen["history_bytes"] = prev_arg.read_bytes() if prev_arg.exists() else None
            out_dir = Path(cmd[cmd.index("--output") + 1]).parent
            for name in (
                "data.json",
                "catalog.json",
                "history.json",
                "highlights.json",
            ):
                (out_dir / name).touch()
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(publish.subprocess, "run", fake_run)
    monkeypatch.setattr(publish, "fetch_prev", lambda url: b'{"snapshots": []}')
    publish.build_site(tmp_path / "site")
    assert seen["history_bytes"] == b'{"snapshots": []}'


def test_build_site_fails_when_expected_artifacts_missing(tmp_path, monkeypatch):
    calls = []

    def fake_run(cmd, cwd=None, stdout=None, check=True):
        calls.append(cmd)
        if "--best" in cmd:
            stdout.write("openrouter/x/y\n")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(publish.subprocess, "run", fake_run)
    monkeypatch.setattr(publish, "fetch_prev", lambda url: None)
    with pytest.raises(RuntimeError):
        publish.build_site(tmp_path / "site")


def test_main_output_dir(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(publish, "build_site", lambda out: seen.setdefault("out", out))
    rc = publish.main(["--output-dir", str(tmp_path / "deploy")])
    assert rc == 0
    assert seen["out"] == tmp_path / "deploy"


def test_main_output_dir_default(monkeypatch):
    seen = {}
    monkeypatch.setattr(publish, "build_site", lambda out: seen.setdefault("out", out))
    rc = publish.main([])
    assert rc == 0
    assert seen["out"] == publish.REPO_ROOT / "_site"
