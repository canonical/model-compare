"""Tests for the web/publish.py orchestrator (all subprocesses stubbed)."""

import subprocess
import urllib.error
from pathlib import Path

import pytest

import publish


def _fake_run(calls):
    def fake_run(cmd, cwd=None, stdout=None, check=True):
        calls.append(cmd)
        if "--best" in cmd:
            stdout.write("openrouter/z-ai/glm-5.3-flash\n")
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
    assert "--priority" not in next(cmd for cmd in calls if "--best" in cmd)
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


def test_build_site_fails_loudly(tmp_path, monkeypatch):
    def boom(cmd, cwd=None, stdout=None, check=True):
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(publish.subprocess, "run", boom)
    with pytest.raises(subprocess.CalledProcessError):
        publish.build_site(tmp_path / "site")


def test_fetch_prev_returns_none_on_error(monkeypatch):
    def boom(req, timeout=None):
        raise urllib.error.URLError("unreachable")

    monkeypatch.setattr(publish.urllib.request, "urlopen", boom)
    assert publish.fetch_prev("http://127.0.0.1:9/history.json") is None


def test_main_output_dir(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(publish, "build_site", lambda out: seen.setdefault("out", out))
    rc = publish.main(["--output-dir", str(tmp_path / "deploy")])
    assert rc == 0
    assert seen["out"] == tmp_path / "deploy"
