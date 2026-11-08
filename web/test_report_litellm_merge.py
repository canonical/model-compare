"""Tests for web/report_litellm_merge.py."""

import json

import report_litellm_merge as report


def _status(needed=True):
    return {
        "_meta": {
            "generated_at": "2026-11-08T00:00:00+00:00",
            "source_revision": "deadbee",
        },
        "upstream": {
            "url": "https://example/upstream.json",
            "sha256": "a" * 64,
            "entries": 4871,
        },
        "merged": {
            "status": "frozen",
            "reason": "collision",
            "sha256": "b" * 64,
            "entries": 4871,
            "rate_card_sha256": "c" * 64,
        },
        "collisions": ["openrouter/z-ai/glm-5.3"],
        "upstream_fetch_failures": 1,
        "report": {"needed": needed, "reason": "collision"},
    }


def test_should_report_only_when_needed():
    assert report.should_report({"report": {"needed": False}}) is False
    assert report.should_report({"report": {"needed": True}}) is True


def test_noops_without_token(monkeypatch, tmp_path):
    path = tmp_path / "status.json"
    path.write_text(json.dumps(_status()))
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_REPOSITORY", raising=False)
    assert report.main(["--status", str(path)]) == 0


def test_noops_when_not_needed(monkeypatch, tmp_path):
    path = tmp_path / "status.json"
    path.write_text(json.dumps(_status(needed=False)))
    monkeypatch.setenv("GITHUB_TOKEN", "x")
    monkeypatch.setenv("GITHUB_REPOSITORY", "canonical/model-compare")
    calls = []
    monkeypatch.setattr(report, "_request", lambda *a, **k: calls.append(a) or [])
    assert report.main(["--status", str(path)]) == 0
    assert calls == []


def test_creates_issue_when_none_matches(monkeypatch, tmp_path):
    path = tmp_path / "status.json"
    path.write_text(json.dumps(_status()))
    monkeypatch.setenv("GITHUB_TOKEN", "x")
    monkeypatch.setenv("GITHUB_REPOSITORY", "canonical/model-compare")
    calls = []

    def fake_request(method, url, payload=None):
        calls.append((method, url, payload))
        if method == "GET":
            return []
        return {"number": 7}

    monkeypatch.setattr(report, "_request", fake_request)
    assert report.main(["--status", str(path)]) == 0
    assert any(method == "POST" and url.endswith("/issues") for method, url, _ in calls)


def test_comments_when_issue_exists(monkeypatch, tmp_path):
    path = tmp_path / "status.json"
    path.write_text(json.dumps(_status()))
    monkeypatch.setenv("GITHUB_TOKEN", "x")
    monkeypatch.setenv("GITHUB_REPOSITORY", "canonical/model-compare")
    calls = []

    def fake_request(method, url, payload=None):
        calls.append((method, url, payload))
        if method == "GET":
            return [{"number": 7, "title": report.ISSUE_TITLE}]
        return {"number": 7}

    monkeypatch.setattr(report, "_request", fake_request)
    assert report.main(["--status", str(path)]) == 0
    assert any(
        method == "POST" and url.endswith("/issues/7/comments")
        for method, url, _ in calls
    )


def test_http_failure_is_a_warning(monkeypatch, tmp_path):
    path = tmp_path / "status.json"
    path.write_text(json.dumps(_status()))
    monkeypatch.setenv("GITHUB_TOKEN", "x")
    monkeypatch.setenv("GITHUB_REPOSITORY", "canonical/model-compare")

    def broken(*args, **kwargs):
        raise OSError("network down")

    monkeypatch.setattr(report, "_request", broken)
    assert report.main(["--status", str(path)]) == 0


def test_missing_status_file_is_a_noop(monkeypatch, tmp_path):
    monkeypatch.setenv("GITHUB_TOKEN", "x")
    monkeypatch.setenv("GITHUB_REPOSITORY", "canonical/model-compare")
    assert report.main(["--status", str(tmp_path / "missing.json")]) == 0
