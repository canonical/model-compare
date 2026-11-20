"""Tests for web/report_litellm_merge.py."""

import http.client
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


def test_should_close_only_when_merged_is_ok():
    assert report.should_close({"merged": {"status": "ok"}}) is False
    assert report.should_close({"merged": {"status": "ok"}}, allow_close=True) is True
    assert (
        report.should_close({"merged": {"status": "frozen"}}, allow_close=True) is False
    )
    assert (
        report.should_close({"merged": {"status": "skipped"}}, allow_close=True)
        is False
    )
    assert report.should_close({}, allow_close=True) is False


def test_issue_body_highlights_blocking_diff():
    status = _status()
    status["collision_details"] = [
        {
            "key": "openrouter/google/gemini-4-hydrogen",
            "resolution": "blocked",
            "price_equal": False,
            "overridden": False,
            "differences": [
                {
                    "field": "input_cost_per_token",
                    "from_tokens": 0,
                    "model_compare": 7.5e-7,
                    "upstream": 1.5e-6,
                },
                {
                    "field": "output_cost_per_token",
                    "from_tokens": 0,
                    "model_compare": 3.75e-6,
                    "upstream": 7.5e-6,
                },
            ],
        }
    ]
    body = report.issue_body(status, "https://example/run")
    assert "Blocking collisions" in body
    assert "| Key | Resolution | Field |" in body
    assert (
        "| `openrouter/google/gemini-4-hydrogen` | blocked | "
        "`input_cost_per_token` |" in body
    )
    assert "$0.75" in body
    assert "$1.5" in body
    assert "$3.75" in body
    assert "$7.5" in body
    assert "https://example/run" in body


def test_issue_body_lists_auto_resolved_collisions():
    status = _status()
    status["collision_details"] = [
        {
            "key": "openrouter/openai/gpt-6.2-sol",
            "resolution": "upstream",
            "price_equal": True,
            "overridden": False,
            "differences": [],
        },
        {
            "key": "openrouter/openai/gpt-6.2-astra",
            "resolution": "fragment",
            "price_equal": False,
            "overridden": True,
            "override_reason": "baseline price is intentional",
            "differences": [
                {
                    "field": "input_cost_per_token",
                    "from_tokens": 0,
                    "model_compare": 7.5e-7,
                    "upstream": 1.5e-6,
                }
            ],
        },
        {
            "key": "openrouter/google/gemini-4-hydrogen",
            "resolution": "upstream",
            "price_equal": False,
            "overridden": True,
            "differences": [],
        },
    ]
    body = report.issue_body(status, "")
    assert "Auto-resolved collisions" in body
    assert "same prices" in body
    assert "reviewed override; model-compare entry kept" in body
    assert "reviewed override; LiteLLM upstream entry kept" in body
    assert "baseline price is intentional" in body
    assert "| model-compare | `input_cost_per_token` |" in body
    assert "$0.75" in body
    assert "$1.5" in body


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


def test_http_exception_is_a_warning(monkeypatch, tmp_path):
    path = tmp_path / "status.json"
    path.write_text(json.dumps(_status()))
    monkeypatch.setenv("GITHUB_TOKEN", "x")
    monkeypatch.setenv("GITHUB_REPOSITORY", "canonical/model-compare")

    def broken(*args, **kwargs):
        raise http.client.IncompleteRead(b"part", 10)

    monkeypatch.setattr(report, "_request", broken)
    assert report.main(["--status", str(path)]) == 0


def test_missing_status_file_is_a_noop(monkeypatch, tmp_path):
    monkeypatch.setenv("GITHUB_TOKEN", "x")
    monkeypatch.setenv("GITHUB_REPOSITORY", "canonical/model-compare")
    assert report.main(["--status", str(tmp_path / "missing.json")]) == 0


def test_closes_issue_when_map_is_healthy(monkeypatch, tmp_path):
    status = _status(needed=False)
    status["merged"]["status"] = "ok"
    status["report"] = {"needed": False, "reason": None}
    path = tmp_path / "status.json"
    path.write_text(json.dumps(status))
    monkeypatch.setenv("GITHUB_TOKEN", "x")
    monkeypatch.setenv("GITHUB_REPOSITORY", "canonical/model-compare")
    calls = []

    def fake_request(method, url, payload=None):
        calls.append((method, url, payload))
        if method == "GET":
            return [{"number": 7, "title": report.ISSUE_TITLE}]
        return {"number": 7}

    monkeypatch.setattr(report, "_request", fake_request)
    assert report.main(["--status", str(path), "--allow-close"]) == 0
    assert any(
        method == "PATCH"
        and url.endswith("/issues/7")
        and payload == {"state": "closed", "state_reason": "completed"}
        for method, url, payload in calls
    )


def test_does_not_close_issue_without_allow_close(monkeypatch, tmp_path):
    status = _status(needed=False)
    status["merged"]["status"] = "ok"
    status["report"] = {"needed": False, "reason": None}
    path = tmp_path / "status.json"
    path.write_text(json.dumps(status))
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
    assert not any(method == "PATCH" for method, _, _ in calls)
    assert calls == []


def test_does_not_close_issue_when_map_is_frozen(monkeypatch, tmp_path):
    status = _status(needed=False)
    path = tmp_path / "status.json"
    path.write_text(json.dumps(status))
    monkeypatch.setenv("GITHUB_TOKEN", "x")
    monkeypatch.setenv("GITHUB_REPOSITORY", "canonical/model-compare")
    calls = []

    def fake_request(method, url, payload=None):
        calls.append((method, url, payload))
        if method == "GET":
            return [{"number": 7, "title": report.ISSUE_TITLE}]
        return {"number": 7}

    monkeypatch.setattr(report, "_request", fake_request)
    assert report.main(["--status", str(path), "--allow-close"]) == 0
    assert not any(method == "PATCH" for method, _, _ in calls)
