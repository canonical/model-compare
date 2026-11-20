#!/usr/bin/env python3
"""Report a LiteLLM merged-map collision or a repeated upstream failure.

Reads litellm_merge_status.json and, when the status asks for a report,
creates or comments on one deduplicated GitHub issue. When `--allow-close`
is passed and the rebuilt map is healthy (`merged.status == "ok"`), closes
any open alert issue. Best-effort: every failure is a warning, never a
deploy failure.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import sys
import urllib.error
import urllib.request

ISSUE_TITLE = "LiteLLM merged map needs attention"
ISSUE_LABEL = "bug"
USER_AGENT = "model-compare (https://github.com/canonical/model-compare)"


def should_report(status: dict) -> bool:
    report = status.get("report") if isinstance(status, dict) else None
    return isinstance(report, dict) and report.get("needed") is True


def should_close(status: dict, allow_close: bool = False) -> bool:
    """A deployed rebuilt map clears the alert; frozen/skipped or unconfirmed runs do not."""
    if not allow_close:
        return False
    merged = status.get("merged") if isinstance(status, dict) else None
    return isinstance(merged, dict) and merged.get("status") == "ok"


def issue_title() -> str:
    return ISSUE_TITLE


def _format_per_million(value) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return str(value)
    text = f"{value * 1_000_000:.9f}".rstrip("0").rstrip(".")
    return f"${text or '0'}"


def _format_tokens(value) -> str:
    if isinstance(value, int) and not isinstance(value, bool):
        return f"{value:,}"
    return str(value)


def _run_url() -> str:
    server = os.environ.get("GITHUB_SERVER_URL") or "https://github.com"
    repo = os.environ.get("GITHUB_REPOSITORY") or ""
    run_id = os.environ.get("GITHUB_RUN_ID") or ""
    if not repo or not run_id:
        return ""
    return f"{server.rstrip('/')}/{repo}/actions/runs/{run_id}"


def issue_body(status: dict, run_url: str) -> str:
    report = status.get("report") if isinstance(status.get("report"), dict) else {}
    upstream = (
        status.get("upstream") if isinstance(status.get("upstream"), dict) else {}
    )
    merged = status.get("merged") if isinstance(status.get("merged"), dict) else {}
    meta = status.get("_meta") if isinstance(status.get("_meta"), dict) else {}
    collisions = status.get("collisions") or []
    details = status.get("collision_details")
    if isinstance(details, list):
        blocking = [
            detail
            for detail in details
            if isinstance(detail, dict) and detail.get("resolution") == "blocked"
        ]
        resolved = [
            detail
            for detail in details
            if isinstance(detail, dict)
            and detail.get("resolution") in ("fragment", "upstream")
        ]
    else:
        # Older status files only carry the colliding keys.
        blocking = [
            {
                "key": key,
                "resolution": "blocked",
                "differences": [],
            }
            for key in collisions
        ]
        resolved = []
    lines = [
        "The published LiteLLM merged map was not updated.",
        "",
        f"Reason: {report.get('reason')}",
        f"Catalog generated_at: {meta.get('generated_at')}",
        f"Source revision: {meta.get('source_revision')}",
        "",
        f"Upstream map: {upstream.get('url')}",
        f"Upstream SHA-256: {upstream.get('sha256')}",
        f"Upstream entries: {upstream.get('entries')}",
        f"Published merged map: {merged.get('status')} "
        f"(SHA-256 {merged.get('sha256')}, entries {merged.get('entries')})",
        f"Upstream fetch failures: {status.get('upstream_fetch_failures')}",
        "",
        f"Blocking collisions: {len(blocking)}",
    ]
    if blocking:
        lines += ["", "Blocking collisions (need a decision):"]
        for detail in blocking:
            note = detail.get("note")
            suffix = f" — {note}" if isinstance(note, str) and note else ""
            lines.append(f"- `{detail.get('key')}`{suffix}")
    if resolved:
        lines += ["", "Auto-resolved collisions (no action needed):"]
        for detail in resolved:
            if detail.get("overridden"):
                kept = (
                    "model-compare entry"
                    if detail.get("resolution") == "fragment"
                    else "LiteLLM upstream entry"
                )
                suffix = f" — reviewed override; {kept} kept"
                reason = detail.get("override_reason")
                if isinstance(reason, str) and reason:
                    suffix += f" ({reason})"
            else:
                suffix = " — same prices; LiteLLM upstream entry kept"
            lines.append(f"- `{detail.get('key')}`{suffix}")
    rows = []
    for detail in blocking + resolved:
        for difference in detail.get("differences") or []:
            if not isinstance(difference, dict):
                continue
            rows.append((detail, difference))
    if rows:
        lines += [
            "",
            "Price differences (per 1M tokens):",
            "",
            "| Key | Resolution | Field | From tokens | model-compare | LiteLLM upstream |",
            "| --- | --- | --- | ---: | ---: | ---: |",
        ]
        resolution_labels = {
            "blocked": "blocked",
            "fragment": "model-compare",
            "upstream": "LiteLLM upstream",
        }
        for detail, difference in rows:
            resolution = detail.get("resolution")
            lines.append(
                f"| `{detail.get('key')}` | "
                f"{resolution_labels.get(resolution, resolution)} | "
                f"`{difference.get('field')}` | "
                f"{_format_tokens(difference.get('from_tokens'))} | "
                f"{_format_per_million(difference.get('model_compare'))} | "
                f"{_format_per_million(difference.get('upstream'))} |"
            )
    if run_url:
        lines += ["", f"Workflow run: {run_url}"]
    return "\n".join(lines)


def _request(method: str, url: str, payload: dict | None = None):
    data = None
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": USER_AGENT,
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=30) as response:
        raw = response.read()
    if not raw:
        return None
    return json.loads(raw)


def _api_base() -> str:
    return (os.environ.get("GITHUB_API_URL") or "https://api.github.com").rstrip("/")


def find_open_issue(repo: str) -> dict | None:
    url = f"{_api_base()}/repos/{repo}/issues?state=open&per_page=100"
    issues = _request("GET", url)
    if not isinstance(issues, list):
        return None
    for issue in issues:
        if not isinstance(issue, dict):
            continue
        if "pull_request" in issue:
            continue
        if issue.get("title") == ISSUE_TITLE:
            return issue
    return None


def create_issue(repo: str, body: str) -> dict | None:
    url = f"{_api_base()}/repos/{repo}/issues"
    payload = {"title": ISSUE_TITLE, "body": body, "labels": [ISSUE_LABEL]}
    result = _request("POST", url, payload)
    return result if isinstance(result, dict) else None


def comment_issue(repo: str, number: int, body: str) -> dict | None:
    url = f"{_api_base()}/repos/{repo}/issues/{number}/comments"
    result = _request("POST", url, {"body": body})
    return result if isinstance(result, dict) else None


def close_issue(repo: str, number: int, body: str) -> dict | None:
    comment_issue(repo, number, body)
    url = f"{_api_base()}/repos/{repo}/issues/{number}"
    result = _request("PATCH", url, {"state": "closed", "state_reason": "completed"})
    return result if isinstance(result, dict) else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="report_litellm_merge.py")
    parser.add_argument("--status", required=True)
    parser.add_argument("--allow-close", action="store_true")
    args = parser.parse_args(argv)

    try:
        with open(args.status) as handle:
            status = json.load(handle)
    except (OSError, ValueError) as exc:
        print(f"report: cannot read {args.status}: {exc}", file=sys.stderr)
        return 0
    if not isinstance(status, dict):
        return 0
    report_needed = should_report(status)
    close_needed = should_close(status, allow_close=args.allow_close)
    if not report_needed and not close_needed:
        return 0

    repo = os.environ.get("GITHUB_REPOSITORY")
    if not os.environ.get("GITHUB_TOKEN") or not repo:
        print(
            "report: GITHUB_TOKEN/GITHUB_REPOSITORY not set; skipping", file=sys.stderr
        )
        return 0

    try:
        existing = find_open_issue(repo)
        if report_needed:
            body = issue_body(status, _run_url())
            if existing is not None and isinstance(existing.get("number"), int):
                comment_issue(repo, existing["number"], body)
                print(f"report: commented on issue #{existing['number']}")
            else:
                created = create_issue(repo, body)
                if created is not None:
                    print(f"report: opened issue #{created.get('number')}")
        elif existing is not None and isinstance(existing.get("number"), int):
            close_issue(
                repo,
                existing["number"],
                "The LiteLLM merged map rebuilt successfully; closing this alert.",
            )
            print(f"report: closed issue #{existing['number']}")
    except (
        urllib.error.URLError,
        http.client.HTTPException,
        OSError,
        ValueError,
    ) as exc:
        print(f"report: warning: GitHub API failed: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
