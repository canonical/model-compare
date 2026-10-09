"""Tests for the web/publish.py orchestrator (all subprocesses stubbed)."""

import http.client
import json
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


STAMP = "2026-09-28T06:00:00Z"
BEST = "openrouter/z-ai/glm-5.3-flash"


def _write_artifacts(out_dir, overrides=None, best_id="z-ai/glm-5.3-flash"):
    """Write minimal valid build_site_data artifacts sharing one stamp.

    `overrides` maps an artifact name to replacement text (for mismatch and
    malformed-JSON cases); these stubs only feed publish's stamp and best
    gates. `best_id` is the bare catalog id ranked first for balanced, so it
    must match what the stubbed `--best` run wrote to best.txt.
    """
    docs = {
        "data.json": {"generated_at": STAMP},
        "catalog.json": {
            "generated_at": STAMP,
            "models": [],
            "rankings": {"balanced": [best_id]},
        },
        "history.json": {"updated_at": STAMP, "snapshots": {}},
        "highlights.json": {"generated_at": "2026-09-27T00:00:00Z"},
    }
    for name, doc in docs.items():
        text = (overrides or {}).get(name)
        (out_dir / name).write_text(json.dumps(doc) if text is None else text)


# What the stubbed `model_compare.py --catalog` run prints: the AA gate
# reads it from scratch before any other step runs.
HEALTHY_AA = {"mode": "openrouter", "fallback": "api"}


@pytest.fixture(autouse=True)
def _no_aa_key(monkeypatch):
    # The AA gate depends on AA_API_KEY; keep the suite hermetic. Gate tests
    # set the key explicitly.
    monkeypatch.delenv("AA_API_KEY", raising=False)


def _is_catalog_run(cmd):
    return Path(cmd[1]).name == "model_compare.py" and "--catalog" in cmd


def _fake_run(
    calls, simulate_build_site_data=True, overrides=None, best=BEST, aa=HEALTHY_AA
):
    def fake_run(cmd, cwd=None, stdout=None, check=True):
        calls.append(cmd)
        if _is_catalog_run(cmd):
            stdout.write(json.dumps({"generated_at": STAMP, "sources": {"aa": aa}}))
        if "--best" in cmd:
            stdout.write(best + "\n")
        if simulate_build_site_data and Path(cmd[1]).name == "build_site_data.py":
            _write_artifacts(Path(cmd[cmd.index("--output") + 1]).parent, overrides)
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
        "model_compare.py",  # --catalog (first: primes the shared caches)
        "model_compare.py",  # --best
        "model_compare.py",  # balanced --json --top 10
        "model_compare.py",  # price --json --top 10
        "model_compare.py",  # quality --json --top 10
        "generate_highlights.py",
        "build_site_data.py",
    ]
    assert "--catalog" in calls[0]
    assert "--best" in calls[1]
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
            _write_artifacts(Path(cmd[cmd.index("--output") + 1]).parent, best_id="x/y")
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


@pytest.mark.parametrize(
    "name, doc",
    [
        ("data.json", {"generated_at": "2026-09-27T06:00:00Z"}),
        ("history.json", {"updated_at": "2026-09-27T06:00:00Z", "snapshots": {}}),
    ],
)
def test_build_site_fails_on_generated_at_mismatch(tmp_path, monkeypatch, name, doc):
    overrides = {name: json.dumps(doc)}
    monkeypatch.setattr(publish.subprocess, "run", _fake_run([], overrides=overrides))
    monkeypatch.setattr(publish, "fetch_prev", lambda url: None)
    with pytest.raises(RuntimeError, match="generated_at mismatch") as exc:
        publish.build_site(tmp_path / "site")
    assert "2026-09-27T06:00:00Z" in str(exc.value)
    assert STAMP in str(exc.value)


def test_build_site_ignores_highlights_stamp(tmp_path, monkeypatch):
    # highlights.json keeps its own writing time (24h LLM-reuse window).
    monkeypatch.setattr(publish.subprocess, "run", _fake_run([]))
    monkeypatch.setattr(publish, "fetch_prev", lambda url: None)
    publish.build_site(tmp_path / "site")


@pytest.mark.parametrize("name", ["data.json", "catalog.json", "history.json"])
def test_build_site_fails_on_malformed_artifact(tmp_path, monkeypatch, name):
    overrides = {name: "{not json"}
    monkeypatch.setattr(publish.subprocess, "run", _fake_run([], overrides=overrides))
    monkeypatch.setattr(publish, "fetch_prev", lambda url: None)
    with pytest.raises(RuntimeError, match=name):
        publish.build_site(tmp_path / "site")


@pytest.mark.parametrize(
    "name, doc",
    [
        ("history.json", {"snapshots": {}}),
        ("history.json", {"updated_at": STAMP}),
        ("data.json", {}),
        ("catalog.json", []),
        # the best.txt gate reads rankings; a catalog without them is broken
        ("catalog.json", {"generated_at": STAMP, "models": []}),
    ],
)
def test_build_site_fails_on_missing_stamp_fields(tmp_path, monkeypatch, name, doc):
    overrides = {name: json.dumps(doc)}
    monkeypatch.setattr(publish.subprocess, "run", _fake_run([], overrides=overrides))
    monkeypatch.setattr(publish, "fetch_prev", lambda url: None)
    with pytest.raises(RuntimeError, match=name):
        publish.build_site(tmp_path / "site")


def _catalog_ranking(balanced):
    return json.dumps(
        {"generated_at": STAMP, "models": [], "rankings": {"balanced": balanced}}
    )


@pytest.mark.parametrize(
    "best", ["z-ai/glm-5.3-flash", "openrouter/z-ai/glm-5.3-flash"]
)
def test_best_gate_accepts_bare_and_qualified_ids(tmp_path, monkeypatch, best):
    monkeypatch.setattr(publish.subprocess, "run", _fake_run([], best=best))
    monkeypatch.setattr(publish, "fetch_prev", lambda url: None)
    publish.build_site(tmp_path / "site")  # must not raise


@pytest.mark.parametrize(
    "best",
    [
        "openrouter/acme/other",
        # exact membership, never prefix stripping
        "openrouter/openrouter/z-ai/glm-5.3-flash",
        "z-ai/glm-5.3",
    ],
)
def test_best_gate_rejects_mismatch_naming_both(tmp_path, monkeypatch, best):
    monkeypatch.setattr(publish.subprocess, "run", _fake_run([], best=best))
    monkeypatch.setattr(publish, "fetch_prev", lambda url: None)
    with pytest.raises(RuntimeError, match="best.txt") as exc:
        publish.build_site(tmp_path / "site")
    assert best in str(exc.value)
    assert "z-ai/glm-5.3-flash" in str(exc.value)


@pytest.mark.parametrize(
    "catalog_text",
    [
        pytest.param(_catalog_ranking([]), id="empty"),
        pytest.param(
            json.dumps({"generated_at": STAMP, "rankings": {}}), id="no-balanced"
        ),
        pytest.param(
            json.dumps({"generated_at": STAMP, "rankings": ["x"]}), id="not-dict"
        ),
        pytest.param(_catalog_ranking([7]), id="non-string-top"),
    ],
)
def test_best_gate_rejects_unusable_rankings(tmp_path, monkeypatch, catalog_text):
    overrides = {"catalog.json": catalog_text}
    monkeypatch.setattr(publish.subprocess, "run", _fake_run([], overrides=overrides))
    monkeypatch.setattr(publish, "fetch_prev", lambda url: None)
    with pytest.raises(RuntimeError, match="rankings.balanced"):
        publish.build_site(tmp_path / "site")


_ABSENT = object()


def _catalog_run(mode, calls, fallback=_ABSENT):
    """Stub the pipeline; the --catalog run prints the given AA sources."""
    aa = {"mode": mode}
    if fallback is not _ABSENT:
        aa["fallback"] = fallback
    return _fake_run(calls, aa=aa)


PUBLISH_ARTIFACTS = (
    "data.json",
    "catalog.json",
    "history.json",
    "highlights.json",
    "best.txt",
    "index.html",
)


@pytest.mark.parametrize(
    "mode, fallback",
    [
        ("api", "api"),
        ("scrape", "scrape"),
        # "openrouter" is what healthy runs publish whenever OpenRouter
        # benchmarks cover a matched model -- the gate must not
        # false-positive on it while the AA fallback works.
        ("openrouter", "api"),
        ("openrouter", "scrape"),
    ],
)
def test_build_site_aa_gate_passes_live_fallback(tmp_path, monkeypatch, mode, fallback):
    calls = []
    monkeypatch.setattr(publish.subprocess, "run", _catalog_run(mode, calls, fallback))
    monkeypatch.setattr(publish, "fetch_prev", lambda url: None)
    monkeypatch.setenv("AA_API_KEY", "dummy-key")
    publish.build_site(tmp_path / "site")


def test_build_site_aa_gate_fails_when_aa_absent_despite_key(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(publish.subprocess, "run", _catalog_run("none", calls, "none"))
    monkeypatch.setattr(publish, "fetch_prev", lambda url: None)
    monkeypatch.setenv("AA_API_KEY", "dummy-key")
    with pytest.raises(RuntimeError, match="AA_API_KEY is set"):
        publish.build_site(tmp_path / "site")


def test_build_site_aa_gate_fails_when_openrouter_masks_failed_fallback(
    tmp_path, monkeypatch
):
    # Regression: mode "openrouter" says only that some candidate's AA data
    # came from OpenRouter. A rejected key plus a failed scrape used to pass.
    calls = []
    monkeypatch.setattr(
        publish.subprocess, "run", _catalog_run("openrouter", calls, "none")
    )
    monkeypatch.setattr(publish, "fetch_prev", lambda url: None)
    monkeypatch.setenv("AA_API_KEY", "dummy-key")
    with pytest.raises(RuntimeError, match="AA_API_KEY is set") as exc:
        publish.build_site(tmp_path / "site")
    assert "'none'" in str(exc.value)
    assert "sources.aa.fallback" in str(exc.value)


@pytest.mark.parametrize("fallback", [_ABSENT, None, "psychic", "openrouter"])
def test_build_site_aa_gate_fails_closed_on_unknown_fallback(
    tmp_path, monkeypatch, fallback
):
    calls = []
    monkeypatch.setattr(
        publish.subprocess, "run", _catalog_run("openrouter", calls, fallback)
    )
    monkeypatch.setattr(publish, "fetch_prev", lambda url: None)
    monkeypatch.setenv("AA_API_KEY", "dummy-key")
    with pytest.raises(RuntimeError, match="AA_API_KEY is set"):
        publish.build_site(tmp_path / "site")


@pytest.mark.parametrize("key", [None, ""])
def test_build_site_aa_gate_inactive_without_key(tmp_path, monkeypatch, key):
    calls = []
    monkeypatch.setattr(publish.subprocess, "run", _catalog_run("none", calls, "none"))
    monkeypatch.setattr(publish, "fetch_prev", lambda url: None)
    if key is not None:
        monkeypatch.setenv("AA_API_KEY", key)
    publish.build_site(tmp_path / "site")


def test_build_site_aa_gate_inactive_without_key_even_if_fallback_missing(
    tmp_path, monkeypatch
):
    calls = []
    monkeypatch.setattr(publish.subprocess, "run", _catalog_run("openrouter", calls))
    monkeypatch.setattr(publish, "fetch_prev", lambda url: None)
    publish.build_site(tmp_path / "site")


def test_build_site_aa_gate_runs_before_any_artifact_is_written(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        publish.subprocess, "run", _catalog_run("openrouter", calls, "none")
    )
    monkeypatch.setattr(publish, "fetch_prev", lambda url: None)
    monkeypatch.setenv("AA_API_KEY", "dummy-key")
    out = tmp_path / "site"
    with pytest.raises(RuntimeError, match="AA_API_KEY is set"):
        publish.build_site(out)
    assert [name for name in PUBLISH_ARTIFACTS if (out / name).exists()] == []
    # nothing after the --catalog run may have started
    assert len(calls) == 1 and _is_catalog_run(calls[0])


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
