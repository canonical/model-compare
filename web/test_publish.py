"""Tests for the web/publish.py orchestrator (all subprocesses stubbed)."""

import hashlib
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
# What resolve_revision's git rev-parse stub answers in pipeline tests.
STUB_REV = "deadbee"


def _write_artifacts(out_dir, overrides=None, best_id="z-ai/glm-5.3-flash"):
    """Write minimal valid build_site_data artifacts sharing one stamp.

    `overrides` maps an artifact name to replacement text (for mismatch and
    malformed-JSON cases); these stubs only feed publish's stamp and best
    gates. `best_id` is the bare catalog id ranked first for balanced, so it
    must match what the stubbed `--best` run wrote to best.txt.
    """
    docs = {
        "data.json": {
            "schema_version": 3,
            "generated_at": STAMP,
            "source_revision": STUB_REV,
        },
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


def _write_litellm_artifacts(out_dir, stamp=STAMP, merged_state="skipped"):
    """Write the three required LiteLLM artifacts with matching hashes."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rate_bytes = b"{}\n"
    (out_dir / "litellm_rate_card.json").write_bytes(rate_bytes)
    rate_hash = hashlib.sha256(rate_bytes).hexdigest()
    steering = {
        "_meta": {
            "schema_version": 1,
            "generated_at": stamp,
            "source_revision": STUB_REV,
            "catalog_schema_version": 6,
            "rate_card_sha256": rate_hash,
            "entries": 0,
            "warnings": [],
        }
    }
    (out_dir / "litellm_steering.json").write_text(json.dumps(steering))
    merged = {
        "status": merged_state,
        "reason": "no_previous_map" if merged_state == "skipped" else None,
        "sha256": None,
        "entries": None,
        "rate_card_sha256": rate_hash,
    }
    status = {
        "_meta": {
            "schema_version": 1,
            "generated_at": stamp,
            "source_revision": STUB_REV,
            "catalog_schema_version": 6,
        },
        "rate_card": {"sha256": rate_hash, "entries": 0},
        "upstream": {
            "url": "https://example/upstream.json",
            "sha256": None,
            "entries": None,
        },
        "merged": merged,
        "collisions": [],
        "upstream_fetch_failures": 0,
        "report": {"needed": False, "reason": None},
    }
    (out_dir / "litellm_merge_status.json").write_text(json.dumps(status))


# What the stubbed `model_compare.py --catalog` run prints: the AA and
# endpoint-drop gates read it from scratch before any other step runs.
# Catalog v5 shape: the keyed run's chain is "aa", its three disjoint match
# buckets sum to the pool's matched entries (12 + 0 + 8 = 20), and the
# depth coverage counts are candidate-scoped (40 rows over 12 models, all
# 20 candidates carrying a default-effort main score).
HEALTHY_AA = {
    "mode": "aa",
    "matched_aa": 12,
    "matched_catalog": 0,
    "matched_openrouter": 8,
    "depth_rows": 40,
    "matched_depth": 12,
    "matched_default": 20,
    "catalog_generated_at": None,
    "intelligence_index_version": 4.3,
}

# The scratch catalog's pool (Task 8): 20 candidates entered the endpoint
# stage and none dropped for endpoint reasons -- the healthy happy path.
HEALTHY_POOL = {
    "candidates": 20,
    "dropped": {"no_baseline_eligible_endpoint": 0, "endpoint_fetch_failed": 0},
}


@pytest.fixture(autouse=True)
def _no_aa_key(monkeypatch):
    # The AA gate depends on AA_API_KEY; keep the suite hermetic. Gate tests
    # set the key explicitly.
    monkeypatch.delenv("AA_API_KEY", raising=False)


@pytest.fixture(autouse=True)
def _no_github_sha(monkeypatch):
    # Same hermeticity for the ambient CI environment: resolve_revision
    # prefers GITHUB_SHA over the stubbed `git rev-parse`, so a leaked
    # runner SHA would replace STUB_REV in the pipeline's --revision
    # argument and reroute resolve_revision away from the git stubs. The
    # resolve_revision tests set GITHUB_SHA explicitly.
    monkeypatch.delenv("GITHUB_SHA", raising=False)


def _is_catalog_run(cmd):
    return Path(cmd[1]).name == "model_compare.py" and "--catalog" in cmd


def _fake_run(
    calls,
    simulate_build_site_data=True,
    overrides=None,
    best=BEST,
    aa=HEALTHY_AA,
    pool=HEALTHY_POOL,
):
    def fake_run(cmd, cwd=None, stdout=None, check=True, **kwargs):
        if Path(cmd[0]).name == "git":
            # resolve_revision's git rev-parse (capture_output, no stdout
            # file): never part of the recorded pipeline, it only supplies
            # the data.json stamp value.
            return subprocess.CompletedProcess(cmd, 0, stdout=STUB_REV + "\n")
        calls.append(cmd)
        if _is_catalog_run(cmd):
            stdout.write(
                json.dumps(
                    {
                        "generated_at": STAMP,
                        "sources": {"aa": aa},
                        "pool": pool,
                        "models": [],
                    }
                )
            )
        if "--best" in cmd:
            stdout.write(best + "\n")
        if Path(cmd[1]).name == "build_litellm_artifacts.py":
            _write_litellm_artifacts(Path(cmd[cmd.index("--output-dir") + 1]))
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
        "model_compare.py",  # balanced --json --top 20
        "model_compare.py",  # price --json --top 20
        "model_compare.py",  # quality --json --top 20
        "generate_highlights.py",
        "build_litellm_artifacts.py",
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
    assert (out / "style.css").is_file()
    assert (out / "app.js").is_file()
    assert (out / "litellm_rate_card.json").is_file()
    assert (out / "litellm_steering.json").is_file()
    assert (out / "litellm_merge_status.json").is_file()
    for name in FAVICON_FILES:
        assert (out / name).is_file()

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
    assert bsd_cmd[12:14] == ["--revision", STUB_REV]
    priority_values = bsd_cmd[15::2]
    assert [p.split("=", 1)[0] for p in priority_values] == [
        "balanced",
        "price",
        "quality",
    ]
    assert all(
        Path(p.split("=", 1)[1]).parent == gh_paths[0].parent for p in priority_values
    )


def test_build_site_removes_stale_merged_map(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(publish.subprocess, "run", _fake_run(calls))
    monkeypatch.setattr(publish, "fetch_prev", lambda url: None)
    out = tmp_path / "site"
    out.mkdir()
    (out / "litellm.json").write_text("{}")
    publish.build_site(out)
    assert not (out / "litellm.json").exists()


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

    def fake_run(cmd, cwd=None, stdout=None, check=True, **kwargs):
        if Path(cmd[0]).name == "git":
            return subprocess.CompletedProcess(cmd, 0, stdout=STUB_REV + "\n")
        if _is_catalog_run(cmd):
            stdout.write(
                json.dumps(
                    {
                        "generated_at": STAMP,
                        "sources": HEALTHY_AA,
                        "pool": HEALTHY_POOL,
                        "models": [],
                    }
                )
            )
        if "--best" in cmd:
            stdout.write("openrouter/x/y\n")
        if Path(cmd[1]).name == "build_litellm_artifacts.py":
            _write_litellm_artifacts(Path(cmd[cmd.index("--output-dir") + 1]))
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

    def fake_run(cmd, cwd=None, stdout=None, check=True, **kwargs):
        if Path(cmd[0]).name == "git":
            return subprocess.CompletedProcess(cmd, 0, stdout=STUB_REV + "\n")
        calls.append(cmd)
        if _is_catalog_run(cmd):
            stdout.write(
                json.dumps(
                    {
                        "generated_at": STAMP,
                        "sources": HEALTHY_AA,
                        "pool": HEALTHY_POOL,
                    }
                )
            )
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
        (
            "data.json",
            {
                "schema_version": 3,
                "generated_at": "2026-09-27T06:00:00Z",
                "source_revision": STUB_REV,
            },
        ),
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
        (
            "data.json",
            {"generated_at": STAMP, "source_revision": STUB_REV},
        ),  # missing only schema_version
        # missing only source_revision
        ("data.json", {"schema_version": 3, "generated_at": STAMP}),
        (
            "data.json",
            {"schema_version": 2, "generated_at": STAMP, "source_revision": STUB_REV},
        ),
        # strict identity: 3.0 compares equal to 3 but is not v3; True is a
        # bool, not an int
        (
            "data.json",
            {
                "schema_version": True,
                "generated_at": STAMP,
                "source_revision": STUB_REV,
            },
        ),
        (
            "data.json",
            {"schema_version": 3.0, "generated_at": STAMP, "source_revision": STUB_REV},
        ),
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


def test_build_site_requires_data_json_schema_version(tmp_path, monkeypatch):
    # Regression pin for the required-tuple half of the gate: a data.json
    # missing only schema_version must fail at load ("lacks"), not slip
    # through to a later check -- reverting the tuple to ("generated_at",)
    # must turn this red. The version VALUE (now 3) is pinned separately in
    # test_data_json_version_pin_is_3.
    overrides = {
        "data.json": json.dumps({"generated_at": STAMP, "source_revision": STUB_REV})
    }
    monkeypatch.setattr(publish.subprocess, "run", _fake_run([], overrides=overrides))
    monkeypatch.setattr(publish, "fetch_prev", lambda url: None)
    with pytest.raises(RuntimeError, match="data.json lacks schema_version"):
        publish.build_site(tmp_path / "site")


def test_build_site_requires_data_json_source_revision(tmp_path, monkeypatch):
    # Regression pin for the required-tuple half of the gate: a data.json
    # missing only source_revision must fail at load ("lacks"), not slip
    # through to a later check -- reverting the tuple to
    # ("schema_version", "generated_at") must turn this red. _load_artifact
    # names every absent key at once, so the exact one-key message proves
    # the other two were present.
    overrides = {"data.json": json.dumps({"schema_version": 3, "generated_at": STAMP})}
    monkeypatch.setattr(publish.subprocess, "run", _fake_run([], overrides=overrides))
    monkeypatch.setattr(publish, "fetch_prev", lambda url: None)
    with pytest.raises(RuntimeError, match="data.json lacks source_revision"):
        publish.build_site(tmp_path / "site")


def test_data_json_version_pin_is_3(tmp_path, monkeypatch):
    # publish imports nothing from build_site_data, so check_stamps pins the
    # expected data.json version itself: a stale v1 document must fail, the
    # pinned v3 must pass.
    overrides = {
        "data.json": json.dumps(
            {"schema_version": 1, "generated_at": STAMP, "source_revision": STUB_REV}
        )
    }
    monkeypatch.setattr(publish.subprocess, "run", _fake_run([], overrides=overrides))
    monkeypatch.setattr(publish, "fetch_prev", lambda url: None)
    with pytest.raises(RuntimeError, match="data.json schema_version must be 3, got 1"):
        publish.build_site(tmp_path / "site")
    # v3 passes: rerun with the happy-path fixtures (no v1 override).
    monkeypatch.setattr(publish.subprocess, "run", _fake_run([], overrides=None))
    publish.build_site(tmp_path / "site")


def _pool_catalog(candidates=0, no_baseline=0, fetch_failed=0, extra_dropped=None):
    """A scratch catalog carrying the pool the endpoint-drop gate reads."""
    dropped = {
        "no_baseline_eligible_endpoint": no_baseline,
        "endpoint_fetch_failed": fetch_failed,
    }
    dropped.update(extra_dropped or {})
    return {
        "generated_at": STAMP,
        "pool": {"candidates": candidates, "dropped": dropped},
    }


def test_endpoint_drop_rate_gate_passes_under_threshold(tmp_path):
    # 2% of entered candidates dropped is provider churn, not an exodus;
    # exactly the 5% threshold passes too (the gate is strict >).
    path = tmp_path / "catalog.json"
    path.write_text(
        json.dumps(_pool_catalog(candidates=98, no_baseline=1, fetch_failed=1))
    )
    publish.check_endpoint_drops(path)  # must not raise
    path.write_text(
        json.dumps(
            _pool_catalog(
                candidates=95,
                no_baseline=3,
                fetch_failed=2,
                extra_dropped={"context": 100},
            )
        )
    )
    publish.check_endpoint_drops(path)  # must not raise


def test_endpoint_drop_rate_gate_fails_over_threshold(tmp_path):
    path = tmp_path / "catalog.json"
    path.write_text(
        json.dumps(_pool_catalog(candidates=94, no_baseline=4, fetch_failed=2))
    )
    with pytest.raises(RuntimeError) as exc:
        publish.check_endpoint_drops(path)
    message = str(exc.value)
    assert "dropped 6 of 100" in message
    assert "6%" in message and "5%" in message
    assert "no_baseline_eligible_endpoint 4" in message
    assert "endpoint_fetch_failed 2" in message


def test_endpoint_drop_rate_gate_zero_denominator_safe(tmp_path):
    # Unreachable in a real publish (check_best fails on empty rankings
    # first) but the gate must still not divide by zero if reached.
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(_pool_catalog()))
    publish.check_endpoint_drops(path)  # must not raise


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


def _catalog_run(aa, calls):
    """Stub the pipeline; the --catalog run prints the given AA sources."""
    return _fake_run(calls, aa=aa)


PUBLISH_ARTIFACTS = (
    "data.json",
    "catalog.json",
    "history.json",
    "highlights.json",
    "best.txt",
    "index.html",
    "faq.html",
    "style.css",
    "app.js",
    "favicon.svg",
    "favicon-16x16.png",
    "favicon-32x32.png",
    "apple-touch-icon.png",
)

# The favicon set copies with the site assets; pinned here and in
# publish.build_site so a renamed icon fails the publish, not the tab.
FAVICON_FILES = (
    "favicon.svg",
    "favicon-16x16.png",
    "favicon-32x32.png",
    "apple-touch-icon.png",
)


def _litellm_catalog():
    return {"generated_at": STAMP, "models": []}


def test_check_litellm_accepts_skipped(tmp_path):
    _write_litellm_artifacts(tmp_path)
    publish.check_litellm(tmp_path, _litellm_catalog())


def test_check_litellm_rejects_hash_mismatch(tmp_path):
    _write_litellm_artifacts(tmp_path)
    steering = json.loads((tmp_path / "litellm_steering.json").read_text())
    steering["_meta"]["rate_card_sha256"] = "0" * 64
    (tmp_path / "litellm_steering.json").write_text(json.dumps(steering))
    with pytest.raises(RuntimeError, match="rate_card_sha256"):
        publish.check_litellm(tmp_path, _litellm_catalog())


def test_check_litellm_rejects_missing_map_when_ok(tmp_path):
    _write_litellm_artifacts(tmp_path)
    status = json.loads((tmp_path / "litellm_merge_status.json").read_text())
    status["merged"]["status"] = "ok"
    (tmp_path / "litellm_merge_status.json").write_text(json.dumps(status))
    with pytest.raises(RuntimeError):
        publish.check_litellm(tmp_path, _litellm_catalog())


def test_check_litellm_rejects_present_map_when_skipped(tmp_path):
    _write_litellm_artifacts(tmp_path)
    (tmp_path / "litellm.json").write_text("{}")
    with pytest.raises(RuntimeError):
        publish.check_litellm(tmp_path, _litellm_catalog())


def test_check_litellm_accepts_frozen_map_with_old_keys(tmp_path):
    _write_litellm_artifacts(tmp_path)
    merged = b'{"old/key": {}}\n'
    (tmp_path / "litellm.json").write_bytes(merged)
    status = json.loads((tmp_path / "litellm_merge_status.json").read_text())
    status["merged"].update(
        {
            "status": "frozen",
            "reason": "collision",
            "sha256": hashlib.sha256(merged).hexdigest(),
            "entries": 1,
            "rate_card_sha256": "0" * 64,
        }
    )
    status["collisions"] = ["openrouter/z-ai/glm-5.3"]
    (tmp_path / "litellm_merge_status.json").write_text(json.dumps(status))
    publish.check_litellm(tmp_path, _litellm_catalog())


def test_check_litellm_rejects_missing_rate_card(tmp_path):
    _write_litellm_artifacts(tmp_path)
    (tmp_path / "litellm_rate_card.json").unlink()
    with pytest.raises(RuntimeError, match="litellm_rate_card.json"):
        publish.check_litellm(tmp_path, _litellm_catalog())


def test_check_litellm_rejects_key_parity_drift(tmp_path):
    _write_litellm_artifacts(tmp_path)
    steering = json.loads((tmp_path / "litellm_steering.json").read_text())
    steering["openrouter/ghost/model"] = {"provider": {}}
    (tmp_path / "litellm_steering.json").write_text(json.dumps(steering))
    with pytest.raises(RuntimeError, match="keys do not match"):
        publish.check_litellm(tmp_path, _litellm_catalog())


def test_check_litellm_rejects_stale_steering_stamp(tmp_path):
    _write_litellm_artifacts(tmp_path)
    steering = json.loads((tmp_path / "litellm_steering.json").read_text())
    steering["_meta"]["generated_at"] = "2000-01-01T00:00:00+00:00"
    (tmp_path / "litellm_steering.json").write_text(json.dumps(steering))
    with pytest.raises(RuntimeError, match="generated_at"):
        publish.check_litellm(tmp_path, _litellm_catalog())


def test_check_litellm_rejects_stale_status_stamp(tmp_path):
    _write_litellm_artifacts(tmp_path)
    status = json.loads((tmp_path / "litellm_merge_status.json").read_text())
    status["_meta"]["generated_at"] = "2000-01-01T00:00:00+00:00"
    (tmp_path / "litellm_merge_status.json").write_text(json.dumps(status))
    with pytest.raises(RuntimeError, match="generated_at"):
        publish.check_litellm(tmp_path, _litellm_catalog())


def test_check_litellm_rejects_frozen_hash_mismatch(tmp_path):
    _write_litellm_artifacts(tmp_path)
    (tmp_path / "litellm.json").write_bytes(b'{"old/key": {}}\n')
    status = json.loads((tmp_path / "litellm_merge_status.json").read_text())
    status["merged"] = {
        "status": "frozen",
        "reason": "collision",
        "sha256": "0" * 64,
        "entries": 1,
        "rate_card_sha256": "0" * 64,
    }
    (tmp_path / "litellm_merge_status.json").write_text(json.dumps(status))
    with pytest.raises(RuntimeError, match="merged hash does not match"):
        publish.check_litellm(tmp_path, _litellm_catalog())


def test_publish_workflow_has_litellm_reporting():
    repo = Path(__file__).resolve().parent.parent
    text = (repo / ".github" / "workflows" / "publish.yml").read_text()
    assert "issues: write" in text
    assert "retention-days: 1" in text
    assert "report_litellm_merge.py" in text
    assert "if: always()" in text


def test_publish_copies_the_faq_page(tmp_path, monkeypatch):
    # The FAQ is a static sibling of index.html; a page that is not in the
    # copy list and the artifact check would 404 in production while
    # passing every local test (issue #20).
    calls = []
    monkeypatch.setattr(publish.subprocess, "run", _fake_run(calls))
    monkeypatch.setattr(publish, "fetch_prev", lambda url: None)
    out = tmp_path / "site"
    publish.build_site(out)
    assert (out / "faq.html").is_file()


def test_faq_page_is_linked_from_the_index_footer_and_answers_routing():
    site = Path(__file__).resolve().parent / "site"
    faq = (site / "faq.html").read_text(encoding="utf-8")
    index = (site / "index.html").read_text(encoding="utf-8")
    # Linked from the footer, not buried in the OpenCode section.
    footer_at = index.index("<footer>")
    assert 'href="faq.html"' in index[footer_at:]
    # Same shell as the index page: shared stylesheet, same CSP posture.
    assert 'href="style.css"' in faq
    assert "Content-Security-Policy" in faq
    assert "default-src 'none'" in faq
    # The four questions the issue asks for, plus depths and variants.
    for needle in (
        "PREC",
        "cheapest",
        "steering",
        "extra_body",
        "provider",
        "--json",
        "catalog.json",
        "#max",
        ":free",
    ):
        assert needle in faq, needle


def test_publish_healthy_aa_v5_fixture(tmp_path, monkeypatch):
    # The fixture every pipeline test publishes through the AA gate must
    # stay producer-faithful: exactly the v5 sources.aa shape build_catalog
    # emits for a keyed run, and the gate accepts it with the key set.
    assert set(HEALTHY_AA) == {
        "mode",
        "matched_aa",
        "matched_catalog",
        "matched_openrouter",
        "depth_rows",
        "matched_depth",
        "matched_default",
        "catalog_generated_at",
        "intelligence_index_version",
    }
    calls = []
    monkeypatch.setattr(publish.subprocess, "run", _catalog_run(HEALTHY_AA, calls))
    monkeypatch.setattr(publish, "fetch_prev", lambda url: None)
    monkeypatch.setenv("AA_API_KEY", "dummy-key")
    publish.build_site(tmp_path / "site")


@pytest.mark.parametrize("matched_aa", [12, 1])
def test_build_site_aa_gate_passes_live_aa(tmp_path, monkeypatch, matched_aa):
    # The only live shape: the AA chain ran (mode "aa") and at least one
    # model's quality came from it (matched_aa > 0).
    calls = []
    monkeypatch.setattr(
        publish.subprocess,
        "run",
        _catalog_run(
            {
                "mode": "aa",
                "matched_aa": matched_aa,
                "depth_rows": 10,
                "matched_default": matched_aa,
            },
            calls,
        ),
    )
    monkeypatch.setattr(publish, "fetch_prev", lambda url: None)
    monkeypatch.setenv("AA_API_KEY", "dummy-key")
    publish.build_site(tmp_path / "site")


def test_build_site_aa_gate_fails_when_aa_absent_despite_key(tmp_path, monkeypatch):
    # mode "none" says no live AA data stands behind the quality -- a keyed
    # publish must not deploy it.
    calls = []
    monkeypatch.setattr(
        publish.subprocess,
        "run",
        _catalog_run({"mode": "none", "matched_aa": 0}, calls),
    )
    monkeypatch.setattr(publish, "fetch_prev", lambda url: None)
    monkeypatch.setenv("AA_API_KEY", "dummy-key")
    with pytest.raises(RuntimeError, match="AA_API_KEY is set"):
        publish.build_site(tmp_path / "site")


def test_build_site_aa_gate_fails_when_aa_yields_no_matches(tmp_path, monkeypatch):
    # Regression (the v3 gate read the fallback label alone): mode "aa"
    # with zero AA-matched models means the quality actually came from
    # OpenRouter's snapshot, not from the live AA API the key paid for.
    calls = []
    monkeypatch.setattr(
        publish.subprocess, "run", _catalog_run({"mode": "aa", "matched_aa": 0}, calls)
    )
    monkeypatch.setattr(publish, "fetch_prev", lambda url: None)
    monkeypatch.setenv("AA_API_KEY", "dummy-key")
    with pytest.raises(RuntimeError, match="AA_API_KEY is set") as exc:
        publish.build_site(tmp_path / "site")
    assert "matched_aa" in str(exc.value)


def test_check_aa_source_requires_depth_rows_and_default_matches(
    tmp_path, monkeypatch, capsys
):
    # The v6 depth gates: a keyed publish must prove the AA API produced
    # per-effort rows and at least one candidate has quality coverage.
    # Coverage below 25% is a warning, never a failure.
    monkeypatch.setenv("AA_API_KEY", "dummy-key")
    path = tmp_path / "catalog.json"

    def write(aa, candidates=20):
        path.write_text(
            json.dumps({"sources": {"aa": aa}, "pool": {"candidates": candidates}})
        )

    healthy = {**HEALTHY_AA, "depth_rows": 10, "matched_default": 10}
    write({**healthy, "depth_rows": 0})
    with pytest.raises(RuntimeError, match="depth_rows"):
        publish.check_aa_source(path)
    write({**healthy, "matched_default": 0})
    with pytest.raises(RuntimeError, match="matched_default"):
        publish.check_aa_source(path)
    write({**healthy, "matched_default": 4})  # 4/20 = 20% < 25%
    publish.check_aa_source(path)
    err = capsys.readouterr().err
    assert "quality-coverage is 4 of 20 candidates" in err
    assert "quality records cover few candidates" in err
    assert "25%" in err


@pytest.mark.parametrize(
    "aa",
    [
        {},  # the block is missing entirely
        {"mode": None, "matched_aa": 3},  # mode is not a chain name
        # the page scrape was removed; its label can no longer be produced
        {"mode": "scrape", "matched_aa": 3},
        {"mode": "psychic", "matched_aa": 1},  # unknown chain
        {"mode": "aa", "matched_aa": "3"},  # a count that is not an integer
        {"mode": "aa", "matched_aa": True},  # bools are not counts
    ],
)
def test_build_site_aa_gate_fails_closed_on_unusable_block(tmp_path, monkeypatch, aa):
    calls = []
    monkeypatch.setattr(publish.subprocess, "run", _catalog_run(aa, calls))
    monkeypatch.setattr(publish, "fetch_prev", lambda url: None)
    monkeypatch.setenv("AA_API_KEY", "dummy-key")
    with pytest.raises(RuntimeError, match="AA_API_KEY is set"):
        publish.build_site(tmp_path / "site")


@pytest.mark.parametrize("key", [None, ""])
def test_build_site_aa_gate_inactive_without_key(tmp_path, monkeypatch, key):
    calls = []
    monkeypatch.setattr(
        publish.subprocess,
        "run",
        _catalog_run({"mode": "none", "matched_aa": 0}, calls),
    )
    monkeypatch.setattr(publish, "fetch_prev", lambda url: None)
    if key is not None:
        monkeypatch.setenv("AA_API_KEY", key)
    publish.build_site(tmp_path / "site")


def test_build_site_aa_gate_inactive_without_key_even_if_block_unusable(
    tmp_path, monkeypatch
):
    calls = []
    monkeypatch.setattr(publish.subprocess, "run", _catalog_run({}, calls))
    monkeypatch.setattr(publish, "fetch_prev", lambda url: None)
    publish.build_site(tmp_path / "site")


def test_build_site_aa_gate_runs_before_any_artifact_is_written(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        publish.subprocess,
        "run",
        _catalog_run({"mode": "none", "matched_aa": 0}, calls),
    )
    monkeypatch.setattr(publish, "fetch_prev", lambda url: None)
    monkeypatch.setenv("AA_API_KEY", "dummy-key")
    out = tmp_path / "site"
    with pytest.raises(RuntimeError, match="AA_API_KEY is set"):
        publish.build_site(out)
    assert [name for name in PUBLISH_ARTIFACTS if (out / name).exists()] == []
    # nothing after the --catalog run may have started
    assert len(calls) == 1 and _is_catalog_run(calls[0])


def test_build_site_endpoint_gate_runs_before_any_artifact_is_written(
    tmp_path, monkeypatch
):
    # Both gates read the scratch catalog before anything else runs: a
    # drop-rate exodus must write no publish artifacts to output_dir.
    exodus = {
        "candidates": 94,
        "dropped": {"no_baseline_eligible_endpoint": 4, "endpoint_fetch_failed": 2},
    }
    calls = []
    monkeypatch.setattr(publish.subprocess, "run", _fake_run(calls, pool=exodus))
    monkeypatch.setattr(publish, "fetch_prev", lambda url: None)
    out = tmp_path / "site"
    with pytest.raises(RuntimeError, match="endpoint stage dropped"):
        publish.build_site(out)
    assert [name for name in PUBLISH_ARTIFACTS if (out / name).exists()] == []
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


# ---------------------------------------------------------------------------
# resolve_revision: the site's source stamp (GITHUB_SHA, else git HEAD)
# ---------------------------------------------------------------------------


def test_resolve_revision_prefers_github_sha(monkeypatch):
    # CI's checkout SHA wins over the local HEAD, and only the first 7
    # characters are kept (the short-hash form stamped into data.json).
    monkeypatch.setenv("GITHUB_SHA", "fedcba9876543210fedcba9876543210fedcba98")
    assert publish.resolve_revision() == "fedcba9"


def test_resolve_revision_rejects_malformed_github_sha(monkeypatch):
    # Garbage must fail before the git fallback is consulted: in CI the
    # fallback would silently attribute the deploy to whatever the
    # runner's checkout HEAD happens to be. The raiser proves the fallback
    # never runs -- consulting it would surface "cannot resolve site
    # revision" from the OSError instead of the hash verdict.
    monkeypatch.setenv("GITHUB_SHA", "not-a-hash")
    monkeypatch.setattr(publish.subprocess, "run", _raise(OSError("git: not found")))
    with pytest.raises(RuntimeError, match="not a hex commit hash"):
        publish.resolve_revision()


def test_resolve_revision_falls_back_to_git_head(monkeypatch):
    monkeypatch.delenv("GITHUB_SHA", raising=False)

    def fake_run(cmd, cwd, check, capture_output, text):
        assert cmd == ["git", "rev-parse", "--short=7", "HEAD"]
        assert cwd == publish.REPO_ROOT
        return subprocess.CompletedProcess(cmd, 0, stdout=STUB_REV + "\n")

    monkeypatch.setattr(publish.subprocess, "run", fake_run)
    assert publish.resolve_revision() == STUB_REV


def test_resolve_revision_rejects_malformed_git_output(monkeypatch):
    # The fallback's stdout is validated like GITHUB_SHA: anything that is
    # not a hex commit hash must fail rather than stamp a dead footer link.
    monkeypatch.delenv("GITHUB_SHA", raising=False)

    def fake_run(cmd, cwd, check, capture_output, text):
        return subprocess.CompletedProcess(cmd, 0, stdout="not-a-hash\n")

    monkeypatch.setattr(publish.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="not a hex commit hash"):
        publish.resolve_revision()


def test_resolve_revision_fails_loudly_when_git_unavailable(monkeypatch):
    # An unattributable deploy is treated like any other unexpected
    # result: the run must fail before anything is published.
    monkeypatch.delenv("GITHUB_SHA", raising=False)
    monkeypatch.setattr(
        publish.subprocess,
        "run",
        _raise(OSError("git: not found")),
    )
    with pytest.raises(RuntimeError, match="cannot resolve site revision"):
        publish.resolve_revision()


@pytest.mark.parametrize(
    "exc",
    [
        pytest.param(OSError("git: not found"), id="git-missing"),
        pytest.param(
            subprocess.CalledProcessError(
                128, ["git", "rev-parse", "--short=7", "HEAD"]
            ),
            id="git-exit-128",
        ),
    ],
)
def test_resolve_revision_fails_loudly_on_any_git_failure(monkeypatch, exc):
    # Every git failure mode (binary absent, non-zero exit) funnels into
    # the same RuntimeError so the deploy fails before anything publishes.
    monkeypatch.delenv("GITHUB_SHA", raising=False)
    monkeypatch.setattr(publish.subprocess, "run", _raise(exc))
    with pytest.raises(RuntimeError, match="cannot resolve site revision"):
        publish.resolve_revision()


def _raise(exc):
    def _run(*args, **kwargs):
        raise exc

    return _run
