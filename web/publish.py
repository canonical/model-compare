#!/usr/bin/env python3
"""Publish the model-compare site.

Runs the standalone model_compare.py, then the web-side generators
(generate_highlights.py, build_site_data.py), and assembles the deploy
directory: data.json, catalog.json, history.json, highlights.json,
litellm_rate_card.json, litellm_steering.json, litellm_merge_status.json,
best.txt, and the static site (index.html, plugin.html, how-to-use.html,
faq.html, style.css, app.js, plugin.js, the favicon set and the two dialog
screenshots). Fails loudly on any unexpected result -- including
data.json/history.json stamps that disagree with catalog.json's
generated_at, or a best.txt that is not catalog.json's balanced rank 1 --
so a broken run never deploys a broken site.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = Path(__file__).resolve().parent
USER_AGENT = "model-compare (https://github.com/canonical/model-compare)"
PRIORITIES = ("balanced", "price", "quality")
# The mode every ranking run shares. The org's guardrail enforces ZDR and
# allows Safety Retention for Google's models only, so the published tables
# use the same rule; see model_compare.py --safety-retention.
MODEL_COMPARE_FILTER_ARGS = ["--safety-retention", "google"]
# The project's own domain, not canonical.github.io: that host 301-redirects
# to plaintext http://model-compare.dev, and the previous-artifact fetch must
# never follow an HTTPS downgrade. See _HTTPSOnlyRedirectHandler.
PREV_HISTORY_URL = "https://model-compare.dev/history.json"
PREV_HIGHLIGHTS_URL = "https://model-compare.dev/highlights.json"
PREV_LITELLM_MERGED_URL = "https://model-compare.dev/litellm.json"
PREV_LITELLM_STATUS_URL = "https://model-compare.dev/litellm_merge_status.json"
LITELLM_UPSTREAM_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/main/"
    "model_prices_and_context_window.json"
)
LITELLM_REQUIRED = (
    "litellm_rate_card.json",
    "litellm_steering.json",
    "litellm_merge_status.json",
)
LITELLM_MERGED = "litellm.json"
# The static site: every file copied verbatim from web/site/ into the deploy
# directory. Named here so the tests can assert each one exists before a
# publish, rather than duplicating the list (a page missing from it 404s in
# production while every local test stays green -- issue #20).
SITE_FILES = (
    "index.html",
    "plugin.html",
    "how-to-use.html",
    "faq.html",
    "style.css",
    "app.js",
    "plugin.js",
    "favicon.svg",
    "favicon-16x16.png",
    "favicon-32x32.png",
    "apple-touch-icon.png",
    "picks-dialog.png",
    "report-dialog.png",
)
# Endpoint-stage failures are normal churn (provider outages, gating), but a
# mass exodus means the catalog is no longer comparable to previous runs.
# publish imports nothing from model_compare/build_site_data (subprocess-only
# by design), so this threshold is pinned here rather than imported.
MAX_ENDPOINT_DROP_RATE = 0.05
# Semantic endpoint drops (no admitted endpoint, or no quoteable one) are
# normal with the endpoint-level capability and policy gates: a model can pass
# every model-level filter and still have no endpoint that advertises 1M prompt
# context, tools, or an allowed policy. The live rate is roughly a third of the
# entered pool, so this is a wipe gate, not a churn gate.
MAX_SEMANTIC_ENDPOINT_DROP_RATE = 0.75
# The only pool.dropped reasons the endpoint stage itself produces; every
# other CATALOG_DROP_REASONS entry (malformed id, context, pricing, ...)
# drops a candidate before the endpoint stage is ever tried. publish
# imports nothing from model_compare (subprocess-only by design), so the
# reasons are named here rather than imported.
ENDPOINT_DROP_REASONS = (
    "no_baseline_eligible_endpoint",
    "endpoint_fetch_failed",
)
# The All-providers pool legitimately drops many models at the endpoint stage
# (unpriceable / unknown-format third-party endpoints). This looser gate only
# catches a wipe, never the ZDR pool's 5% churn threshold.
MAX_NONZDR_ENDPOINT_DROP_RATE = 0.5
# A git commit hash in short (7) to full (40) lowercase hex form. Both
# revision sources (GITHUB_SHA, `git rev-parse` output) are validated
# against it: whatever lands in data.json is what the site footer links
# to, so garbage must fail the publish rather than ship a dead link.
REVISION_RE = re.compile(r"[0-9a-f]{7,40}")


def resolve_revision() -> str:
    """Return the revision that built the site (short commit hash).

    CI sets GITHUB_SHA; locally, fall back to the checkout's git HEAD.
    The revision is stamped into data.json so the site is always
    traceable to the code that produced it, without versioning the web
    itself. Fails loudly when unresolvable or not a hex commit hash: an
    unattributable deploy is treated like any other unexpected result.
    """
    sha = os.environ.get("GITHUB_SHA")
    if sha:
        sha = sha.strip()
        if not REVISION_RE.fullmatch(sha):
            raise RuntimeError(
                "publish: cannot resolve site revision: "
                f"GITHUB_SHA {sha!r} is not a hex commit hash"
            )
        return sha[:7]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short=7", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"publish: cannot resolve site revision: {exc}") from exc
    rev = result.stdout.strip()
    if not REVISION_RE.fullmatch(rev):
        raise RuntimeError(
            "publish: cannot resolve site revision: "
            f"git rev-parse output {rev!r} is not a hex commit hash"
        )
    return rev


def run_script(script: Path, args: list[str], out: Path | None = None) -> None:
    """Run a repo script, optionally redirecting stdout into `out`."""
    cmd = [sys.executable, str(script), *args]
    if out is None:
        subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    else:
        with open(out, "w") as fh:
            subprocess.run(cmd, cwd=REPO_ROOT, stdout=fh, check=True)


def run_pool_json(script: Path, args: list[str], out: Path) -> list:
    """Run a --json pool query; a legitimately empty pool writes [].

    Exit 2 is the producer's "no models" signal. Only "no models satisfy the
    filters" is a legitimate empty pool; "no models after the endpoint stage"
    is a regression and must fail the publish, so stderr is inspected.
    """
    cmd = [sys.executable, str(script), *args]
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if result.returncode == 0:
        out.write_text(result.stdout)
        return json.loads(result.stdout)
    if result.returncode == 2 and "no models satisfy the filters" in result.stderr:
        out.write_text("[]\n")
        return []
    raise RuntimeError(
        f"publish: pool query failed ({' '.join(cmd)}): {result.stderr.strip()}"
    )


def run_catalog(script: Path, args: list[str], out: Path) -> None:
    """Run a --catalog build into `out`; any failure (including exit 2) fails."""
    cmd = [sys.executable, str(script), *args]
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"publish: catalog build failed ({' '.join(cmd)}): {result.stderr.strip()}"
        )
    out.write_text(result.stdout)


def run_validate_catalog(script: Path, path: Path) -> None:
    """Validate a catalog with build_site_data's validation-only entry point."""
    subprocess.run(
        [sys.executable, str(script), "--validate-catalog", str(path)],
        cwd=REPO_ROOT,
        check=True,
    )


class _HTTPSOnlyRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Refuse a redirect that would leave HTTPS.

    canonical.github.io 301-redirects to http://model-compare.dev today;
    urllib follows that downgrade by default. The previous-artifact fetch
    feeds published state (history, highlights, LiteLLM map), so a plaintext
    hop would be tamperable. The project-owned HTTPS URLs avoid the redirect
    in normal operation; this handler makes a future redirect fail loudly.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        scheme = urllib.parse.urlsplit(newurl).scheme.lower()
        if scheme != "https":
            raise urllib.error.URLError(
                f"refusing non-HTTPS redirect from {req.full_url} to {newurl}"
            )
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _open_prev(req, timeout: int):
    """Open a previous-artifact URL through the HTTPS-only redirect policy."""
    opener = urllib.request.build_opener(_HTTPSOnlyRedirectHandler)
    return opener.open(req, timeout=timeout)


def fetch_prev(url: str, timeout: int = 60, attempts: int = 3) -> bytes | None:
    """Fetch a previously published file; None only when it is absent (404).

    A transient failure must abort the publish, not reset accumulated state:
    returning None here used to collapse the 10-day history to one snapshot
    or delete a live LiteLLM map. Retries cover short blips, but a final
    failure raises RuntimeError. A 404 is the one honest "no previous
    artifact" answer (first run, or a deliberately removed file).
    """
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    last_exc: Exception = RuntimeError("previous artifact fetch failed")
    for attempt in range(attempts):
        try:
            with _open_prev(req, timeout) as resp:
                final_scheme = urllib.parse.urlsplit(resp.geturl()).scheme.lower()
                if final_scheme != "https":
                    raise RuntimeError(
                        f"previous artifact {url} resolved to non-HTTPS {resp.geturl()}"
                    )
                return resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            last_exc = exc
        except (urllib.error.URLError, OSError, http.client.HTTPException) as exc:
            last_exc = exc
        if attempt < attempts - 1:
            time.sleep(2**attempt)
    raise RuntimeError(f"publish: could not fetch previous artifact {url}: {last_exc}")


def _load_artifact(path: Path, required: tuple[str, ...]) -> dict:
    """json.load an artifact, raising RuntimeError that names it on any defect."""
    try:
        with open(path) as fh:
            doc = json.load(fh)
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"publish: unreadable {path.name}: {exc}") from exc
    if not isinstance(doc, dict):
        raise RuntimeError(f"publish: {path.name} is not a JSON object")
    absent = [key for key in required if key not in doc]
    if absent:
        raise RuntimeError(f"publish: {path.name} lacks {', '.join(absent)}")
    return doc


def check_stamps(output_dir: Path) -> dict:
    """Fail unless data.json and history.json carry the catalog's generated_at.

    The catalog is the single clock: build_site_data stamps data.json from
    it and merge_history sets updated_at to the newest snapshot's stamp,
    which is today's catalog. highlights.json is deliberately excluded --
    its generated_at is its own writing time (the 24h LLM-reuse window).
    data.json must also carry source_revision: the site footer links to
    the commit that produced it, so an unattributable document is as
    broken as one with a wrong stamp. Returns the loaded catalog (which
    must carry rankings) for check_best.
    """
    catalog = _load_artifact(output_dir / "catalog.json", ("generated_at", "rankings"))
    data = _load_artifact(
        output_dir / "data.json", ("schema_version", "generated_at", "source_revision")
    )
    # .get: fail closed with the clean RuntimeError below even if the
    # required tuple above ever stops listing schema_version.
    version = data.get("schema_version")
    # Strict identity: True and 3.0 compare equal to 3 but are not v3.
    if type(version) is not int or version != 4:
        # publish imports nothing from build_site_data (subprocess-only by
        # design); the expected data.json version is pinned here, like the
        # opencode id prefix in check_best.
        raise RuntimeError(
            f"publish: data.json schema_version must be 4, got {version!r}"
        )
    history = _load_artifact(output_dir / "history.json", ("updated_at", "snapshots"))
    stamp = catalog["generated_at"]
    if data["generated_at"] != stamp or history["updated_at"] != stamp:
        raise RuntimeError(
            "publish: generated_at mismatch: "
            f"catalog={stamp!r} data={data['generated_at']!r} "
            f"history={history['updated_at']!r}"
        )
    return catalog


def check_best(output_dir: Path, catalog: dict) -> None:
    """Fail unless best.txt names the catalog's balanced rank-1 model.

    `--best` is the argmax of the same ranking the catalog carries, so any
    disagreement means the invocations saw different inputs.
    """
    rankings = catalog["rankings"]
    balanced = rankings.get("balanced") if isinstance(rankings, dict) else None
    if not isinstance(balanced, list) or not balanced:
        raise RuntimeError(
            "publish: catalog.json rankings.balanced is empty or missing"
        )
    top_id = balanced[0]
    if not isinstance(top_id, str) or not top_id:
        raise RuntimeError(
            f"publish: catalog.json rankings.balanced[0] is not an id: {top_id!r}"
        )
    try:
        best_clean = (output_dir / "best.txt").read_text().strip()
    except (OSError, UnicodeDecodeError) as exc:
        raise RuntimeError(f"publish: unreadable best.txt: {exc}") from exc
    # "openrouter/" duplicates model_compare.OPENCODE_PROVIDER; publish is
    # deliberately subprocess-only and imports nothing from it. Exact
    # membership, never prefix stripping (openrouter/openrouter/x must fail).
    if best_clean not in (top_id, "openrouter/" + top_id):
        raise RuntimeError(
            f"publish: best.txt {best_clean!r} disagrees with catalog "
            f"rankings.balanced[0] {top_id!r}"
        )


AA_LIVE_MODE = "aa"
# Quality-coverage regression threshold (spec 9.4): below this share of
# candidates with a quality record the publish warns but still deploys.
MIN_MATCHED_DEFAULT_COVERAGE = 0.25


def check_aa_source(catalog_path: Path) -> None:
    """With AA_API_KEY set, fail unless the AA API itself produced quality.

    sources.aa.mode records the run's quality chain: "aa" when the AA API
    returned usable entries, "catalog" for published-catalog hydration
    (keyed runs never hydrate), "none" when neither tier produced data. A
    keyed publish must have live AA data standing behind its quality:
    mode "aa" AND at least one AA-matched model (matched_aa > 0), at least
    one per-effort row (depth_rows > 0), and at least one candidate with a
    quality-coverage record (matched_default > 0) -- a rejected key, an
    empty result, a catalog-or-OR-only chain, an unknown mode or a
    malformed count all fail closed. Coverage below
    MIN_MATCHED_DEFAULT_COVERAGE warns without failing: a coverage
    regression is visible but not a deploy blocker. Without a key the
    catalog chain (or OR-only data) is expected, so the gate never raises.
    """
    if not os.environ.get("AA_API_KEY"):
        return
    catalog = _load_artifact(catalog_path, ())
    sources = catalog.get("sources")
    aa_sources = sources.get("aa") if isinstance(sources, dict) else None
    if not isinstance(aa_sources, dict):
        aa_sources = {}
    mode = aa_sources.get("mode")
    matched_aa = aa_sources.get("matched_aa")
    depth_rows = aa_sources.get("depth_rows")
    matched_default = aa_sources.get("matched_default")
    if mode != AA_LIVE_MODE:
        raise RuntimeError(
            "publish: AA_API_KEY is set but catalog sources.aa is "
            f"{{mode: {mode!r}, matched_aa: {matched_aa!r}}}: the AA API "
            "did not yield data. Check the key and artificialanalysis.ai, "
            "or unset AA_API_KEY, before deploying."
        )
    for label, value in (
        ("matched_aa", matched_aa),
        ("depth_rows", depth_rows),
        ("matched_default", matched_default),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise RuntimeError(
                f"publish: AA_API_KEY is set but catalog sources.aa.{label} "
                f"is {value!r}: the AA API did not yield usable data. Check "
                "the key and artificialanalysis.ai, or unset AA_API_KEY, "
                "before deploying."
            )
    pool = catalog.get("pool")
    candidates = pool.get("candidates") if isinstance(pool, dict) else None
    if (
        isinstance(candidates, int)
        and not isinstance(candidates, bool)
        and candidates > 0
        and matched_default / candidates < MIN_MATCHED_DEFAULT_COVERAGE
    ):
        print(
            "publish: warning: AA quality-coverage is "
            f"{matched_default} of {candidates} candidates "
            f"({matched_default / candidates:.0%} < "
            f"{MIN_MATCHED_DEFAULT_COVERAGE:.0%}): quality records cover few candidates",
            file=sys.stderr,
        )
    print(
        f"publish: AA source mode {mode}, matched_aa {matched_aa}, "
        f"depth_rows {depth_rows}, matched_default {matched_default}"
    )


def check_endpoint_drops(catalog_path: Path) -> None:
    """Fail unless the endpoint stage kept most of the pool it entered.

    `pool.candidates` counts the models that survived the endpoint stage, so
    the models that entered it are `candidates` plus the stage's own drop
    reasons: a model dropped before the endpoint stage never reached it, and
    one that dropped during it is one of these two reasons. A few endpoint
    failures are normal churn (provider outages, per-key gating), but a mass
    exodus means the ranking no longer reflects the same candidate pool as
    previous runs -- fail the publish rather than ship it silently.
    Publish imports nothing from model_compare (subprocess-only by design),
    so the reasons are listed here rather than imported.
    """
    catalog = _load_artifact(catalog_path, ("generated_at", "pool"))
    pool = catalog["pool"]
    if not isinstance(pool, dict):
        pool = {}
    dropped = pool.get("dropped")
    if not isinstance(dropped, dict):
        dropped = {}

    def _count(value: object, label: str) -> int:
        # Missing counts default to 0 (an older catalog never recorded
        # them); a present-but-malformed count is a producer bug, and the
        # module contract is to fail loudly on any unexpected result.
        if value is None:
            return 0
        if type(value) is not int or value < 0:
            raise RuntimeError(
                f"publish: {catalog_path.name} {label} is not a count: {value!r}"
            )
        return value

    counts = {
        reason: _count(dropped.get(reason), f"pool.dropped.{reason}")
        for reason in ENDPOINT_DROP_REASONS
    }
    candidates = _count(pool.get("candidates"), "pool.candidates")
    dropped_total = sum(counts.values())
    entered = candidates + dropped_total
    if entered == 0:
        return
    detail = ", ".join(f"{counts[reason]} {reason}" for reason in ENDPOINT_DROP_REASONS)
    fetch_failed = counts["endpoint_fetch_failed"]
    if fetch_failed / entered > MAX_ENDPOINT_DROP_RATE:
        raise RuntimeError(
            f"publish: {catalog_path.name} endpoint fetches dropped "
            f"{fetch_failed} of {entered} "
            f"({fetch_failed / entered:.0%} > "
            f"{MAX_ENDPOINT_DROP_RATE:.0%} limit): {detail}"
        )
    semantic = counts["no_baseline_eligible_endpoint"]
    if semantic / entered > MAX_SEMANTIC_ENDPOINT_DROP_RATE:
        raise RuntimeError(
            f"publish: {catalog_path.name} endpoint stage dropped "
            f"{semantic} of {entered} "
            f"({semantic / entered:.0%} > "
            f"{MAX_SEMANTIC_ENDPOINT_DROP_RATE:.0%} limit): {detail}"
        )


def check_nonzdr_catalog(catalog_path: Path) -> None:
    """Fail unless the All-providers catalog is non-empty and not a wipe.

    The non-ZDR pool legitimately drops many models at the endpoint stage
    (third-party endpoints with no usable price or format), so the 5% ZDR
    churn gate would be wrong here. A wipe (>50%) still means the artifact
    is unusable.
    """
    catalog = _load_artifact(catalog_path, ("pool", "models", "parameters", "sources"))
    parameters = catalog["parameters"]
    sources = catalog["sources"]
    if not isinstance(parameters, dict):
        raise RuntimeError(
            f"publish: {catalog_path.name} parameters is not an object: {parameters!r}"
        )
    if parameters.get("zdr_required") is not False:
        raise RuntimeError(
            f"publish: {catalog_path.name} is not a non-ZDR catalog "
            f"(zdr_required={parameters.get('zdr_required')!r})"
        )
    if not isinstance(sources, dict) or sources.get("zdr") != "skipped":
        raise RuntimeError(
            f"publish: {catalog_path.name} sources.zdr is not 'skipped': "
            f"{sources.get('zdr') if isinstance(sources, dict) else sources!r}"
        )
    models = catalog["models"]
    if not isinstance(models, list):
        raise RuntimeError(f"publish: {catalog_path.name} models is not a list")
    for i, model in enumerate(models):
        if not isinstance(model, dict):
            raise RuntimeError(
                f"publish: {catalog_path.name} models[{i}] is not an object"
            )
        if model.get("zdr") is not None:
            raise RuntimeError(
                f"publish: {catalog_path.name} models[{i}] has zdr="
                f"{model.get('zdr')!r}; expected null"
            )
        if model.get("safety_retention") is not None:
            raise RuntimeError(
                f"publish: {catalog_path.name} models[{i}] has "
                f"safety_retention={model.get('safety_retention')!r}; expected null"
            )
    pool = catalog["pool"]
    if not isinstance(pool, dict):
        pool = {}
    dropped = pool.get("dropped")
    if not isinstance(dropped, dict):
        dropped = {}

    def _count(value: object, label: str) -> int:
        if value is None:
            return 0
        if type(value) is not int or value < 0:
            raise RuntimeError(
                f"publish: {catalog_path.name} {label} is not a count: {value!r}"
            )
        return value

    candidates = _count(pool.get("candidates"), "pool.candidates")
    if candidates <= 0:
        raise RuntimeError(f"publish: {catalog_path.name} has no ranked models")
    counts = {
        reason: _count(dropped.get(reason), reason) for reason in ENDPOINT_DROP_REASONS
    }
    dropped_total = sum(counts.values())
    entered = candidates + dropped_total
    if entered and dropped_total / entered > MAX_NONZDR_ENDPOINT_DROP_RATE:
        raise RuntimeError(
            f"publish: {catalog_path.name} endpoint stage dropped "
            f"{dropped_total} of {entered} "
            f"({dropped_total / entered:.0%} > "
            f"{MAX_NONZDR_ENDPOINT_DROP_RATE:.0%} limit)"
        )


def _artifact_bytes(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise RuntimeError(f"publish: unreadable {path.name}: {exc}") from exc


def _artifact_object(raw: bytes, name: str) -> dict:
    try:
        doc = json.loads(raw)
    except ValueError as exc:
        raise RuntimeError(f"publish: {name} is not JSON: {exc}") from exc
    if not isinstance(doc, dict):
        raise RuntimeError(f"publish: {name} is not a JSON object")
    return doc


def check_litellm(output_dir: Path, catalog: dict) -> None:
    """Fail unless the LiteLLM artifacts agree with each other and the catalog.

    The rate card is the price contract: its exact bytes are hashed into the
    steering `_meta` and the status file. The merged map is optional and its
    key set is compared with the rate card only when it was rebuilt (`ok`);
    a frozen map is deliberately allowed to carry an older candidate set.
    """
    rate_bytes = _artifact_bytes(output_dir / LITELLM_REQUIRED[0])
    rate_card = _artifact_object(rate_bytes, LITELLM_REQUIRED[0])
    steering = _artifact_object(
        _artifact_bytes(output_dir / LITELLM_REQUIRED[1]), LITELLM_REQUIRED[1]
    )
    status = _artifact_object(
        _artifact_bytes(output_dir / LITELLM_REQUIRED[2]), LITELLM_REQUIRED[2]
    )

    meta = steering.get("_meta")
    if not isinstance(meta, dict):
        raise RuntimeError("publish: litellm_steering.json lacks _meta")
    if meta.get("generated_at") != catalog.get("generated_at"):
        raise RuntimeError(
            "publish: litellm_steering.json generated_at disagrees with catalog"
        )
    rate_hash = hashlib.sha256(rate_bytes).hexdigest()
    if meta.get("rate_card_sha256") != rate_hash:
        raise RuntimeError(
            "publish: litellm_steering.json rate_card_sha256 does not match "
            "litellm_rate_card.json"
        )
    keys = set(steering) - {"_meta"}
    if keys != set(rate_card):
        raise RuntimeError(
            "publish: litellm_steering.json keys do not match the rate card"
        )
    models = catalog.get("models")
    expected = len(models) if isinstance(models, list) else None
    if meta.get("entries") != len(rate_card):
        raise RuntimeError(
            "publish: litellm_steering.json entries disagrees with the rate card"
        )
    if expected is not None and len(rate_card) != expected:
        raise RuntimeError(
            "publish: litellm_rate_card.json entries disagree with catalog models"
        )

    status_meta = status.get("_meta")
    if not isinstance(status_meta, dict):
        raise RuntimeError("publish: litellm_merge_status.json lacks _meta")
    if status_meta.get("generated_at") != catalog.get("generated_at"):
        raise RuntimeError(
            "publish: litellm_merge_status.json generated_at disagrees with catalog"
        )
    rate_status = status.get("rate_card")
    if not isinstance(rate_status, dict):
        raise RuntimeError("publish: litellm_merge_status.json lacks rate_card")
    if rate_status.get("sha256") != rate_hash:
        raise RuntimeError(
            "publish: litellm_merge_status.json rate_card hash does not match"
        )
    if rate_status.get("entries") != len(rate_card):
        raise RuntimeError(
            "publish: litellm_merge_status.json rate_card entries disagree"
        )

    merged = status.get("merged")
    if not isinstance(merged, dict):
        raise RuntimeError("publish: litellm_merge_status.json lacks merged")
    state = merged.get("status")
    merged_path = output_dir / LITELLM_MERGED
    if state == "skipped":
        if merged_path.exists():
            raise RuntimeError("publish: litellm.json exists but status says skipped")
        return
    if state not in ("ok", "frozen"):
        raise RuntimeError(f"publish: unknown merged status {state!r}")
    raw = _artifact_bytes(merged_path)
    digest = hashlib.sha256(raw).hexdigest()
    if merged.get("sha256") != digest:
        raise RuntimeError(
            "publish: litellm_merge_status.json merged hash does not match litellm.json"
        )
    merged_doc = _artifact_object(raw, LITELLM_MERGED)
    if state == "ok":
        missing = set(rate_card) - set(merged_doc)
        if missing:
            raise RuntimeError(
                "publish: litellm.json is missing rate-card keys: "
                + ", ".join(sorted(missing)[:5])
            )
        if merged.get("rate_card_sha256") != rate_hash:
            raise RuntimeError(
                "publish: litellm_merge_status.json merged rate-card hash "
                "does not match the current rate card"
            )


def build_site(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = (
        "data.json",
        "catalog.json",
        "catalog-nonzdr.json",
        "history.json",  # self-feeds the next run via PREV_HISTORY_URL
        "highlights.json",  # ditto via PREV_HIGHLIGHTS_URL
        *LITELLM_REQUIRED,
        "best.txt",
        *SITE_FILES,
    )
    with tempfile.TemporaryDirectory(prefix="mc-build-") as tmp:
        scratch = Path(tmp)
        model_compare = REPO_ROOT / "model_compare.py"

        # --catalog first: it fetches OpenRouter/AA data and primes the shared
        # caches, so later invocations very likely rank identical inputs.
        # Not guaranteed: degraded fetches (empty discounts, no AA entries)
        # are never cached, so a later run may fetch different data. That
        # fails loudly (order_rows, check_best) and the next run self-heals.
        run_script(
            model_compare,
            ["--catalog", *MODEL_COMPARE_FILTER_ARGS],
            out=scratch / "catalog.json",
        )
        # Gates before anything else runs, so a failed AA source gate or an
        # endpoint-drop exodus writes no publish artifacts to output_dir.
        check_aa_source(scratch / "catalog.json")
        check_endpoint_drops(scratch / "catalog.json")
        run_script(
            model_compare,
            ["--best", *MODEL_COMPARE_FILTER_ARGS],
            out=scratch / "best.txt",
        )
        for priority in PRIORITIES:
            run_script(
                model_compare,
                [
                    "--priority",
                    priority,
                    "--json",
                    "--top",
                    "20",
                    *MODEL_COMPARE_FILTER_ARGS,
                ],
                out=scratch / f"{priority}.json",
            )

        for url, name in (
            (PREV_HISTORY_URL, "history-prev.json"),
            (PREV_HIGHLIGHTS_URL, "highlights-prev.json"),
            (PREV_LITELLM_MERGED_URL, "litellm-merged-prev.json"),
            (PREV_LITELLM_STATUS_URL, "litellm-status-prev.json"),
        ):
            prev = fetch_prev(url)
            if prev is not None:
                (scratch / name).write_bytes(prev)

        # The BerriAI map is optional input: a transient GitHub failure must
        # degrade to the "upstream failed" freeze path, not abort the whole
        # site publish. Previous project artifacts, by contrast, are
        # required state and fetch_prev raises on anything but 404.
        try:
            upstream = fetch_prev(LITELLM_UPSTREAM_URL)
        except RuntimeError as exc:
            print(
                f"publish: LiteLLM upstream unavailable this run: {exc}",
                file=sys.stderr,
            )
            upstream = None
        if upstream is not None:
            (scratch / "litellm-upstream.json").write_bytes(upstream)

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

        litellm_args = [
            "--catalog",
            str(scratch / "catalog.json"),
            "--output-dir",
            str(scratch / "litellm"),
            "--revision",
            resolve_revision(),
            "--upstream-status",
            "ok" if upstream is not None else "failed",
            "--upstream-url",
            LITELLM_UPSTREAM_URL,
        ]
        if upstream is not None:
            litellm_args += ["--upstream-map", str(scratch / "litellm-upstream.json")]
        if (scratch / "litellm-merged-prev.json").is_file():
            litellm_args += [
                "--prev-merged",
                str(scratch / "litellm-merged-prev.json"),
            ]
        if (scratch / "litellm-status-prev.json").is_file():
            litellm_args += [
                "--prev-status",
                str(scratch / "litellm-status-prev.json"),
            ]
        run_script(WEB_DIR / "build_litellm_artifacts.py", litellm_args)

        for name in LITELLM_REQUIRED + (LITELLM_MERGED,):
            source = scratch / "litellm" / name
            if source.is_file():
                shutil.copyfile(source, scratch / name)

        priority_args: list[str] = []
        for p in PRIORITIES:
            priority_args += ["--priority", f"{p}={scratch / f'{p}.json'}"]

        # All-providers catalog first, then its rows: both share the endpoint
        # cache, and the catalog is the row-ordering authority the site
        # validator checks against (a transient row/catalog divergence must
        # fail the publish, never ship an empty or misordered tab).
        nonzdr_catalog = scratch / "catalog-nonzdr.json"
        run_catalog(
            model_compare,
            ["--catalog", "--no-zdr", "--zdr-facts", *MODEL_COMPARE_FILTER_ARGS],
            nonzdr_catalog,
        )
        run_validate_catalog(WEB_DIR / "build_site_data.py", nonzdr_catalog)
        check_nonzdr_catalog(nonzdr_catalog)
        nonzdr_args: list[str] = []
        for p in PRIORITIES:
            rows_path = scratch / f"{p}-nonzdr.json"
            run_pool_json(
                model_compare,
                [
                    "--priority",
                    p,
                    "--json",
                    "--top",
                    "20",
                    "--no-zdr",
                    "--zdr-facts",
                    *MODEL_COMPARE_FILTER_ARGS,
                ],
                rows_path,
            )
            nonzdr_args += ["--nonzdr-priority", f"{p}={rows_path}"]

        run_script(
            WEB_DIR / "build_site_data.py",
            [
                "--best-file",
                str(scratch / "best.txt"),
                "--output",
                str(scratch / "data.json"),
                "--catalog-file",
                str(scratch / "catalog.json"),
                "--nonzdr-catalog-file",
                str(nonzdr_catalog),
                "--history-prev-file",
                str(scratch / "history-prev.json"),
                "--highlights-file",
                str(scratch / "highlights-new.json"),
                "--revision",
                resolve_revision(),
                *priority_args,
                *nonzdr_args,
            ],
        )

        for name in SITE_FILES:
            shutil.copyfile(WEB_DIR / "site" / name, scratch / name)

        # Everything above was assembled and gated in the scratch directory;
        # only a fully valid build may touch the deploy directory. This keeps
        # local --output-dir trees and CI's _site free of half-updated
        # artifacts when a late gate fails.
        missing = [name for name in artifacts if not (scratch / name).is_file()]
        if missing:
            raise RuntimeError(
                f"publish: expected artifacts not written: {', '.join(missing)}"
            )
        catalog = check_stamps(scratch)
        check_best(scratch, catalog)
        check_litellm(scratch, catalog)
        for name in artifacts:
            shutil.copyfile(scratch / name, output_dir / name)
        merged_path = output_dir / LITELLM_MERGED
        if (scratch / LITELLM_MERGED).is_file():
            shutil.copyfile(scratch / LITELLM_MERGED, merged_path)
            print(f"publish: wrote {merged_path}")
        elif merged_path.exists():
            # No merged map this run (first run, or an honest 404): do not
            # leave a stale deployed map pretending to be current.
            merged_path.unlink()
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
