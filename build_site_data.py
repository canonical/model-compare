#!/usr/bin/env python3
"""Build data.json for the published model-compare site.

Reads the JSON output of `model_compare.py --priority P --json --top 10` for
each priority plus the `--best` output, validates everything, and writes a
single data.json consumed by site/index.html. Fails loudly on anything
unexpected so a broken run never deploys a broken site.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from datetime import date, datetime, timezone

PRIORITIES = ("balanced", "price", "quality")
ROW_KEYS = (
    "model",
    "opencode_model",
    "name",
    "score",
    "quality_index",
    "input_usd_per_m",
    "output_usd_per_m",
    "blended_usd_per_m",
    "discount",
    "context_tokens",
    "age_days",
)
# provider-qualified opencode id (openrouter/<provider/model>), the plain
# catalog id (<provider/model>), each with an optional :variant suffix.
MODEL_ID_RE = re.compile(
    r"^(?:[a-zA-Z0-9_.-]+/)?[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+"
    r"(?::[a-zA-Z0-9_.-]+)?$"
)

CATALOG_SCHEMA_VERSION = 1

CATALOG_ENTRY_KEYS = (
    "id",
    "name",
    "provider",
    "family",
    "pricing",
    "context",
    "listed_at",
    "age_days",
    "tool_calling",
    "zdr",
    "discount",
    "expired",
    "quality",
    "aa",
    "quality_match",
    "scores",
)
CATALOG_PRICING_KEYS = ("input_per_1m", "output_per_1m", "blended_per_1m")
CATALOG_SCORE_KEYS = ("price", "quality", "context", "age")
CATALOG_OVERALL_KEYS = ("balanced", "price", "quality")
CATALOG_AA_KEYS = ("intelligence_index", "coding_index", "agentic_index")
CATALOG_QUALITY_MATCH_VALUES = ("openrouter", "api", "scrape")
CATALOG_AA_MODES = ("openrouter", "api", "scrape", "none")


def _is_score(value) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and 0.0 <= value <= 1.0
    )


def _is_aa_value(value) -> bool:
    if value is None:
        return True
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and 0.0 <= value <= 100.0
    )


def validate_catalog(document) -> None:
    """Validate a raw model_compare.py --catalog document.

    Raises ValueError on any breach so a broken run fails the build and
    never deploys. The contract is additive-only: unknown schema_version
    or missing fields fail here rather than downstream.
    """
    if not isinstance(document, dict):
        raise ValueError("catalog document is not an object")
    if document.get("schema_version") != CATALOG_SCHEMA_VERSION:
        raise ValueError(
            f"unknown catalog schema_version: {document.get('schema_version')!r}"
        )
    if document.get("tool") != "model-compare":
        raise ValueError(f"unexpected catalog tool: {document.get('tool')!r}")
    for key in ("generated_at", "parameters", "sources", "pool", "models", "filtered"):
        if key not in document:
            raise ValueError(f"catalog missing key: {key}")
    parameters = document["parameters"]
    if not isinstance(parameters, dict) or "zdr_required" not in parameters:
        raise ValueError("catalog parameters must set zdr_required")
    # scores.overall is documented as reproducible from parameters.weights
    # alone, so the weights must cover every priority and sum to 1.
    weights = parameters.get("weights")
    if not isinstance(weights, dict) or any(
        key not in weights for key in CATALOG_OVERALL_KEYS
    ):
        raise ValueError(
            "catalog parameters.weights must cover: " + ", ".join(CATALOG_OVERALL_KEYS)
        )
    for priority in CATALOG_OVERALL_KEYS:
        w = weights[priority]
        if not isinstance(w, dict) or not w:
            raise ValueError(
                f"catalog parameters.weights.{priority} must map score names to weights"
            )
        bad_weights = [
            name for name, value in w.items() if not _is_score(value) or value == 0
        ]
        if bad_weights or abs(sum(w.values()) - 1.0) > 1e-6:
            raise ValueError(
                f"catalog parameters.weights.{priority} must be positive weights summing to 1"
            )
    pool = document["pool"]
    if not isinstance(pool, dict) or not isinstance(pool.get("candidates"), int):
        raise ValueError("catalog pool.candidates must be an integer")
    dropped = pool.get("dropped")
    if not isinstance(dropped, dict) or not all(
        isinstance(value, int) for value in dropped.values()
    ):
        raise ValueError("catalog pool.dropped must map reasons to integers")
    listed = pool.get("listed")
    if not isinstance(listed, int):
        raise ValueError("catalog pool.listed must be an integer")
    if sum(dropped.values()) + pool["candidates"] != listed:
        raise ValueError(
            "catalog pool accounting mismatch: "
            f"{sum(dropped.values())} dropped + {pool['candidates']} candidates"
            f" != {listed} listed"
        )
    sources = document["sources"]
    aa_sources = sources.get("aa") if isinstance(sources, dict) else None
    if (
        not isinstance(aa_sources, dict)
        or aa_sources.get("mode") not in CATALOG_AA_MODES
    ):
        raise ValueError("catalog sources.aa.mode is unknown")
    for key in ("matched", "matched_openrouter"):
        value = aa_sources.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"catalog sources.aa.{key} must be a non-negative integer")
    if aa_sources["matched_openrouter"] > aa_sources["matched"]:
        raise ValueError("catalog sources.aa.matched_openrouter exceeds matched")
    models = document["models"]
    if not isinstance(models, list):
        raise ValueError("catalog models must be a list")
    if len(models) != pool["candidates"]:
        raise ValueError(
            f"catalog has {len(models)} models but pool.candidates is {pool['candidates']}"
        )
    for i, entry in enumerate(models):
        if not isinstance(entry, dict):
            raise ValueError(f"models[{i}] is not an object")
        missing = [key for key in CATALOG_ENTRY_KEYS if key not in entry]
        if missing:
            raise ValueError(f"models[{i}] is missing keys: {', '.join(missing)}")
        pricing = entry["pricing"]
        if not isinstance(pricing, dict) or any(
            key not in pricing for key in CATALOG_PRICING_KEYS
        ):
            raise ValueError(f"models[{i}] pricing is incomplete")
        scores = entry["scores"]
        if not isinstance(scores, dict):
            raise ValueError(f"models[{i}] scores is not an object")
        bad = [key for key in CATALOG_SCORE_KEYS if not _is_score(scores.get(key))]
        if bad:
            raise ValueError(f"models[{i}] scores out of range: {', '.join(bad)}")
        overall = scores.get("overall")
        if not isinstance(overall, dict) or any(
            key not in overall for key in CATALOG_OVERALL_KEYS
        ):
            raise ValueError(f"models[{i}] scores.overall is incomplete")
        bad_overall = [
            key for key in CATALOG_OVERALL_KEYS if not _is_score(overall[key])
        ]
        if bad_overall:
            raise ValueError(
                f"models[{i}] scores.overall out of range: {', '.join(bad_overall)}"
            )
        aa = entry["aa"]
        if not isinstance(aa, dict) or any(key not in aa for key in CATALOG_AA_KEYS):
            raise ValueError(f"models[{i}] aa is incomplete")
        bad_aa = [key for key in CATALOG_AA_KEYS if not _is_aa_value(aa[key])]
        if bad_aa:
            raise ValueError(f"models[{i}] aa out of range: {', '.join(bad_aa)}")
        if (
            entry["quality_match"] is not None
            and entry["quality_match"] not in CATALOG_QUALITY_MATCH_VALUES
        ):
            raise ValueError(f"models[{i}] has unknown quality_match")
        if parameters["zdr_required"] and entry["zdr"] is not True:
            raise ValueError(f"models[{i}] is not zdr=true under zdr_required")
    # ids double as the document's primary key for downstream consumers:
    # unique within models, and never also present in filtered.
    model_ids = [entry["id"] for entry in models]
    if not all(isinstance(model_id, str) for model_id in model_ids):
        raise ValueError("catalog models id must be a string")
    if len(set(model_ids)) != len(model_ids):
        raise ValueError("catalog models contain duplicate ids")
    filtered_ids = []
    for i, entry in enumerate(document["filtered"]):
        if not isinstance(entry, dict):
            raise ValueError(f"filtered[{i}] is not an object")
        if not isinstance(entry.get("id"), str) or not entry["id"]:
            raise ValueError(f"filtered[{i}] has no id")
        if not isinstance(entry.get("reasons"), list) or not entry["reasons"]:
            raise ValueError(f"filtered[{i}] has no reasons")
        filtered_ids.append(entry["id"])
    in_both = set(model_ids) & set(filtered_ids)
    if in_both:
        raise ValueError(
            f"catalog id in both models and filtered: {sorted(in_both)[0]}"
        )


HISTORY_SCHEMA_VERSION = 1
HISTORY_RETENTION = 10
HISTORY_TOP_N = 10
HISTORY_TAB_KEYS = ("balanced", "price", "quality")


def _is_iso_date(value) -> bool:
    try:
        date.fromisoformat(str(value))
    except (TypeError, ValueError):
        return False
    return True


def build_snapshot(catalog) -> dict:
    """Project a validated catalog document into one daily history snapshot.

    tabs derive from the catalog itself: per priority, models sorted by
    scores.overall descending (id tiebreak), top 10, rank 1-based. aa and
    prices cover candidates only -- filtered entries carry neither.
    """
    models = catalog["models"]
    tabs = {}
    for priority in HISTORY_TAB_KEYS:
        ranked = sorted(
            models, key=lambda e: (-e["scores"]["overall"][priority], e["id"])
        )
        tabs[priority] = [
            {
                "id": entry["id"],
                "rank": i + 1,
                "quality": entry["quality"],
                "blended": entry["pricing"]["blended_per_1m"],
            }
            for i, entry in enumerate(ranked[:HISTORY_TOP_N])
        ]
    return {
        "generated_at": catalog["generated_at"],
        "pool_ids": sorted(
            [e["id"] for e in models] + [e["id"] for e in catalog["filtered"]]
        ),
        "tabs": tabs,
        "aa": {
            e["id"]: e["aa"]["intelligence_index"]
            for e in models
            if e["aa"]["intelligence_index"] is not None
        },
        "prices": {
            e["id"]: [
                e["pricing"]["input_per_1m"],
                e["pricing"]["output_per_1m"],
                e["pricing"]["blended_per_1m"],
                e["discount"],
            ]
            for e in models
        },
    }


def merge_history(prev, snapshot) -> dict:
    """Upsert the snapshot under its UTC date and prune to the newest 10.

    A malformed previous document starts fresh; garbage entries inside a
    well-formed one are dropped per date. Same-day upserts are
    last-write-wins.
    """
    snapshots = {}
    if isinstance(prev, dict) and isinstance(prev.get("snapshots"), dict):
        for date_key, snap in prev["snapshots"].items():
            if _is_iso_date(date_key) and isinstance(snap, dict):
                snapshots[date_key] = snap
    today = snapshot["generated_at"][:10]
    snapshots[today] = snapshot
    kept = sorted(snapshots)[-HISTORY_RETENTION:]
    return {
        "schema_version": HISTORY_SCHEMA_VERSION,
        "updated_at": snapshots[kept[-1]]["generated_at"],
        "snapshots": {date_key: snapshots[date_key] for date_key in kept},
    }


def validate_history(document) -> None:
    """Validate the merged history document (strict identity, <=10 dates)."""
    if not isinstance(document, dict):
        raise ValueError("history document is not an object")
    if document.get("schema_version") != HISTORY_SCHEMA_VERSION:
        raise ValueError(
            f"unknown history schema_version: {document.get('schema_version')!r}"
        )
    snapshots = document.get("snapshots")
    if not isinstance(snapshots, dict) or not snapshots:
        raise ValueError("history snapshots must be a non-empty object")
    if not all(_is_iso_date(d) for d in snapshots):
        raise ValueError("history snapshot keys must be ISO dates")
    if len(snapshots) > HISTORY_RETENTION:
        raise ValueError(
            f"history holds {len(snapshots)} snapshots (max {HISTORY_RETENTION})"
        )
    for date_key, snap in snapshots.items():
        if not isinstance(snap, dict):
            raise ValueError(f"history snapshot {date_key} is not an object")
        for key in ("generated_at", "pool_ids", "tabs", "aa", "prices"):
            if key not in snap:
                raise ValueError(f"history snapshot {date_key} missing key: {key}")
        for priority in HISTORY_TAB_KEYS:
            rows = snap["tabs"].get(priority)
            if not isinstance(rows, list):
                raise ValueError(f"history snapshot {date_key} tabs.{priority} missing")
            for i, row in enumerate(rows):
                if not isinstance(row, dict) or row.get("rank") != i + 1:
                    raise ValueError(
                        f"history snapshot {date_key} tabs.{priority} rank mismatch"
                    )


def build_data(best, priorities, now=None) -> dict:
    if not isinstance(best, str) or not MODEL_ID_RE.fullmatch(best.strip()):
        raise ValueError(f"best model id looks wrong: {best!r}")
    best = best.strip()
    for name in PRIORITIES:
        rows = priorities.get(name)
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"no rows for priority {name!r}")
        for i, row in enumerate(rows):
            if not isinstance(row, dict):
                raise ValueError(f"{name}[{i}] is not an object")
            missing = [key for key in ROW_KEYS if key not in row]
            if missing:
                raise ValueError(f"{name}[{i}] is missing keys: {', '.join(missing)}")
    return {
        "generated_at": (now or datetime.now(timezone.utc)).isoformat(
            timespec="seconds"
        ),
        "best": best,
        "priorities": {name: list(priorities[name]) for name in PRIORITIES},
    }


def parse_priority_pair(pair):
    name, sep, path = pair.partition("=")
    if not sep or name not in PRIORITIES:
        raise ValueError(
            f"bad --priority {pair!r} (want NAME=FILE, NAME one of {PRIORITIES})"
        )
    with open(path) as fh:
        rows = json.load(fh)
    return name, rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Build data.json for the published model-compare site"
    )
    parser.add_argument(
        "--best-file", required=True, help="file holding the `--best` output"
    )
    parser.add_argument("--output", required=True, help="data.json destination path")
    parser.add_argument(
        "--catalog-file",
        help="raw model_compare.py --catalog output; validated and written as catalog.json next to --output",
    )
    parser.add_argument(
        "--history-prev-file",
        help="previous history.json (fetched from the live site); merged, pruned and rewritten as history.json next to --output",
    )
    parser.add_argument(
        "--priority",
        action="append",
        required=True,
        metavar="NAME=FILE",
        help="priority name and its model_compare.py --json output file",
    )
    args = parser.parse_args(argv)
    catalog = None
    if args.catalog_file:
        try:
            with open(args.catalog_file) as fh:
                catalog = json.load(fh)
            validate_catalog(catalog)
        except (OSError, ValueError) as exc:
            print(f"error: invalid catalog: {exc}", file=sys.stderr)
            return 1
    try:
        with open(args.best_file) as fh:
            best = fh.read()
        priorities = {}
        for pair in args.priority:
            name, rows = parse_priority_pair(pair)
            if name in priorities:
                raise ValueError(
                    f"duplicate --priority {name!r} (want each priority once)"
                )
            priorities[name] = rows
        data = build_data(best, priorities)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    try:
        with open(args.output, "w") as fh:
            json.dump(data, fh, indent=2)
            fh.write("\n")
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if catalog is not None:
        catalog_path = os.path.join(
            os.path.dirname(os.path.abspath(args.output)), "catalog.json"
        )
        try:
            with open(catalog_path, "w") as fh:
                json.dump(catalog, fh, indent=2)
                fh.write("\n")
        except OSError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
    if args.history_prev_file is not None:
        if catalog is None:
            print(
                "error: --history-prev-file requires --catalog-file",
                file=sys.stderr,
            )
            return 1
        prev = None
        if args.history_prev_file:
            try:
                with open(args.history_prev_file) as fh:
                    prev = json.load(fh)
            except (OSError, ValueError):
                prev = None  # malformed/missing previous: start fresh
        history = merge_history(prev, build_snapshot(catalog))
        try:
            validate_history(history)
        except ValueError as exc:
            print(f"error: invalid history: {exc}", file=sys.stderr)
            return 1
        history_path = os.path.join(
            os.path.dirname(os.path.abspath(args.output)), "history.json"
        )
        try:
            with open(history_path, "w") as fh:
                json.dump(history, fh, indent=2)
                fh.write("\n")
        except OSError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
