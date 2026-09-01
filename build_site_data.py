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
import re
import sys
from datetime import datetime, timezone

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
        "--priority",
        action="append",
        required=True,
        metavar="NAME=FILE",
        help="priority name and its model_compare.py --json output file",
    )
    args = parser.parse_args(argv)
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
