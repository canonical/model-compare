#!/usr/bin/env bash
# preview.sh — serve site/index.html locally for a visual check.
#
# Data sources, in order of preference:
#   1. --build    run the real pipeline (model_compare.py + build_site_data.py)
#                 locally; set AA_API_KEY for better quality scores
#   2. default    fetch live data.json/best.txt from the deployed site
#   3. fallback   synthesize placeholder rows if the live fetch fails
#
# Ctrl-C stops the server; the temporary preview dir is removed on exit.

set -euo pipefail

PORT=8734
BUILD=0
LIVE_URL="https://canonical.github.io/model-compare"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SITE_DIR="$SCRIPT_DIR/site"
SRV_PID=""
BUILD_DIR=""

die_usage() {
	echo "preview.sh: $1 (see --help)" >&2
	exit 2
}

usage() {
	cat <<EOF
usage: preview.sh [--port N] [--build] [--live-url URL] [--site-dir DIR]

Serves site/index.html at http://127.0.0.1:PORT (default 8734) together
with data.json and best.txt.

  --port N        port to serve on (default: 8734)
  --build         build fresh data locally with model_compare.py +
                  build_site_data.py instead of using the live data
  --live-url URL  deployed site to fetch data from
  --site-dir DIR  directory holding index.html (default: ./site)
EOF
	exit 0
}

while [ $# -gt 0 ]; do
	case "$1" in
	--port)
		[ $# -ge 2 ] || die_usage "--port requires a value"
		case "$2" in '' | *[!0-9]*) die_usage "--port wants a number, got '$2'" ;; esac
		PORT="$2"
		shift 2
		;;
	--build)
		BUILD=1
		shift
		;;
	--live-url)
		[ $# -ge 2 ] || die_usage "--live-url requires a value"
		LIVE_URL="$2"
		shift 2
		;;
	--site-dir)
		[ $# -ge 2 ] || die_usage "--site-dir requires a value"
		SITE_DIR="$2"
		shift 2
		;;
	-h | --help)
		usage
		;;
	*)
		die_usage "unknown option: $1"
		;;
	esac
done

command -v python3 >/dev/null || {
	echo "preview.sh: python3 is required" >&2
	exit 1
}
[ -f "$SITE_DIR/index.html" ] || {
	echo "preview.sh: no index.html in $SITE_DIR" >&2
	exit 1
}

PREVIEW_DIR="$(mktemp -d)"
trap '
	if [ -n "$SRV_PID" ]; then
		kill "$SRV_PID" 2>/dev/null || true
	fi
	if [ -n "$BUILD_DIR" ]; then
		rm -rf "$BUILD_DIR"
	fi
	rm -rf "$PREVIEW_DIR"
' EXIT

cp "$SITE_DIR/index.html" "$PREVIEW_DIR/index.html"

if [ "$BUILD" -eq 1 ]; then
	echo "building data locally with model_compare.py + build_site_data.py..."
	BUILD_DIR="$(mktemp -d)"
	for p in balanced price quality; do
		python3 "$SCRIPT_DIR/model_compare.py" --priority "$p" --json --top 10 \
			>"$BUILD_DIR/$p.json"
	done
	python3 "$SCRIPT_DIR/model_compare.py" --best >"$PREVIEW_DIR/best.txt"
	python3 "$SCRIPT_DIR/build_site_data.py" \
		--best-file "$PREVIEW_DIR/best.txt" \
		--output "$PREVIEW_DIR/data.json" \
		--priority "balanced=$BUILD_DIR/balanced.json" \
		--priority "price=$BUILD_DIR/price.json" \
		--priority "quality=$BUILD_DIR/quality.json"
elif command -v curl >/dev/null &&
	curl -fsSL --max-time 10 "$LIVE_URL/data.json" -o "$PREVIEW_DIR/data.json" &&
	curl -fsSL --max-time 10 "$LIVE_URL/best.txt" -o "$PREVIEW_DIR/best.txt"; then
	echo "using live data from $LIVE_URL"
else
	echo "live data unavailable; using synthetic placeholder data" >&2
	python3 - "$PREVIEW_DIR/data.json" <<'PY'
import json
import sys
from datetime import datetime, timezone


def row(model, q, inp, outp, disc, ctx, age, score):
    return {
        "model": model,
        "opencode_model": "openrouter/" + model,
        "name": model.split("/")[1],
        "score": score,
        "quality_index": q,
        "input_usd_per_m": inp,
        "output_usd_per_m": outp,
        "blended_usd_per_m": (inp + outp) / 2,
        "discount": disc,
        "context_tokens": ctx,
        "age_days": age,
    }


rows = [
    row("google/gemini-2.5-pro", 71.2, 1.25, 10.0, 0, 1048576, 320, 0.862),
    row("deepseek/deepseek-v4", 68.9, 0.27, 1.1, 0.5, 163840, 210, 0.847),
    row("qwen/qwen3-max", 66.4, 0.6, 2.4, 0.75, 262144, 190, 0.831),
    row("openai/gpt-5.2-mini", 64.1, 0.55, 4.4, 0, 400000, 150, 0.812),
    row("anthropic/claude-haiku-4.5", 62.0, 1.0, 5.0, 0, 200000, 400, 0.795),
    row("mistralai/mistral-large-3", 59.8, 2.0, 6.0, 0.5, 128000, 260, 0.774),
    row("meta-llama/llama-4-maverick", 57.5, 0.22, 0.85, 0, 1048576, 180, 0.751),
    row("x-ai/grok-4-fast", 55.9, 0.2, 0.5, 0.75, 2000000, 120, 0.733),
    row("amazon/nova-pro-2", 53.1, 0.4, 1.6, 0.5, 300000, 240, 0.705),
    row("cohere/command-a-2", 50.4, 2.5, 10.0, None, 256000, 300, 0.682),
]
data = {
    "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "best": "openrouter/google/gemini-2.5-pro",
    "priorities": {
        "balanced": rows,
        "price": sorted(rows, key=lambda r: (r["input_usd_per_m"], -r["score"])),
        "quality": sorted(rows, key=lambda r: -r["quality_index"]),
    },
}
with open(sys.argv[1], "w") as fh:
    json.dump(data, fh, indent=2)
    fh.write("\n")
PY
	python3 -c "import json, sys; print(json.load(open(sys.argv[1]))['best'])" \
		"$PREVIEW_DIR/data.json" >"$PREVIEW_DIR/best.txt"
fi

python3 -m http.server "$PORT" --directory "$PREVIEW_DIR" >/dev/null 2>&1 &
SRV_PID=$!

probe() {
	python3 -c "import urllib.request as u; u.install_opener(u.build_opener(u.ProxyHandler({}))); u.urlopen('http://127.0.0.1:$PORT/index.html', timeout=1)" 2>/dev/null
}

ready=0
for _ in $(seq 1 25); do
	if probe; then
		ready=1
		break
	fi
	if ! kill -0 "$SRV_PID" 2>/dev/null; then
		echo "preview.sh: server failed to start (is port $PORT already in use?)" >&2
		exit 1
	fi
	sleep 0.2
done
[ "$ready" -eq 1 ] || {
	echo "preview.sh: server did not become ready" >&2
	exit 1
}

echo "preview: http://127.0.0.1:$PORT  (Ctrl-C to stop)"
wait "$SRV_PID"
