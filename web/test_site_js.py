"""Node-backed tests for the inline script in web/site/index.html.

No npm toolchain: the test extracts the page's single <script> block,
prepends the web/js_stubs.js shims, appends a per-test driver, and
evaluates it under node (skipped when node is absent; GitHub's ubuntu
runners ship node preinstalled). The harness exists because the script
duplicates model_compare.py's fmt_* helpers under a "keep them in sync"
comment and has regressed before -- parity is enforced here, and site
behaviour (tooltips, guards) gets real tests instead of code reading.
"""

import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

WEB = Path(__file__).parent
SITE_HTML = WEB / "site" / "index.html"
STUBS_JS = WEB / "js_stubs.js"

sys.path.insert(0, str(WEB.parent))
import model_compare as mc  # noqa: E402

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None, reason="node is not installed"
)

# The page carries exactly one inline <script> block (its CSP forbids
# external scripts), so this match is the whole program.
SCRIPT_RE = re.compile(r"<script>(.*)</script>", re.DOTALL)

NODE_TIMEOUT = 30


def run_js(expression: str):
    """Evaluate stubs + inline script + expression under node; return JSON."""
    match = SCRIPT_RE.search(SITE_HTML.read_text())
    if match is None:
        raise AssertionError("no inline <script> block in web/site/index.html")
    program = "\n".join(
        [STUBS_JS.read_text(), match.group(1), f"var __out = ({expression});"]
    )
    with tempfile.NamedTemporaryFile(
        "w", suffix=".js", dir="/tmp/opencode", delete=False
    ) as fh:
        fh.write(program + "\nconsole.log(JSON.stringify(__out));\n")
        path = fh.name
    proc = subprocess.run(
        ["node", path],
        capture_output=True,
        text=True,
        timeout=NODE_TIMEOUT,
    )
    if proc.returncode != 0:
        raise AssertionError(f"node driver failed:\n{proc.stderr}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


# ---------------------------------------------------------------------------
# Harness sanity + Python parity (existing behaviour)


def test_harness_reaches_the_inline_script():
    out = run_js(
        "{price: fmtPrice(0.1), context: fmtContext(1048576), age: fmtAge(null)}"
    )
    assert out == {"price": "0.100", "context": "1.05M", "age": "-"}


@pytest.mark.parametrize(
    "value",
    [0, 0.001, 0.014, 0.1, 0.27, 0.5, 1, 1.25, 9.99, 10.0, 27.5, 100, 1234.5678],
)
def test_fmtprice_matches_python(value):
    assert run_js(f"fmtPrice({value!r})") == mc.fmt_price(value)


@pytest.mark.parametrize("value", [0, 512, 1000, 163840, 1048576, 2000000])
def test_fmtcontext_matches_python(value):
    assert run_js(f"fmtContext({value!r})") == mc.fmt_context(value)


@pytest.mark.parametrize("value", [None, 0, 7, 59, 60, 90, 105, 320, 364, 365, 400])
def test_fmtage_matches_python(value):
    assert run_js(f"fmtAge({json.dumps(value)})") == mc.fmt_age(value)


# ---------------------------------------------------------------------------
# TIER tooltip: below-threshold (base) prices


def test_tier_tooltip_appends_short_context_rate():
    out = run_js(
        "tierCellData({tier_note: '>272k', sched_note: null, sched_detail: null,"
        " base_input_usd_per_m: 0.1, base_output_usd_per_m: 0.5})"
    )
    assert out == {
        "text": ">272k",
        "hover": (
            "Charges a higher rate once the prompt exceeds 272k tokens;"
            " price shown: that long-context rate."
            " The short-context rate is $0.100 in / $0.500 out per 1M tokens."
        ),
    }


def test_tier_tooltip_without_base_prices_stays_short():
    out = run_js(
        "tierCellData({tier_note: '>272k', sched_note: null, sched_detail: null})"
    )
    assert out["hover"] == (
        "Charges a higher rate once the prompt exceeds 272k tokens;"
        " price shown: that long-context rate."
    )


def test_tier_tooltip_with_null_base_prices_stays_short():
    out = run_js(
        "tierCellData({tier_note: '>272k', sched_note: null, sched_detail: null,"
        " base_input_usd_per_m: null, base_output_usd_per_m: null})"
    )
    assert out["hover"] == (
        "Charges a higher rate once the prompt exceeds 272k tokens;"
        " price shown: that long-context rate."
    )


def test_sched_only_tooltip_gains_no_rate_sentence():
    out = run_js(
        "tierCellData({tier_note: null, sched_note: '-30%',"
        " sched_detail: '00:00-08:00 UTC daily',"
        " base_input_usd_per_m: 0.1, base_output_usd_per_m: 0.5})"
    )
    assert out == {
        "text": "-30%",
        "hover": (
            "Runs up to 30% cheaper during scheduled off-peak hours"
            " (00:00-08:00 UTC daily); price shown: the standard rate."
        ),
    }


def test_sched_marker_tooltip_gains_no_rate_sentence():
    out = run_js(
        "tierCellData({tier_note: null, sched_note: 'SCHED',"
        " sched_detail: null,"
        " base_input_usd_per_m: 0.1, base_output_usd_per_m: 0.5})"
    )
    assert out["hover"] == (
        "Price varies on a schedule by less than 1%; price shown: the standard rate."
    )


def test_mixed_row_joins_both_sentences_with_rate():
    out = run_js(
        "tierCellData({tier_note: '>100k', sched_note: '-30%',"
        " sched_detail: '01:00-05:00 UTC daily',"
        " base_input_usd_per_m: 0.2, base_output_usd_per_m: 2})"
    )
    assert out["hover"] == (
        "Charges a higher rate once the prompt exceeds 100k tokens;"
        " price shown: that long-context rate."
        " The short-context rate is $0.200 in / $2.00 out per 1M tokens."
        " Runs up to 30% cheaper during scheduled off-peak hours"
        " (01:00-05:00 UTC daily); price shown: the standard rate."
    )
