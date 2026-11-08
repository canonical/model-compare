"""Unit tests for model_compare scoring, matching, and filtering logic.

These cover pure functions only -- no network access is performed. Run with:

    pytest -q
"""

from __future__ import annotations

import json
import io
import math
import os
import re
import subprocess
import sys
import threading
import time
from argparse import Namespace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / "web"))
import build_site_data as bsd  # noqa: E402  (web/ module, contract validator)
import model_compare as mc
from model_compare import (
    PRECISION_LADDER,
    is_first_party,
    normalize_provider,
)


@pytest.fixture(autouse=True)
def _isolate_user_cache(monkeypatch, tmp_path):
    """Point XDG_CACHE_HOME at a per-test directory for every test.

    Tests must never read or write the real user cache
    (~/.cache/model-compare). A test once saved fixture AA entries there,
    and a CI publish then loaded them on a silent cache hit and published
    them. Tests that set XDG_CACHE_HOME themselves still win (same
    tmp_path, later setenv).
    """
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    return tmp_path


def make_args(**overrides):
    """Build an args namespace with the defaults build_candidates/compute_scores expect."""
    base = dict(
        min_context=0,
        priority="balanced",
        top=5,
        best=False,
        json=False,
        catalog=False,
        input_share=0.75,
        recency_half_life=120.0,
        max_age_days=0.0,
        quality_ref=70.0,
        aa_api_key=None,
        no_require_tools=True,
        exclude_free=False,
        include_batch=False,
        discount=False,
        no_zdr=False,
        no_cache=True,
        cache_ttl=0,
        # parse_args always provides this (mirror the production shape);
        # None means the keyless fast path in fetch_session_cost.
        openrouter_api_key=None,
        reasoning_effort="default",
        # --no-depths arrives with the depth-output task; inert until then.
        no_depths=False,
        detail="expert",
        off_peak=False,
        bare=False,
        color="auto",
        no_pager=False,
    )
    base.update(overrides)
    return Namespace(**base)


def make_model(**overrides):
    model = {
        "id": "acme/model-a",
        "name": "Acme: Model A",
        "context_length": 2_000_000,
        "pricing": {"prompt": "0.000001", "completion": "0.000002"},
        "architecture": {"modality": "text->text"},
        "supported_parameters": ["tools", "tool_choice"],
        "created": 0,
    }
    model.update(overrides)
    return model


# ---------------------------------------------------------------------------
# coerce_int (Fix #1)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        (128000, 128000),
        ("128000", 128000),
        (128000.9, 128000),
        (None, 0),
        ("", 0),
        ("nan-ish", 0),
        ([], 0),
    ],
)
def test_coerce_int(value, expected):
    assert mc.coerce_int(value) == expected


def test_coerce_int_custom_default():
    assert mc.coerce_int(None, default=-1) == -1


# ---------------------------------------------------------------------------
# Tiered pricing: effective_pricing + display-string formatters
# ---------------------------------------------------------------------------


def haiku_pricing():
    """Verbatim live anthropic/claude-haiku-5.5 pricing shape (2026-10-09)."""
    return {
        "prompt": "0.0000001",
        "completion": "0.0000005",
        "overrides": [
            {
                "min_prompt_tokens": 100000,
                "prompt": "0.0000005",
                "completion": "0.0000025",
                "input_cache_read": "0.00000005",
                "input_cache_write": "0.000000625",
            }
        ],
    }


def deepseek_pricing():
    """Verbatim live deepseek/deepseek-v4-pro-0813 overrides (2026-10-09).

    Weekday peak windows tile 01:00-10:00 UTC at 2x; the rest of the week
    is off-peak, including the days-only weekend window.
    """
    return {
        "prompt": "0.00000066",
        "completion": "0.00000198",
        "input_cache_read": "0.000000022",
        "overrides": [
            {
                "utc_days": ["saturday", "sunday"],
                "prompt": "0.00000066",
                "completion": "0.00000198",
                "input_cache_read": "0.000000022",
            },
            {
                "utc_days": ["monday", "tuesday", "wednesday", "thursday", "friday"],
                "utc_start": 0,
                "utc_end": 100,
                "prompt": "0.00000066",
                "completion": "0.00000198",
                "input_cache_read": "0.000000022",
            },
            {
                "utc_days": ["monday", "tuesday", "wednesday", "thursday", "friday"],
                "utc_start": 100,
                "utc_end": 400,
                "prompt": "0.00000132",
                "completion": "0.00000396",
                "input_cache_read": "0.000000044",
            },
            {
                "utc_days": ["monday", "tuesday", "wednesday", "thursday", "friday"],
                "utc_start": 400,
                "utc_end": 600,
                "prompt": "0.00000066",
                "completion": "0.00000198",
                "input_cache_read": "0.000000022",
            },
            {
                "utc_days": ["monday", "tuesday", "wednesday", "thursday", "friday"],
                "utc_start": 600,
                "utc_end": 1000,
                "prompt": "0.00000132",
                "completion": "0.00000396",
                "input_cache_read": "0.000000044",
            },
            {
                "utc_days": ["monday", "tuesday", "wednesday", "thursday", "friday"],
                "utc_start": 1000,
                "utc_end": 0,
                "prompt": "0.00000066",
                "completion": "0.00000198",
                "input_cache_read": "0.000000022",
            },
        ],
    }


def hy4_pricing():
    """tencent/hy4-preview shape: two wrap windows tiling the day."""
    return {
        "prompt": "0.000000834",
        "completion": "0.000002501",
        "overrides": [
            {
                "utc_start": 0,
                "utc_end": 1600,
                "prompt": "0.000000834",
                "completion": "0.000002501",
            },
            {
                "utc_start": 1600,
                "utc_end": 0,
                "prompt": "0.0000007506",
                "completion": "0.0000022509",
            },
        ],
    }


def test_no_overrides_passthrough():
    eff = mc.effective_pricing(make_model()["pricing"], 1_000_000, 0.75)
    assert eff["price_in"] == eff["base_price_in"] == 1e-6
    assert eff["price_out"] == eff["base_price_out"] == 2e-6
    assert eff["tier_prompt_tokens"] is None
    assert eff["tiers"] == []
    assert eff["schedule"] is None
    assert eff["schedule_error"] is False
    assert eff["max_discount"] is None
    assert eff["peak_blended"] is None


def test_haiku_single_tier_at_min_context():
    eff = mc.effective_pricing(haiku_pricing(), 1_000_000, 0.75)
    assert eff["price_in"] == 5e-7
    assert eff["price_out"] == 2.5e-6
    assert eff["tier_prompt_tokens"] == 100000
    assert eff["tiers"] == [
        {"min_prompt_tokens": 100000, "price_in": 5e-7, "price_out": 2.5e-6}
    ]
    assert eff["base_price_in"] == 1e-7
    assert eff["base_price_out"] == 5e-7
    assert eff["schedule"] is None
    assert eff["schedule_error"] is False


def test_tier_below_prompt_size_not_applied():
    eff = mc.effective_pricing(haiku_pricing(), 50000, 0.75)
    assert eff["price_in"] == 1e-7
    assert eff["price_out"] == 5e-7
    assert eff["tier_prompt_tokens"] is None


def test_threshold_equal_to_prompt_size_not_applied():
    eff = mc.effective_pricing(haiku_pricing(), 100000, 0.75)
    assert eff["price_in"] == 1e-7
    assert eff["tier_prompt_tokens"] is None


def test_multi_tier_later_wins():
    pricing = {
        "prompt": "0.000001",
        "completion": "0.000002",
        "overrides": [
            {"min_prompt_tokens": 100000, "prompt": "0.000002"},
            {
                "min_prompt_tokens": 200000,
                "prompt": "0.000003",
                "completion": "0.000006",
            },
        ],
    }
    eff = mc.effective_pricing(pricing, 1_000_000, 0.75)
    assert eff["price_in"] == 3e-6
    assert eff["price_out"] == 6e-6
    assert eff["tier_prompt_tokens"] == 200000
    assert [
        (t["min_prompt_tokens"], t["price_in"], t["price_out"]) for t in eff["tiers"]
    ] == [
        (100000, 2e-6, 2e-6),
        (200000, 3e-6, 6e-6),
    ]
    at_150k = mc.effective_pricing(pricing, 150000, 0.75)
    assert at_150k["price_in"] == 2e-6
    assert at_150k["price_out"] == 2e-6  # per-key inheritance: base completion
    assert at_150k["tier_prompt_tokens"] == 100000


def test_duplicate_thresholds_deduped_later_wins_per_key():
    pricing = {
        "prompt": "0.000001",
        "completion": "0.000002",
        "overrides": [
            {"min_prompt_tokens": 100000, "prompt": "0.000002"},
            {"min_prompt_tokens": 100000, "completion": "0.000004"},
        ],
    }
    eff = mc.effective_pricing(pricing, 1_000_000, 0.75)
    assert eff["tiers"] == [
        {"min_prompt_tokens": 100000, "price_in": 2e-6, "price_out": 4e-6}
    ]
    assert eff["price_in"] == 2e-6
    assert eff["price_out"] == 4e-6


def test_per_key_inheritance_from_base():
    pricing = {
        "prompt": "0.000001",
        "completion": "0.000002",
        "overrides": [{"min_prompt_tokens": 100000, "prompt": "0.000003"}],
    }
    eff = mc.effective_pricing(pricing, 1_000_000, 0.75)
    assert eff["price_in"] == 3e-6
    assert eff["price_out"] == 2e-6


def test_key_whitelist_audio_price_keys_apply():
    pricing = {
        "prompt": "0.000001",
        "completion": "0.000002",
        "overrides": [
            {
                "min_prompt_tokens": 100000,
                "prompt": "0.000002",
                "audio": "0.00003",
                "input_audio_cache": "0.000001",
            }
        ],
    }
    eff = mc.effective_pricing(pricing, 1_000_000, 0.75)
    assert eff["price_in"] == 2e-6


def test_key_whitelist_unknown_key_skips_entry():
    pricing = {
        "prompt": "0.000001",
        "completion": "0.000002",
        "overrides": [
            {
                "min_prompt_tokens": 100000,
                "prompt": "0.000002",
                "quantum_flavor": "high",
            }
        ],
    }
    eff = mc.effective_pricing(pricing, 1_000_000, 0.75)
    assert eff["price_in"] == 1e-6
    assert eff["tiers"] == []


def test_conditionless_entry_is_ignored():
    pricing = {
        "prompt": "0.000001",
        "completion": "0.000002",
        "overrides": [{"prompt": "0.000009", "completion": "0.000009"}],
    }
    eff = mc.effective_pricing(pricing, 1_000_000, 0.75)
    assert eff["price_in"] == 1e-6
    assert eff["tiers"] == []
    assert eff["schedule_error"] is False


@pytest.mark.parametrize(
    "bad_threshold",
    [None, "100000", True, -1, 100000.5],
)
def test_fail_soft_bad_thresholds(bad_threshold):
    pricing = {
        "prompt": "0.000001",
        "completion": "0.000002",
        "overrides": [{"min_prompt_tokens": bad_threshold, "prompt": "0.000002"}],
    }
    eff = mc.effective_pricing(pricing, 1_000_000, 0.75)
    assert eff["price_in"] == 1e-6
    assert eff["tiers"] == []
    assert eff["schedule_error"] is False


@pytest.mark.parametrize(
    "bad_entry",
    [
        {"utc_start": 2400, "utc_end": 0},
        {"utc_start": 1075, "utc_end": 0},  # minutes 75
        {"utc_start": 1630.5, "utc_end": 0},
        {"utc_start": True, "utc_end": 0},
        {"utc_start": 100},  # one-sided
        {"utc_end": 100},
        {"utc_days": []},
        {"utc_days": "saturday"},
        {"utc_days": ["funday"]},
    ],
)
def test_fail_soft_bad_time_conditions_cascade(bad_entry):
    # A skipped TIME window always breaks the tiling invariant (spec rule 9),
    # so the schedule is invalid even though the entry itself is merely skipped.
    entry = {"prompt": "0.000001", "completion": "0.000002"}
    entry.update(bad_entry)
    pricing = {
        "prompt": "0.000001",
        "completion": "0.000002",
        "overrides": [entry],
    }
    eff = mc.effective_pricing(pricing, 1_000_000, 0.75)
    assert eff["schedule_error"] is True
    assert eff["schedule"] is None


@pytest.mark.parametrize("bad_overrides", [None, "x", {"a": 1}, []])
def test_overrides_not_a_list_treated_as_none(bad_overrides):
    pricing = {
        "prompt": "0.000001",
        "completion": "0.000002",
        "overrides": bad_overrides,
    }
    eff = mc.effective_pricing(pricing, 1_000_000, 0.75)
    assert eff["price_in"] == 1e-6
    assert eff["schedule_error"] is False


@pytest.mark.parametrize("bad_price", ["-1", "nan", "inf", "x"])
def test_fail_soft_prices(bad_price):
    pricing = {
        "prompt": "0.000001",
        "completion": "0.000002",
        "overrides": [{"min_prompt_tokens": 100000, "prompt": bad_price}],
    }
    eff = mc.effective_pricing(pricing, 1_000_000, 0.75)
    assert eff["price_in"] == 1e-6
    assert eff["tiers"] == []


def test_non_finite_base_passes_through_for_upstream_drop():
    # Base validity belongs to build_candidates; the parser must not crash
    # and must pass the parsed (possibly non-finite) values through.
    eff = mc.effective_pricing({"prompt": "nan", "completion": "0.000002"}, 0, 0.75)
    assert math.isnan(eff["base_price_in"])
    assert eff["price_out"] == 2e-6


HUGE_INT = 10**400  # what json.loads yields for a 401-digit integer literal


@pytest.mark.parametrize(
    "entry",
    [
        {"min_prompt_tokens": HUGE_INT, "prompt": "0.000005"},
        {"min_prompt_tokens": 1000, "prompt": HUGE_INT},
        {"min_prompt_tokens": 1000, "completion": HUGE_INT},
        {"min_prompt_tokens": 1000, "prompt": True},
    ],
    ids=["huge-threshold", "huge-prompt", "huge-completion", "bool-prompt"],
)
def test_overflowing_or_bool_tier_entry_skipped(entry):
    pricing = {"prompt": "0.000001", "completion": "0.000002", "overrides": [entry]}
    eff = mc.effective_pricing(pricing, 1_000_000, 0.75)  # must not raise
    assert eff["price_in"] == 1e-6
    assert eff["price_out"] == 2e-6
    assert eff["tier_prompt_tokens"] is None
    assert eff["tiers"] == []
    assert eff["schedule_error"] is False


@pytest.mark.parametrize("key", ["utc_start", "utc_end"])
def test_overflowing_clock_cascades_to_schedule_error(key):
    pricing = hy4_pricing()
    pricing["overrides"][0][key] = HUGE_INT
    eff = mc.effective_pricing(pricing, 1_000_000, 0.75)  # must not raise
    assert eff["schedule_error"] is True
    assert eff["schedule"] is None


def test_parse_price_rejects_overflow_and_bool():
    assert mc.parse_price(HUGE_INT) is None
    assert mc.parse_price(True) is None
    assert mc.parse_price(False) is None
    assert mc.parse_price("0.000001") == 1e-6
    assert mc.parse_price(0) == 0.0


def test_days_only_window_is_whole_day():
    pricing = {
        "prompt": "0.00000099",
        "completion": "0.00000198",
        "overrides": [
            {
                "utc_days": ["saturday", "sunday"],
                "prompt": "0.00000066",
                "completion": "0.00000198",
            }
        ],
    }
    # Days-only windows cannot tile the week alone: schedule_error is set.
    eff = mc.effective_pricing(pricing, 1_000_000, 0.75)
    assert eff["schedule_error"] is True


def half_hour_pricing():
    """48 half-hour windows tiling the day (utc_days absent), alternating
    peak/off-peak -- a valid schedule whose 4-decimal-rounded coverages
    drift past 1e-3 (each 7*30/10080 = 0.020833.. rounds to 0.0208)."""
    overrides = []
    for i in range(48):
        start = (i // 2) * 100 + (i % 2) * 30
        end = ((i + 1) // 2) * 100 + ((i + 1) % 2) * 30
        overrides.append(
            {
                "utc_start": start,
                "utc_end": 0 if end == 2400 else end,
                "prompt": "0.000001" if i % 2 else "0.000002",
                "completion": "0.000004",
            }
        )
    return {"prompt": "0.000002", "completion": "0.000004", "overrides": overrides}


def test_half_hour_schedule_is_valid_despite_coverage_drift():
    eff = mc.effective_pricing(half_hour_pricing(), 1_000_000, 0.75)
    assert eff["schedule_error"] is False
    assert len(eff["schedule"]) == 48
    expected = sum(round(7 * 30 / 10080, 4) for _ in range(48))
    total = sum(w["coverage"] for w in eff["schedule"])
    assert total == pytest.approx(expected)
    assert abs(total - 1.0) > 1e-3  # the drift the site validator must tolerate


def test_deepseek_fixture_tiling_and_coverage():
    eff = mc.effective_pricing(deepseek_pricing(), 1_000_000, 0.75)
    assert eff["schedule_error"] is False
    schedule = eff["schedule"]
    assert len(schedule) == 6
    assert sorted(w["coverage"] for w in schedule) == [
        0.0298,
        0.0595,
        0.0893,
        0.1190,
        0.2857,
        0.4167,
    ]
    # unrounded coverages must sum to exactly 1 week (10080 minutes)
    minutes = sum(
        1440 * len(w["utc_days"])
        if w["utc_start"] == w["utc_end"]
        else len(w["utc_days"]) * mc._window_minutes(w["utc_start"], w["utc_end"])
        for w in schedule
    )
    assert minutes == 10080
    # frozen base = peak window prices (1.32e-6 / 3.96e-6)
    assert eff["base_price_in"] == 1.32e-6
    assert eff["base_price_out"] == 3.96e-6
    assert eff["price_in"] == 1.32e-6  # schedule-only: effective == peak
    assert eff["peak_blended"] == pytest.approx(0.75 * 1.32e-6 + 0.25 * 3.96e-6)
    assert eff["max_discount"] == 0.5


def test_hy4_preview_wrapping_windows():
    eff = mc.effective_pricing(hy4_pricing(), 1_000_000, 0.75)
    assert eff["schedule_error"] is False
    schedule = eff["schedule"]
    assert [(w["utc_start"], w["utc_end"], w["coverage"]) for w in schedule] == [
        (0, 1600, 0.6667),
        (1600, 0, 0.3333),
    ]
    assert eff["base_price_in"] == 8.34e-7  # peak = the 00:00-16:00 window
    assert eff["price_in"] == 8.34e-7
    assert eff["max_discount"] == 0.1


def test_invalid_schedule_fail_closed():
    # a window missing completion
    pricing = {
        "prompt": "0.000001",
        "completion": "0.000002",
        "overrides": [{"utc_start": 0, "utc_end": 1200, "prompt": "0.000001"}],
    }
    assert mc.effective_pricing(pricing, 1_000_000, 0.75)["schedule_error"] is True
    # a skipped (unknown-key) window leaves a tiling gap -> same cascade
    pricing2 = {
        "prompt": "0.000001",
        "completion": "0.000002",
        "overrides": [
            {"utc_start": 0, "utc_end": 1200, "prompt": "0.000001", "weird": 1},
            {
                "utc_start": 1200,
                "utc_end": 0,
                "prompt": "0.000001",
                "completion": "0.000002",
            },
        ],
    }
    assert mc.effective_pricing(pricing2, 1_000_000, 0.75)["schedule_error"] is True
    # a combined token+time entry invalidates the schedule
    pricing3 = {
        "prompt": "0.000001",
        "completion": "0.000002",
        "overrides": [
            {
                "min_prompt_tokens": 100000,
                "utc_start": 0,
                "utc_end": 0,
                "prompt": "0.000001",
                "completion": "0.000002",
            }
        ],
    }
    assert mc.effective_pricing(pricing3, 1_000_000, 0.75)["schedule_error"] is True


def test_mixed_model_uses_frozen_base():
    pricing = {
        "prompt": "0.0000001",
        "completion": "0.0000002",
        "overrides": [
            {
                "utc_start": 0,
                "utc_end": 1200,
                "prompt": "0.000001",
                "completion": "0.000002",
            },
            {
                "utc_start": 1200,
                "utc_end": 0,
                "prompt": "0.000002",
                "completion": "0.000004",
            },
            {"min_prompt_tokens": 100000, "prompt": "0.00001"},
        ],
    }
    eff = mc.effective_pricing(pricing, 1_000_000, 0.75)
    assert eff["schedule_error"] is False
    # frozen base = peak window (12:00-00:00: blended 0.75*2e-6+0.25*4e-6 = 2e-6
    # beats 00:00-12:00's 1.25e-6)
    assert eff["base_price_in"] == 2e-6
    assert eff["base_price_out"] == 4e-6
    # tier applies over the frozen base: prompt replaced, completion inherited
    assert eff["price_in"] == 1e-5
    assert eff["price_out"] == 4e-6
    assert eff["tier_prompt_tokens"] == 100000
    # schedule annotation carries base-layer window prices, not tier-adjusted
    assert eff["schedule"][0]["price_in"] == 1e-6
    assert eff["schedule"][1]["price_in"] == 2e-6


def test_schedule_only_effective_is_peak():
    eff = mc.effective_pricing(hy4_pricing(), 1_000_000, 0.75)
    assert eff["price_in"] == eff["base_price_in"] == 8.34e-7
    assert eff["price_out"] == eff["base_price_out"] == 2.501e-6


def test_peak_tie_breaks_to_first_in_api_order():
    # Two windows with equal blended price (0.75*in + 0.25*out == 1e-6):
    # A(1e-6, 1e-6) vs B(1.2e-6, 4e-7). First in API order wins the freeze.
    pricing = {
        "prompt": "0.000001",
        "completion": "0.000002",
        "overrides": [
            {
                "utc_start": 0,
                "utc_end": 1200,
                "prompt": "0.000001",
                "completion": "0.000001",
            },
            {
                "utc_start": 1200,
                "utc_end": 0,
                "prompt": "0.0000012",
                "completion": "0.0000004",
            },
        ],
    }
    eff = mc.effective_pricing(pricing, 1_000_000, 0.75)
    assert eff["base_price_in"] == 1e-6
    assert eff["base_price_out"] == 1e-6


def test_input_share_extremes_select_peak():
    pricing = {
        "prompt": "0.000001",
        "completion": "0.000002",
        "overrides": [
            {
                "utc_start": 0,
                "utc_end": 1200,
                "prompt": "0.000001",
                "completion": "0.000001",
            },
            {
                "utc_start": 1200,
                "utc_end": 0,
                "prompt": "0.000002",
                "completion": "0.0000005",
            },
        ],
    }
    # input_share=1.0: blended == input price -> window B (2e-6) is peak
    eff_in = mc.effective_pricing(pricing, 1_000_000, 1.0)
    assert eff_in["base_price_in"] == 2e-6
    assert eff_in["base_price_out"] == 5e-7
    # input_share=0.0: blended == output price -> window A (1e-6) is peak
    eff_out = mc.effective_pricing(pricing, 1_000_000, 0.0)
    assert eff_out["base_price_in"] == 1e-6
    assert eff_out["base_price_out"] == 1e-6


def test_max_discount_rounded_once_before_formatting():
    # off-peak at 0.6250001x of peak -> raw discount 0.3749999 -> rounds to 0.375
    pricing = {
        "prompt": "0.000001",
        "completion": "0.000001",
        "overrides": [
            {
                "utc_start": 0,
                "utc_end": 1200,
                "prompt": "0.000001",
                "completion": "0.000001",
            },
            {
                "utc_start": 1200,
                "utc_end": 0,
                "prompt": "0.0000006250001",
                "completion": "0.0000006250001",
            },
        ],
    }
    eff = mc.effective_pricing(pricing, 1_000_000, 0.75)
    assert eff["max_discount"] == 0.375
    assert mc.fmt_sched_note(eff["max_discount"]) == "-38%"


def test_zero_peak_blended_gives_zero_discount():
    pricing = {
        "prompt": "0",
        "completion": "0",
        "overrides": [
            {"utc_start": 0, "utc_end": 1200, "prompt": "0", "completion": "0"},
            {"utc_start": 1200, "utc_end": 0, "prompt": "0", "completion": "0"},
        ],
    }
    eff = mc.effective_pricing(pricing, 1_000_000, 0.75)
    assert eff["peak_blended"] == 0
    assert eff["max_discount"] == 0.0
    assert mc.fmt_sched_note(eff["max_discount"]) == "SCHED"


def test_fmt_tier_note_cases():
    assert mc.fmt_tier_note(100000) == ">100k"
    assert mc.fmt_tier_note(100001) == ">100k"
    assert mc.fmt_tier_note(272000) == ">272k"
    assert mc.fmt_tier_note(500) == ">500"
    assert mc.fmt_tier_note(None) is None


def test_fmt_sched_note_cases():
    assert mc.fmt_sched_note(0.375) == "-38%"
    assert mc.fmt_sched_note(0.0) == "SCHED"
    assert mc.fmt_sched_note(None) is None
    # the half-even boundary is pinned to Python's own formatting
    expected_pct = f"{0.005:.0%}"
    assert mc.fmt_sched_note(0.005) == (
        "SCHED" if expected_pct == "0%" else "-" + expected_pct
    )


def test_fmt_sched_detail_grammar():
    def window(days, start, end, price_in=1e-6, price_out=1e-6):
        return {
            "utc_days": days,
            "utc_start": start,
            "utc_end": end,
            "coverage": 0.5,
            "price_in": price_in,
            "price_out": price_out,
            "blended": price_in,
        }

    # daily, single off-peak wrap window
    assert mc.fmt_sched_detail([window(None, 1600, 0)], 2e-6) == "daily 16:00-00:00 UTC"
    # deepseek: weekday off-peak group + weekend whole-day group
    ds = mc.effective_pricing(deepseek_pricing(), 1_000_000, 0.75)
    detail = mc.fmt_sched_detail(ds["schedule"], ds["peak_blended"])
    assert (
        detail == "weekdays 00:00-01:00, 04:00-06:00, 10:00-00:00; weekends all day UTC"
    )
    # custom day set abbreviations, week order (parser-normalized input)
    custom = [
        window(["monday", "wednesday", "friday"], 800, 900),
    ]
    assert mc.fmt_sched_detail(custom, 2e-6) == "mon,wed,fri 08:00-09:00 UTC"
    # whole-day window renders "all day"
    wholeday = [window(["monday"], 0, 0)]
    assert mc.fmt_sched_detail(wholeday, 2e-6) == "mon all day UTC"
    # groups ordered by first weekday; windows ascending within a group
    multi = [
        window(["saturday", "sunday"], 1200, 1400),
        window(None, 100, 200),
    ]
    assert (
        mc.fmt_sched_detail(multi, 2e-6)
        == "daily 01:00-02:00; weekends 12:00-14:00 UTC"
    )
    # ASCII only
    assert all(s.isascii() for s in [detail])


def test_fmt_sched_detail_seven_days_is_daily():
    def window(days, start, end):
        return {
            "utc_days": days,
            "utc_start": start,
            "utc_end": end,
            "coverage": 0.3333,
            "price_in": 1e-6,
            "price_out": 1e-6,
            "blended": 1e-6,
        }

    week = list(mc.WEEKDAYS)
    # an explicit all-seven-days list labels exactly like utc_days absent
    assert mc.fmt_sched_detail([window(week, 0, 800)], 2e-6) == (
        "daily 00:00-08:00 UTC"
    )
    # ... and shares one group with absent-days windows
    mixed = [window(None, 1600, 0), window(week, 0, 800)]
    assert mc.fmt_sched_detail(mixed, 2e-6) == "daily 00:00-08:00, 16:00-00:00 UTC"


def test_fmt_sched_detail_none_cases():
    assert mc.fmt_sched_detail(None, None) is None
    # a schedule whose every window is at the peak price -> no off-peak -> None
    same = [
        {
            "utc_days": None,
            "utc_start": 0,
            "utc_end": 1200,
            "coverage": 0.5,
            "price_in": 1e-6,
            "price_out": 1e-6,
            "blended": 1e-6,
        },
        {
            "utc_days": None,
            "utc_start": 1200,
            "utc_end": 0,
            "coverage": 0.5,
            "price_in": 1e-6,
            "price_out": 1e-6,
            "blended": 1e-6,
        },
    ]
    assert mc.fmt_sched_detail(same, 1e-6) is None


# ---------------------------------------------------------------------------
# Tiered pricing: build_candidates integration
# ---------------------------------------------------------------------------


def test_candidates_carry_effective_and_base_prices():
    models = [make_model(id="acme/model-a", pricing=haiku_pricing())]
    candidates, dropped = mc.build_candidates(
        models, make_args(min_context=1_000_000), {}, {"acme/model-a"}, []
    )
    (cand,) = candidates
    assert cand["price_in"] == 0.5  # effective, USD per 1M
    assert cand["price_out"] == 2.5
    assert cand["blended"] == pytest.approx(0.75 * 0.5 + 0.25 * 2.5)
    assert cand["base_price_in"] == pytest.approx(0.1)  # base tier, USD per 1M
    assert cand["base_price_out"] == pytest.approx(0.5)
    assert cand["tier_prompt_tokens"] == 100000
    assert cand["tier_note"] == ">100k"
    assert cand["sched_note"] is None
    assert cand["sched_detail"] is None


def test_schedule_drop_reason():
    broken = {
        "prompt": "0.000001",
        "completion": "0.000002",
        "overrides": [{"utc_start": 0, "utc_end": 1200, "prompt": "0.000001"}],
    }
    models = [make_model(id="acme/model-a", pricing=broken)]
    candidates, dropped = mc.build_candidates(
        models, make_args(min_context=0), {}, {"acme/model-a"}, []
    )
    assert candidates == []
    assert dropped["schedule"] == 1
    # mixed model with an invalid schedule: no top-level-base fallback
    mixed = {
        "prompt": "0.000001",
        "completion": "0.000002",
        "overrides": [
            {"min_prompt_tokens": 100000, "prompt": "0.000002"},
            {"utc_start": 0, "utc_end": 1200, "prompt": "0.000001"},
        ],
    }
    models = [make_model(id="acme/model-b", pricing=mixed)]
    candidates, dropped = mc.build_candidates(
        models, make_args(min_context=0), {}, {"acme/model-b"}, []
    )
    assert candidates == []
    assert dropped["schedule"] == 1


def test_free_drop_runs_on_effective_prices():
    tiered_free = {
        "prompt": "0",
        "completion": "0",
        "overrides": [
            {
                "min_prompt_tokens": 100000,
                "prompt": "0.000001",
                "completion": "0.000002",
            }
        ],
    }
    models = [
        make_model(id="acme/tiered-free", pricing=tiered_free),
        make_model(id="acme/plain-free", pricing={"prompt": "0", "completion": "0"}),
    ]
    candidates, dropped = mc.build_candidates(
        models,
        make_args(min_context=1_000_000, exclude_free=True),
        {},
        {"acme/tiered-free", "acme/plain-free"},
        [],
    )
    assert [cand["id"] for cand in candidates] == ["acme/tiered-free"]
    assert dropped["free"] == 1


def test_base_validity_uses_base_prices():
    pricing = {
        "prompt": "nan",
        "completion": "0.000002",
        "overrides": [
            {
                "min_prompt_tokens": 100000,
                "prompt": "0.000002",
                "completion": "0.000004",
            }
        ],
    }
    models = [make_model(id="acme/model-a", pricing=pricing)]
    candidates, dropped = mc.build_candidates(
        models, make_args(min_context=1_000_000), {}, {"acme/model-a"}, []
    )
    assert candidates == []
    assert dropped["pricing"] == 1


def test_catalog_drop_reasons_constant():
    reasons = mc.CATALOG_DROP_REASONS
    assert "schedule" in reasons
    assert reasons.index("schedule") == reasons.index("pricing") + 1


def test_windowed_model_reports_raw_top_level_prices():
    # Spec rules 1+7: a windowed model's top-level prices are parsed and
    # validated, then discarded in favor of the frozen peak base. The parser
    # must therefore expose the RAW top level for the validity gate.
    pricing = {
        "prompt": "garbage",
        "completion": None,
        "overrides": [
            {
                "utc_start": 0,
                "utc_end": 1200,
                "prompt": "0.000001",
                "completion": "0.000002",
            },
            {
                "utc_start": 1200,
                "utc_end": 0,
                "prompt": "0.000001",
                "completion": "0.000002",
            },
        ],
    }
    eff = mc.effective_pricing(pricing, 1_000_000, 0.75)
    assert eff["schedule_error"] is False  # the windows themselves are valid
    assert eff["top_price_in"] is None  # "garbage" does not parse
    assert eff["top_price_out"] is None
    assert eff["base_price_in"] == 1e-6  # frozen peak base still drives pricing
    assert eff["price_in"] == 1e-6


def test_windowed_model_with_invalid_top_level_dropped():
    pricing = {
        "prompt": "garbage",
        "completion": None,
        "overrides": [
            {
                "utc_start": 0,
                "utc_end": 1200,
                "prompt": "0.000001",
                "completion": "0.000002",
            },
            {
                "utc_start": 1200,
                "utc_end": 0,
                "prompt": "0.000001",
                "completion": "0.000002",
            },
        ],
    }
    models = [make_model(id="acme/model-a", pricing=pricing)]
    candidates, dropped = mc.build_candidates(
        models, make_args(min_context=0), {}, {"acme/model-a"}, []
    )
    assert candidates == []
    assert dropped["pricing"] == 1


def test_schedules_skip_invalid_top_level_models():
    broken_top = {
        "prompt": "garbage",
        "completion": None,
        "overrides": [
            {
                "utc_start": 0,
                "utc_end": 1200,
                "prompt": "0.000001",
                "completion": "0.000002",
            },
            {
                "utc_start": 1200,
                "utc_end": 0,
                "prompt": "0.000001",
                "completion": "0.000002",
            },
        ],
    }
    doc = _doc_with(
        [
            make_model(id="acme/model-a"),
            make_model(id="acme/hy4-ok", pricing=hy4_pricing()),
            make_model(id="acme/hy4-broken", pricing=broken_top),
        ]
    )
    assert [s["id"] for s in doc["schedules"]] == ["acme/hy4-ok"]


# ---------------------------------------------------------------------------
# Provider-aware eval: precision ladder + first-party detection
# ---------------------------------------------------------------------------


def test_precision_ladder_fp6_outranks_int8():
    assert PRECISION_LADDER["fp6"] > PRECISION_LADDER["int8"]
    assert PRECISION_LADDER["fp32"] == max(PRECISION_LADDER.values())
    assert PRECISION_LADDER["int4"] == min(PRECISION_LADDER.values())
    assert "unknown" not in PRECISION_LADDER


def test_normalize_provider_strips_punctuation():
    assert normalize_provider("Z.AI") == normalize_provider("z-ai") == "zai"
    assert normalize_provider("AWS Bedrock") == normalize_provider("aws-bedrock")


def test_first_party_author_and_partner():
    assert is_first_party("Z.AI", "z-ai")
    assert is_first_party("AWS Bedrock", "anthropic")
    assert not is_first_party("OpenInference", "z-ai")


@pytest.mark.parametrize(
    ("provider_name", "author"),
    [
        ("Amazon Bedrock", "amazon"),
        ("Google AI Studio", "google"),
        ("Mistral", "mistralai"),
    ],
)
def test_first_party_alias_matches_verified_live_names(provider_name, author):
    # Live-verified §4.2 pairs (2026-10-20, task-13-fix-report.md 8b): these
    # makers self-name differently than their author slugs -- normalization
    # alone cannot bridge "Amazon Bedrock" to "amazon". The alias pair does.
    assert is_first_party(provider_name, author)
    # Both sides compare normalized, so case/punctuation variants match.
    assert is_first_party(provider_name.upper(), author.upper())
    assert is_first_party(provider_name.lower(), author.replace("-", "_"))


@pytest.mark.parametrize(
    ("provider_name", "author"),
    [
        ("Amazon Bedrock", "google"),
        ("Google AI Studio", "mistralai"),
        ("Mistral", "amazon"),
    ],
)
def test_first_party_alias_requires_author_match(provider_name, author):
    # The alias binds a provider name to ONE author slug: the same provider
    # name stays third-party for every other author (an alias must never
    # hand a model a first-party carve-out it did not verify for).
    assert not is_first_party(provider_name, author)


def test_default_precision_derivation_uses_alias():
    # The live amazon/nova-2-lite-v1 shape, offline: an all-unknown-quant
    # maker endpoint gets default_precision None (nothing ranks) and is
    # ineligible for a third-party author -- but eligible through the
    # carve-out for the alias-matched author, becoming the baseline.
    args = make_args()
    eps = mc.parse_endpoints(
        {
            "data": {
                "endpoints": [
                    _endpoint(
                        provider_name="Amazon Bedrock",
                        tag="amazon-bedrock",
                        quantization="unknown",
                    )
                ]
            }
        }
    )
    assert mc.derive_default_precision(eps, "amazon") is None
    # A different author gets no carve-out from this provider.
    assert not mc.baseline_eligible(eps[0], None, "acme", args)
    # The alias-matched author does, and the endpoint is the only baseline.
    assert mc.baseline_eligible(eps[0], None, "amazon", args)
    assert mc.pick_baseline(eps, [True], args)["tag"] == "amazon-bedrock"


# ---------------------------------------------------------------------------
# Provider-aware eval: endpoint fetch, parsing, variant + schedule gate
# ---------------------------------------------------------------------------


def _endpoint(**overrides):
    """One entry of OpenRouter's per-model endpoints list (live shape)."""
    base = {
        "provider_name": "Z.AI",
        "tag": "z-ai/fp8",
        "quantization": "fp8",
        "pricing": {"prompt": "0.00000015", "completion": "0.0000005"},
        "context_length": 262144,
        "status": 0,
        "service_tier": None,
    }
    base.update(overrides)
    return base


def test_parse_endpoints_keeps_verified_fields():
    # Live envelope (Task 13 lesson): the list lives ONLY at data.endpoints.
    # The first fixtures fed the models/find shape (a bare list under
    # "data"), so the suite stayed green while every live parse returned []
    # and the whole pool died as no_baseline_eligible_endpoint. No
    # shape-compat: every other shape keeps parsing to [] (fail closed).
    assert mc.parse_endpoints({"data": [_endpoint()]}) == []
    assert mc.parse_endpoints({"data": {"models": [{"slug": "acme/a"}]}}) == []
    assert mc.parse_endpoints({"data": None}) == []
    assert mc.parse_endpoints({}) == []
    assert mc.parse_endpoints([]) == []
    assert mc.parse_endpoints(None) == []
    eps = mc.parse_endpoints({"data": {"endpoints": [_endpoint()]}})
    assert eps[0]["provider_name"] == "Z.AI" and eps[0]["status"] == 0
    assert eps[0]["pricing"]["prompt"] == 0.00000015
    assert eps[0]["pricing"]["completion"] == 0.0000005
    assert eps[0]["tag"] == "z-ai/fp8"
    assert eps[0]["quantization"] == "fp8"
    assert eps[0]["context_length"] == 262144
    assert eps[0]["service_tier"] is None


def test_variant_detection_marker_and_zero_price():
    assert mc.is_variant_endpoint(_endpoint(tag="z-ai:free"))
    assert mc.is_variant_endpoint(_endpoint(tag="z-ai:batch"))  # marker, not price
    assert mc.is_variant_endpoint(_endpoint(pricing={"prompt": "0", "completion": "0"}))
    assert not mc.is_variant_endpoint(_endpoint())


def test_endpoint_schedule_gate_uses_full_schedule_error():
    assert mc.endpoint_schedule_ok(_endpoint(), make_args())
    broken = _endpoint(
        pricing={
            "prompt": "0.1",
            "completion": "0.1",
            "overrides": [
                # overrides entries are a LIST (live shape, haiku_pricing());
                # this one is a malformed window ("25:99" is no valid HHMM)
                {"utc_days": ["monday"], "utc_start": "25:99", "utc_end": "01:00"}
            ],
        }
    )
    assert not mc.endpoint_schedule_ok(broken, make_args())


def test_load_cache_expired_returns_stale_payload(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    mc.save_cache("k", {"v": 1})
    path = Path(mc.cache_path("k"))
    blob = json.loads(path.read_text())
    blob["fetched_at"] = time.time() - 99999
    path.write_text(json.dumps(blob))
    assert mc.load_cache("k", 3600) is None  # past TTL: the fresh load drops it
    assert mc.load_cache_expired("k") == {"v": 1}  # ...but the stale load keeps it


def test_fetch_model_endpoints_round_trip(monkeypatch):
    """Task 3 deferred coverage: the per-model cache round trip.

    Cache name flattens the model id ("/" -> "_"); --no-cache bypasses the
    read (refetches) but still writes; a second call serves from the cache
    without hitting the network again.
    """
    payload = {"data": {"endpoints": [_endpoint()]}}
    calls = []

    def fake_fetch_json(url, headers=None, timeout=30, retries=1):
        calls.append(url)
        return payload

    monkeypatch.setattr(mc, "fetch_json", fake_fetch_json)
    args = make_args(no_cache=False, cache_ttl=3600)
    eps = mc.fetch_model_endpoints("z-ai/glm", args)
    assert calls == [mc.OPENROUTER_ENDPOINTS_URL.format(id="z-ai/glm")]
    assert eps == mc.parse_endpoints(payload)
    # Saved under the flattened cache name ("/" -> "_", hyphen kept).
    assert Path(mc.cache_path("endpoints:z-ai_glm")).exists()
    # Second call: served from cache, no refetch.
    assert mc.fetch_model_endpoints("z-ai/glm", args) == eps
    assert len(calls) == 1
    # --no-cache bypasses the cache READ but the write still happens.
    assert mc.fetch_model_endpoints("z-ai/glm", make_args(no_cache=True)) == eps
    assert len(calls) == 2
    assert mc.fetch_model_endpoints("z-ai/glm", args) == eps
    assert len(calls) == 2  # the write refreshed the cache: still no refetch


# ---------------------------------------------------------------------------
# Provider-aware eval: default precision, eligibility, baseline selection
# ---------------------------------------------------------------------------


def _schedule_broken(**fields):
    """Endpoint whose pricing.overrides fails the schedule gate.

    Same malformed window as test_endpoint_schedule_gate_uses_full_schedule_error:
    "25:99" is no valid HHMM, so effective_pricing reports schedule_error.
    """
    fields.setdefault(
        "pricing",
        {
            "prompt": "0.00000015",
            "completion": "0.0000005",
            "overrides": [
                {"utc_days": ["monday"], "utc_start": "25:99", "utc_end": "01:00"}
            ],
        },
    )
    return _endpoint(**fields)


def test_maker_quantization_pins_default():
    eps = [
        _endpoint(provider_name="Z.AI", quantization="fp8"),
        _endpoint(provider_name="DeepInfra", quantization="fp4"),
    ]
    assert mc.derive_default_precision(eps, "z-ai") == "fp8"


def test_variant_and_schedule_broken_never_pin_default():
    eps = [
        _endpoint(provider_name="Z.AI", quantization="fp8", tag="z-ai:free"),
        _schedule_broken(provider_name="Z.AI", quantization="fp8"),
        _endpoint(provider_name="DeepInfra", quantization="fp4"),
    ]
    assert mc.derive_default_precision(eps, "z-ai") == "fp4"  # best-available fallback


def test_all_unknown_model_default_is_null():
    eps = [_endpoint(provider_name="Anthropic", quantization="unknown")]
    assert mc.derive_default_precision(eps, "anthropic") is None


def test_partner_never_pins_maker_default():
    # Fallback semantics (design section 4.1): a partner's known quantization
    # IS the best-available fallback when the maker itself has none -- partners
    # never pin via the MAKER rule, but they do feed the fallback universe.
    eps = [_endpoint(provider_name="AWS Bedrock", quantization="fp8")]
    assert mc.derive_default_precision(eps, "anthropic") == "fp8"


def test_maker_own_endpoint_wins_over_better_partner():
    # The other half of "partner never pins": the maker's OWN endpoint wins
    # the primary rule even when a partner offers a better-ranked one.
    eps = [
        _endpoint(provider_name="Z.AI", quantization="int8"),
        _endpoint(provider_name="DeepInfra", quantization="fp8"),
    ]
    assert mc.derive_default_precision(eps, "z-ai") == "int8"


def test_empty_provider_name_never_first_party():
    args = make_args()
    # is_first_party("", "") holds literally (Task 2 deferred note), but an
    # endpoint without a provider name must never be the maker nor count as
    # first-party -- fail-closed even when the author string is empty too.
    # A name that only normalizes to empty (" ") is guarded the same way.
    anon = _endpoint(provider_name=None, quantization="unknown")
    blank = _endpoint(provider_name=" ", quantization="unknown")
    assert not mc.baseline_eligible(anon, None, "", args)
    assert not mc.baseline_eligible(blank, None, "", args)
    # Matching no maker, it cannot pin one: an unknown quantization alone
    # leaves the default null.
    assert mc.derive_default_precision([anon], "") is None
    assert mc.derive_default_precision([blank], "") is None


def test_first_party_unknown_eligible_third_party_not():
    args = make_args()
    d = None
    assert mc.baseline_eligible(
        _endpoint(provider_name="Anthropic", quantization="unknown"),
        d,
        "anthropic",
        args,
    )
    assert not mc.baseline_eligible(
        _endpoint(provider_name="Relace", quantization="unknown"), d, "anthropic", args
    )
    # With a null default the rank branch is meaningless: only the carve-out
    # applies, so even a known third-party quantization is not eligible.
    assert not mc.baseline_eligible(
        _endpoint(provider_name="DeepInfra", quantization="fp8"), d, "anthropic", args
    )


def test_variant_never_baseline_eligible():
    args = make_args()
    assert not mc.baseline_eligible(_endpoint(tag="z-ai:free"), "fp8", "z-ai", args)
    assert not mc.baseline_eligible(
        _endpoint(pricing={"prompt": "0", "completion": "0"}), "fp8", "z-ai", args
    )
    assert not mc.baseline_eligible(_endpoint(status=1), "fp8", "z-ai", args)
    assert not mc.baseline_eligible(_schedule_broken(), "fp8", "z-ai", args)


def test_sub_default_quantization_not_eligible():
    args = make_args()
    # A known quantization at/above the default is eligible WITHOUT
    # first-party (the rank branch is provider-blind); below default is not.
    assert mc.baseline_eligible(
        _endpoint(provider_name="DeepInfra", quantization="fp8"),
        "fp8",
        "anthropic",
        args,
    )
    assert not mc.baseline_eligible(
        _endpoint(provider_name="DeepInfra", quantization="int8"),
        "fp8",
        "anthropic",
        args,
    )
    # A better-than-default endpoint clears the bar trivially.
    assert mc.baseline_eligible(
        _endpoint(provider_name="DeepInfra", quantization="fp8"),
        "int8",
        "anthropic",
        args,
    )


def test_unrecognized_quantization_fails_closed():
    args = make_args()
    # Neither a ladder label nor the "unknown" sentinel: no rank comparison
    # and no carve-out -- fail closed even for the maker itself.
    assert not mc.baseline_eligible(
        _endpoint(quantization="int16"), "fp8", "z-ai", args
    )
    assert (
        mc.derive_default_precision([_endpoint(quantization="int16")], "z-ai") is None
    )


def test_baseline_cheapest_with_pinned_tiebreak():
    args = make_args()

    def eligible(eps):
        return [mc.baseline_eligible(ep, "fp8", "z-ai", args) for ep in eps]

    # Cheapest blended price wins regardless of name order.
    eps = [
        _endpoint(
            provider_name="Alpha LLC",
            tag="alpha/fp8",
            pricing={"prompt": "0.0000002", "completion": "0.0000004"},
        ),
        _endpoint(
            provider_name="Zed Inc",
            tag="zed/fp8",
            pricing={"prompt": "0.0000001", "completion": "0.0000002"},
        ),
    ]
    assert mc.pick_baseline(eps, eligible(eps), args)["tag"] == "zed/fp8"

    # Equal price: the normalized provider name breaks the tie.
    eps = [
        _endpoint(provider_name="Zed Inc", tag="zed/fp8"),
        _endpoint(provider_name="Alpha LLC", tag="alpha/fp8"),
    ]
    assert mc.pick_baseline(eps, eligible(eps), args)["tag"] == "alpha/fp8"

    # Equal price and provider: the tag breaks the tie.
    eps = [
        _endpoint(provider_name="Alpha LLC", tag="alpha/zeta"),
        _endpoint(provider_name="Alpha LLC", tag="alpha/alpha"),
    ]
    assert mc.pick_baseline(eps, eligible(eps), args)["tag"] == "alpha/alpha"

    # Fully equal keys: first in list order wins (deterministic final tie).
    eps = [
        _endpoint(provider_name="Z.AI", tag="same"),
        _endpoint(provider_name="z ai", tag="same"),
    ]
    assert mc.pick_baseline(eps, eligible(eps), args)["provider_name"] == "Z.AI"

    # A non-eligible endpoint never wins, however cheap ("cheap:free" is a
    # :free variant marker, so baseline_eligible rejects it).
    eps = [
        _endpoint(tag="cheap:free"),
        _endpoint(provider_name="Zed Inc", tag="zed/fp8"),
    ]
    flags = eligible(eps)
    assert flags == [False, True]
    assert mc.pick_baseline(eps, flags, args)["tag"] == "zed/fp8"

    # Unpriceable (no parseable prices): skipped even when flagged eligible.
    assert mc.pick_baseline([_endpoint(pricing={})], [True], args) is None

    # Nothing eligible at all: no baseline (the caller drops the model).
    assert mc.pick_baseline(eps, [False, False], args) is None


# ---------------------------------------------------------------------------
# Provider-aware eval: annotations + steering builder
# ---------------------------------------------------------------------------


def test_price_spread_direction_and_null():
    args = make_args()
    # Completion = 2x prompt on every endpoint, so the blended ratio equals
    # the prompt-price ratio regardless of --input-share: eligible fp8 at
    # 0.15/M, non-eligible fp4 at 0.075/M -> spread = 0.075/0.15 = 0.5. A
    # second, pricier non-eligible endpoint pins "cheapest", not first/mean.
    eps = [
        _endpoint(
            provider_name="Z.AI",
            tag="z-ai/fp8",
            quantization="fp8",
            pricing={"prompt": "0.00000015", "completion": "0.0000003"},
        ),
        _endpoint(
            provider_name="DeepInfra",
            tag="deepinfra/fp4",
            quantization="fp4",
            pricing={"prompt": "0.000000075", "completion": "0.00000015"},
        ),
        _endpoint(
            provider_name="BetaHost",
            tag="beta/fp4",
            quantization="fp4",
            pricing={"prompt": "0.0000003", "completion": "0.0000006"},
        ),
    ]
    flags = [mc.baseline_eligible(ep, "fp8", "z-ai", args) for ep in eps]
    assert flags == [True, False, False]
    annotations = mc.build_annotations(eps, flags, eps[0], 262144, args)
    assert annotations["price_spread"] == pytest.approx(0.5)
    # The quantization-caused subset is the same two fp4 endpoints here.
    assert annotations["cheapest_quantized"] == {
        "provider": "DeepInfra",
        "quantization": "fp4",
        "price": pytest.approx(0.75 * 0.000000075 + 0.25 * 0.00000015),
    }

    # Null guard: an unpriceable endpoint on either side leaves no ratio.
    broken = _endpoint(
        provider_name="Broken",
        tag="broken/fp4",
        quantization="fp4",
        pricing={},
    )
    eps = [_endpoint(), broken]
    flags = [True, mc.baseline_eligible(broken, "fp8", "z-ai", args)]
    assert flags == [True, False]
    assert (
        mc.build_annotations(eps, flags, eps[0], 262144, args)["price_spread"] is None
    )
    # ...and on the eligible side (flags are contract inputs: an eligible
    # endpoint may still be unpriceable -- pick_baseline's None case).
    eps = [_endpoint(pricing={}), _endpoint(tag="second/fp4", quantization="fp4")]
    assert (
        mc.build_annotations(eps, [True, False], eps[0], 262144, args)["price_spread"]
        is None
    )


def test_price_spread_null_when_no_non_eligible():
    args = make_args()
    eps = [_endpoint(), _endpoint(provider_name="Alpha", tag="alpha/fp8")]
    flags = [mc.baseline_eligible(ep, "fp8", "z-ai", args) for ep in eps]
    assert flags == [True, True]
    # Empty non-eligible universe: nothing to compare against, and no
    # quantization-caused risk to report. Exact shape: exactly four keys.
    assert mc.build_annotations(eps, flags, eps[0], 262144, args) == {
        "price_spread": None,
        "cheapest_quantized": None,
        "risk_flag": False,
        "baseline_context_below_catalog": False,
    }


def test_cheapest_quantized_excludes_schedule_gated():
    args = make_args()
    eps = [
        _endpoint(),  # eligible maker baseline (blended 0.2375/M)
        _endpoint(
            provider_name="DeepInfra",
            tag="deepinfra/fp4",
            quantization="fp4",
            pricing={"prompt": "0.00000005", "completion": "0.0000001"},
        ),  # sub-default known quantization (blended 0.0625/M)
        _endpoint(
            provider_name="Relace",
            tag="relace/unknown",
            quantization="unknown",
            pricing={"prompt": "0.00000002", "completion": "0.00000004"},
        ),  # third-party unknown (blended 0.025/M)
        _schedule_broken(
            provider_name="CheapHost",
            tag="cheaphost/fp4",
            quantization="fp4",
            pricing={
                "prompt": "0.00000001",
                "completion": "0.00000002",
                "overrides": [
                    {"utc_days": ["monday"], "utc_start": "25:99", "utc_end": "01:00"}
                ],
            },
        ),  # schedule-gate-failed (blended 0.0125/M) -- in the universe, but
        # NOT in the quantization-caused subset despite being cheapest overall
    ]
    flags = [mc.baseline_eligible(ep, "fp8", "z-ai", args) for ep in eps]
    assert flags == [True, False, False, False]
    annotations = mc.build_annotations(eps, flags, eps[0], 262144, args)
    # Cheapest member of the quantization subset (fp4 @0.0625 vs unknown
    # @0.025): the third-party unknown endpoint -- the 0.0125 schedule-gated
    # endpoint must not win even though it is the cheapest in the universe.
    assert annotations["cheapest_quantized"] == {
        "provider": "Relace",
        "quantization": "unknown",
        "price": pytest.approx(0.75 * 0.00000002 + 0.25 * 0.00000004),
    }


def test_risk_flag_weight_formula():
    args = make_args()
    # eligible fp8 @0.15/M and @0.16/M vs non-eligible fp4 @0.0258/M:
    # 1/0.0258^2 (~1.5e3) >> 1/0.15^2 + 1/0.16^2 (~83.5) -> True. Equal
    # in/out prices keep each blended equal to its per-M headline price.
    eps = [
        _endpoint(
            provider_name="Z.AI",
            tag="z-ai/fp8",
            pricing={"prompt": "0.00000015", "completion": "0.00000015"},
        ),
        _endpoint(
            provider_name="Alpha LLC",
            tag="alpha/fp8",
            pricing={"prompt": "0.00000016", "completion": "0.00000016"},
        ),
        _endpoint(
            provider_name="DeepInfra",
            tag="deepinfra/fp4",
            quantization="fp4",
            pricing={"prompt": "0.0000000258", "completion": "0.0000000258"},
        ),
    ]
    flags = [mc.baseline_eligible(ep, "fp8", "z-ai", args) for ep in eps]
    assert flags == [True, True, False]
    assert mc.build_annotations(eps, flags, eps[0], 262144, args)["risk_flag"] is True

    # Eligible-only model: the non-eligible sum is empty -> False.
    eps = [_endpoint()]
    flags = [mc.baseline_eligible(eps[0], "fp8", "z-ai", args)]
    assert mc.build_annotations(eps, flags, eps[0], 262144, args)["risk_flag"] is False

    # Exact tie -> False (strict inequality). Identical pricing dicts make
    # the two blended prices bit-identical, so the sums are exactly equal:
    # Z.AI int8 (rank 0 >= rank 0) is eligible; DeepInfra int4 (rank -2 < 0)
    # is not.
    tie_pricing = {"prompt": "0.0000001", "completion": "0.0000002"}
    eps = [
        _endpoint(
            provider_name="Z.AI",
            tag="z-ai/int8",
            quantization="int8",
            pricing=tie_pricing,
        ),
        _endpoint(
            provider_name="DeepInfra",
            tag="deepinfra/int4",
            quantization="int4",
            pricing=tie_pricing,
        ),
    ]
    flags = [mc.baseline_eligible(ep, "int8", "z-ai", args) for ep in eps]
    assert flags == [True, False]
    assert mc.build_annotations(eps, flags, eps[0], 262144, args)["risk_flag"] is False

    # Determinism under malformed data: an unpriceable non-eligible endpoint
    # is skipped from BOTH sums, so it cannot manufacture a risk flag.
    eps = [
        _endpoint(),
        _endpoint(
            provider_name="Broken",
            tag="broken/fp4",
            quantization="fp4",
            pricing={},
        ),
    ]
    flags = [mc.baseline_eligible(ep, "fp8", "z-ai", args) for ep in eps]
    assert flags == [True, False]
    assert mc.build_annotations(eps, flags, eps[0], 262144, args)["risk_flag"] is False


def test_baseline_context_below_catalog_flag():
    args = make_args()
    baseline = _endpoint()  # context_length 262144
    eps, flags = [baseline], [True]
    # Smaller than the catalog context -> True (strict comparison).
    assert (
        mc.build_annotations(eps, flags, baseline, 1_000_000, args)[
            "baseline_context_below_catalog"
        ]
        is True
    )
    # Equal is not below -> False; larger -> False.
    assert (
        mc.build_annotations(eps, flags, baseline, 262144, args)[
            "baseline_context_below_catalog"
        ]
        is False
    )
    assert (
        mc.build_annotations(eps, flags, baseline, 131072, args)[
            "baseline_context_below_catalog"
        ]
        is False
    )
    # None baseline -> False (pinned); missing context on either side is
    # missing data, not a positive finding -> False.
    assert (
        mc.build_annotations(eps, flags, None, 1_000_000, args)[
            "baseline_context_below_catalog"
        ]
        is False
    )
    assert (
        mc.build_annotations(eps, flags, baseline, None, args)[
            "baseline_context_below_catalog"
        ]
        is False
    )
    assert (
        mc.build_annotations(
            eps, flags, _endpoint(context_length=None), 1_000_000, args
        )["baseline_context_below_catalog"]
        is False
    )


def test_steering_order_ignores_and_quantizations():
    args = make_args()
    eps = [
        _endpoint(provider_name="Z.AI", tag="z-ai/fp8"),  # eligible (0.2375/M)
        _endpoint(
            provider_name="Alpha LLC",
            tag="alpha/fp8",
            pricing={"prompt": "0.0000001", "completion": "0.0000002"},
        ),  # eligible, cheaper (0.1/M)
        _endpoint(
            provider_name="DeepInfra", tag="deepinfra/fp4", quantization="fp4"
        ),  # sub-default quantization
        _endpoint(
            provider_name="Relace", tag="relace/unknown", quantization="unknown"
        ),  # third-party unknown
        _endpoint(tag="z-ai:free"),  # live variant
        _schedule_broken(
            provider_name="CheapHost", tag="cheaphost/mxfp8", quantization="mxfp8"
        ),  # schedule-gate-failed (mxfp8: rank == fp8, but the gate fails)
        _endpoint(provider_name="GoneHost", tag="gone/fp8", status=1),  # not live
    ]
    flags = [mc.baseline_eligible(ep, "fp8", "z-ai", args) for ep in eps]
    assert flags == [True, True, False, False, False, False, False]

    steering = mc.build_steering("z-ai/glm-5.3-flash", eps, flags, "fp8", args)
    assert steering["model"] == "z-ai/glm-5.3-flash"
    provider = steering["provider"]
    # order: eligible tags cheapest first.
    assert provider["order"] == ["alpha/fp8", "z-ai/fp8"]
    # ignore: EVERY live non-eligible tag -- sub-default, third-party
    # unknown, schedule-gated AND the live variant; the non-live endpoint is
    # never listed. List order (the API's).
    assert provider["ignore"] == [
        "deepinfra/fp4",
        "relace/unknown",
        "z-ai:free",
        "cheaphost/mxfp8",
    ]
    # quantizations: KNOWN quants >= rank(fp8) among live non-variant
    # endpoints -- deduped, sorted; the schedule-gated mxfp8 still counts
    # (no schedule filter here), the fp4/unknown/non-live/variant do not.
    assert provider["quantizations"] == ["fp8", "mxfp8"]
    assert provider["allow_fallbacks"] is True

    # The annotations universe is the STRICTER set: live, non-variant AND
    # non-eligible -- the variant and the non-live endpoint stay out, unlike
    # in ignore.
    universe = mc.NON_ELIGIBLE_UNIVERSE(eps, flags)
    assert [ep["tag"] for ep in universe] == [
        "deepinfra/fp4",
        "relace/unknown",
        "cheaphost/mxfp8",
    ]

    # Price tie between eligible endpoints: (normalized provider name, tag)
    # breaks it -- the same comparator as pick_baseline.
    eps_tie = [
        _endpoint(provider_name="Zed Inc", tag="zed/fp8"),
        _endpoint(provider_name="Alpha LLC", tag="alpha/fp8"),
    ]
    flags_tie = [mc.baseline_eligible(ep, "fp8", "z-ai", args) for ep in eps_tie]
    steering = mc.build_steering("m", eps_tie, flags_tie, "fp8", args)
    assert steering["provider"]["order"] == ["alpha/fp8", "zed/fp8"]

    # An eligible endpoint with no computable price cannot be price-ranked;
    # it is appended last (still provider/tag-ordered), never dropped.
    eps_unpriced = [
        _endpoint(provider_name="Zed Inc", tag="zed/fp8"),
        _endpoint(provider_name="NoPrice", tag="noprice/fp8", pricing={}),
    ]
    flags_unpriced = [
        mc.baseline_eligible(ep, "fp8", "z-ai", args) for ep in eps_unpriced
    ]
    assert flags_unpriced == [True, True]
    steering = mc.build_steering("m", eps_unpriced, flags_unpriced, "fp8", args)
    assert steering["provider"]["order"] == ["zed/fp8", "noprice/fp8"]


def test_steering_omits_quantizations_when_default_null():
    args = make_args()
    # All-unknown closed model: the maker's endpoint is eligible via the
    # first-party carve-out, the third-party one is not. default_precision
    # None -> the quantizations key is OMITTED entirely, never an empty list
    # (OpenRouter semantics would filter out every endpoint).
    eps = [
        _endpoint(
            provider_name="Anthropic",
            tag="anthropic/unknown",
            quantization="unknown",
        ),
        _endpoint(
            provider_name="Relace",
            tag="relace/unknown",
            quantization="unknown",
        ),
        _endpoint(
            provider_name="Anthropic",
            tag="anthropic:free",
            quantization="unknown",
        ),  # live variant: ignored, never ordered
    ]
    flags = [mc.baseline_eligible(ep, None, "anthropic", args) for ep in eps]
    assert flags == [True, False, False]
    steering = mc.build_steering("anthropic/claude-haiku-5.5", eps, flags, None, args)
    provider = steering["provider"]
    assert provider["order"] == ["anthropic/unknown"]
    assert provider["ignore"] == ["relace/unknown", "anthropic:free"]
    assert "quantizations" not in provider
    assert provider["allow_fallbacks"] is True


def test_flag_list_mismatch_raises():
    # Parallel-list hardening (Task 4 review ruling): a flags list that does
    # not match endpoints must raise, never truncate silently.
    eps = [_endpoint(), _endpoint(tag="second/fp8")]
    with pytest.raises(ValueError):
        mc.build_annotations(eps, [True], _endpoint(), 262144, make_args())
    with pytest.raises(ValueError):
        mc.NON_ELIGIBLE_UNIVERSE(eps, [True])
    with pytest.raises(ValueError):
        mc.build_steering("m", eps, [True], "fp8", make_args())


# ---------------------------------------------------------------------------
# Provider-aware eval: CLI wiring (endpoint stage, drops, --json keys)
# ---------------------------------------------------------------------------


def _endpoint_stage_eps():
    """A maker fp8 endpoint plus a cheaper sub-default fp4 partner.

    The fp8 endpoint is the only eligible one and becomes the baseline; the
    fp4 endpoint feeds the annotations universe. Prices are chosen so the
    baseline blended price (0.25/M) is clearly below the models-API blended
    price a candidate carries before the stage runs (1.25/M with make_model's
    default pricing).

    Parsed through mc.parse_endpoints: this is the stand-in for
    fetch_model_endpoints' return value, which is parse_endpoints' output
    (numeric prices, the overrides key) -- not the raw API payload. The
    catalog document carries these dicts verbatim, and the site validator
    demands the parsed shape.
    """
    return mc.parse_endpoints(
        {
            "data": {
                "endpoints": [
                    _endpoint(
                        provider_name="Z.AI",
                        tag="z-ai/fp8",
                        quantization="fp8",
                        pricing={"prompt": "0.0000002", "completion": "0.0000004"},
                    ),
                    _endpoint(
                        provider_name="DeepInfra",
                        tag="deepinfra/fp4",
                        quantization="fp4",
                        pricing={"prompt": "0.0000001", "completion": "0.0000002"},
                    ),
                ]
            }
        }
    )


def _stub_endpoint_pipeline(
    monkeypatch, models, endpoints_by_id, fail_ids=(), frontend_models=()
):
    """Stub every fetcher so mc.run() exercises the endpoint stage offline."""
    zdr = {m["id"] for m in models}
    monkeypatch.setattr(mc, "fetch_openrouter_models", lambda a: (models, False))
    monkeypatch.setattr(
        mc,
        "fetch_openrouter_frontend",
        lambda a: ({}, zdr, {}, set(), list(frontend_models)),
    )
    monkeypatch.setattr(mc, "fetch_aa_entries", lambda a: ([], None, False, None))
    # Task 4's unkeyed chain hydrates quality from the published catalog;
    # these tests are about the endpoint/session-cost/zdr stages, so the
    # catalog stays unavailable (None) and the OR trio keeps covering.
    monkeypatch.setattr(mc, "fetch_quality_catalog", lambda a: None)
    calls = []

    def fake_fetch(model_id, args):
        calls.append(model_id)
        if model_id in fail_ids:
            raise RuntimeError("endpoint fetch failed")
        endpoints = endpoints_by_id[model_id]
        # Mirror fetch_model_endpoints: a successful fetch writes the cache
        # the stale-annotations rule reads later.
        mc.save_cache(f"endpoints:{model_id.replace('/', '_')}", endpoints)
        return endpoints

    monkeypatch.setattr(mc, "fetch_model_endpoints", fake_fetch)
    return calls


def test_pool_accounting_balances_with_endpoint_drops(monkeypatch, capsys):
    # run() now fetches session cost once per invocation (Task 8); pin the
    # keyless path so the suite never touches the network for it.
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    models = [
        make_model(id="z-ai/glm", name="Z.AI: GLM"),
        make_model(id="acme/plain", name="Acme: Plain"),
    ]
    endpoints_by_id = {
        "z-ai/glm": _endpoint_stage_eps(),
        # Third-party unknown-quantization endpoint: nothing eligible, no
        # baseline -> fail-closed drop.
        "acme/plain": [
            _endpoint(provider_name="Relace", tag="relace/x", quantization="unknown")
        ],
    }
    _stub_endpoint_pipeline(monkeypatch, models, endpoints_by_id)
    args = make_args(min_context=0, catalog=True)
    assert mc.run(args, models, False) == 0
    doc = json.loads(capsys.readouterr().out)
    pool = doc["pool"]
    assert pool["listed"] == len(models)
    assert pool["candidates"] == 1
    assert pool["listed"] == pool["candidates"] + sum(pool["dropped"].values())
    assert pool["dropped"]["no_baseline_eligible_endpoint"] == 1
    (entry,) = doc["filtered"]
    assert entry["id"] == "acme/plain"
    assert entry["name"] == "Acme: Plain"
    assert entry["reasons"] == ["no_baseline_eligible_endpoint"]
    # The fetch succeeded, so the cache written moments ago backs the
    # display annotations, marked stale (the model is not ranked on them).
    assert entry["annotations"]["stale"] is True
    # The producer and site validator must always agree on the contract.
    if bsd.CATALOG_SCHEMA_VERSION == mc.CATALOG_SCHEMA_VERSION:
        bsd.validate_catalog(doc)


def test_catalog_schema_versions_are_in_lockstep():
    assert mc.VERSION == "0.7.4"
    assert mc.CATALOG_QUALITY_SCHEMA_VERSION == mc.CATALOG_SCHEMA_VERSION == 6
    assert (
        mc.CATALOG_QUALITY_SCHEMA_VERSION
        == mc.CATALOG_SCHEMA_VERSION
        == bsd.CATALOG_SCHEMA_VERSION
    )


def test_empty_pool_after_endpoint_drops_warns_with_drop_counts(monkeypatch, capsys):
    # The pre-endpoint "no models satisfy the filters" wording would point
    # at filters that already passed; when the endpoint stage wipes the
    # pool, the warning must name the endpoint drop counts instead (Task
    # 13 fix: 73 no_baseline_eligible_endpoint used to print as a filter
    # complaint). The message is the operator's diagnosis entry point.
    models = [
        make_model(id="z-ai/glm", name="Z.AI: GLM"),
        make_model(id="acme/plain", name="Acme: Plain"),
    ]
    endpoints_by_id = {
        "z-ai/glm": [
            _endpoint(provider_name="Relace", tag="relace/x", quantization="unknown")
        ],
        "acme/plain": [
            _endpoint(provider_name="Relace", tag="relace/x", quantization="unknown")
        ],
    }
    _stub_endpoint_pipeline(monkeypatch, models, endpoints_by_id)
    args = make_args(min_context=0, catalog=True)
    assert mc.run(args, models, False) == 2
    captured = capsys.readouterr()
    assert captured.out == ""  # no document
    assert (
        "no models after the endpoint stage: "
        "no_baseline_eligible_endpoint=2, endpoint_fetch_failed=0" in captured.err
    )


def test_run_filters_at_requested_reasoning_effort(monkeypatch, capsys):
    # Without the run() wiring the metadata is unknown and the model would
    # be dropped by the effort filter; supporting max must keep it.
    model = make_model(
        id="z-ai/glm",
        name="Z.AI: GLM",
        reasoning={
            "mandatory": False,
            "default_enabled": True,
            "supported_efforts": ["max", "high"],
            "default_effort": "high",
        },
    )
    _stub_endpoint_pipeline(monkeypatch, [model], {"z-ai/glm": _endpoint_stage_eps()})
    args = make_args(min_context=0, reasoning_effort="max")
    assert mc.run(args, [model], False) == 0
    assert "z-ai/glm" in capsys.readouterr().out


def test_run_uses_frontend_reasoning_fallback_for_the_filter(monkeypatch, capsys):
    # The model has no models-API reasoning object; only the frontend says
    # it is non-reasoning, which is what makes --reasoning-effort none keep it.
    model = make_model(id="z-ai/glm", name="Z.AI: GLM")
    frontend = [
        {
            "slug": "z-ai/glm",
            "permaslug": "z-ai/glm-20261017",
            "supports_reasoning": False,
            "reasoning_config": {},
        }
    ]
    _stub_endpoint_pipeline(
        monkeypatch,
        [model],
        {"z-ai/glm": _endpoint_stage_eps()},
        frontend_models=frontend,
    )
    args = make_args(min_context=0, reasoning_effort="none")
    assert mc.run(args, [model], False) == 0
    assert "z-ai/glm" in capsys.readouterr().out


def test_candidate_pricing_replaced_by_baseline(monkeypatch):
    args = make_args(min_context=0)
    eps = _endpoint_stage_eps()
    monkeypatch.setattr(mc, "fetch_model_endpoints", lambda mid, a: eps)
    models = [make_model(id="z-ai/glm", name="Z.AI: GLM")]
    candidates, dropped = mc.build_candidates(models, args, {}, {"z-ai/glm"}, [])
    assert dropped == {}
    # Before the stage the candidate carries the models-API prices.
    assert candidates[0]["blended"] == pytest.approx(1.25)
    candidates, extra = mc.apply_endpoint_pricing(candidates, args, [])
    assert extra == {}
    cand = candidates[0]
    # Catalog pricing block == baseline endpoint prices, not models-API prices.
    assert cand["price_in"] == pytest.approx(0.2)
    assert cand["price_out"] == pytest.approx(0.4)
    assert cand["blended"] == pytest.approx(0.75 * 0.2 + 0.25 * 0.4)
    assert cand["base_price_in"] == pytest.approx(0.2)
    assert cand["base_price_out"] == pytest.approx(0.4)
    assert cand["tier_prompt_tokens"] is None
    assert cand["tiers"] == []
    assert cand["schedule"] is None
    assert cand["max_discount"] is None
    assert cand["tier_note"] is None
    assert cand["sched_note"] is None
    assert cand["sched_detail"] is None
    # The advertised models-API scalars are retained for comparison only.
    assert cand["advertised_pricing"] == {
        "input_per_1m": 1.0,
        "output_per_1m": 2.0,
        "blended_per_1m": 1.25,
        "source": "models-api",
    }
    assert cand["default_precision"] == "fp8"
    assert cand["endpoints"] == eps
    assert cand["annotations"] == {
        "price_spread": pytest.approx(0.5),
        "cheapest_quantized": {
            "provider": "DeepInfra",
            "quantization": "fp4",
            "price": pytest.approx(0.75 * 0.0000001 + 0.25 * 0.0000002),
        },
        "risk_flag": True,
        "baseline_context_below_catalog": True,
    }
    assert cand["steering"] == {
        "model": "z-ai/glm",
        "provider": {
            "order": ["z-ai/fp8"],
            "ignore": ["deepinfra/fp4"],
            "quantizations": ["fp8"],
            "allow_fallbacks": True,
        },
    }


def test_fetch_failure_drops_model_with_stale_annotations(monkeypatch):
    models = [
        make_model(id="z-ai/glm", name="Z.AI: GLM"),
        make_model(id="acme/nocache", name="Acme: No Cache"),
    ]
    args = make_args(min_context=0)

    def boom(model_id, a):
        raise RuntimeError("endpoint fetch failed")

    monkeypatch.setattr(mc, "fetch_model_endpoints", boom)
    # Only z-ai/glm has an (expired) cached endpoint list to fall back to.
    stale_eps = [_endpoint(provider_name="Z.AI", tag="z-ai/fp8", quantization="fp8")]
    mc.save_cache("endpoints:z-ai_glm", stale_eps)
    path = Path(mc.cache_path("endpoints:z-ai_glm"))
    blob = json.loads(path.read_text())
    blob["fetched_at"] = time.time() - 999999
    path.write_text(json.dumps(blob))

    filtered = []
    candidates, dropped = mc.build_candidates(
        models, args, {}, set(m["id"] for m in models), filtered
    )
    assert dropped == {}
    candidates, extra = mc.apply_endpoint_pricing(candidates, args, filtered)
    assert candidates == []
    assert dropped == {}
    assert extra == {"endpoint_fetch_failed": 2}
    by_id = {entry["id"]: entry for entry in filtered}
    glm = by_id["z-ai/glm"]
    assert glm["reasons"] == ["endpoint_fetch_failed"]
    assert glm["name"] == "Z.AI: GLM"
    assert glm["annotations"]["stale"] is True
    # Built from the stale cache: the single eligible endpoint yields no
    # non-eligible universe, so risk_flag is False.
    assert glm["annotations"]["risk_flag"] is False
    nocache = by_id["acme/nocache"]
    assert nocache["reasons"] == ["endpoint_fetch_failed"]
    assert "annotations" not in nocache


def test_json_rows_carry_new_additive_keys(monkeypatch, capsys):
    args = make_args(min_context=0)
    monkeypatch.setattr(
        mc, "fetch_model_endpoints", lambda mid, a: _endpoint_stage_eps()
    )
    models = [make_model(id="z-ai/glm", name="Z.AI: GLM")]
    candidates, _ = mc.build_candidates(models, args, {}, {"z-ai/glm"}, [])
    candidates, _ = mc.apply_endpoint_pricing(candidates, args, [])
    mc.compute_scores(candidates, args, {})
    mc.print_json(candidates)
    (row,) = json.loads(capsys.readouterr().out)
    # The missing-key check build_site_data does still passes, and the five
    # new keys close the row in the exact order Task 9's ROW_KEYS extension
    # pins (quant_note trails: Task 6b appended it after Task 6's four) --
    # with Task 12b's steering trailing them all (an extra key NOT in
    # ROW_KEYS; the row check is missing-key-only, so it passes through).
    assert all(key in row for key in bsd.ROW_KEYS)
    assert list(row)[-6:] == [
        "default_precision",
        "risk_flag",
        "price_spread",
        "session_cost_usd",
        "quant_note",
        "steering",
    ]
    assert row["default_precision"] == "fp8"
    assert row["risk_flag"] is True
    assert row["price_spread"] == pytest.approx(0.5)
    assert row["session_cost_usd"] is None  # Task 7 fills it


def test_json_rows_tolerate_pre_endpoint_candidates(capsys):
    # Candidates that never went through the endpoint stage (the historical
    # shape) still serialize: the new keys degrade to null.
    mc.print_json(_scored_candidates())
    for row in json.loads(capsys.readouterr().out):
        assert row["default_precision"] is None
        assert row["risk_flag"] is None
        assert row["price_spread"] is None
        assert row["session_cost_usd"] is None
        assert row["quant_note"] is None


# ---------------------------------------------------------------------------
# Provider-aware eval: CLI/site surface (PREC column, risk marker, quant_note)
# ---------------------------------------------------------------------------


def test_fmt_quant_note_formats_cheapest_non_baseline():
    # cheapest_quantized.price is stored per-token (blended, exactly as
    # build_annotations leaves it); the note shows USD per 1M like every
    # other price the CLI prints. The exact string is the site tooltip's
    # verbatim content (Task 11), so it is pinned here.
    annotations = {
        "price_spread": 0.17,
        "cheapest_quantized": {
            "provider": "open-inference",
            "quantization": "fp4",
            "price": 0.000000026,
        },
        "risk_flag": True,
        "baseline_context_below_catalog": False,
    }
    assert mc.fmt_quant_note(annotations) == (
        "cheapest non-baseline: fp4 @ open-inference, $0.026/M in (0.17x baseline)"
    )
    # Defensive: a quantized endpoint without a comparable spread drops the
    # parenthetical only.
    assert mc.fmt_quant_note({**annotations, "price_spread": None}) == (
        "cheapest non-baseline: fp4 @ open-inference, $0.026/M"
    )


def test_fmt_quant_note_none_without_non_eligible():
    # No non-eligible live endpoint: nothing cheaper to route to, no note.
    annotations = {
        "price_spread": None,
        "cheapest_quantized": None,
        "risk_flag": False,
        "baseline_context_below_catalog": False,
    }
    assert mc.fmt_quant_note(annotations) is None
    # Candidates that never went through the endpoint stage carry no
    # annotations at all.
    assert mc.fmt_quant_note({}) is None
    assert mc.fmt_quant_note(None) is None


def test_fmt_quant_note_renders_unknown_for_null_quantization():
    # A third-party null-quantization endpoint can be the cheapest
    # non-baseline (build_annotations puts third-party unknown/null in the
    # quantization-caused subset; baseline_eligible treats null == "unknown").
    # The design's sentinel must be rendered -- never Python's None repr,
    # which Task 11 would show verbatim on the site. Same guard for a null
    # provider_name on malformed data (Task 6's own fixture has that case).
    annotations = {
        "price_spread": 0.17,
        "cheapest_quantized": {
            "provider": "Relace",
            "quantization": None,
            "price": 0.000000026,
        },
        "risk_flag": True,
        "baseline_context_below_catalog": False,
    }
    assert mc.fmt_quant_note(annotations) == (
        "cheapest non-baseline: unknown @ Relace, $0.026/M in (0.17x baseline)"
    )
    assert mc.fmt_quant_note(
        {
            **annotations,
            "cheapest_quantized": {
                "provider": None,
                "quantization": None,
                "price": 0.000000026,
            },
        }
    ) == ("cheapest non-baseline: unknown @ unknown, $0.026/M in (0.17x baseline)")


def test_cli_table_shows_prec_column_and_risk_marker(monkeypatch, capsys):
    models = [
        make_model(id="z-ai/glm", name="Z.AI: GLM"),
        # First-party all-unknown model: eligibility survives the carve-out
        # while the default precision stays null (PREC cell "n/a").
        make_model(id="anthropic/claude", name="Anthropic: Claude"),
    ]

    def fake_fetch(model_id, a):
        if model_id == "z-ai/glm":
            return _endpoint_stage_eps()
        return [_endpoint(provider_name="Anthropic", quantization="unknown")]

    monkeypatch.setattr(mc, "fetch_model_endpoints", fake_fetch)
    args = make_args(min_context=0)
    candidates, _ = mc.build_candidates(models, args, {}, {m["id"] for m in models}, [])
    candidates, _ = mc.apply_endpoint_pricing(candidates, args, [])
    mc.compute_scores(candidates, args, {})
    mc.print_table(candidates[:2], mc.PRIORITY_WEIGHTS["balanced"], None)
    out = capsys.readouterr().out
    header = out.splitlines()[0]
    assert "PREC" in header
    assert header.index("DISC") < header.index("PREC") < header.index("TIER")
    risk_row = next(line for line in out.splitlines() if "z-ai/glm" in line)
    assert "fp8!" in risk_row
    plain_row = next(line for line in out.splitlines() if "anthropic/claude" in line)
    assert "n/a" in plain_row
    assert 'PREC      quoted precision ("!" = default routing favors cheaper' in out
    assert "sub-default endpoints)" in out


def test_json_rows_carry_quant_note(monkeypatch, capsys):
    args = make_args(min_context=0)
    monkeypatch.setattr(
        mc, "fetch_model_endpoints", lambda mid, a: _endpoint_stage_eps()
    )
    models = [make_model(id="z-ai/glm", name="Z.AI: GLM")]
    candidates, _ = mc.build_candidates(models, args, {}, {"z-ai/glm"}, [])
    candidates, _ = mc.apply_endpoint_pricing(candidates, args, [])
    mc.compute_scores(candidates, args, {})
    mc.print_json(candidates)
    (row,) = json.loads(capsys.readouterr().out)
    # Appended after Task 6's four keys; Task 9's ROW_KEYS extension pins
    # quant_note as the fifth (emit-first keeps the missing-key check green).
    # Task 12b's steering trails it as an extra key NOT in ROW_KEYS.
    assert list(row)[-2:] == ["quant_note", "steering"]
    assert row["quant_note"] == (
        "cheapest non-baseline: fp4 @ DeepInfra, $0.125/M in (0.50x baseline)"
    )


def test_json_rows_carry_steering_block(monkeypatch, capsys):
    args = make_args(min_context=0)
    monkeypatch.setattr(
        mc, "fetch_model_endpoints", lambda mid, a: _endpoint_stage_eps()
    )
    models = [make_model(id="z-ai/glm", name="Z.AI: GLM")]
    candidates, _ = mc.build_candidates(models, args, {}, {"z-ai/glm"}, [])
    candidates, _ = mc.apply_endpoint_pricing(candidates, args, [])
    mc.compute_scores(candidates, args, {})
    mc.print_json(candidates)
    (row,) = json.loads(capsys.readouterr().out)
    # The build_steering output shape (design section 4.7), verbatim from
    # the endpoint stage's attachment: the baseline-eligible fp8 endpoint
    # first in order, the live-but-ineligible fp4 endpoint ignored, and
    # quantizations at or above the fp8 default.
    assert row["steering"] == {
        "model": "z-ai/glm",
        "provider": {
            "order": ["z-ai/fp8"],
            "ignore": ["deepinfra/fp4"],
            "quantizations": ["fp8"],
            "allow_fallbacks": True,
        },
    }
    # Pre-endpoint candidates (historical test paths) degrade to None --
    # the same pattern as the other additive row keys.
    mc.print_json(_scored_candidates())
    by_model = {r["model"]: r for r in json.loads(capsys.readouterr().out)}
    assert by_model["acme/tiered"]["steering"] is None
    assert by_model["acme/plain"]["steering"] is None


def test_endpoint_stage_applies_results_in_candidate_order(monkeypatch):
    # Results must land per candidate id in the original candidates order,
    # never in completion order. The gate makes "a/slow" finish only after
    # "a/fast" has returned, so completion order is the reverse of input
    # order -- a completion-order implementation would swap the survivors.
    models = [make_model(id="a/slow"), make_model(id="a/fast")]
    args = make_args(min_context=0)
    gate = threading.Event()

    def fake_fetch(model_id, a):
        if model_id == "a/slow":
            assert gate.wait(timeout=5)
        else:
            gate.set()
        return [_endpoint(provider_name="A", tag="a/fp8", quantization="fp8")]

    monkeypatch.setattr(mc, "fetch_model_endpoints", fake_fetch)
    candidates, dropped = mc.build_candidates(
        models, args, {}, {"a/slow", "a/fast"}, []
    )
    assert dropped == {}
    survivors, extra = mc.apply_endpoint_pricing(candidates, args, [])
    assert extra == {}
    assert [c["id"] for c in survivors] == ["a/slow", "a/fast"]


def test_catalog_drop_reasons_carry_endpoint_stage_drops():
    # The registry stays load-bearing: pool.dropped zero-fills from it, and
    # the publish drop-rate gate reads the endpoint-derived reasons.
    for reason in ("no_baseline_eligible_endpoint", "endpoint_fetch_failed"):
        assert reason in mc.CATALOG_DROP_REASONS


def test_parse_args_openrouter_api_key_flag():
    args = mc.parse_args(["--openrouter-api-key", "sk-or"])
    assert args.openrouter_api_key == "sk-or"
    assert mc.parse_args([]).openrouter_api_key is None


# ---------------------------------------------------------------------------
# Tiered pricing: CLI display (TIER column, legend, --json keys)
# ---------------------------------------------------------------------------


def _scored_candidates():
    models = [
        make_model(id="acme/tiered", pricing=haiku_pricing()),
        make_model(id="acme/plain"),
    ]
    args = make_args(min_context=1_000_000)
    candidates, dropped = mc.build_candidates(
        models, args, {}, {"acme/tiered", "acme/plain"}, []
    )
    mc.compute_scores(candidates, args, {})
    return candidates


def test_json_additive_tier_keys(capsys):
    mc.print_json(_scored_candidates())
    by_model = {row["model"]: row for row in json.loads(capsys.readouterr().out)}
    tiered = by_model["acme/tiered"]
    assert tiered["pricing_tier_prompt_tokens"] == 100000
    assert tiered["base_input_usd_per_m"] == pytest.approx(0.1)
    assert tiered["base_output_usd_per_m"] == pytest.approx(0.5)
    assert tiered["tier_note"] == ">100k"
    assert tiered["sched_note"] is None
    assert tiered["sched_detail"] is None
    assert tiered["time_schedule"] is None
    plain = by_model["acme/plain"]
    assert plain["pricing_tier_prompt_tokens"] is None
    assert plain["tier_note"] is None
    assert plain["time_schedule"] is None


def test_json_time_schedule_string(capsys):
    models = [make_model(id="acme/hy4", pricing=hy4_pricing())]
    args = make_args(min_context=1_000_000)
    candidates, _ = mc.build_candidates(models, args, {}, {"acme/hy4"}, [])
    mc.compute_scores(candidates, args, {})
    mc.print_json(candidates)
    (row,) = json.loads(capsys.readouterr().out)
    assert row["sched_note"] == "-10%"
    assert row["sched_detail"] == "daily 16:00-00:00 UTC"
    assert row["time_schedule"] == "-10% daily 16:00-00:00 UTC"
    assert row["pricing_tier_prompt_tokens"] is None


def test_table_has_tier_column(capsys):
    mc.print_table(_scored_candidates(), mc.PRIORITY_WEIGHTS["balanced"], None)
    out = capsys.readouterr().out
    lines = out.splitlines()
    header = lines[0]
    assert "DISC" in header and "TIER" in header and "CTX" in header
    assert header.index("DISC") < header.index("TIER") < header.index("CTX")
    tiered_row = next(line for line in lines if "acme/tiered" in line)
    assert ">100k" in tiered_row
    plain_row = next(line for line in lines if "acme/plain" in line)
    assert ">100k" not in plain_row


@pytest.mark.parametrize("detail", ["expert", "simple"])
def test_print_table_legend_describes_best_selectable_depth(detail, capsys):
    mc.print_table(
        _scored_candidates(), mc.PRIORITY_WEIGHTS["balanced"], None, detail=detail
    )
    out = capsys.readouterr().out
    assert "best measured selectable depth" in out
    if detail == "expert":
        assert "OpenRouter's default (informational)" in out
    else:
        assert "suffix after · names the effort used for QUAL" in out


def test_print_table_expert_columns_and_legend(capsys):
    mc.print_table(_scored_candidates(), mc.PRIORITY_WEIGHTS["balanced"], None)
    out = capsys.readouterr().out
    header = out.splitlines()[0]
    for column in (
        "RANK",
        "MODEL",
        "QUAL",
        "$IN/M",
        "$OUT/M",
        "DISC",
        "PREC",
        "TIER",
        "CTX",
        "AGE",
        "SCORE",
    ):
        assert column in header
    assert (
        header.index("DISC")
        < header.index("PREC")
        < header.index("TIER")
        < header.index("CTX")
    )
    assert "PREC      quoted precision" in out
    assert "TIER      how prices vary" in out
    assert "SCORE     weighted blend of the four criteria above" in out
    assert "Weights: " in out


def test_print_table_simple_view_drops_expert_columns(capsys):
    mc.print_table(
        _scored_candidates(), mc.PRIORITY_WEIGHTS["balanced"], None, detail="simple"
    )
    out = capsys.readouterr().out
    header = out.splitlines()[0]
    assert "QUALITY" in header and "DISCOUNT" in header
    for column in ("PREC", "TIER", "CTX", "SCORE"):
        assert column not in header
    assert "RANK        position in this ranking" in out
    assert "IN/OUT $/M  USD per 1M tokens" in out
    assert "PREC" not in out


def test_print_table_color_rules(capsys):
    top = _scored_candidates()
    mc.print_table(top, mc.PRIORITY_WEIGHTS["balanced"], None, style=mc.Style(True))
    lines = capsys.readouterr().out.splitlines()
    header = lines[0]
    assert header.startswith("\x1b[1mRANK\x1b[0m")
    assert "SCORE" in header
    assert not re.search(r"\x1b\[1m\s*SCORE", header)
    first_row = next(line for line in lines if top[0]["id"] in line)
    assert "\x1b[32m" in first_row
    rank_cell = first_row.split("  ")[0]
    assert "\x1b[1m" in rank_cell  # rank is bold inside the green #1 row
    assert "\x1b[32m" in rank_cell
    second_row = next(line for line in lines if top[1]["id"] in line)
    assert "\x1b[32m" not in second_row


def test_print_table_green_wins_over_dim_on_top_row(capsys):
    top = _scored_candidates()
    top[0]["quality"] = None
    mc.print_table(top, mc.PRIORITY_WEIGHTS["balanced"], None, style=mc.Style(True))
    first_row = next(
        line for line in capsys.readouterr().out.splitlines() if top[0]["id"] in line
    )
    assert "\x1b[32m" in first_row
    assert "\x1b[2m" not in first_row


def test_print_table_color_preserves_alignment(capsys):
    top = _scored_candidates()
    mc.print_table(top, mc.PRIORITY_WEIGHTS["balanced"], None)
    plain = capsys.readouterr().out
    mc.print_table(top, mc.PRIORITY_WEIGHTS["balanced"], None, style=mc.Style(True))
    colored = capsys.readouterr().out
    assert re.sub(r"\x1b\[[0-9;]*m", "", colored) == plain


def test_print_table_dims_unknown_quality(capsys):
    top = _scored_candidates()
    top[1]["quality"] = None
    mc.print_table(top, mc.PRIORITY_WEIGHTS["balanced"], None, style=mc.Style(True))
    row = next(
        line for line in capsys.readouterr().out.splitlines() if top[1]["id"] in line
    )
    assert re.search(r"\x1b\[2m\s*-\x1b\[0m", row)


def test_print_table_yellow_quality_source_block_when_warn(capsys):
    note = {
        "sources": ["live AA (1 model, checked 13:15Z)"],
        "clauses": ["1 model unmatched (no quality data; quality score 0)"],
        "warn": True,
    }
    mc.print_table(
        _scored_candidates(),
        mc.PRIORITY_WEIGHTS["balanced"],
        note,
        style=mc.Style(True),
    )
    lines = capsys.readouterr().out.splitlines()
    source_line = next(line for line in lines if "Quality source:" in line)
    clause_line = next(line for line in lines if "unmatched" in line)
    for line in (source_line, clause_line):
        assert line.startswith("\x1b[33m")
        assert line.endswith("\x1b[0m")


def test_print_table_quality_source_not_yellow_when_not_warn(capsys):
    note = {
        "sources": ["live AA (1 model, checked 13:15Z)"],
        "clauses": [],
        "warn": False,
    }
    mc.print_table(
        _scored_candidates(),
        mc.PRIORITY_WEIGHTS["balanced"],
        note,
        style=mc.Style(True),
    )
    source_line = next(
        line
        for line in capsys.readouterr().out.splitlines()
        if "Quality source:" in line
    )
    assert "\x1b[33m" not in source_line


def test_print_table_dims_dash_rule_under_header(capsys):
    mc.print_table(
        _scored_candidates(),
        mc.PRIORITY_WEIGHTS["balanced"],
        None,
        style=mc.Style(True),
    )
    lines = capsys.readouterr().out.splitlines()
    rule = lines[1]
    assert rule.startswith("\x1b[2m")
    assert rule.endswith("\x1b[0m")
    stripped = rule.replace("\x1b[2m", "").replace("\x1b[0m", "").replace(" ", "")
    assert set(stripped) == {"-"}


def test_print_table_bolds_legend_terms(capsys):
    mc.print_table(
        _scored_candidates(),
        mc.PRIORITY_WEIGHTS["balanced"],
        None,
        style=mc.Style(True),
    )
    out = capsys.readouterr().out
    assert re.search(r"\x1b\[1mQUALITY\s*\x1b\[0m", out)
    assert re.search(r"\x1b\[1mRANK\s*\x1b\[0m", out)


def test_print_table_bare_prints_table_only(capsys):
    mc.print_table(
        _scored_candidates(), mc.PRIORITY_WEIGHTS["balanced"], None, bare=True
    )
    out = capsys.readouterr().out.rstrip("\n")
    assert out.splitlines()[0].startswith("RANK")
    assert "Weights:" not in out
    assert "QUALITY" not in out  # no legend
    assert "SCORE     weighted" not in out


class _FakeStream:
    def __init__(self, tty):
        self._tty = tty

    def isatty(self):
        return self._tty


def test_style_disabled_passes_text_through():
    style = mc.Style(False)
    for method in ("bold", "dim", "green", "yellow"):
        assert getattr(style, method)("text") == "text"


def test_style_enabled_wraps_in_sgr():
    style = mc.Style(True)
    assert style.bold("RANK") == "\x1b[1mRANK\x1b[0m"
    assert style.dim("-") == "\x1b[2m-\x1b[0m"
    assert style.green("row") == "\x1b[32mrow\x1b[0m"
    assert style.yellow("warn") == "\x1b[33mwarn\x1b[0m"


def test_color_auto_follows_tty(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    assert mc.color_enabled("auto", _FakeStream(True)) is True
    assert mc.color_enabled("auto", _FakeStream(False)) is False


def test_color_auto_honors_nonempty_no_color(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    assert mc.color_enabled("auto", _FakeStream(True)) is False


def test_color_auto_ignores_empty_no_color(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "")
    assert mc.color_enabled("auto", _FakeStream(True)) is True


def test_color_on_overrides_no_color(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    assert mc.color_enabled("on", _FakeStream(False)) is True


def test_color_off_forces_plain():
    assert mc.color_enabled("off", _FakeStream(True)) is False


class _TTYStringIO(io.StringIO):
    """StringIO that pretends to be a terminal for color_enabled(auto)."""

    def isatty(self):
        return True


def test_run_auto_colors_when_streams_are_ttys(monkeypatch):
    models = [
        make_model(id="acme/model-a", name="Acme: A"),
        make_model(id="acme/short", name="Acme: Short", context_length=1000),
    ]
    _stub_quality_run(monkeypatch, models, aa_raises=True)
    procs = _stub_pager(monkeypatch)
    out, err = _TTYStringIO(), _TTYStringIO()
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)
    monkeypatch.delenv("NO_COLOR", raising=False)
    assert mc.run(make_args(min_context=1_000_000), models, False) == 0
    (proc,) = procs  # the human path paged; bold header went to the pager
    assert "\x1b[1m" in proc.stdin.getvalue()
    assert "\x1b[2m(dropped: 1 context)\x1b[0m" in err.getvalue()


def test_run_auto_honors_no_color_when_streams_are_ttys(monkeypatch):
    models = [
        make_model(id="acme/model-a", name="Acme: A"),
        make_model(id="acme/short", name="Acme: Short", context_length=1000),
    ]
    _stub_quality_run(monkeypatch, models, aa_raises=True)
    procs = _stub_pager(monkeypatch)
    out, err = _TTYStringIO(), _TTYStringIO()
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)
    monkeypatch.setenv("NO_COLOR", "1")
    assert mc.run(make_args(min_context=1_000_000), models, False) == 0
    (proc,) = procs
    assert "\x1b[" not in proc.stdin.getvalue()
    assert "\x1b[" not in err.getvalue()


# ---------------------------------------------------------------------------
# pager
# ---------------------------------------------------------------------------


def test_resolve_pager_off_when_no_pager_flag(monkeypatch):
    monkeypatch.setattr(sys, "stdout", _FakeStream(True))
    assert mc.resolve_pager(make_args(no_pager=True)) is None


def test_resolve_pager_off_without_tty(monkeypatch):
    monkeypatch.setattr(sys, "stdout", _FakeStream(False))
    monkeypatch.delenv("PAGER", raising=False)
    assert mc.resolve_pager(make_args()) is None


def test_resolve_pager_less_fallback_without_pager_env(monkeypatch):
    monkeypatch.setattr(sys, "stdout", _FakeStream(True))
    monkeypatch.delenv("PAGER", raising=False)
    assert mc.resolve_pager(make_args()) == [
        "less",
        "--quit-if-one-screen",
        "--RAW-CONTROL-CHARS",
        "--no-init",
    ]


@pytest.mark.parametrize("empty", ["", "   "])
def test_resolve_pager_empty_env_counts_as_unset(monkeypatch, empty):
    monkeypatch.setattr(sys, "stdout", _FakeStream(True))
    monkeypatch.setenv("PAGER", empty)
    assert mc.resolve_pager(make_args())[0] == "less"


def test_resolve_pager_uses_pager_env_verbatim(monkeypatch):
    monkeypatch.setattr(sys, "stdout", _FakeStream(True))
    monkeypatch.setenv("PAGER", "pager -X -R")
    assert mc.resolve_pager(make_args()) == ["pager", "-X", "-R"]


@pytest.mark.parametrize("malformed", ['less "', "a\\"])
def test_resolve_pager_malformed_env_counts_as_unset(monkeypatch, malformed):
    # shlex.split raises ValueError on unbalanced quotes / trailing
    # backslashes; a hostile $PAGER must degrade to the fallback, never
    # abort the run (spec invariant: pager startup problems never lose
    # output).
    monkeypatch.setattr(sys, "stdout", _FakeStream(True))
    monkeypatch.setenv("PAGER", malformed)
    assert mc.resolve_pager(make_args()) == [
        "less",
        "--quit-if-one-screen",
        "--RAW-CONTROL-CHARS",
        "--no-init",
    ]


class _PagerStdin(io.StringIO):
    """Pager-stdin double: close() records itself instead of sealing the buffer.

    The real contract is that the pager's stdin is closed after the output
    (the EOF signal); a plain StringIO would then refuse getvalue(), so the
    double keeps the bytes readable for assertions.
    """

    def __init__(self):
        super().__init__()
        self.close_called = False

    def close(self):
        self.close_called = True


class _FakePagerProc:
    """Minimal Popen double: writable stdin, poll/terminate/wait bookkeeping."""

    def __init__(self, argv):
        self.argv = argv
        self.stdin = _PagerStdin()
        self.terminated = False
        self.reaped = False

    def poll(self):
        return 0 if self.reaped else None

    def terminate(self):
        self.terminated = True

    def wait(self):
        self.reaped = True
        return 0


def _stub_pager(monkeypatch):
    """Replace subprocess.Popen with a fake pager; return its procs list.

    The list fills as the code under test spawns, so callers that only
    assert on spawns must unpack after the call.
    """
    procs = []

    def fake_popen(argv, **kwargs):
        procs.append(_FakePagerProc(argv))
        return procs[-1]

    monkeypatch.setattr(mc.subprocess, "Popen", fake_popen)
    return procs


def _stub_pager_proc(monkeypatch, proc):
    """Like _stub_pager but every spawn returns the given pre-built double.

    For tests that must customize the double (stdin, wait) BEFORE entering
    the pager context.
    """

    def fake_popen(argv, **kwargs):
        proc.argv = argv
        return proc

    monkeypatch.setattr(mc.subprocess, "Popen", fake_popen)


def test_paged_stdout_swaps_swaps_back_and_reaps(monkeypatch):
    proc = _FakePagerProc(["less"])
    _stub_pager_proc(monkeypatch, proc)
    original = sys.stdout
    with mc.paged_stdout(["less"]):
        assert sys.stdout is proc.stdin
        print("hello")
    assert sys.stdout is original
    assert proc.stdin.getvalue() == "hello\n"
    assert proc.stdin.close_called is True  # EOF signaled to the pager
    assert proc.reaped is True and proc.terminated is False


def test_paged_stdout_spawn_failure_yields_false(monkeypatch, capsys):
    def broken_popen(argv, **kwargs):
        raise OSError("no pager")

    monkeypatch.setattr(mc.subprocess, "Popen", broken_popen)
    with mc.paged_stdout(["less"]) as paged:
        assert paged is False
        print("direct")
    assert capsys.readouterr().out == "direct\n"


def test_paged_stdout_no_argv_never_spawns(monkeypatch):
    spawned = []

    def spy_popen(argv, **kwargs):
        spawned.append(argv)
        raise AssertionError("Popen must not run without a pager argv")

    monkeypatch.setattr(mc.subprocess, "Popen", spy_popen)
    with mc.paged_stdout(None) as paged:
        assert paged is False
        print("direct")
    assert spawned == []


def test_paged_stdout_sets_less_frx_for_child_when_unset(monkeypatch):
    # A bare `PAGER=less` reads no flags and would show raw SGR as junk
    # (the human-reported bug). Git's convention: give the child
    # LESS=FRX when the user has none — colors render, one-screen output
    # exits immediately, the stderr notes survive quitting.
    captured = {}

    def spy_popen(argv, **kwargs):
        captured["env"] = kwargs.get("env")
        return _FakePagerProc(argv)

    monkeypatch.setattr(mc.subprocess, "Popen", spy_popen)
    monkeypatch.delenv("LESS", raising=False)
    with mc.paged_stdout(["less"]):
        pass
    assert captured["env"]["LESS"] == "FRX"


def test_paged_stdout_keeps_user_less_env(monkeypatch):
    # The user's own $LESS always wins: inherit the environment untouched.
    captured = {}

    def spy_popen(argv, **kwargs):
        captured["env"] = kwargs.get("env")
        return _FakePagerProc(argv)

    monkeypatch.setattr(mc.subprocess, "Popen", spy_popen)
    monkeypatch.setenv("LESS", "-R")
    with mc.paged_stdout(["less"]):
        pass
    assert captured["env"] is None


def test_paged_stdout_keeps_empty_less_env(monkeypatch):
    # git parity: set-but-empty $LESS counts as set -- the child inherits it
    # untouched; only a genuinely unset $LESS triggers the LESS=FRX injection.
    captured = {}

    def spy_popen(argv, **kwargs):
        captured["env"] = kwargs.get("env")
        return _FakePagerProc(argv)

    monkeypatch.setattr(mc.subprocess, "Popen", spy_popen)
    monkeypatch.setenv("LESS", "")
    with mc.paged_stdout(["less"]):
        pass
    assert captured["env"] is None


def test_paged_stdout_pipe_encoding_defaults_to_utf8(monkeypatch):
    # The pager pipe must write with the encoding the direct path would have
    # used, not the locale's: Popen's text mode uses
    # locale.getpreferredencoding, which can disagree with sys.stdout.encoding
    # (e.g. PYTHONIOENCODING=utf-8 over a latin-1 locale) and then a non-ASCII
    # model name raises UnicodeEncodeError mid-table only when paged.
    # _TTYStringIO carries StringIO's encoding=None -> the utf-8 fallback.
    captured = {}

    def spy_popen(argv, **kwargs):
        captured.update(kwargs)
        return _FakePagerProc(argv)

    monkeypatch.setattr(mc.subprocess, "Popen", spy_popen)
    monkeypatch.setattr(sys, "stdout", _TTYStringIO())
    with mc.paged_stdout(["less"]):
        pass
    assert captured.get("encoding") == "utf-8"


def test_paged_stdout_pipe_encoding_follows_stdout(monkeypatch):
    # A stream with its own encoding pins the pipe's encoding (and error
    # handler) to exactly what a direct print would have used.
    captured = {}

    def spy_popen(argv, **kwargs):
        captured.update(kwargs)
        return _FakePagerProc(argv)

    monkeypatch.setattr(mc.subprocess, "Popen", spy_popen)

    class _AsciiTTY(_TTYStringIO):
        encoding = "ascii"

    fake = _AsciiTTY()
    monkeypatch.setattr(sys, "stdout", fake)
    expected = (fake.encoding, fake.errors)
    with mc.paged_stdout(["less"]):
        pass
    assert (captured.get("encoding"), captured.get("errors")) == expected


def test_paged_stdout_block_interrupt_terminates_and_signals(monkeypatch):
    proc = _FakePagerProc(["less"])
    _stub_pager_proc(monkeypatch, proc)
    with pytest.raises(mc.PagerInterrupted):
        with mc.paged_stdout(["less"]):
            raise KeyboardInterrupt
    assert proc.terminated is True and proc.reaped is True


def test_paged_stdout_close_broken_pipe_is_swallowed(monkeypatch):
    proc = _FakePagerProc(["less"])
    _stub_pager_proc(monkeypatch, proc)

    class _EpipeClose(io.StringIO):
        # Fire EPIPE once -- the CM's guarded close; a later close (the
        # destructor's) must seal quietly or GC logs an unraisable error.
        def __init__(self):
            super().__init__()
            self.epipe_fired = False

        def close(self):
            if not self.epipe_fired:
                self.epipe_fired = True
                raise BrokenPipeError
            super().close()

    proc.stdin = _EpipeClose()
    with mc.paged_stdout(["less"]):
        pass
    assert proc.reaped is True


def test_paged_stdout_close_interrupt_terminates_and_signals(monkeypatch):
    # A Ctrl-C landing inside the guarded stdin close() must count as an
    # interrupt: terminate + reap the pager and surface PagerInterrupted
    # (run() -> 130), never a raw KeyboardInterrupt traceback.
    proc = _FakePagerProc(["less"])
    _stub_pager_proc(monkeypatch, proc)

    class _InterruptClose(_PagerStdin):
        # Fire KI once -- the CM's guarded close; a later close (the
        # destructor's) must seal quietly or GC logs an unraisable error.
        def __init__(self):
            super().__init__()
            self.ki_fired = False

        def close(self):
            if not self.ki_fired:
                self.ki_fired = True
                raise KeyboardInterrupt
            super().close()

    proc.stdin = _InterruptClose()
    original = sys.stdout
    with pytest.raises(mc.PagerInterrupted):
        with mc.paged_stdout(["less"]):
            pass
    assert sys.stdout is original
    assert proc.terminated is True and proc.reaped is True


def test_paged_stdout_wait_interrupt_exits_via_sentinel(monkeypatch):
    proc = _FakePagerProc(["less"])
    _stub_pager_proc(monkeypatch, proc)
    raised = []

    def flaky_wait():
        if not raised:
            raised.append(True)
            raise KeyboardInterrupt
        proc.reaped = True
        return 0

    proc.wait = flaky_wait
    with pytest.raises(mc.PagerInterrupted):
        with mc.paged_stdout(["less"]):
            pass
    assert proc.terminated is True


def test_run_pages_human_output_on_tty(monkeypatch):
    models = [
        make_model(id="acme/model-a", name="Acme: A"),
        make_model(id="acme/short", name="Acme: Short", context_length=1000),
    ]
    _stub_quality_run(monkeypatch, models, aa_raises=True)
    procs = _stub_pager(monkeypatch)
    monkeypatch.setattr(sys, "stdout", _TTYStringIO())
    monkeypatch.setattr(sys, "stderr", _TTYStringIO())
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("PAGER", raising=False)
    assert mc.run(make_args(min_context=1_000_000), models, False) == 0
    (proc,) = procs
    assert proc.argv == [
        "less",
        "--quit-if-one-screen",
        "--RAW-CONTROL-CHARS",
        "--no-init",
    ]
    assert "RANK" in proc.stdin.getvalue()


@pytest.mark.parametrize("mode", [{"best": True}, {"json": True}, {"catalog": True}])
def test_run_machine_modes_never_page(monkeypatch, mode):
    models = [make_model(id="acme/model-a", name="Acme: A")]
    _stub_quality_run(monkeypatch, models, aa_raises=True)
    procs = _stub_pager(monkeypatch)
    out, err = _TTYStringIO(), _TTYStringIO()
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)
    assert mc.run(make_args(min_context=1_000_000, **mode), models, False) == 0
    assert procs == []
    assert out.getvalue() != ""


def test_run_never_pages_without_tty(monkeypatch, capsys):
    models = [make_model(id="acme/model-a", name="Acme: A")]
    _stub_quality_run(monkeypatch, models, aa_raises=True)
    procs = _stub_pager(monkeypatch)
    assert mc.run(make_args(min_context=1_000_000), models, False) == 0
    assert procs == []
    assert "RANK" in capsys.readouterr().out


def test_run_no_pager_flag_prints_directly(monkeypatch):
    models = [make_model(id="acme/model-a", name="Acme: A")]
    _stub_quality_run(monkeypatch, models, aa_raises=True)
    procs = _stub_pager(monkeypatch)
    out, err = _TTYStringIO(), _TTYStringIO()
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)
    assert mc.run(make_args(min_context=1_000_000, no_pager=True), models, False) == 0
    assert procs == []
    assert "RANK" in out.getvalue()


def test_run_pager_start_failure_prints_directly(monkeypatch):
    models = [make_model(id="acme/model-a", name="Acme: A")]
    _stub_quality_run(monkeypatch, models, aa_raises=True)

    def broken_popen(argv, **kwargs):
        raise OSError("no pager binary")

    monkeypatch.setattr(mc.subprocess, "Popen", broken_popen)
    out, err = _TTYStringIO(), _TTYStringIO()
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)
    assert mc.run(make_args(min_context=1_000_000), models, False) == 0
    assert "RANK" in out.getvalue()


def test_run_malformed_pager_env_degrades_to_fallback(monkeypatch):
    models = [make_model(id="acme/model-a", name="Acme: A")]
    _stub_quality_run(monkeypatch, models, aa_raises=True)
    procs = _stub_pager(monkeypatch)
    monkeypatch.setattr(sys, "stdout", _TTYStringIO())
    monkeypatch.setattr(sys, "stderr", _TTYStringIO())
    monkeypatch.setenv("PAGER", 'less "')
    assert mc.run(make_args(min_context=1_000_000), models, False) == 0
    (proc,) = procs
    assert proc.argv == [
        "less",
        "--quit-if-one-screen",
        "--RAW-CONTROL-CHARS",
        "--no-init",
    ]
    assert "RANK" in proc.stdin.getvalue()


def test_run_pager_early_quit_does_not_crash(monkeypatch):
    models = [make_model(id="acme/model-a", name="Acme: A")]
    _stub_quality_run(monkeypatch, models, aa_raises=True)

    class _ExplodingSink:
        def write(self, text):
            raise BrokenPipeError

        def close(self):
            pass

    proc = _FakePagerProc(["less"])
    _stub_pager_proc(monkeypatch, proc)
    proc.stdin = _ExplodingSink()
    out, err = _TTYStringIO(), _TTYStringIO()
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)
    assert mc.run(make_args(min_context=1_000_000), models, False) == 0
    assert proc.reaped is True


def test_run_pager_exit_status_not_propagated(monkeypatch):
    class _FailingPagerProc(_FakePagerProc):
        def wait(self):
            self.reaped = True
            return 1  # pager exited nonzero; run()'s own status is unaffected

    def fake_popen(argv, **kwargs):
        return _FailingPagerProc(argv)

    monkeypatch.setattr(mc.subprocess, "Popen", fake_popen)
    models = [make_model(id="acme/model-a", name="Acme: A")]
    _stub_quality_run(monkeypatch, models, aa_raises=True)
    monkeypatch.setattr(sys, "stdout", _TTYStringIO())
    monkeypatch.setattr(sys, "stderr", _TTYStringIO())
    assert mc.run(make_args(min_context=1_000_000), models, False) == 0


def test_run_pager_off_peak_and_bare(monkeypatch):
    models = [
        make_model(id="acme/deal", pricing=hy4_pricing(), context_length=262144),
        make_model(id="acme/plain"),
    ]
    _stub_quality_run(monkeypatch, models, aa_raises=True)
    procs = _stub_pager(monkeypatch)
    monkeypatch.setattr(sys, "stdout", _TTYStringIO())
    monkeypatch.setattr(sys, "stderr", _TTYStringIO())
    assert (
        mc.run(
            make_args(off_peak=True, bare=True, min_context=1_000_000), models, False
        )
        == 0
    )
    (proc,) = procs
    # --bare drops the deals title and legend (DESIGN.md "CLI output");
    # the deals table itself must still reach the pager.
    assert "OFF-PEAK" in proc.stdin.getvalue()


@pytest.mark.parametrize("flags", [{"bare": True}, {"no_depths": True}])
def test_run_pager_output_variants(monkeypatch, flags):
    models = [make_model(id="acme/model-a", name="Acme: A")]
    _stub_quality_run(monkeypatch, models, aa_raises=True)
    procs = _stub_pager(monkeypatch)
    monkeypatch.setattr(sys, "stdout", _TTYStringIO())
    monkeypatch.setattr(sys, "stderr", _TTYStringIO())
    assert mc.run(make_args(min_context=1_000_000, **flags), models, False) == 0
    (proc,) = procs
    assert "RANK" in proc.stdin.getvalue()


def test_run_interrupted_while_paging_exits_130(monkeypatch):
    models = [make_model(id="acme/model-a", name="Acme: A")]
    _stub_quality_run(monkeypatch, models, aa_raises=True)
    procs = _stub_pager(monkeypatch)
    monkeypatch.setattr(sys, "stdout", _TTYStringIO())
    monkeypatch.setattr(sys, "stderr", _TTYStringIO())

    def interrupted(*a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr(mc, "print_table", interrupted)
    assert mc.run(make_args(min_context=1_000_000), models, False) == 130
    (proc,) = procs
    assert proc.terminated is True and proc.reaped is True


@pytest.mark.parametrize(
    "env,flag,expected_color",
    [
        ("", None, True),  # auto + TTY: color forced on through the pager
        ("1", None, False),  # NO_COLOR silences auto
        ("1", "on", True),  # explicit on beats NO_COLOR, through the pager too
        (None, "off", False),
    ],
)
def test_run_pager_color_resolution(monkeypatch, env, flag, expected_color):
    models = [make_model(id="acme/model-a", name="Acme: A")]
    _stub_quality_run(monkeypatch, models, aa_raises=True)
    procs = _stub_pager(monkeypatch)
    monkeypatch.setattr(sys, "stdout", _TTYStringIO())
    monkeypatch.setattr(sys, "stderr", _TTYStringIO())
    if env is None:
        monkeypatch.delenv("NO_COLOR", raising=False)
    else:
        monkeypatch.setenv("NO_COLOR", env)
    kwargs = {} if flag is None else {"color": flag}
    assert mc.run(make_args(min_context=1_000_000, **kwargs), models, False) == 0
    (proc,) = procs
    sgr = "\x1b[" in proc.stdin.getvalue()
    assert sgr is expected_color


def test_footer_legend_lines(capsys):
    mc.print_table(_scored_candidates(), mc.PRIORITY_WEIGHTS["balanced"], None)
    out = capsys.readouterr().out
    assert (
        "TIER      how prices vary: >Nk = pricier above that prompt size; -N% =" in out
    )
    assert "SCHED = schedule with a discount under 1% (--off-peak lists them)" in out
    assert "A higher rate" in out
    legend_lines = [
        line for line in out.splitlines() if line.startswith(("TIER", "$IN/$OUT"))
    ]
    assert legend_lines and all(line.isascii() for line in legend_lines)


def test_build_candidates_survives_string_context_length():
    # Regression: string context_length must not raise a TypeError.
    models = [make_model(context_length="2000000")]
    candidates, dropped = mc.build_candidates(
        models, make_args(min_context=1_000_000), {}, {"acme/model-a"}
    )
    assert len(candidates) == 1
    assert candidates[0]["context"] == 2_000_000
    assert "context" not in dropped


# ---------------------------------------------------------------------------
# build_candidates filtering
# ---------------------------------------------------------------------------


def test_build_candidates_drops_malformed_id():
    models = [make_model(id="no-slash")]
    candidates, dropped = mc.build_candidates(models, make_args(), {}, {"acme/model-a"})
    assert candidates == []
    assert dropped["malformed id"] == 1


def test_build_candidates_drops_low_context():
    models = [make_model(context_length=1000)]
    candidates, dropped = mc.build_candidates(
        models, make_args(min_context=1_000_000), {}, {"acme/model-a"}
    )
    assert candidates == []
    assert dropped["context"] == 1


def test_build_candidates_drops_bad_pricing():
    models = [make_model(pricing={"prompt": "x", "completion": "0.1"})]
    candidates, dropped = mc.build_candidates(models, make_args(), {}, {"acme/model-a"})
    assert candidates == []
    assert dropped["pricing"] == 1


def test_build_candidates_drops_negative_pricing():
    models = [make_model(pricing={"prompt": "-1", "completion": "0.1"})]
    candidates, dropped = mc.build_candidates(models, make_args(), {}, {"acme/model-a"})
    assert candidates == []
    assert dropped["pricing"] == 1


def test_build_candidates_exclude_free():
    models = [make_model(pricing={"prompt": "0", "completion": "0"})]
    candidates, dropped = mc.build_candidates(
        models, make_args(exclude_free=True), {}, {"acme/model-a"}
    )
    assert candidates == []
    assert dropped["free"] == 1


def test_build_candidates_drops_non_text_output():
    models = [make_model(architecture={"modality": "text->image"})]
    candidates, dropped = mc.build_candidates(models, make_args(), {}, {"acme/model-a"})
    assert candidates == []
    assert dropped["modality"] == 1


def test_build_candidates_requires_tools_when_asked():
    models = [make_model(supported_parameters=["tools"])]  # missing tool_choice
    candidates, dropped = mc.build_candidates(
        models, make_args(no_require_tools=False), {}, {"acme/model-a"}
    )
    assert candidates == []
    assert dropped["tool calling"] == 1


def test_build_candidates_blended_price():
    models = [make_model(pricing={"prompt": "0.000001", "completion": "0.000005"})]
    candidates, _ = mc.build_candidates(
        models, make_args(input_share=0.75), {}, {"acme/model-a"}
    )
    cand = candidates[0]
    assert cand["price_in"] == pytest.approx(1.0)
    assert cand["price_out"] == pytest.approx(5.0)
    # 0.75*1 + 0.25*5 = 2.0
    assert cand["blended"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# batch variants and discounts
# ---------------------------------------------------------------------------


def test_build_candidates_drops_batch_by_default():
    models = [make_model(id="acme/model-a:batch")]
    candidates, dropped = mc.build_candidates(models, make_args(), {}, {"acme/model-a"})
    assert candidates == []
    assert dropped["batch"] == 1


def test_build_candidates_keeps_batch_with_include_batch():
    models = [make_model(id="acme/model-a:batch")]
    candidates, dropped = mc.build_candidates(
        models, make_args(include_batch=True), {}, {"acme/model-a:batch"}
    )
    assert [c["id"] for c in candidates] == ["acme/model-a:batch"]
    assert not dropped


def test_build_candidates_discount_filter_keeps_discounted_only():
    models = [make_model(id="acme/discounted"), make_model(id="acme/normal")]
    candidates, dropped = mc.build_candidates(
        models, make_args(discount=True), {"acme/discounted": 0.5}, {"acme/discounted"}
    )
    assert [c["id"] for c in candidates] == ["acme/discounted"]
    assert dropped["no discount"] == 1
    assert candidates[0]["discount"] == 0.5


def test_build_candidates_discount_filter_without_data_drops_all():
    models = [make_model()]
    candidates, dropped = mc.build_candidates(
        models, make_args(discount=True), {}, {"acme/model-a"}
    )
    assert candidates == []
    assert dropped["no discount"] == 1


def test_build_candidates_discount_filter_ignores_negligible_discount():
    models = [make_model(id="acme/sliver")]
    candidates, dropped = mc.build_candidates(
        models, make_args(discount=True), {"acme/sliver": 0.004}, {"acme/sliver"}
    )
    assert candidates == []
    assert dropped["no discount"] == 1


def test_build_candidates_stores_discount():
    models = [make_model(id="acme/model-a")]
    candidates, _ = mc.build_candidates(
        models, make_args(), {"acme/model-a": 0.75}, {"acme/model-a"}
    )
    assert candidates[0]["discount"] == 0.75


# ---------------------------------------------------------------------------
# OpenRouter frontend fetch (discounts + AA benchmarks + ZDR)
# ---------------------------------------------------------------------------


def frontend_payload():
    """Base-URL payload with discounts and benchmarks in one response.

    Mirrors the live shape: models entries carry the slug/permaslug pair,
    and several dated permaslugs can pair with one slug -- the newest
    snapshot's trio wins for that slug.
    """
    return {
        "data": {
            "models": [
                {
                    "slug": "acme/a",
                    "permaslug": "acme/a-20260826",
                    "endpoint": {
                        "variant": "standard",
                        "pricing": {"discount": 0.5, "prompt": "0.1"},
                    },
                },
                {
                    "slug": "acme/a",
                    "permaslug": "acme/a-20260915",
                    "endpoint": {
                        "variant": "standard",
                        "pricing": {"discount": 0.5, "prompt": "0.1"},
                    },
                },
                {
                    "slug": "acme/b",
                    "endpoint": {"variant": "free", "pricing": {"discount": 0}},
                },
                {
                    "slug": "acme/c",
                    "endpoint": {"variant": "batch", "pricing": {"discount": 0.75}},
                },
                {
                    "slug": "~acme/private",
                    "endpoint": {"variant": "standard", "pricing": {"discount": 0.9}},
                },
                {"slug": "acme/no-endpoint", "endpoint": None},
                {
                    "slug": "acme/no-discount-field",
                    "endpoint": {"variant": "standard", "pricing": {}},
                },
            ],
            "benchmarks": {
                "acme/a-20260826": {
                    "aa": {
                        "intelligence_index": 57.5,
                        "coding_index": 71.5,
                        "agentic_index": 58.2,
                    }
                },
                "acme/a-20260915": {
                    "aa": {
                        "intelligence_index": 60,
                        "coding_index": 72,
                        "agentic_index": 59,
                    }
                },
            },
        }
    }


def zdr_endpoint(model_id, provider="Acme"):
    """One entry of OpenRouter's public per-endpoint ZDR list (live shape)."""
    entry = {
        "name": f"{provider} | {model_id}-20260901",
        "model_id": model_id,
        "model_name": f"Acme: {model_id}",
        "context_length": 131072,
        "pricing": {"prompt": "0.0000001", "completion": "0.0000002", "discount": 0},
        "provider_name": provider,
        "tag": provider.lower(),
        "status": 0,
    }
    return entry


def zdr_payload():
    """GET /api/v1/endpoints/zdr: a flat list, one entry per ZDR endpoint."""
    no_id = zdr_endpoint("acme/no-id")
    del no_id["model_id"]
    return {
        "data": [
            zdr_endpoint("acme/a"),
            # several ZDR endpoints for the same model are normal
            zdr_endpoint("acme/a", provider="Other"),
            zdr_endpoint("acme/b:batch"),
            no_id,
            {**zdr_endpoint("acme/null-id"), "model_id": None},
            {**zdr_endpoint("acme/empty-id"), "model_id": ""},
            {**zdr_endpoint("acme/non-str-id"), "model_id": {"slug": "acme/x"}},
            "not-a-dict",
            None,
        ]
    }


def stub_frontend(monkeypatch, calls, base_payload, zdr_payload=None):
    """URL-dispatching fetch_json stub; zdr_payload None means the ZDR URL raises."""

    def fake_fetch(url, *a, **k):
        calls.append(url)
        if url == mc.OPENROUTER_ZDR_URL:
            if zdr_payload is None:
                raise RuntimeError("zdr down")
            return zdr_payload
        if isinstance(base_payload, Exception):
            raise base_payload
        return base_payload

    monkeypatch.setattr(mc, "fetch_json", fake_fetch)


def test_fetch_openrouter_frontend_derives_all_payloads(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    calls = []
    stub_frontend(monkeypatch, calls, frontend_payload(), zdr_payload())
    discounts, zdr_ids, aa_by_id, cache_hits, frontend_models = (
        mc.fetch_openrouter_frontend(make_args(no_cache=True))
    )
    assert discounts == {"acme/a": 0.5, "acme/b:free": 0.0, "acme/c:batch": 0.75}
    assert zdr_ids == {"acme/a", "acme/b:batch"}
    assert aa_by_id == {
        "acme/a": {
            "intelligence_index": 60.0,
            "coding_index": 72.0,
            "agentic_index": 59.0,
            "as_of": "20260915",
        }
    }
    assert cache_hits == set()
    assert mc.OPENROUTER_DISCOUNTS_URL in calls
    assert mc.OPENROUTER_ZDR_URL in calls
    # The frontend listing is projected to the reasoning fields; "~" router
    # aliases are not models and never enter the fallback.
    assert [entry["slug"] for entry in frontend_models] == [
        "acme/a",
        "acme/a",
        "acme/b",
        "acme/c",
        "acme/no-endpoint",
        "acme/no-discount-field",
    ]
    assert set(frontend_models[0]) == set(mc.FRONTEND_REASONING_FIELDS)


def test_fetch_openrouter_frontend_caches_reasoning_entries(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    payload = {
        "data": {
            "models": [
                {
                    "slug": "acme/a",
                    "permaslug": "acme/a-20261017",
                    "supports_reasoning": True,
                    "reasoning_config": {"supported_reasoning_efforts": ["high"]},
                    "endpoint": {"variant": "standard", "pricing": {"discount": 0}},
                }
            ],
            # A benchmark entry keeps the AA cache warm too, so the second
            # run is served entirely from cache (the point of this test).
            "benchmarks": {"acme/a-20261017": {"aa": {"intelligence_index": 50}}},
        }
    }
    calls = []
    stub_frontend(monkeypatch, calls, payload, zdr_payload())
    args = make_args(no_cache=False, cache_ttl=3600)
    first = mc.fetch_openrouter_frontend(args)
    second = mc.fetch_openrouter_frontend(args)
    expected = [
        {
            "slug": "acme/a",
            "permaslug": "acme/a-20261017",
            "supports_reasoning": True,
            "reasoning_config": {"supported_reasoning_efforts": ["high"]},
        }
    ]
    assert first[4] == expected
    assert second[4] == expected
    assert len(calls) == 2  # base + zdr once each; the second run is cached
    saved = json.loads(
        (
            tmp_path / "model-compare" / "openrouter-frontend-reasoning-v1.json"
        ).read_text()
    )
    assert saved["payload"] == expected


def test_openrouter_zdr_url_is_the_per_endpoint_list():
    # The models/find ?zdr=true response is model-level and its embedded
    # endpoint is the default route, so it cannot tell which variants are
    # ZDR. The authoritative source is the public per-endpoint list.
    assert mc.OPENROUTER_ZDR_URL == "https://openrouter.ai/api/v1/endpoints/zdr"
    assert mc.OPENROUTER_ZDR_URL != mc.OPENROUTER_DISCOUNTS_URL


def test_fetch_openrouter_frontend_zdr_excludes_unlisted_variant(monkeypatch, tmp_path):
    # Regression: nvidia/nemotron-3-ultra-550b-a55b has a ZDR endpoint, but
    # its :free variant does not (those endpoints train on prompts) and was
    # still published with zdr: true. A variant counts only if it is listed
    # itself, even when its base slug is present and the find response
    # (discount URL) carries the variant.
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    base = {
        "data": {
            "models": [
                {
                    "slug": "nvidia/nemotron-3-ultra-550b-a55b",
                    "endpoint": {"variant": "standard", "pricing": {"discount": 0}},
                },
                {
                    "slug": "nvidia/nemotron-3-ultra-550b-a55b",
                    "endpoint": {"variant": "free", "pricing": {"discount": 0}},
                },
            ]
        }
    }
    zdr = {
        "data": [
            zdr_endpoint("nvidia/nemotron-3-ultra-550b-a55b", provider="BaseTen"),
        ]
    }
    calls = []
    stub_frontend(monkeypatch, calls, base, zdr)
    discounts, zdr_ids, aa_by_id, cache_hits, frontend_models = (
        mc.fetch_openrouter_frontend(make_args(no_cache=True))
    )
    assert zdr_ids == {"nvidia/nemotron-3-ultra-550b-a55b"}
    assert "nvidia/nemotron-3-ultra-550b-a55b:free" in discounts  # find has it


@pytest.mark.parametrize(
    "payload",
    [
        {"data": []},
        {"data": None},
        {},
        [],
        # the old models/find shape is not the per-endpoint list
        {"data": {"models": [{"slug": "acme/a", "endpoint": {"variant": "standard"}}]}},
        # entries without a usable model_id
        {
            "data": [
                {"name": "Acme | acme/a", "provider_name": "Acme"},
                {"model_id": ""},
            ]
        },
    ],
)
def test_fetch_openrouter_frontend_zdr_fails_closed_without_entries(
    monkeypatch, tmp_path, capsys, payload
):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    calls = []
    stub_frontend(monkeypatch, calls, frontend_payload(), payload)
    discounts, zdr_ids, aa_by_id, cache_hits, frontend_models = (
        mc.fetch_openrouter_frontend(make_args(no_cache=False, cache_ttl=3600))
    )
    assert zdr_ids == set()
    assert discounts  # base data unaffected
    assert "no ZDR entries found" in capsys.readouterr().err
    assert not (tmp_path / "model-compare" / "openrouter-zdr-v3.json").exists()


def test_fetch_openrouter_frontend_ignores_stale_v2_zdr_cache(monkeypatch, tmp_path):
    # v1/v2 sets were derived from the find response's default-route policy;
    # the semantics changed, so they must never be read.
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    cache_dir = tmp_path / "model-compare"
    cache_dir.mkdir(parents=True)
    for stale in ("openrouter-zdr.json", "openrouter-zdr-v2.json"):
        (cache_dir / stale).write_text(
            json.dumps({"fetched_at": time.time(), "payload": ["acme/stale"]})
        )
    calls = []
    stub_frontend(monkeypatch, calls, frontend_payload(), zdr_payload())
    discounts, zdr_ids, aa_by_id, cache_hits, frontend_models = (
        mc.fetch_openrouter_frontend(make_args(no_cache=False, cache_ttl=3600))
    )
    assert zdr_ids == {"acme/a", "acme/b:batch"}
    assert "zdr" not in cache_hits
    saved = json.loads((cache_dir / "openrouter-zdr-v3.json").read_text())
    assert saved["payload"] == ["acme/a", "acme/b:batch"]


def test_fetch_openrouter_frontend_caches_per_payload(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    calls = []
    stub_frontend(monkeypatch, calls, frontend_payload(), zdr_payload())
    args = make_args(no_cache=False, cache_ttl=3600)
    first = mc.fetch_openrouter_frontend(args)
    second = mc.fetch_openrouter_frontend(args)
    assert len(calls) == 2  # base + zdr once each; second run fully cached
    assert first[3] == set()
    assert second[3] == {"discounts", "aa", "zdr"}
    assert first[:3] == second[:3]


def test_fetch_openrouter_frontend_base_failure_keeps_zdr_decoupled(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    calls = []
    stub_frontend(monkeypatch, calls, RuntimeError("network down"), zdr_payload())
    discounts, zdr_ids, aa_by_id, cache_hits, frontend_models = (
        mc.fetch_openrouter_frontend(make_args(no_cache=True))
    )
    assert discounts == {}
    assert aa_by_id == {}
    assert zdr_ids == {"acme/a", "acme/b:batch"}
    assert cache_hits == set()
    err = capsys.readouterr().err
    assert "could not fetch discount data" in err
    assert "could not fetch AA benchmark data" in err
    assert "could not fetch ZDR data" not in err  # zdr succeeded


def test_fetch_openrouter_frontend_does_not_cache_empty(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    calls = []
    stub_frontend(monkeypatch, calls, {"data": {"models": []}}, {"data": []})
    args = make_args(no_cache=False, cache_ttl=3600)
    first = mc.fetch_openrouter_frontend(args)
    second = mc.fetch_openrouter_frontend(args)
    assert first[:3] == ({}, set(), {})
    assert second[:3] == ({}, set(), {})
    assert len(calls) == 4  # 2 per run (base + zdr); nothing cacheable
    cache_dir = tmp_path / "model-compare"
    assert not (cache_dir / "openrouter-frontend-discounts.json").exists()
    assert not (cache_dir / "openrouter-frontend-aa-v2.json").exists()
    assert not (cache_dir / "openrouter-frontend-reasoning-v1.json").exists()
    assert not (cache_dir / "openrouter-zdr-v3.json").exists()


def test_fetch_openrouter_frontend_treats_empty_cached_payloads_as_miss(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    calls = []
    stub_frontend(monkeypatch, calls, frontend_payload(), zdr_payload())
    cache_dir = tmp_path / "model-compare"
    cache_dir.mkdir(parents=True)
    now = time.time()
    (cache_dir / "openrouter-frontend-discounts.json").write_text(
        json.dumps({"fetched_at": now, "payload": {}})
    )
    (cache_dir / "openrouter-frontend-aa-v2.json").write_text(
        json.dumps({"fetched_at": now, "payload": {}})
    )
    (cache_dir / "openrouter-zdr-v3.json").write_text(
        json.dumps({"fetched_at": now, "payload": []})
    )
    discounts, zdr_ids, aa_by_id, cache_hits, frontend_models = (
        mc.fetch_openrouter_frontend(make_args(no_cache=False, cache_ttl=3600))
    )
    assert len(calls) == 2  # all three payloads empty in cache -> both fetches
    assert discounts == {"acme/a": 0.5, "acme/b:free": 0.0, "acme/c:batch": 0.75}
    assert zdr_ids == {"acme/a", "acme/b:batch"}
    assert cache_hits == set()


def test_fetch_openrouter_frontend_fresh_fetch_replaces_cached_discounts(
    monkeypatch, tmp_path
):
    # the discounts cache hit is provisional: the base URL is still fetched
    # for the AA benchmarks, and the fresh discounts win and overwrite it
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    calls = []
    stub_frontend(monkeypatch, calls, frontend_payload(), zdr_payload())
    cache_dir = tmp_path / "model-compare"
    cache_dir.mkdir(parents=True)
    (cache_dir / "openrouter-frontend-discounts.json").write_text(
        json.dumps({"fetched_at": time.time(), "payload": {"acme/cached": 0.1}})
    )
    discounts, zdr_ids, aa_by_id, cache_hits, frontend_models = (
        mc.fetch_openrouter_frontend(make_args(no_cache=False, cache_ttl=3600))
    )
    assert len(calls) == 2  # aa not cached, so the base URL is hit anyway
    assert discounts == {"acme/a": 0.5, "acme/b:free": 0.0, "acme/c:batch": 0.75}
    assert aa_by_id == {
        "acme/a": {
            "intelligence_index": 60.0,
            "coding_index": 72.0,
            "agentic_index": 59.0,
            "as_of": "20260915",
        }
    }
    assert zdr_ids == {"acme/a", "acme/b:batch"}
    assert cache_hits == set()  # fresh discounts discard the cache hit
    saved = json.loads((cache_dir / "openrouter-frontend-discounts.json").read_text())
    assert saved["payload"] == discounts


# ---------------------------------------------------------------------------
# ZDR (zero data retention) filter
# ---------------------------------------------------------------------------


def test_build_candidates_drops_non_zdr_by_default():
    models = [make_model(id="acme/zdr"), make_model(id="acme/plain")]
    candidates, dropped = mc.build_candidates(models, make_args(), {}, {"acme/zdr"})
    assert [c["id"] for c in candidates] == ["acme/zdr"]
    assert dropped["not ZDR"] == 1


def test_build_candidates_no_zdr_considers_all():
    models = [make_model(id="acme/zdr"), make_model(id="acme/plain")]
    candidates, dropped = mc.build_candidates(models, make_args(no_zdr=True), {}, set())
    assert len(candidates) == 2
    assert not dropped


def test_build_candidates_zdr_variant_keys():
    # a :batch variant counts as ZDR only if that variant itself is listed
    models = [make_model(id="acme/m:batch"), make_model(id="acme/n:batch")]
    candidates, dropped = mc.build_candidates(
        models, make_args(include_batch=True), {}, {"acme/m:batch"}
    )
    assert [c["id"] for c in candidates] == ["acme/m:batch"]
    assert dropped["not ZDR"] == 1


def test_build_candidates_collects_filtered_entries():
    models = [
        make_model(id="acme/model-a"),
        make_model(id="acme/model-b", name="B corp: Model B", context_length=100),
        make_model(id="no-slash", name="No Slash"),
    ]
    filtered = []
    candidates, dropped = mc.build_candidates(
        models, make_args(min_context=1_000_000), {}, {"acme/model-a"}, filtered
    )
    assert len(candidates) == 1
    # "no-slash" counts under malformed id but is withheld from filtered,
    # which only ever carries valid provider/model ids.
    assert filtered == [
        {"id": "acme/model-b", "name": "B corp: Model B", "reasons": ["context"]}
    ]
    assert dropped == {"context": 1, "malformed id": 1}


def test_build_candidates_empty_id_counted_not_listed():
    # An empty id counts under "malformed id" but is never emitted into
    # filtered, where it would fail the site validator's no-empty-id rule.
    args = make_args(min_context=0)
    models = [make_model(id=""), make_model(id="acme/model-a")]
    filtered = []
    candidates, dropped = mc.build_candidates(
        models, args, {}, {"acme/model-a"}, filtered
    )
    assert dropped == {"malformed id": 1}
    assert filtered == []
    mc.compute_scores(candidates, args, {})
    doc = mc.build_catalog(
        args, models, candidates, dropped, filtered, {}, {}, None, {}, {}
    )
    assert doc["filtered"] == []
    assert [e["id"] for e in doc["models"]] == ["acme/model-a"]


def test_build_candidates_without_collector_matches_old_signature():
    # 2-tuple unpacking stays valid; no filtered list, no behavior change.
    candidates, dropped = mc.build_candidates(
        [make_model(id="acme/model-b", context_length=100)],
        make_args(min_context=1_000_000),
        {},
        {"acme/model-a"},
    )
    assert candidates == []
    assert dropped == {"context": 1}


def test_build_candidates_additive_fields():
    created = 1_750_000_000
    candidates, _ = mc.build_candidates(
        [make_model(id="acme/model-a", created=created)],
        make_args(),
        {},
        {"acme/model-a"},
    )
    cand = candidates[0]
    assert cand["created"] == float(created)
    assert cand["tool_calling"] is True
    assert cand["zdr"] is True
    assert cand["expired"] is False


def test_build_candidates_zdr_null_under_no_zdr():
    candidates, _ = mc.build_candidates(
        [make_model(id="acme/model-a")], make_args(no_zdr=True), {}, set()
    )
    assert candidates[0]["zdr"] is None


def test_build_candidates_tool_calling_false_when_relaxed():
    candidates, _ = mc.build_candidates(
        [make_model(supported_parameters=[])],
        make_args(no_require_tools=True),
        {},
        {"acme/model-a"},
    )
    assert candidates[0]["tool_calling"] is False


# ---------------------------------------------------------------------------
# model_family / catalog_weights (catalog helpers)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model_id,expected",
    [
        ("z-ai/glm-5.3", "glm"),
        ("openai/gpt-5.2-mini", "gpt"),
        ("anthropic/claude-opus-4.6", "claude"),
        ("alibaba/qwen3-max", "qwen"),
        ("deepseek/deepseek-chat-v4", "deepseek"),
        ("amazon/nova-pro-2", "nova"),
        ("google/gemini_2_5_pro", "gemini"),
        ("x-ai/o4-mini", "o"),  # documented oddball
        ("kimi/k2", None),
        ("acme/model-a", "model"),
        ("acme/model-a:variant", "model"),
        ("weird/model", "model"),
    ],
)
def test_model_family(model_id, expected):
    assert mc.model_family(model_id) == expected


def test_catalog_weights_match_base_when_quality_present():
    weights = mc.catalog_weights([{"id": "a/b"}], {"a/b": 50.0})
    assert weights == mc.PRIORITY_WEIGHTS


def test_catalog_weights_renormalize_when_quality_blind():
    weights = mc.catalog_weights([{"id": "a/b"}], {})
    for priority, base in mc.PRIORITY_WEIGHTS.items():
        assert "quality" not in weights[priority]
        total = sum(weights[priority].values())
        assert total == pytest.approx(1.0)
        for name, value in base.items():
            if name != "quality":
                assert weights[priority][name] == pytest.approx(
                    value / (sum(base.values()) - base["quality"])
                )


def test_fetch_openrouter_frontend_no_zdr_skips_zdr_fetch_but_still_loads_base(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    calls = []
    stub_frontend(monkeypatch, calls, frontend_payload(), zdr_payload())
    discounts, zdr_ids, aa_by_id, cache_hits, frontend_models = (
        mc.fetch_openrouter_frontend(make_args(no_cache=True, no_zdr=True))
    )
    assert zdr_ids == set()
    assert discounts != {}
    assert aa_by_id != {}
    assert calls == [mc.OPENROUTER_DISCOUNTS_URL]


def test_fetch_openrouter_frontend_normal_run_after_no_zdr_run(monkeypatch, tmp_path):
    # a --no-zdr run writes no ZDR cache; a later normal run on the warm
    # cache must still derive ZDR ids instead of failing closed
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    calls = []
    stub_frontend(monkeypatch, calls, frontend_payload(), zdr_payload())
    first = mc.fetch_openrouter_frontend(
        make_args(no_cache=False, cache_ttl=3600, no_zdr=True)
    )
    assert first[1] == set()
    assert not (tmp_path / "model-compare" / "openrouter-zdr-v3.json").exists()
    second = mc.fetch_openrouter_frontend(make_args(no_cache=False, cache_ttl=3600))
    assert second[1] == {"acme/a", "acme/b:batch"}
    assert calls == [mc.OPENROUTER_DISCOUNTS_URL, mc.OPENROUTER_ZDR_URL]


def test_fetch_openrouter_frontend_aa_cache_hit_skips_base_fetch(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    calls = []
    stub_frontend(monkeypatch, calls, frontend_payload(), zdr_payload())
    cache_dir = tmp_path / "model-compare"
    cache_dir.mkdir(parents=True)
    (cache_dir / "openrouter-frontend-discounts.json").write_text(
        json.dumps({"fetched_at": time.time(), "payload": {"acme/a": 0.5}})
    )
    (cache_dir / "openrouter-frontend-aa-v2.json").write_text(
        json.dumps(
            {
                "fetched_at": time.time(),
                "payload": {"acme/a": {"intelligence_index": 57.5}},
            }
        )
    )
    discounts, zdr_ids, aa_by_id, cache_hits, frontend_models = (
        mc.fetch_openrouter_frontend(make_args(no_cache=False, cache_ttl=3600))
    )
    assert len(calls) == 1  # only the ZDR URL
    assert discounts == {"acme/a": 0.5}
    assert aa_by_id == {"acme/a": {"intelligence_index": 57.5}}
    assert cache_hits == {"discounts", "aa"}
    # A warm discounts+AA cache predating the reasoning cache does not
    # trigger a base refetch; the fallback self-heals when the TTL expires.
    assert frontend_models == []


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, "--"),
        (0, "--"),
        (0.0, "--"),
        (0.004, "--"),
        (0.005, "--"),
        (0.006, "1%"),
        (0.049, "5%"),
        (0.5, "50%"),
        (0.561, "56%"),
        (0.75, "75%"),
        (-0.1, "--"),
    ],
)
def test_fmt_discount(value, expected):
    assert mc.fmt_discount(value) == expected


def test_print_table_shows_disc_column(capsys):
    top = [_cand("a/x", 1.0)]
    top[0].update({"score": 0.9, "quality": None, "discount": 0.5})
    mc.print_table(top, {"price": 1.0}, None)
    out = capsys.readouterr().out
    assert "DISC" in out
    assert "50%" in out


def test_print_table_dash_without_discount(capsys):
    top = [_cand("a/x", 1.0)]
    top[0].update({"score": 0.9, "quality": None, "discount": None})
    mc.print_table(top, {"price": 1.0}, None)
    out = capsys.readouterr().out
    data_row = next(line for line in out.splitlines() if line.strip().startswith("1"))
    assert "DISC" in out
    assert "--" in data_row


def test_print_json_includes_discount(capsys):
    rows = [_cand("a/x", 1.0)]
    rows[0].update({"score": 0.9, "discount": 0.25})
    mc.print_json(rows)
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["discount"] == 0.25
    assert payload[0]["discount_pct"] == "25%"


def test_print_json_discount_null_when_absent(capsys):
    rows = [_cand("a/x", 1.0)]
    rows[0].update({"score": 0.9, "discount": None})
    mc.print_json(rows)
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["discount"] is None
    assert payload[0]["discount_pct"] == "--"


def test_print_json_discount_null_for_negligible(capsys):
    rows = [_cand("a/x", 1.0)]
    rows[0].update({"score": 0.9, "discount": 0.004})
    mc.print_json(rows)
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["discount"] is None
    assert payload[0]["discount_pct"] == "--"


def test_print_json_discount_pct_matches_table_rounding(capsys):
    # The site renders discount_pct verbatim, so it must use the DISC column's
    # half-to-even rounding: 0.025 is "2%" here, not the "3%" Math.round gave.
    rows = [_cand("a/x", 1.0)]
    rows[0].update({"score": 0.9, "discount": 0.025})
    mc.print_json(rows)
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["discount_pct"] == "2%"
    assert payload[0]["discount_pct"] == mc.fmt_discount(0.025)


# ---------------------------------------------------------------------------
# opencode model namespace
# ---------------------------------------------------------------------------


def test_opencode_model_id():
    assert mc.opencode_model_id("z-ai/glm-5.3-flash") == "openrouter/z-ai/glm-5.3-flash"
    assert mc.opencode_model_id("nvidia/x:free") == "openrouter/nvidia/x:free"


def test_print_json_includes_opencode_model(capsys):
    rows = [_cand("a/x", 1.0)]
    rows[0].update({"score": 0.9})
    mc.print_json(rows)
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["model"] == "a/x"
    assert payload[0]["opencode_model"] == "openrouter/a/x"


def test_best_prints_provider_qualified_id(monkeypatch, capsys):
    monkeypatch.setattr(
        mc, "fetch_openrouter_models", lambda a: ([make_model()], False)
    )
    monkeypatch.setattr(
        mc, "fetch_openrouter_frontend", lambda a: ({}, {"acme/model-a"}, {}, set(), [])
    )
    monkeypatch.setattr(mc, "fetch_aa_entries", lambda a: ([], None, False, None))
    # Unkeyed runs hydrate quality from the published catalog (Task 4);
    # this test is about the --best id, so the catalog stays unavailable.
    monkeypatch.setattr(mc, "fetch_quality_catalog", lambda a: None)
    # The endpoint stage fetches per candidate: one eligible endpoint keeps
    # the model in the pool (offline stub, like every other fetcher here).
    monkeypatch.setattr(mc, "fetch_model_endpoints", lambda mid, a: [_endpoint()])
    argv = ["--best", "--no-cache", "--min-context", "0", "--no-require-tools"]
    assert mc.main(argv) == 0
    assert capsys.readouterr().out.strip() == "openrouter/acme/model-a"


def test_table_mode_prints_composed_quality_note(monkeypatch, capsys):
    models = [
        make_model(),
        make_model(
            id="acme/model-b",
            pricing={"prompt": "0.000002", "completion": "0.000004"},
            context_length=4_000_000,
        ),
    ]
    monkeypatch.setattr(mc, "fetch_openrouter_models", lambda a: (models, False))
    monkeypatch.setattr(
        mc,
        "fetch_openrouter_frontend",
        lambda a: (
            {},
            {"acme/model-a", "acme/model-b"},
            {"acme/model-a": {"intelligence_index": 57.5}},
            set(),
            [
                {
                    "slug": "acme/model-a",
                    "supports_reasoning": False,
                    "reasoning_config": {},
                },
                {
                    "slug": "acme/model-b",
                    "supports_reasoning": False,
                    "reasoning_config": {},
                },
            ],
        ),
    )
    monkeypatch.setattr(mc, "fetch_aa_entries", lambda a: ([], None, False, None))
    # Unkeyed runs hydrate quality from the published catalog (Task 4);
    # this test pins the table footer, so the catalog stays unavailable.
    monkeypatch.setattr(mc, "fetch_quality_catalog", lambda a: None)
    # The endpoint stage fetches per candidate: one eligible endpoint keeps
    # both models in the pool (offline stub, like every other fetcher here).
    monkeypatch.setattr(mc, "fetch_model_endpoints", lambda mid, a: [_endpoint()])
    argv = ["--top", "1", "--no-cache", "--min-context", "0", "--no-require-tools"]
    assert mc.main(argv) == 0
    captured = capsys.readouterr()
    # Per-source counts + as_of + index version when known. The OR
    # benchmark carries no as_of here, so the block shows the bare count;
    # model-b is unmatched and gets the trailing clause.
    assert "Quality source: OpenRouter snapshot (1 model);" in captured.out
    assert "1 model unmatched (no quality data; quality score 0)" in captured.out
    assert "note:" not in captured.err  # --no-cache: nothing served from cache


# ---------------------------------------------------------------------------
# compute_scores
# ---------------------------------------------------------------------------


def _cand(model_id, blended, context=2_000_000, age_days: float | None = 0.0):
    return {
        "id": model_id,
        "name": model_id,
        "context": context,
        "price_in": blended,
        "price_out": blended,
        "blended": blended,
        "base_price_in": blended,
        "base_price_out": blended,
        "tier_prompt_tokens": None,
        "tiers": [],
        "schedule": None,
        "max_discount": None,
        "tier_note": None,
        "sched_note": None,
        "sched_detail": None,
        "age_days": age_days,
        "quality": None,
        "discount": None,
        "score": 0.0,
    }


def test_compute_scores_cheapest_gets_top_price_score():
    candidates = [_cand("a/cheap", 1.0), _cand("a/pricey", 100.0)]
    mc.compute_scores(candidates, make_args(priority="price"), {})
    by_id = {c["id"]: c for c in candidates}
    assert by_id["a/cheap"]["price_score"] == pytest.approx(1.0)
    assert by_id["a/pricey"]["price_score"] < by_id["a/cheap"]["price_score"]
    # price priority -> cheap should rank first
    assert candidates[0]["id"] == "a/cheap"


def test_compute_scores_drops_quality_weight_when_all_unmatched():
    candidates = [_cand("a/x", 1.0), _cand("a/y", 2.0)]
    weights = mc.compute_scores(candidates, make_args(), {})
    assert "quality" not in weights
    assert sum(weights.values()) == pytest.approx(1.0)


def test_compute_scores_keeps_quality_weight_on_partial_match():
    candidates = [_cand("a/x", 1.0), _cand("a/y", 2.0)]
    weights = mc.compute_scores(candidates, make_args(), {"a/x": 70.0})
    assert "quality" in weights
    by_id = {c["id"]: c for c in candidates}
    # unmatched candidate is penalized with quality_score 0, not reweighted
    assert by_id["a/y"]["quality_score"] == 0.0
    assert by_id["a/x"]["quality_score"] == pytest.approx(1.0)


def test_compute_scores_quality_ref_clamps():
    candidates = [_cand("a/x", 1.0)]
    mc.compute_scores(candidates, make_args(quality_ref=70.0), {"a/x": 140.0})
    assert candidates[0]["quality_score"] == 1.0


def test_compute_scores_context_score_bounds():
    floor = 1_000_000
    candidates = [
        _cand("a/at-floor", 1.0, context=floor),
        _cand("a/4x", 1.0, context=4 * floor),
        _cand("a/8x", 1.0, context=8 * floor),
    ]
    mc.compute_scores(candidates, make_args(min_context=floor), {})
    by_id = {c["id"]: c for c in candidates}
    assert by_id["a/at-floor"]["context_score"] == 0.0
    assert by_id["a/4x"]["context_score"] == pytest.approx(1.0)
    assert by_id["a/8x"]["context_score"] == 1.0  # capped


def test_compute_scores_age_score_decay():
    candidates = [
        _cand("a/fresh", 1.0, age_days=0.0),
        _cand("a/half", 1.0, age_days=120.0),
        _cand("a/unknown", 1.0, age_days=None),
    ]
    mc.compute_scores(candidates, make_args(recency_half_life=120.0), {})
    by_id = {c["id"]: c for c in candidates}
    assert by_id["a/fresh"]["age_score"] == pytest.approx(1.0)
    assert by_id["a/half"]["age_score"] == pytest.approx(0.5)
    assert by_id["a/unknown"]["age_score"] == pytest.approx(0.5)


def test_compute_scores_single_candidate_pool():
    candidates = [_cand("a/only", 5.0)]
    weights = mc.compute_scores(candidates, make_args(), {})
    assert candidates[0]["price_score"] == pytest.approx(1.0)
    assert sum(weights.values()) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# AA lookup + matching
# ---------------------------------------------------------------------------


def test_parse_aa_effort_reads_name_then_slug():
    assert mc.parse_aa_effort(
        "Claude Sonnet 5.5 (High, Default Fallback)", "claude-sonnet-5-5-high"
    ) == {
        "base_name": "Claude Sonnet 5.5",
        "base_slug": "claude-sonnet-5-5",
        "effort": "high",
        "ambiguous": False,
    }
    assert mc.parse_aa_effort("GLM-5.3-Flash (Max)", "glm-5-3-flash") == {
        "base_name": "GLM-5.3-Flash",
        "base_slug": "glm-5-3-flash",
        "effort": "max",
        "ambiguous": False,
    }


def test_parse_aa_effort_ambiguous_and_unknown():
    assert mc.parse_aa_effort("Model (Non-reasoning, High)", "model")["effort"] is None
    assert (
        mc.parse_aa_effort("Model (Non-reasoning, High)", "model")["ambiguous"] is True
    )
    # An ambiguous name blocks the slug fallback even when the slug names an
    # effort: the name is authoritative and the conflict is not guessed away.
    ambiguous_slug = mc.parse_aa_effort("Model (Non-reasoning, High)", "model-high")
    assert ambiguous_slug["effort"] is None
    assert ambiguous_slug["ambiguous"] is True
    assert mc.parse_aa_effort("Model (High, Max)", "model")["effort"] is None
    assert mc.parse_aa_effort("Model (High, Max)", "model")["ambiguous"] is True
    assert mc.parse_aa_effort("Model (Something)", "model")["effort"] is None
    assert mc.parse_aa_effort("Model", "model")["effort"] is None


def test_parse_aa_effort_falls_back_to_slug_suffix():
    assert mc.parse_aa_effort("Model A", "model-a-high")["effort"] == "high"
    assert mc.parse_aa_effort("Model A", "model-a-xhigh")["effort"] == "xhigh"
    assert mc.parse_aa_effort("Model A", "model-a-non-reasoning")["effort"] == "none"
    reasoning = mc.parse_aa_effort("Model A", "model-a-reasoning")
    assert reasoning["effort"] == "reasoning"
    assert reasoning["base_slug"] == "model-a"
    # The name still wins over a suffixed slug.
    assert mc.parse_aa_effort("Model A (High)", "model-a-reasoning")["effort"] == "high"


def test_parse_aa_effort_warns_once_per_unknown_shape(monkeypatch, capsys):
    monkeypatch.setattr(mc, "_AA_WARNED_UNKNOWN_SHAPES", set())
    # Two rows sharing one unrecognized parenthetical shape: one warning.
    mc.parse_aa_effort("Model A (Novel Shape)", "model-a")
    mc.parse_aa_effort("Model B (Novel Shape)", "model-b")
    err = capsys.readouterr().err
    lines = [line for line in err.splitlines() if line]
    assert len(lines) == 1, err
    assert "Novel Shape" in lines[0]
    # A legitimately labeled row produces no warning.
    mc.parse_aa_effort("Model C (High)", "model-c-high")
    assert capsys.readouterr().err == ""


def test_build_aa_depth_lookup_groups_variants_and_drops_conflicts():
    entries = [
        {
            "id": "1",
            "name": "Model A (Max)",
            "slug": "model-a",
            "index": 56.0,
            "as_of": "t1",
        },
        {
            "id": "2",
            "name": "Model A (High)",
            "slug": "model-a-high",
            "index": 46.8,
            "as_of": "t1",
        },
        {
            "id": "3",
            "name": "Model A (High, Default Fallback)",
            "slug": "model-a-high-2",
            "index": 40.0,
            "as_of": "t1",
        },
    ]
    lookup = mc.build_aa_depth_lookup(entries)
    assert lookup["model a"]["max"]["index"] == 56.0
    assert lookup["model a"]["high"] is None
    assert (
        mc.build_aa_depth_lookup(
            [
                {
                    "id": "4",
                    "name": "Model B",
                    "slug": "model-b",
                    "index": 50.0,
                },
            ]
        )
        == {}
    )


def test_build_aa_depth_lookup_normalizes_cost_and_creator_key():
    lookup = mc.build_aa_depth_lookup(
        [
            {
                "id": "1",
                "name": "Model A (High)",
                "slug": "model-a-high",
                "index": 46.8,
                "creator": "Z.AI",
                "cost_per_task": 0.8844,
                "as_of": "t1",
            }
        ]
    )
    record = lookup["model a"]["high"]
    assert record["effort"] == "high"
    assert record["cost_per_task_usd"] == 0.8844
    assert record["as_of"] == "t1"
    # creator-qualified alias uses normalize_provider on model_creator.name
    assert lookup["zai:model a"]["high"] is record


def test_build_aa_depth_lookup_skips_overflowing_index():
    lookup = mc.build_aa_depth_lookup(
        [
            {
                "id": "1",
                "name": "Model A (High)",
                "slug": "model-a-high",
                "index": 10**400,
                "as_of": "t1",
            }
        ]
    )
    # huge ints overflow float(); the row is skipped, no effort group created
    assert lookup == {}


def test_build_aa_unlabeled_lookup_keeps_only_unlabeled_rows():
    lookup = mc.build_aa_unlabeled_lookup(
        [
            {
                "id": "1",
                "name": "GLM-5.3-Flash",
                "slug": "glm-5-3-flash",
                "index": 27.8,
                "creator": "Z.AI",
                "cost_per_task": 0.58,
                "as_of": "t1",
            },
            {
                "id": "2",
                "name": "GLM-5.3 (Max)",
                "slug": "glm-5-3",
                "index": 31.7,
                "as_of": "t1",
            },
            {
                "id": "3",
                "name": "Model (Non-reasoning, High)",
                "slug": "model",
                "index": 40.0,
                "as_of": "t1",
            },
        ]
    )
    record = lookup["glm 5 3 flash"]
    assert record == {
        "index": 27.8,
        "cost_per_task_usd": 0.58,
        "source": "aa",
        "as_of": "t1",
    }
    assert lookup["zai:glm 5 3 flash"] is record
    # Labeled rows and ambiguous labels never enter the reference map.
    assert "glm 5 3" not in lookup
    assert "model" not in lookup


def test_build_aa_unlabeled_lookup_drops_conflicting_duplicates():
    lookup = mc.build_aa_unlabeled_lookup(
        [
            {"id": "1", "name": "Model A", "slug": "model-a", "index": 40.0},
            {"id": "2", "name": "Model A", "slug": "model-a", "index": 41.0},
        ]
    )
    # Two unlabeled rows for one base key cannot be told apart: never guess.
    assert lookup.get("model a") is None


def test_build_aa_unlabeled_lookup_skips_overflowing_index():
    lookup = mc.build_aa_unlabeled_lookup(
        [
            {
                "id": "1",
                "name": "GLM-5.3-Flash",
                "slug": "glm-5-3-flash",
                "index": 10**400,
                "as_of": "t1",
            }
        ]
    )
    # huge ints overflow float(); the row is skipped and the lookup stays
    # empty for its base key
    assert lookup == {}


def test_legacy_aa_entries_dedupe_by_key_first_wins():
    # The depth path keeps every effort variant (id-first dedupe), but the
    # legacy lookup must see the key-unique list it always saw: one entry per
    # key, first occurrence in list order wins.
    entries = [
        {
            "id": "1",
            "key": "model-a",
            "name": "Model A (Max)",
            "slug": "model-a",
            "index": 56.0,
        },
        {
            "id": "2",
            "key": "model-a",
            "name": "Model A (High)",
            "slug": "model-a",
            "index": 46.8,
        },
        {
            "id": "3",
            "key": "model-b",
            "name": "Model B",
            "slug": "model-b",
            "index": 50.0,
        },
    ]
    legacy = mc.dedupe_aa_entries_by_key(entries)
    assert [entry["id"] for entry in legacy] == ["1", "3"]

    exact, fuzzy = mc.build_aa_lookup(legacy)
    # Without the seam the variant name "Model A (High)" would win tier 2;
    # with it the candidate falls through to the first-wins max entry.
    assert mc.match_quality(
        {"id": "other/unrelated", "name": "Acme: Model A (High)"}, exact, fuzzy
    ) == (56.0, None)


def test_build_aa_lookup_skips_non_numeric_index():
    entries = [
        {"key": "acme/model-a", "name": "Model A", "index": "not-a-number"},
        {"key": "acme/model-b", "name": "Model B", "index": 55.0},
    ]
    exact, fuzzy = mc.build_aa_lookup(entries)
    # only the numeric entry survives into the fuzzy list
    assert len(fuzzy) == 1
    assert mc.match_quality(
        {"id": "acme/model-b", "name": "Model B"}, exact, fuzzy
    ) == (55.0, None)


def test_build_aa_lookup_skips_non_finite_index():
    entries = [
        {"key": "acme/model-a", "name": "Model A", "index": float("nan")},
        {"key": "acme/model-b", "name": "Model B", "index": 55.0},
    ]
    exact, fuzzy = mc.build_aa_lookup(entries)
    # a NaN index would serialize as JSON NaN and score 1.0; never match it
    assert (
        mc.match_quality({"id": "acme/model-a", "name": "Acme: Model A"}, exact, fuzzy)
        is None
    )
    assert mc.match_quality(
        {"id": "acme/model-b", "name": "Model B"}, exact, fuzzy
    ) == (55.0, None)


def test_build_aa_lookup_skips_overflowing_index():
    entries = [
        {"key": "acme/model-a", "name": "Model A", "index": 10**400},
        {"key": "acme/model-b", "name": "Model B", "index": 55.0},
    ]
    exact, fuzzy = mc.build_aa_lookup(entries)
    # huge ints overflow float(); the entry never reaches the exact map
    assert all("model a" not in key for _, key in exact)
    assert len(fuzzy) == 1
    assert mc.match_quality(
        {"id": "acme/model-b", "name": "Model B"}, exact, fuzzy
    ) == (55.0, None)


def test_match_quality_exact_by_id():
    entries = [{"key": "acme/model-a", "name": "Model A", "index": 60.0}]
    exact, fuzzy = mc.build_aa_lookup(entries)
    assert mc.match_quality(
        {"id": "acme/model-a", "name": "Acme: Model A"}, exact, fuzzy
    ) == (60.0, None)


def test_match_quality_fuzzy_overlap():
    entries = [{"key": "model-a", "name": "Model A", "index": 55.0}]
    exact, fuzzy = mc.build_aa_lookup(entries)
    result = mc.match_quality(
        {"id": "acme/model-a-extra", "name": "Model A Extra"},
        exact,
        fuzzy,
        allow_fuzzy=True,
    )
    assert result == (55.0, None)


def test_match_quality_no_match_returns_none():
    entries = [{"key": "acme/model-a", "name": "Model A", "index": 60.0}]
    exact, fuzzy = mc.build_aa_lookup(entries)
    assert (
        mc.match_quality({"id": "other/unrelated-xyz", "name": "Zzz"}, exact, fuzzy)
        is None
    )


def test_match_quality_fuzzy_disabled_by_default():
    # Regression: the fuzzy-matching path paired z-ai/glm-5.3-flash with the
    # single AA entry glm-5-3 (Jaccard 0.6 >= 0.5). Exact-only must refuse.
    entries = [{"key": "glm-5-3", "name": "GLM-5.3 (max)", "index": 59.5}]
    exact, fuzzy = mc.build_aa_lookup(entries)
    model = {"id": "z-ai/glm-5.3-flash", "name": "Z.AI: GLM 5.3 Flash"}
    assert mc.match_quality(model, exact, fuzzy) is None
    assert mc.match_quality(model, exact, fuzzy, allow_fuzzy=True) == (59.5, None)


# ---------------------------------------------------------------------------
# OR-published AA benchmarks (build_aa_benchmarks)
# ---------------------------------------------------------------------------


def test_build_aa_benchmarks_strips_dated_permaslugs():
    benchmarks = {
        "z-ai/glm-5.3-flash-20260826": {
            "aa": {
                "intelligence_index": 57.5,
                "coding_index": 71.5,
                "agentic_index": 58.2,
            }
        },
        "acme/plain": {"aa": {"intelligence_index": 40}},
    }
    aa = mc.build_aa_benchmarks(benchmarks, None)
    assert aa["z-ai/glm-5.3-flash"] == {
        "intelligence_index": 57.5,
        "coding_index": 71.5,
        "agentic_index": 58.2,
        "as_of": "20260826",
    }
    assert aa["acme/plain"] == {"intelligence_index": 40.0, "as_of": None}


def test_build_aa_benchmarks_two_listings_no_borrowing():
    # Regression for #17: two dated permaslugs of one model family strip to
    # the SAME bare id, so blind date-stripping let the base listing borrow
    # the snapshot's score. The catalog's own slug/permaslug pairs keep each
    # listing's trio on its own slug (values are the live-verified pair).
    benchmarks = {
        "deepseek/deepseek-v4.1-flash-20261017": {"aa": {"intelligence_index": 39.5}},
        "deepseek/deepseek-v4.1-flash-20260910": {"aa": {"intelligence_index": 24.7}},
    }
    models = [
        {
            "slug": "deepseek/deepseek-v4.1-flash",
            "permaslug": "deepseek/deepseek-v4.1-flash-20260910",
        },
        # standard and batch variants repeat the same slug/permaslug pair
        {
            "slug": "deepseek/deepseek-v4.1-flash",
            "permaslug": "deepseek/deepseek-v4.1-flash-20260910",
        },
        {
            "slug": "deepseek/deepseek-v4.1-flash-1017",
            "permaslug": "deepseek/deepseek-v4.1-flash-20261017",
        },
    ]
    aa = mc.build_aa_benchmarks(benchmarks, models)
    assert aa == {
        "deepseek/deepseek-v4.1-flash": {
            "intelligence_index": 24.7,
            "as_of": "20260910",
        },
        "deepseek/deepseek-v4.1-flash-1017": {
            "intelligence_index": 39.5,
            "as_of": "20261017",
        },
    }


def test_build_aa_benchmarks_undated_key_self_maps():
    # An undated key is a bare catalog id and maps to itself.
    benchmarks = {"acme/plain": {"aa": {"intelligence_index": 40}}}
    assert mc.build_aa_benchmarks(benchmarks, []) == {
        "acme/plain": {"intelligence_index": 40.0, "as_of": None}
    }


def test_build_aa_benchmarks_undated_key_drops_when_pass1_claimed():
    # Pass 1 claims the slug through the catalog's permaslug pairing, and
    # the undated key for that id defers -- in either benchmarks order,
    # pinning that all of pass 1 resolves before pass 2 runs.
    dated = {"aa": {"intelligence_index": 20}}
    undated = {"aa": {"intelligence_index": 10}}
    models = [{"slug": "acme/m", "permaslug": "acme/m-20260820"}]
    for benchmarks in (
        {"acme/m": undated, "acme/m-20260820": dated},
        {"acme/m-20260820": dated, "acme/m": undated},
    ):
        aa = mc.build_aa_benchmarks(benchmarks, models)
        assert aa == {"acme/m": {"intelligence_index": 20.0, "as_of": "20260820"}}


def test_build_aa_benchmarks_unresolvable_dated_key_strips_when_unambiguous():
    # A dated key the catalog does not pair still maps by stripping, as
    # long as the stripped id is unclaimed and no other key strips there.
    benchmarks = {"acme/legacy-20250101": {"aa": {"intelligence_index": 11}}}
    assert mc.build_aa_benchmarks(benchmarks, []) == {
        "acme/legacy": {"intelligence_index": 11.0, "as_of": "20250101"}
    }


def test_build_aa_benchmarks_ambiguous_strip_dropped():
    # The #17 shape with no catalog pairing to disambiguate: two dated
    # keys strip to one id, so neither may claim it (no borrowing).
    benchmarks = {
        "deepseek/deepseek-v4.1-flash-20261017": {"aa": {"intelligence_index": 39.5}},
        "deepseek/deepseek-v4.1-flash-20260910": {"aa": {"intelligence_index": 24.7}},
    }
    assert mc.build_aa_benchmarks(benchmarks, []) == {}


def test_build_aa_benchmarks_same_slug_latest_date_wins():
    # glm-style: several dated permaslugs legitimately pair with ONE slug
    # (snapshot variants of one listing); the latest date wins.
    benchmarks = {
        "z-ai/glm-5.3-flash-20260826": {"aa": {"intelligence_index": 57.5}},
        "z-ai/glm-5.3-flash-20260915": {"aa": {"intelligence_index": 59.5}},
    }
    models = [
        {"slug": "z-ai/glm-5.3-flash", "permaslug": "z-ai/glm-5.3-flash-20260826"},
        {"slug": "z-ai/glm-5.3-flash", "permaslug": "z-ai/glm-5.3-flash-20260915"},
    ]
    assert mc.build_aa_benchmarks(benchmarks, models) == {
        "z-ai/glm-5.3-flash": {"intelligence_index": 59.5, "as_of": "20260915"}
    }


def test_build_aa_benchmarks_invalid_trio_skipped():
    # Non-finite and null index values fail the finite-number guard, so
    # the whole entry is treated as absent.
    benchmarks = {
        "acme/null": {"aa": {"intelligence_index": None}},
        "acme/nan": {"aa": {"intelligence_index": float("nan")}},
        "acme/inf": {"aa": {"intelligence_index": float("inf")}},
        "acme/valid": {"aa": {"intelligence_index": 42}},
    }
    assert mc.build_aa_benchmarks(benchmarks, []) == {
        "acme/valid": {"intelligence_index": 42.0, "as_of": None}
    }


def test_build_aa_benchmarks_skips_invalid_values_but_keeps_valid_ones():
    benchmarks = {
        "acme/partial": {
            "aa": {
                "intelligence_index": float("nan"),
                "coding_index": 70,
                "agentic_index": True,
            }
        },
        "acme/no-aa": {"da": {"default_elo": 1300}},
        "acme/not-a-node": "garbage",
    }
    aa = mc.build_aa_benchmarks(benchmarks)
    assert aa == {"acme/partial": {"coding_index": 70.0, "as_of": None}}


def test_build_aa_benchmarks_empty_inputs():
    assert mc.build_aa_benchmarks({}) == {}
    assert mc.build_aa_benchmarks(None) == {}


def test_base_model_id_strips_variant():
    assert mc.base_model_id("z-ai/glm-5.3-flash:free") == "z-ai/glm-5.3-flash"
    assert mc.base_model_id("z-ai/glm-5.3") == "z-ai/glm-5.3"


# ---------------------------------------------------------------------------
# OpenRouter reasoning metadata (effective default + effort filter)
# ---------------------------------------------------------------------------


def test_effective_default_is_none_when_disabled():
    assert (
        mc.effective_default_effort(
            {
                "supported_efforts": ["max", "high", "none"],
                "default_enabled": False,
                "default_effort": "medium",
                "mandatory": False,
                "is_reasoning": True,
            }
        )
        == "none"
    )


def test_effective_default_single_state_and_mandatory():
    assert (
        mc.effective_default_effort(
            {
                "supported_efforts": [],
                "default_enabled": True,
                "default_effort": None,
                "mandatory": True,
                "is_reasoning": True,
            }
        )
        == "reasoning"
    )


def test_supported_efforts_null_expands_but_omitted_stays_empty():
    meta = mc.build_reasoning_metadata(
        [{"id": "a/m", "reasoning": {"supported_efforts": None}}]
    )
    assert meta["a/m"]["supported_efforts"] == list(mc.REASONING_GATEWAY_EFFORTS)
    meta = mc.build_reasoning_metadata(
        [{"id": "a/m", "reasoning": {"mandatory": True}}]
    )
    assert meta["a/m"]["supported_efforts"] == []


def test_build_candidates_drops_unsupported_effort():
    model = make_model(
        reasoning={"supported_efforts": ["high"], "default_effort": "high"}
    )
    args = make_args(reasoning_effort="max")
    candidates, dropped = mc.build_candidates(
        [model],
        args,
        {},
        set(),
        reasoning_by_id={
            "acme/model-a": {
                "supported_efforts": ["high"],
                "default_effort": "high",
                "default_enabled": True,
                "mandatory": False,
                "supports_max_tokens": False,
                "is_reasoning": True,
            }
        },
    )
    assert candidates == []
    assert dropped["reasoning effort"] == 1


def test_build_candidates_keeps_effective_default_effort_outside_supported_list():
    # BUG-03: default_enabled false resolves to the effective default "none"
    # even though the supported list omits it. Asking for that same effort
    # must keep the model: it runs at "none" by default.
    reasoning = {
        "supported_efforts": ["max", "high"],
        "default_effort": "none",
        "default_enabled": False,
        "mandatory": False,
        "supports_max_tokens": False,
        "is_reasoning": True,
    }
    candidates, dropped = mc.build_candidates(
        [make_model()],
        make_args(reasoning_effort="none", no_zdr=True),
        {},
        set(),
        reasoning_by_id={"acme/model-a": dict(reasoning)},
    )
    assert [c["id"] for c in candidates] == ["acme/model-a"]
    assert "reasoning effort" not in dropped


def test_build_candidates_effort_filter_control_cases():
    # Control 1: a supported effort is still kept.
    supported = {
        "supported_efforts": ["max", "high"],
        "default_effort": "none",
        "default_enabled": False,
        "mandatory": False,
        "supports_max_tokens": False,
        "is_reasoning": True,
    }
    candidates, dropped = mc.build_candidates(
        [make_model()],
        make_args(reasoning_effort="max", no_zdr=True),
        {},
        set(),
        reasoning_by_id={"acme/model-a": dict(supported)},
    )
    assert [c["id"] for c in candidates] == ["acme/model-a"]
    assert "reasoning effort" not in dropped

    # Control 2: the equality carve-out only covers the effective default.
    # A model whose default is "high" cannot run "none".
    high_default = {
        "supported_efforts": ["high"],
        "default_effort": "high",
        "default_enabled": True,
        "mandatory": False,
        "supports_max_tokens": False,
        "is_reasoning": True,
    }
    candidates, dropped = mc.build_candidates(
        [make_model()],
        make_args(reasoning_effort="none"),
        {},
        set(),
        reasoning_by_id={"acme/model-a": dict(high_default)},
    )
    assert candidates == []
    assert dropped["reasoning effort"] == 1


def test_effective_default_discrete_effort_rules():
    base = {
        "supported_efforts": ["max", "high", "none"],
        "default_enabled": True,
        "default_effort": "high",
        "mandatory": False,
        "is_reasoning": True,
    }
    assert mc.effective_default_effort(base) == "high"
    # default_enabled null is the same as true for a default request
    assert mc.effective_default_effort({**base, "default_enabled": None}) == "high"
    assert mc.effective_default_effort({**base, "default_effort": "none"}) == "none"
    assert mc.effective_default_effort({**base, "default_effort": None}) is None
    # default_enabled false wins over a non-none pre-select effort
    assert mc.effective_default_effort({**base, "default_enabled": False}) == "none"


def test_effective_default_single_state_disabled_and_token_budget():
    single = {
        "supported_efforts": [],
        "default_enabled": False,
        "default_effort": None,
        "mandatory": False,
        "is_reasoning": True,
    }
    assert mc.effective_default_effort(single) == "none"
    # A token-budget model has no discrete efforts and follows the same rule.
    assert (
        mc.effective_default_effort(
            {**single, "default_enabled": None, "supports_max_tokens": True}
        )
        == "reasoning"
    )


def test_effective_default_mandatory_disabled_is_contradictory(capsys):
    discrete = {
        "supported_efforts": ["high"],
        "default_enabled": False,
        "default_effort": "high",
        "mandatory": True,
        "is_reasoning": True,
    }
    assert mc.effective_default_effort(discrete) is None
    assert "contradict" in capsys.readouterr().err

    single = {
        "supported_efforts": [],
        "default_enabled": False,
        "default_effort": None,
        "mandatory": True,
        "is_reasoning": True,
    }
    assert mc.effective_default_effort(single) is None
    assert "contradict" in capsys.readouterr().err


def test_effective_default_non_reasoning_and_unknown():
    assert (
        mc.effective_default_effort({"supported_efforts": [], "is_reasoning": False})
        == "none"
    )
    assert (
        mc.effective_default_effort({"supported_efforts": [], "is_reasoning": None})
        is None
    )
    assert mc.effective_default_effort(None) is None


def test_supports_effort_branches():
    discrete = {
        "supported_efforts": ["high", "none"],
        "is_reasoning": True,
        "mandatory": False,
    }
    assert mc.supports_effort(discrete, "high") is True
    assert mc.supports_effort(discrete, "max") is False

    non_reasoning = {
        "supported_efforts": [],
        "is_reasoning": False,
        "mandatory": False,
    }
    assert mc.supports_effort(non_reasoning, "none") is True
    assert mc.supports_effort(non_reasoning, "high") is False

    single = {"supported_efforts": [], "is_reasoning": True, "mandatory": False}
    assert mc.supports_effort(single, "none") is True
    assert mc.supports_effort(single, "reasoning") is True
    assert mc.supports_effort(single, "high") is False
    assert mc.supports_effort({**single, "mandatory": True}, "none") is False
    assert mc.supports_effort({**single, "mandatory": True}, "reasoning") is True

    assert mc.supports_effort(None, "none") is False


def test_build_reasoning_metadata_reads_models_api_fields():
    meta = mc.build_reasoning_metadata(
        [
            {
                "id": "acme/m",
                "reasoning": {
                    "mandatory": False,
                    "default_enabled": True,
                    "supported_efforts": ["high", "none"],
                    "default_effort": "high",
                    "supports_max_tokens": False,
                },
            }
        ]
    )
    assert meta["acme/m"] == {
        "default_effort": "high",
        "supported_efforts": ["high", "none"],
        "default_enabled": True,
        "mandatory": False,
        "supports_max_tokens": False,
        "is_reasoning": True,
    }


def test_build_reasoning_metadata_supports_max_tokens_is_bool_or_null(capsys):
    meta = mc.build_reasoning_metadata(
        [
            {"id": "a/unknown", "reasoning": {"mandatory": True}},
            {"id": "a/false", "reasoning": {"supports_max_tokens": False}},
            {"id": "a/true", "reasoning": {"supports_max_tokens": True}},
            {"id": "a/string", "reasoning": {"supports_max_tokens": "false"}},
            {"id": "a/bad", "reasoning": {"supports_max_tokens": "maybe"}},
        ]
    )
    # Spec 9.1: the published field is bool or null. An omitted field is
    # unknown, not false; an explicit bool stays verbatim.
    assert meta["a/unknown"]["supports_max_tokens"] is None
    assert meta["a/false"]["supports_max_tokens"] is False
    assert meta["a/true"]["supports_max_tokens"] is True
    assert meta["a/string"]["supports_max_tokens"] is False
    assert meta["a/bad"]["supports_max_tokens"] is None
    err = capsys.readouterr().err
    assert "supports_max_tokens" in err


def test_build_reasoning_metadata_coerces_malformed_fields(capsys):
    meta = mc.build_reasoning_metadata(
        [
            {
                "id": "a/bad",
                "reasoning": {
                    "supported_efforts": ["high"],
                    "default_effort": 3.5,
                    "mandatory": [],
                    "default_enabled": 3,
                    "supports_max_tokens": "maybe",
                },
            }
        ]
    )
    # Malformed external metadata degrades to unknown; it must never reach
    # the published catalog, where a non-string default_effort would fail
    # the whole artifact.
    assert meta["a/bad"]["default_effort"] is None
    assert meta["a/bad"]["mandatory"] is None
    assert meta["a/bad"]["default_enabled"] is None
    assert meta["a/bad"]["supports_max_tokens"] is None
    err = capsys.readouterr().err
    assert "default_effort" in err
    assert "mandatory" in err
    assert "default_enabled" in err


def test_build_reasoning_metadata_frontend_fallback_and_non_reasoning():
    models = [
        {"id": "acme/m", "canonical_slug": "acme/m-20261001"},
        {"id": "acme/n", "canonical_slug": "acme/n-20261002"},
        {"id": "acme/plain"},
        {"id": "acme/unknown"},
    ]
    frontend = [
        {
            "slug": "acme/m",
            "permaslug": "acme/m-20261001",
            "supports_reasoning": True,
            "reasoning_config": {
                "supports_reasoning_effort": True,
                "supported_reasoning_efforts": ["low", "high"],
                "default_reasoning_effort": "high",
                "default_reasoning_enabled": True,
                "is_mandatory_reasoning": False,
                "supports_reasoning_max_tokens": False,
            },
        },
        {
            "slug": "acme/n",
            "permaslug": "acme/n-20261002",
            "supports_reasoning": False,
            "reasoning_config": {},
        },
        {
            "slug": "acme/plain",
            "permaslug": "acme/plain-20261003",
            "supports_reasoning": False,
            "reasoning_config": {},
        },
    ]
    meta = mc.build_reasoning_metadata(models, frontend)
    assert meta["acme/m"]["default_effort"] == "high"
    assert meta["acme/m"]["supported_efforts"] == ["low", "high"]
    assert meta["acme/m"]["is_reasoning"] is True
    assert meta["acme/n"]["default_effort"] == "none"
    assert meta["acme/n"]["is_reasoning"] is False
    # exact slug fallback, even though the permaslug is dated
    assert meta["acme/plain"]["default_effort"] == "none"
    assert meta["acme/plain"]["is_reasoning"] is False
    # no models API field and no frontend entry -> unknown, not non-reasoning
    assert meta["acme/unknown"]["default_effort"] is None
    assert meta["acme/unknown"]["is_reasoning"] is None


def test_build_reasoning_metadata_models_api_wins_over_frontend():
    models = [
        {
            "id": "acme/m",
            "canonical_slug": "acme/m-1",
            "reasoning": {
                "mandatory": False,
                "default_enabled": True,
                "supported_efforts": ["max"],
                "default_effort": "max",
            },
        }
    ]
    frontend = [
        {
            "slug": "acme/m",
            "permaslug": "acme/m-1",
            "supports_reasoning": True,
            "reasoning_config": {
                "supported_reasoning_efforts": ["low"],
                "default_reasoning_effort": "low",
                "default_reasoning_enabled": True,
            },
        }
    ]
    meta = mc.build_reasoning_metadata(models, frontend)["acme/m"]
    assert meta["supported_efforts"] == ["max"]
    assert meta["default_effort"] == "max"


def test_build_reasoning_metadata_fields_fall_back_individually():
    # A models API reasoning object that omits a field takes that field
    # from the frontend config (spec §5.2 precedence per field).
    models = [
        {
            "id": "acme/m",
            "canonical_slug": "acme/m-1",
            "reasoning": {"supported_efforts": ["high"]},
        }
    ]
    frontend = [
        {
            "slug": "acme/m",
            "permaslug": "acme/m-1",
            "supports_reasoning": True,
            "reasoning_config": {
                "default_reasoning_effort": "high",
                "default_reasoning_enabled": True,
                "is_mandatory_reasoning": False,
            },
        }
    ]
    meta = mc.build_reasoning_metadata(models, frontend)["acme/m"]
    assert meta["supported_efforts"] == ["high"]
    assert meta["default_enabled"] is True
    assert meta["mandatory"] is False
    assert meta["default_effort"] == "high"


def test_build_reasoning_metadata_primary_omitted_efforts_stay_single_state():
    # Live shape (google/gemini-2.5-pro, openai/o3): the models API reasoning
    # object omits supported_efforts while the frontend's null marker says
    # "all efforts". The primary object is authoritative: no discrete
    # efforts, so the mandatory single-state rule applies.
    models = [
        {
            "id": "acme/m",
            "canonical_slug": "acme/m-1",
            "reasoning": {"mandatory": True},
        }
    ]
    frontend = [
        {
            "slug": "acme/m",
            "permaslug": "acme/m-1",
            "supports_reasoning": True,
            "reasoning_config": {
                "supported_reasoning_efforts": None,
                "is_mandatory_reasoning": True,
            },
        }
    ]
    meta = mc.build_reasoning_metadata(models, frontend)["acme/m"]
    assert meta["supported_efforts"] == []
    assert meta["default_effort"] == "reasoning"


def test_build_reasoning_metadata_frontend_null_efforts_expand():
    frontend = [
        {
            "slug": "acme/m",
            "permaslug": "acme/m-1",
            "supports_reasoning": True,
            "reasoning_config": {
                "supported_reasoning_efforts": None,
                "default_reasoning_enabled": True,
            },
        }
    ]
    meta = mc.build_reasoning_metadata(
        [{"id": "acme/m", "canonical_slug": "acme/m-1"}], frontend
    )["acme/m"]
    assert meta["supported_efforts"] == list(mc.REASONING_GATEWAY_EFFORTS)
    # no pre-select effort -> unknown, even though reasoning is enabled
    assert meta["default_effort"] is None


def test_build_reasoning_metadata_frontend_single_state():
    frontend = [
        {
            "slug": "acme/m",
            "permaslug": "acme/m-1",
            "supports_reasoning": True,
            "reasoning_config": {"is_mandatory_reasoning": False},
        }
    ]
    meta = mc.build_reasoning_metadata(
        [{"id": "acme/m", "canonical_slug": "acme/m-1"}], frontend
    )["acme/m"]
    assert meta["supported_efforts"] == []
    assert meta["default_effort"] == "reasoning"


def test_build_reasoning_metadata_frontend_pairs_permaslug_across_slugs():
    # The frontend's own slug differs from the candidate id; the catalog's
    # permaslug pairing (pass 1) still attaches the config.
    models = [{"id": "acme/snapshot", "canonical_slug": "acme/snapshot-20261017"}]
    frontend = [
        {
            "slug": "acme/base",
            "permaslug": "acme/snapshot-20261017",
            "supports_reasoning": True,
            "reasoning_config": {
                "supported_reasoning_efforts": ["high"],
                "default_reasoning_effort": "high",
                "default_reasoning_enabled": True,
            },
        }
    ]
    meta = mc.build_reasoning_metadata(models, frontend)["acme/snapshot"]
    assert meta["supported_efforts"] == ["high"]
    assert meta["default_effort"] == "high"


def test_build_reasoning_metadata_frontend_pairs_by_date_strip():
    # An unpaired dated permaslug strips to the candidate id (pass 3).
    models = [{"id": "acme/m"}]
    frontend = [
        {
            "slug": "acme/other",
            "permaslug": "acme/m-20261017",
            "supports_reasoning": True,
            "reasoning_config": {
                "supported_reasoning_efforts": ["low"],
                "default_reasoning_effort": "low",
                "default_reasoning_enabled": True,
            },
        }
    ]
    meta = mc.build_reasoning_metadata(models, frontend)["acme/m"]
    assert meta["supported_efforts"] == ["low"]
    assert meta["default_effort"] == "low"


def test_build_reasoning_metadata_exact_slug_beats_shared_canonical():
    # Base and :batch share a canonical slug, and the permaslug pairing
    # assigns it to the last listing. The undated exact slug must still
    # resolve the base (live: openai/gpt-4o and its :batch variant).
    models = [
        {"id": "acme/m", "canonical_slug": "acme/m"},
        {"id": "acme/m:batch", "canonical_slug": "acme/m"},
    ]
    frontend = [
        {
            "slug": "acme/m",
            "permaslug": "acme/m",
            "supports_reasoning": False,
            "reasoning_config": {},
        }
    ]
    meta = mc.build_reasoning_metadata(models, frontend)
    assert meta["acme/m"]["is_reasoning"] is False
    assert meta["acme/m"]["default_effort"] == "none"
    assert meta["acme/m:batch"]["is_reasoning"] is False


def test_build_candidates_attaches_reasoning_fields():
    reasoning = {
        "supported_efforts": ["high", "max"],
        "default_effort": "high",
        "default_enabled": True,
        "mandatory": False,
        "supports_max_tokens": False,
        "is_reasoning": True,
    }
    candidates, dropped = mc.build_candidates(
        [make_model()],
        make_args(reasoning_effort="high"),
        {},
        {"acme/model-a"},
        reasoning_by_id={"acme/model-a": reasoning},
    )
    assert dropped == {}
    assert candidates[0]["reasoning"] == reasoning
    assert candidates[0]["default_effort"] == "high"
    assert candidates[0]["supported_efforts"] == ["high", "max"]


def test_build_candidates_default_effort_does_not_filter():
    candidates, dropped = mc.build_candidates(
        [make_model()],
        make_args(),
        {},
        {"acme/model-a"},
        reasoning_by_id={
            "acme/model-a": {
                "supported_efforts": [],
                "default_effort": None,
                "default_enabled": None,
                "mandatory": None,
                "supports_max_tokens": False,
                "is_reasoning": None,
            }
        },
    )
    assert len(candidates) == 1
    assert dropped == {}
    assert candidates[0]["default_effort"] is None
    assert candidates[0]["supported_efforts"] == []


def test_build_candidates_reasoning_filter_appends_filtered_entry():
    filtered = []
    candidates, dropped = mc.build_candidates(
        [make_model()],
        make_args(reasoning_effort="max"),
        {},
        {"acme/model-a"},
        filtered,
        reasoning_by_id={
            "acme/model-a": {
                "supported_efforts": ["high"],
                "default_effort": "high",
                "default_enabled": True,
                "mandatory": False,
                "supports_max_tokens": False,
                "is_reasoning": True,
            }
        },
    )
    assert candidates == []
    assert dropped == {"reasoning effort": 1}
    assert filtered == [
        {
            "id": "acme/model-a",
            "name": "Acme: Model A",
            "reasons": ["reasoning effort"],
        }
    ]


def test_catalog_drop_reasons_includes_reasoning_effort():
    assert "reasoning effort" in mc.CATALOG_DROP_REASONS


def test_parse_args_reasoning_effort():
    assert mc.parse_args([]).reasoning_effort == "default"
    assert mc.parse_args(["--reasoning-effort", "max"]).reasoning_effort == "max"
    with pytest.raises(SystemExit):
        mc.parse_args(["--reasoning-effort", "reasoning"])


# ---------------------------------------------------------------------------
# aa_api_entries (AA V2 language models/free parser)
# ---------------------------------------------------------------------------


def aa_page(items, page=1, total_pages=1, has_more=False, with_pagination=True):
    out = {"tier": "free", "intelligence_index_version": 4.3, "data": items}
    if with_pagination:
        out["pagination"] = {
            "page": page,
            "page_size": 200,
            "total_pages": total_pages,
            "has_more": has_more,
        }
    return out


def aa_item(slug="acme/m", name="M", index=42.0, *, id=None, cost=None, creator=None):
    item = {"name": name}
    if id is not None:
        item["id"] = id
    if slug is not None:
        item["slug"] = slug
    if index is not None:
        item["evaluations"] = {"artificial_analysis_intelligence_index": index}
    if cost is not None:
        item["artificial_analysis_intelligence_index_cost"] = {
            "cost_per_task": {"total_cost": cost}
        }
    if creator is not None:
        item["model_creator"] = {"id": "creator-uuid", "name": creator}
    return item


def _without_as_of(entries):
    """Strip the dynamic per-fetch as_of stamp so tests can compare entries."""
    return [{k: v for k, v in entry.items() if k != "as_of"} for entry in entries]


_AA_ENTRY_UNSET = object()


def aa_entry(
    key,
    name,
    index,
    *,
    id=None,
    slug=_AA_ENTRY_UNSET,
    base=None,
    effort=None,
    creator=None,
    cost=None,
):
    """Expected aa_api_entries shape (minus the dynamic as_of stamp)."""
    return {
        "key": key,
        "name": name,
        "index": index,
        "id": id,
        "slug": key if slug is _AA_ENTRY_UNSET else slug,
        "base": key if base is None else base,
        "effort": effort,
        "creator": creator,
        "cost_per_task": cost,
    }


def test_aa_api_sends_auth_header_and_timeout(monkeypatch):
    seen = {}

    def fake_fetch(url, headers=None, timeout=None, **k):
        seen.update(headers=headers, timeout=timeout)
        return aa_page([aa_item()])

    monkeypatch.setattr(mc, "fetch_json", fake_fetch)
    mc.aa_api_entries("dummy-key")
    assert seen["headers"] == {"x-api-key": "dummy-key"}
    assert seen["timeout"] == 30


def test_aa_api_entries_carry_fetched_at_as_of_and_index_version(monkeypatch):
    # The AA API fetch stamps every entry with a single as_of (the fetch
    # timestamp, ISO format) and propagates the envelope's
    # intelligence_index_version as the whole-fetch provenance fact.
    monkeypatch.setattr(mc, "fetch_json", lambda *a, **k: aa_page([aa_item()]))
    entries, version = mc.aa_api_entries("dummy-key")
    assert version == 4.3
    assert len(entries) == 1
    assert entries[0]["index"] == 42.0
    # one fetch timestamp shared by the whole fetch, ISO format
    fetched_at = datetime.fromisoformat(entries[0]["as_of"])
    assert (datetime.now(timezone.utc) - fetched_at).total_seconds() < 60


def test_aa_api_parses_documented_shape(monkeypatch):
    payload = aa_page([aa_item("acme/a", "A", 42.5), aa_item("acme/b", "B", None)])
    monkeypatch.setattr(mc, "fetch_json", lambda *a, **k: payload)
    entries, _ = mc.aa_api_entries("dummy-key")
    assert _without_as_of(entries) == [aa_entry("acme/a", "A", 42.5)]


def test_aa_api_entries_carry_effort_creator_and_cost(monkeypatch):
    payload = aa_page(
        [
            aa_item(
                "claude-sonnet-5-5-high",
                "Claude Sonnet 5.5 (High, Default Fallback)",
                46.8,
                id="uuid-high",
                cost=0.8844,
                creator="Anthropic",
            ),
            aa_item(
                "claude-sonnet-5-5",
                "Claude Sonnet 5.5 (Max, Default Fallback)",
                56.0,
                id="uuid-max",
                cost=5.4609,
                creator="Anthropic",
            ),
        ]
    )
    monkeypatch.setattr(mc, "fetch_json", lambda *a, **k: payload)
    entries, _ = mc.aa_api_entries("dummy-key")
    high, max_row = entries
    assert high["id"] == "uuid-high"
    assert high["slug"] == "claude-sonnet-5-5-high"
    assert high["base"] == "claude-sonnet-5-5"
    assert high["effort"] == "high"
    assert high["creator"] == "Anthropic"
    assert high["cost_per_task"] == 0.8844
    assert max_row["effort"] == "max"
    assert max_row["base"] == "claude-sonnet-5-5"


def test_aa_api_dedupes_variants_by_id_not_slug(monkeypatch):
    # Two effort rows can share a slug (the primary max row often has no
    # suffix); they must survive as separate entries, deduped by AA id.
    payload = aa_page(
        [
            aa_item("model-a", "Model A (Max, Default Fallback)", 56.0, id="1"),
            aa_item("model-a", "Model A (High, Default Fallback)", 46.8, id="2"),
        ]
    )
    monkeypatch.setattr(mc, "fetch_json", lambda *a, **k: payload)
    entries, _ = mc.aa_api_entries("dummy-key")
    assert [entry["effort"] for entry in entries] == ["max", "high"]


def test_aa_api_skips_unmeasurable_indexes(monkeypatch):
    payload = aa_page(
        [
            aa_item("acme/ok", "OK", 5.0),  # the only measurable item
            aa_item("acme/a", "A", None),  # no evaluations at all
            {
                "slug": "acme/b",
                "name": "B",
                "evaluations": {"artificial_analysis_intelligence_index": None},
            },
            {
                "slug": "acme/c",
                "name": "C",
                "evaluations": {"artificial_analysis_intelligence_index": "42"},
            },
            {
                "slug": "acme/d",
                "name": "D",
                "evaluations": {"artificial_analysis_intelligence_index": True},
            },
            {
                "slug": "acme/e",
                "name": "E",
                "evaluations": {"artificial_analysis_intelligence_index": float("nan")},
            },
            {"slug": "acme/f", "name": "F", "evaluations": {}},  # empty dict
            {"slug": "acme/g", "name": "G", "evaluations": [42.0]},  # non-dict
        ]
    )
    monkeypatch.setattr(mc, "fetch_json", lambda *a, **k: payload)
    assert _without_as_of(mc.aa_api_entries("dummy-key")[0]) == [
        aa_entry("acme/ok", "OK", 5.0)
    ]


def test_aa_api_skips_index_too_large_for_float(monkeypatch):
    # A huge JSON integer is a valid int but overflows float conversion; it
    # must be skipped like any unmeasurable value, not abort the dataset.
    payload = aa_page(
        [aa_item("acme/huge", "Huge", 10**400), aa_item("acme/ok", "OK", 5.0)]
    )
    monkeypatch.setattr(mc, "fetch_json", lambda *a, **k: payload)
    assert _without_as_of(mc.aa_api_entries("dummy-key")[0]) == [
        aa_entry("acme/ok", "OK", 5.0)
    ]


def test_aa_api_non_list_data_on_page_one_yields_nothing(monkeypatch):
    fake, calls = aa_url_stub({1: aa_page(5)})
    monkeypatch.setattr(mc, "fetch_json", fake)
    assert mc.aa_api_entries("dummy-key")[0] == []
    assert calls == [f"{mc.AA_API_URL}?page=1"]


def test_aa_api_non_list_data_mid_pagination_keeps_earlier_pages(monkeypatch):
    fake, calls = aa_url_stub(
        {
            1: aa_page([aa_item("acme/a", "A", 40.0)], 1, 2, True),
            2: aa_page(5, 2, 2, False),
        }
    )
    monkeypatch.setattr(mc, "fetch_json", fake)
    assert _without_as_of(mc.aa_api_entries("dummy-key")[0]) == [
        aa_entry("acme/a", "A", 40.0)
    ]
    assert calls == [f"{mc.AA_API_URL}?page=1", f"{mc.AA_API_URL}?page=2"]


def test_aa_api_key_falls_back_to_name(monkeypatch):
    payload = aa_page(
        [
            {
                "name": "Some Model",
                "evaluations": {"artificial_analysis_intelligence_index": 7.0},
            }
        ]
    )
    monkeypatch.setattr(mc, "fetch_json", lambda *a, **k: payload)
    entries, _ = mc.aa_api_entries("dummy-key")
    assert _without_as_of(entries) == [
        aa_entry("Some Model", "Some Model", 7.0, slug=None)
    ]


def test_aa_api_ignores_non_dict_data_items(monkeypatch):
    payload = aa_page(["junk", 3, aa_item()])
    monkeypatch.setattr(mc, "fetch_json", lambda *a, **k: payload)
    assert len(mc.aa_api_entries("dummy-key")[0]) == 1


def aa_url_stub(payloads_by_page):
    calls = []

    def fake_fetch(url, *a, **k):
        calls.append(url)
        page = int(url.split("page=")[-1])
        return payloads_by_page[page]

    return fake_fetch, calls


def test_aa_api_follows_pagination_and_dedupes(monkeypatch):
    pages = {
        1: aa_page([aa_item("acme/a", "A", 40.0)], 1, 2, True),
        2: aa_page(
            [aa_item("acme/a", "A", 41.0), aa_item("acme/b", "B", 9.0)], 2, 2, False
        ),
    }
    fake, calls = aa_url_stub(pages)
    monkeypatch.setattr(mc, "fetch_json", fake)
    entries, _ = mc.aa_api_entries("dummy-key")
    assert calls == [f"{mc.AA_API_URL}?page=1", f"{mc.AA_API_URL}?page=2"]
    assert _without_as_of(entries) == [
        aa_entry("acme/a", "A", 40.0),  # first page wins
        aa_entry("acme/b", "B", 9.0),
    ]


def test_aa_api_caps_runaway_pagination(monkeypatch, capsys):
    fake, calls = aa_url_stub(
        {p: aa_page([aa_item()], p, 999, True) for p in range(1, 30)}
    )
    monkeypatch.setattr(mc, "fetch_json", fake)
    mc.aa_api_entries("dummy-key")
    assert len(calls) == 25
    assert calls[-1] == f"{mc.AA_API_URL}?page=25"


def test_aa_api_degrades_when_pagination_missing(monkeypatch):
    fake, calls = aa_url_stub({1: aa_page([aa_item()], with_pagination=False)})
    monkeypatch.setattr(mc, "fetch_json", fake)
    entries, _ = mc.aa_api_entries("dummy-key")
    assert calls == [f"{mc.AA_API_URL}?page=1"]
    assert len(entries) == 1


def test_aa_api_survives_null_total_pages(monkeypatch):
    page1 = aa_page([aa_item()], 1, 2, True)
    page1["pagination"]["total_pages"] = None
    fake, calls = aa_url_stub({1: page1, 2: aa_page([], 2, None, False)})
    monkeypatch.setattr(mc, "fetch_json", fake)
    mc.aa_api_entries("dummy-key")
    assert len(calls) == 2  # no TypeError; continued past the null


def test_aa_api_stops_when_page_reaches_total_pages(monkeypatch):
    # has_more=True on the last page is the documented terminal state: page
    # == total_pages must stop, not loop (or error past the last fixture).
    fake, calls = aa_url_stub(
        {
            1: aa_page([aa_item()], 1, 2, True),
            2: aa_page([aa_item("acme/b", "B", 9.0)], 2, 2, True),
        }
    )
    monkeypatch.setattr(mc, "fetch_json", fake)
    entries, _ = mc.aa_api_entries("dummy-key")
    assert calls == [f"{mc.AA_API_URL}?page=1", f"{mc.AA_API_URL}?page=2"]
    assert len(entries) == 2


def test_aa_api_treats_truthy_non_true_has_more_as_stop(monkeypatch):
    # The contract is has_more == True exactly; a truthy non-True value is
    # an undocumented shape and must stop, not keep paginating.
    fake, calls = aa_url_stub({1: aa_page([aa_item()], 1, 5, "true")})
    monkeypatch.setattr(mc, "fetch_json", fake)
    mc.aa_api_entries("dummy-key")
    assert calls == [f"{mc.AA_API_URL}?page=1"]


def test_aa_api_best_effort_on_mid_pagination_failure(monkeypatch, capsys):
    def fake_fetch(url, *a, **k):
        if url.endswith("page=2"):
            raise RuntimeError("boom")
        return aa_page([aa_item("acme/a", "A", 40.0)], 1, 2, True)

    monkeypatch.setattr(mc, "fetch_json", fake_fetch)
    entries, _ = mc.aa_api_entries("dummy-key")
    assert _without_as_of(entries) == [aa_entry("acme/a", "A", 40.0)]
    assert (
        "AA API page 2 failed (boom); using entries collected so far"
        in capsys.readouterr().err
    )


def _no_page_fetch(monkeypatch):
    """Record any raw http_get (the removed AA page scrape used it)."""
    pages = []

    def fake_http_get(url, *a, **k):
        pages.append(url)
        raise AssertionError(f"unexpected page fetch: {url}")

    monkeypatch.setattr(mc, "http_get", fake_http_get)
    return pages


def test_fetch_aa_entries_api_failure_returns_empty(monkeypatch, capsys):
    monkeypatch.delenv("AA_API_KEY", raising=False)
    pages = _no_page_fetch(monkeypatch)
    monkeypatch.setattr(
        mc,
        "fetch_json",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("401")),
    )
    result = mc.fetch_aa_entries(make_args(aa_api_key="dummy-key"))
    assert result == ([], None, False, None)
    assert pages == []  # no page-scrape fallback
    err = capsys.readouterr().err
    assert "AA API request failed (401)" in err
    assert "scrape" not in err


def test_fetch_aa_entries_api_empty_returns_empty(monkeypatch, capsys):
    monkeypatch.delenv("AA_API_KEY", raising=False)
    pages = _no_page_fetch(monkeypatch)
    monkeypatch.setattr(mc, "fetch_json", lambda *a, **k: aa_page([]))
    result = mc.fetch_aa_entries(make_args(aa_api_key="dummy-key"))
    assert result == ([], None, False, None)
    assert pages == []
    err = capsys.readouterr().err
    assert "AA API returned no intelligence scores" in err
    assert "scrape" not in err


def test_fetch_aa_entries_no_key_is_silent(monkeypatch, capsys):
    # Keyless runs get AA data only through OpenRouter benchmarks; the
    # catalog records the absence, so nothing is printed and nothing fetched.
    monkeypatch.delenv("AA_API_KEY", raising=False)
    pages = _no_page_fetch(monkeypatch)
    calls = []
    monkeypatch.setattr(
        mc,
        "fetch_json",
        lambda *a, **k: (
            calls.append(1)
            or (_ for _ in ()).throw(AssertionError("API called without key"))
        ),
    )
    result = mc.fetch_aa_entries(make_args())
    assert result == ([], None, False, None)
    assert calls == [] and pages == []
    assert capsys.readouterr().err == ""


def test_fetch_aa_entries_ignores_cached_scrape_source(monkeypatch, tmp_path):
    # A cache entry written by the removed page scrape carries a source
    # build_catalog no longer knows (it would raise); it must never be read.
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.delenv("AA_API_KEY", raising=False)
    _no_page_fetch(monkeypatch)
    mc.save_cache(
        mc.AA_ENTRIES_CACHE,
        {
            "entries": [{"key": "s", "name": "S", "index": 1.0, "estimated": False}],
            "source": "AA page scrape",
        },
    )
    result = mc.fetch_aa_entries(make_args(no_cache=False, cache_ttl=3600))
    assert result == ([], None, False, None)


def test_aa_page_scrape_is_gone():
    assert not hasattr(mc, "aa_scrape_entries")
    assert not hasattr(mc, "AA_MODELS_PAGE_URL")


# ---------------------------------------------------------------------------
# realistic OpenRouter frontend payload (discount + ZDR key shape)
# ---------------------------------------------------------------------------


def _realistic_openrouter_payload():
    """Mirror the real frontend `models/find` response shape with real slugs."""
    return {
        "data": {
            "models": [
                {
                    "slug": "openai/gpt-4o",
                    "endpoint": {
                        "variant": "standard",
                        "pricing": {"prompt": "0.1", "discount": 0.5},
                    },
                },
                {
                    "slug": "openai/gpt-4o",
                    "endpoint": {
                        "variant": "free",
                        "pricing": {"prompt": "0", "discount": 0},
                    },
                },
                {
                    "slug": "anthropic/claude-sonnet-4-20250514",
                    "endpoint": {
                        "variant": "batch",
                        "pricing": {"prompt": "0.1", "discount": 0.25},
                    },
                },
                {
                    "slug": "~acme/private",
                    "endpoint": {
                        "variant": "standard",
                        "pricing": {"prompt": "0.1", "discount": 0.9},
                    },
                },
            ]
        }
    }


def _realistic_zdr_payload():
    """Mirror the real `/api/v1/endpoints/zdr` response with real ids."""
    return {
        "data": [
            zdr_endpoint("openai/gpt-4o", provider="Azure"),
            zdr_endpoint("openai/gpt-4o", provider="OpenAI"),
            zdr_endpoint("openai/gpt-4o:free", provider="OpenAI"),
            zdr_endpoint("anthropic/claude-sonnet-4-20250514:batch", provider="Google"),
        ]
    }


def test_fetch_openrouter_frontend_realistic_slugs(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    calls = []
    stub_frontend(
        monkeypatch, calls, _realistic_openrouter_payload(), _realistic_zdr_payload()
    )
    discounts, zdr_ids, aa_by_id, cache_hits, frontend_models = (
        mc.fetch_openrouter_frontend(make_args(no_cache=True))
    )
    assert discounts == {
        "openai/gpt-4o": 0.5,
        "openai/gpt-4o:free": 0.0,
        "anthropic/claude-sonnet-4-20250514:batch": 0.25,
    }
    assert zdr_ids == {
        "openai/gpt-4o",
        "openai/gpt-4o:free",
        "anthropic/claude-sonnet-4-20250514:batch",
    }
    assert aa_by_id == {}


# ---------------------------------------------------------------------------
# resolve_quality (records with source/as_of/drift; AA primary, OR last resort)
# ---------------------------------------------------------------------------


def test_resolve_quality_returns_records_with_source_and_as_of():
    candidates = [{"id": "acme/model-a", "name": "Acme: Model A"}]
    exact, fuzzy = mc.build_aa_lookup(
        [
            {
                "key": "acme/model-a",
                "name": "Model A",
                "index": 55.0,
                "as_of": "2026-10-23T12:00:00+00:00",
            }
        ]
    )
    quality, source = mc.resolve_quality(candidates, {}, exact, fuzzy, "AA API v2")
    assert quality == {
        "acme/model-a": {
            "index": 55.0,
            "source": "aa",
            "as_of": "2026-10-23T12:00:00+00:00",
            "drift": None,
            "or_index": None,
            "or_as_of": None,
        }
    }
    assert source == {"acme/model-a": "aa"}


def test_resolve_quality_aa_beats_openrouter_for_same_model():
    # SUPERSEDES test_resolve_quality_prefers_openrouter_benchmarks: the AA
    # API tier is now primary, OpenRouter's benchmark trio is the last resort.
    candidates = [{"id": "acme/model-a", "name": "Acme: Model A"}]
    aa_by_id = {
        "acme/model-a": {
            "intelligence_index": 57.5,
            "coding_index": 60.0,
            "agentic_index": 55.0,
            "as_of": "20260915",
        }
    }
    exact, fuzzy = mc.build_aa_lookup(
        [
            {
                "key": "acme/model-a",
                "name": "Model A",
                "index": 55.0,
                "as_of": "2026-10-23T00:00:00+00:00",
            }
        ]
    )
    quality, source = mc.resolve_quality(
        candidates, aa_by_id, exact, fuzzy, "AA API v2"
    )
    assert quality["acme/model-a"]["source"] == "aa"
    assert quality["acme/model-a"]["index"] == 55.0
    assert quality["acme/model-a"]["as_of"] == "2026-10-23T00:00:00+00:00"
    # delta |55.0 - 57.5| = 2.5 < 3.0 -> not drifted, but the OR pair is kept
    assert quality["acme/model-a"]["drift"] is False
    assert quality["acme/model-a"]["or_index"] == 57.5
    assert quality["acme/model-a"]["or_as_of"] == "20260915"
    assert source == {"acme/model-a": "aa"}


def test_resolve_quality_or_tier_labeled_openrouter_with_permaslug_as_of():
    # OR-only hit -> source "openrouter", as_of comes straight from the
    # Task 1 record (the permaslug snapshot date).
    candidates = [{"id": "acme/model-a:free", "name": "Acme: Model A (free)"}]
    aa_by_id = {
        "acme/model-a": {
            "intelligence_index": 57.5,
            "coding_index": 60.0,
            "agentic_index": 55.0,
            "as_of": "20260915",
        }
    }
    quality, source = mc.resolve_quality(candidates, aa_by_id, {}, {}, None)
    assert quality["acme/model-a:free"] == {
        "index": 57.5,
        "source": "openrouter",
        "as_of": "20260915",
        "drift": None,
        "or_index": None,
        "or_as_of": None,
    }
    assert source == {"acme/model-a:free": "openrouter"}


def test_resolve_quality_drift_computed_against_or_trio():
    candidates = [
        {"id": "acme/model-a", "name": "Acme: Model A"},
        {"id": "acme/model-b", "name": "Acme: Model B"},
        {"id": "acme/model-c", "name": "Acme: Model C"},
    ]
    aa_by_id = {
        "acme/model-a": {"intelligence_index": 39.5, "as_of": "20260915"},
        "acme/model-b": {"intelligence_index": 42.2, "as_of": "20260915"},
        "acme/model-c": {"intelligence_index": 50.0, "as_of": "20260915"},
    }
    exact, fuzzy = mc.build_aa_lookup(
        [
            {
                "key": "acme/model-a",
                "name": "Model A",
                "index": 45.1,
                "as_of": "2026-10-23T00:00:00+00:00",
            },
            {
                "key": "acme/model-b",
                "name": "Model B",
                "index": 45.1,
                "as_of": "2026-10-23T00:00:00+00:00",
            },
        ]
    )
    quality, source = mc.resolve_quality(
        candidates, aa_by_id, exact, fuzzy, "AA API v2"
    )
    # delta 45.1 - 39.5 = 5.6 >= 3.0 -> drifted; the OR pair is populated
    assert quality["acme/model-a"]["drift"] is True
    assert quality["acme/model-a"]["or_index"] == 39.5
    assert quality["acme/model-a"]["or_as_of"] == "20260915"
    # delta 45.1 - 42.2 = 2.9 < 3.0 -> not drifted; the OR pair is populated
    assert quality["acme/model-b"]["drift"] is False
    assert quality["acme/model-b"]["or_index"] == 42.2
    assert quality["acme/model-b"]["or_as_of"] == "20260915"
    # the OR winner has no second source to check against
    assert quality["acme/model-c"]["drift"] is None
    assert quality["acme/model-c"]["or_index"] is None
    assert quality["acme/model-c"]["or_as_of"] is None
    assert source["acme/model-c"] == "openrouter"


def test_resolve_quality_falls_back_to_exact_aa_and_records_provenance():
    # AA exact tier (tier 1: base slug) matches; the record carries the
    # entry's provenance (source "aa", as_of from the entry itself).
    candidates = [{"id": "acme/model-b", "name": "Acme: Model B"}]
    exact, fuzzy = mc.build_aa_lookup(
        [
            {
                "key": "openai/model-b",
                "name": "Model B",
                "index": 51.2,
                "as_of": "2026-10-23T00:00:00+00:00",
            }
        ]
    )
    quality, source = mc.resolve_quality(candidates, {}, exact, fuzzy, "AA API v2")
    assert quality == {
        "acme/model-b": {
            "index": 51.2,
            "source": "aa",
            "as_of": "2026-10-23T00:00:00+00:00",
            "drift": None,
            "or_index": None,
            "or_as_of": None,
        }
    }
    assert source == {"acme/model-b": "aa"}


def test_resolve_quality_variant_inherits_base_trio():
    candidates = [{"id": "acme/model-a:free", "name": "Acme: Model A (free)"}]
    aa_by_id = {"acme/model-a": {"intelligence_index": 57.5, "as_of": "20260915"}}
    quality, source = mc.resolve_quality(candidates, aa_by_id, {}, {}, None)
    assert quality["acme/model-a:free"] == {
        "index": 57.5,
        "source": "openrouter",
        "as_of": "20260915",
        "drift": None,
        "or_index": None,
        "or_as_of": None,
    }
    assert source == {"acme/model-a:free": "openrouter"}


def test_resolve_quality_trio_without_intelligence_falls_through():
    # A trio without intelligence_index is not a quality source: the AA API
    # entry below it wins.
    candidates = [{"id": "acme/model-a", "name": "Acme: Model A"}]
    aa_by_id = {"acme/model-a": {"coding_index": 70.0, "as_of": "20260915"}}
    exact, fuzzy = mc.build_aa_lookup(
        [
            {
                "key": "acme/model-a",
                "name": "Model A",
                "index": 44.0,
                "as_of": "2026-10-23T00:00:00+00:00",
            }
        ]
    )
    quality, source = mc.resolve_quality(
        candidates, aa_by_id, exact, fuzzy, "AA API v2"
    )
    assert quality["acme/model-a"] == {
        "index": 44.0,
        "source": "aa",
        "as_of": "2026-10-23T00:00:00+00:00",
        "drift": None,
        "or_index": None,
        "or_as_of": None,
    }
    assert source == {"acme/model-a": "aa"}


def test_resolve_quality_unmatched_is_none():
    candidates = [{"id": "acme/model-a", "name": "Acme: Model A"}]
    quality, source = mc.resolve_quality(candidates, {}, {}, {}, None)
    assert quality == {}
    assert source == {}


def test_select_default_depth_skips_huge_integer_index():
    assert (
        mc.select_default_depth(
            {},
            {
                "max": {"effort": "max", "index": 10**400},
                "high": {"effort": "high", "index": 42.0},
            },
        )
        == "high"
    )
    assert (
        mc.select_default_depth({}, {"max": {"effort": "max", "index": 10**400}})
        is None
    )


def test_resolve_quality_depth_best_selectable_beats_default():
    # Default `high` measured at 46.8 but selectable max measures 56.0:
    # the score uses the best selectable depth.
    cand = {
        "id": "acme/model-a",
        "name": "Acme: Model A",
        "default_effort": "high",
        "supported_efforts": ["max", "high", "low"],
        "reasoning": {"is_reasoning": True},
    }
    lookup = mc.build_aa_depth_lookup(
        [
            {
                "id": "1",
                "name": "Model A (High)",
                "slug": "model-a-high",
                "index": 46.8,
                "as_of": "t1",
            },
            {
                "id": "2",
                "name": "Model A (Max)",
                "slug": "model-a",
                "index": 56.0,
                "as_of": "t1",
            },
        ]
    )
    quality, source = mc.resolve_quality(
        [cand], {}, {}, {}, "AA API v2", depth_lookup=lookup, requested_effort="default"
    )
    assert quality["acme/model-a"]["index"] == 56.0
    assert quality["acme/model-a"]["effort"] == "max"


def test_resolve_quality_depth_unselectable_deeper_record_ignored():
    # AA measured max 56.0 but OpenRouter cannot run max: high 46.8 wins.
    cand = {
        "id": "acme/model-a",
        "name": "Acme: Model A",
        "default_effort": "high",
        "supported_efforts": ["high", "low"],
        "reasoning": {"is_reasoning": True},
    }
    lookup = mc.build_aa_depth_lookup(
        [
            {
                "id": "1",
                "name": "Model A (High)",
                "slug": "model-a-high",
                "index": 46.8,
                "as_of": "t1",
            },
            {
                "id": "2",
                "name": "Model A (Max)",
                "slug": "model-a",
                "index": 56.0,
                "as_of": "t1",
            },
        ]
    )
    quality, _ = mc.resolve_quality(
        [cand], {}, {}, {}, "AA API v2", depth_lookup=lookup, requested_effort="default"
    )
    assert quality["acme/model-a"]["index"] == 46.8
    assert quality["acme/model-a"]["effort"] == "high"


def test_resolve_quality_depth_tie_goes_to_deeper_effort():
    cand = {
        "id": "acme/model-a",
        "name": "Acme: Model A",
        "default_effort": "medium",
        "supported_efforts": ["max", "high", "medium"],
        "reasoning": {"is_reasoning": True},
    }
    lookup = mc.build_aa_depth_lookup(
        [
            {
                "id": "1",
                "name": "Model A (High)",
                "slug": "model-a-high",
                "index": 50.0,
                "as_of": "t1",
            },
            {
                "id": "2",
                "name": "Model A (Max)",
                "slug": "model-a",
                "index": 50.0,
                "as_of": "t1",
            },
        ]
    )
    quality, _ = mc.resolve_quality(
        [cand], {}, {}, {}, "AA API v2", depth_lookup=lookup, requested_effort="default"
    )
    assert quality["acme/model-a"]["effort"] == "max"


def test_resolve_quality_depth_missing_supported_efforts_uses_all_measured():
    # No supported_efforts metadata (older unit-style candidate): never zero
    # a measured model -- all measured records are eligible.
    cand = {
        "id": "acme/model-a",
        "name": "Acme: Model A",
        "default_effort": "high",
        "reasoning": {"is_reasoning": True},
    }
    lookup = mc.build_aa_depth_lookup(
        [
            {
                "id": "1",
                "name": "Model A (High)",
                "slug": "model-a-high",
                "index": 46.8,
                "as_of": "t1",
            },
            {
                "id": "2",
                "name": "Model A (Max)",
                "slug": "model-a",
                "index": 56.0,
                "as_of": "t1",
            },
        ]
    )
    quality, _ = mc.resolve_quality(
        [cand], {}, {}, {}, "AA API v2", depth_lookup=lookup, requested_effort="default"
    )
    assert quality["acme/model-a"]["index"] == 56.0


def test_resolve_quality_depth_scored_at_best_selectable():
    cand = {
        "id": "acme/model-a",
        "name": "Acme: Model A",
        "default_effort": "high",
        "reasoning": {"is_reasoning": True},
    }
    lookup = mc.build_aa_depth_lookup(
        [
            {
                "id": "1",
                "name": "Model A (High)",
                "slug": "model-a-high",
                "index": 46.8,
                "as_of": "t1",
            },
            {
                "id": "2",
                "name": "Model A (Max)",
                "slug": "model-a",
                "index": 56.0,
                "as_of": "t1",
            },
        ]
    )
    quality, source = mc.resolve_quality(
        [cand], {}, {}, {}, "AA API v2", depth_lookup=lookup, requested_effort="default"
    )
    assert quality["acme/model-a"]["index"] == 56.0
    assert quality["acme/model-a"]["effort"] == "max"
    assert source == {"acme/model-a": "aa"}


def test_resolve_quality_depth_default_unmeasured_uses_best_measured():
    cand = {
        "id": "acme/model-a",
        "name": "Acme: Model A",
        "default_effort": "high",
        "reasoning": {"is_reasoning": True},
    }
    lookup = mc.build_aa_depth_lookup(
        [
            {
                "id": "1",
                "name": "Model A (Max)",
                "slug": "model-a",
                "index": 56.0,
                "as_of": "t1",
            },
        ]
    )
    quality, source = mc.resolve_quality(
        [cand], {}, {}, {}, "AA API v2", depth_lookup=lookup, requested_effort="default"
    )
    assert quality["acme/model-a"]["index"] == 56.0
    assert quality["acme/model-a"]["effort"] == "max"
    assert source == {"acme/model-a": "aa"}


def test_resolve_quality_depth_catalog_best_selectable():
    cand = {
        "id": "acme/model-a",
        "name": "Acme: Model A",
        "default_effort": "high",
        "reasoning": {"is_reasoning": True},
    }
    catalog_depths = {
        "acme/model-a": {
            "default_effort": "max",
            "depths": {
                "max": {
                    "effort": "max",
                    "index": 56.0,
                    "as_of": "t1",
                    "source": "catalog",
                },
                "high": {
                    "effort": "high",
                    "index": 46.8,
                    "as_of": "t1",
                    "source": "catalog",
                },
            },
        }
    }
    quality, source = mc.resolve_quality(
        [cand],
        {},
        {},
        {},
        None,
        catalog_depths=catalog_depths,
        requested_effort="default",
    )
    assert quality["acme/model-a"]["index"] == 56.0
    assert quality["acme/model-a"]["effort"] == "max"
    assert quality["acme/model-a"]["source"] == "catalog"
    assert source == {"acme/model-a": "catalog"}


def test_resolve_quality_attaches_unlabeled_aa_reference():
    cand = {
        "id": "z-ai/glm-5.3-flash",
        "name": "Z.AI: GLM 5.3 Flash",
        "default_effort": "max",
        "supported_efforts": ["max", "high", "low"],
        "reasoning": {"is_reasoning": True},
    }
    lookup = mc.build_aa_unlabeled_lookup(
        [
            {
                "id": "1",
                "name": "GLM-5.3-Flash",
                "slug": "glm-5-3-flash",
                "index": 27.8,
                "creator": "Z.AI",
                "cost_per_task": 0.58,
                "as_of": "t1",
            }
        ]
    )
    quality, source = mc.resolve_quality(
        [cand], {}, {}, {}, "AA API v2", depth_lookup={}, unlabeled_lookup=lookup
    )
    # QUAL stays blank: the reference never fills the main record.
    assert quality == {}
    assert source == {}
    assert cand["quality_unlabeled"] == {
        "index": 27.8,
        "cost_per_task_usd": 0.58,
        "source": "aa",
        "as_of": "t1",
    }


def test_resolve_quality_ambiguous_unlabeled_key_terminates_tier_walk():
    # The base tier ("glm 5 3 flash") holds an ambiguous duplicate (None);
    # the lower display tier ("glm53 flash") holds a real record. A present
    # key terminates the walk -- the ambiguity must suppress the reference
    # rather than fall through to the lower tier.
    cand = {
        "id": "z-ai/glm-5.3-flash",
        "name": "Z.AI: GLM5.3 Flash",
        "default_effort": "max",
        "supported_efforts": ["max", "high", "low"],
        "reasoning": {"is_reasoning": True},
    }
    lookup = {
        "glm 5 3 flash": None,
        "glm53 flash": {
            "index": 27.8,
            "cost_per_task_usd": 0.58,
            "source": "aa",
            "as_of": "t1",
        },
    }
    quality, source = mc.resolve_quality(
        [cand], {}, {}, {}, "AA API v2", depth_lookup={}, unlabeled_lookup=lookup
    )
    assert quality == {}
    assert source == {}
    assert "quality_unlabeled" not in cand


def test_resolve_quality_keeps_unlabeled_reference_out_of_main_quality():
    cand = {
        "id": "acme/model-a",
        "name": "Acme: Model A",
        "default_effort": "high",
        "reasoning": {"is_reasoning": True},
    }
    lookup = mc.build_aa_depth_lookup(
        [
            {
                "id": "1",
                "name": "Model A (High)",
                "slug": "model-a-high",
                "index": 46.8,
                "as_of": "t1",
            }
        ]
    )
    unlabeled = mc.build_aa_unlabeled_lookup(
        [{"id": "2", "name": "Model A", "slug": "model-a", "index": 40.0}]
    )
    quality, source = mc.resolve_quality(
        [cand],
        {},
        {},
        {},
        "AA API v2",
        depth_lookup=lookup,
        unlabeled_lookup=unlabeled,
    )
    assert quality["acme/model-a"]["index"] == 46.8
    # A resolved QUAL makes the reference redundant: never attach it.
    assert "quality_unlabeled" not in cand


def test_resolve_quality_clears_stale_unlabeled_reference():
    cand = {
        "id": "acme/model-a",
        "name": "Acme: Model A",
        "default_effort": "high",
        "reasoning": {"is_reasoning": True},
        "quality_unlabeled": {
            "index": 99.0,
            "cost_per_task_usd": None,
            "source": "aa",
            "as_of": "stale",
        },
    }
    lookup = mc.build_aa_depth_lookup(
        [
            {
                "id": "1",
                "name": "Model A (High)",
                "slug": "model-a-high",
                "index": 46.8,
                "as_of": "t1",
            }
        ]
    )
    quality, _ = mc.resolve_quality(
        [cand], {}, {}, {}, "AA API v2", depth_lookup=lookup
    )
    assert quality["acme/model-a"]["index"] == 46.8
    assert "quality_unlabeled" not in cand


def test_resolve_quality_hydrates_unlabeled_reference_from_catalog():
    cand = {
        "id": "acme/model-a",
        "name": "Acme: Model A",
        "default_effort": "max",
        "reasoning": {"is_reasoning": True},
    }
    catalog_depths = {
        "acme/model-a": {
            "default_effort": "max",
            "depths": {},
            "unlabeled": {
                "index": 27.8,
                "cost_per_task_usd": 0.58,
                "source": "aa",
                "as_of": "t1",
            },
        }
    }
    quality, source = mc.resolve_quality(
        [cand], {}, {}, {}, None, catalog_depths=catalog_depths
    )
    assert quality == {}
    assert cand["quality_unlabeled"]["index"] == 27.8
    assert cand["quality_unlabeled"]["as_of"] == "t1"


def test_resolve_quality_single_state_reasoning_uses_reasoning_row():
    cand = {
        "id": "acme/model-a",
        "name": "Acme: Model A",
        "default_effort": "reasoning",
        "reasoning": {"is_reasoning": True, "supported_efforts": []},
    }
    lookup = mc.build_aa_depth_lookup(
        [
            {
                "id": "1",
                "name": "Model A (Reasoning)",
                "slug": "model-a",
                "index": 44.0,
                "as_of": "t1",
            },
        ]
    )
    quality, source = mc.resolve_quality(
        [cand], {}, {}, {}, "AA API v2", depth_lookup=lookup, requested_effort="default"
    )
    assert quality["acme/model-a"]["index"] == 44.0
    assert quality["acme/model-a"]["effort"] == "reasoning"


def test_resolve_quality_depth_explicit_effort_does_not_borrow_other_depth():
    cand = {
        "id": "acme/model-a",
        "name": "Acme: Model A",
        "default_effort": "max",
        "supported_efforts": ["max", "high"],
        "reasoning": {"is_reasoning": True},
    }
    lookup = mc.build_aa_depth_lookup(
        [
            {
                "id": "1",
                "name": "Model A (Max)",
                "slug": "model-a",
                "index": 56.0,
                "as_of": "t1",
            },
        ]
    )
    quality, source = mc.resolve_quality(
        [cand], {}, {}, {}, "AA API v2", depth_lookup=lookup, requested_effort="high"
    )
    assert quality == {}
    assert source == {}


def test_resolve_quality_depth_explicit_effort_selects_that_depth():
    cand = {
        "id": "acme/model-a",
        "name": "Acme: Model A",
        "default_effort": "high",
        "reasoning": {"is_reasoning": True},
    }
    lookup = mc.build_aa_depth_lookup(
        [
            {
                "id": "1",
                "name": "Model A (High)",
                "slug": "model-a-high",
                "index": 46.8,
                "as_of": "t1",
            },
            {
                "id": "2",
                "name": "Model A (Max)",
                "slug": "model-a",
                "index": 56.0,
                "as_of": "t1",
            },
        ]
    )
    quality, source = mc.resolve_quality(
        [cand], {}, {}, {}, "AA API v2", depth_lookup=lookup, requested_effort="max"
    )
    assert quality["acme/model-a"]["index"] == 56.0
    assert quality["acme/model-a"]["effort"] == "max"


def test_match_depth_quality_prefers_creator_qualified_key():
    lookup = mc.build_aa_depth_lookup(
        [
            {
                "id": "1",
                "name": "Model A (High)",
                "slug": "model-a-high",
                "index": 46.8,
                "as_of": "t1",
                "creator": "Beta",
            },
            {
                "id": "2",
                "name": "Model A (High)",
                "slug": "model-a-high",
                "index": 40.0,
                "as_of": "t1",
                "creator": "Acme",
            },
        ]
    )
    # The unqualified key is ambiguous (two creators); the qualified key is not.
    assert lookup["model a"]["high"] is None
    record = mc.match_depth_quality(
        {"id": "acme/model-a", "name": "Acme: Model A"}, lookup, "high"
    )
    assert record["index"] == 40.0
    beta = mc.match_depth_quality(
        {"id": "beta/model-a", "name": "Beta: Model A"}, lookup, "high"
    )
    assert beta["index"] == 46.8


def test_match_depth_quality_missing_effort_does_not_fall_through():
    lookup = mc.build_aa_depth_lookup(
        [
            {
                "id": "1",
                "name": "Model A (Max)",
                "slug": "model-a",
                "index": 56.0,
                "as_of": "t1",
            },
        ]
    )
    model = {"id": "acme/model-a", "name": "Acme: Model A"}
    assert mc.match_depth_quality(model, lookup, "high") is None
    assert mc.match_depth_quality(model, lookup, None) is None


def test_match_depth_quality_ambiguous_tier_does_not_fall_through():
    # The base tier's high record is ambiguous (None). A lower display tier
    # exists, but a matched-but-ambiguous tier yields nothing.
    lookup = {
        "model a": {"high": None},
        "something else": {
            "high": {
                "effort": "high",
                "index": 50.0,
                "cost_per_task_usd": None,
                "source": "aa",
                "as_of": "t1",
            }
        },
    }
    model = {"id": "acme/model-a", "name": "Acme: Something Else"}
    assert mc.match_depth_quality(model, lookup, "high") is None


def test_resolve_quality_reasoning_model_without_depth_ignores_or_trio():
    cand = {
        "id": "acme/model-a",
        "name": "Acme: Model A",
        "default_effort": "high",
        "reasoning": {"is_reasoning": True},
    }
    aa_by_id = {
        "acme/model-a": {"intelligence_index": 57.5, "as_of": "20260915"},
    }
    quality, source = mc.resolve_quality(
        [cand], aa_by_id, {}, {}, None, depth_lookup={}, requested_effort="default"
    )
    assert quality == {} and source == {}


def test_resolve_quality_non_reasoning_uses_or_trio_with_none_effort():
    cand = {
        "id": "acme/model-a",
        "name": "Acme: Model A",
        "default_effort": "none",
        "reasoning": {"is_reasoning": False},
    }
    aa_by_id = {
        "acme/model-a": {"intelligence_index": 57.5, "as_of": "20260915"},
    }
    quality, source = mc.resolve_quality(
        [cand], aa_by_id, {}, {}, None, depth_lookup={}, requested_effort="default"
    )
    assert quality["acme/model-a"]["source"] == "openrouter"
    assert quality["acme/model-a"]["effort"] == "none"
    assert cand["reasoning_depths"] == [
        {
            "effort": "none",
            "index": 57.5,
            "cost_per_task_usd": None,
            "source": "openrouter",
            "as_of": "20260915",
        }
    ]
    assert source == {"acme/model-a": "openrouter"}


def test_resolve_quality_depth_drift_only_when_like_for_like():
    cands = [
        {
            "id": "acme/model-a",
            "name": "Acme: Model A",
            "default_effort": "none",
            "reasoning": {"is_reasoning": True},
        },
        {
            "id": "acme/model-b",
            "name": "Acme: Model B",
            "default_effort": "high",
            "reasoning": {"is_reasoning": True},
        },
        {
            "id": "acme/model-c",
            "name": "Acme: Model C",
            "default_effort": "none",
            "reasoning": {"is_reasoning": False},
        },
    ]
    lookup = mc.build_aa_depth_lookup(
        [
            {
                "id": "1",
                "name": "Model A (Non-reasoning)",
                "slug": "model-a",
                "index": 45.1,
                "as_of": "t1",
            },
            {
                "id": "2",
                "name": "Model B (High)",
                "slug": "model-b-high",
                "index": 45.1,
                "as_of": "t1",
            },
            {
                "id": "3",
                "name": "Model C (Non-reasoning)",
                "slug": "model-c",
                "index": 45.1,
                "as_of": "t1",
            },
        ]
    )
    aa_by_id = {
        "acme/model-a": {"intelligence_index": 39.5, "as_of": "20260915"},
        "acme/model-b": {"intelligence_index": 39.5, "as_of": "20260915"},
        "acme/model-c": {"intelligence_index": 39.5, "as_of": "20260915"},
    }
    quality, _source = mc.resolve_quality(
        cands, aa_by_id, {}, {}, "AA API v2", depth_lookup=lookup
    )
    # A reasoning model at effort "none" is still a reasoning model: the OR
    # snapshot is measured at its reasoning state, so the pair is a
    # reference only (not a like-for-like drift check).
    assert quality["acme/model-a"]["or_index"] == 39.5
    assert quality["acme/model-a"]["or_as_of"] == "20260915"
    assert quality["acme/model-a"]["drift"] is None
    # A reasoning model at a non-none effort: same.
    assert quality["acme/model-b"]["or_index"] == 39.5
    assert quality["acme/model-b"]["drift"] is None
    # A non-reasoning model at effort "none": both sides are non-reasoning,
    # so the old drift rule applies.
    assert quality["acme/model-c"]["drift"] is True


def test_resolve_quality_depth_attaches_full_measured_depth_list():
    cand = {
        "id": "acme/model-a",
        "name": "Acme: Model A",
        "default_effort": "high",
        "reasoning": {"is_reasoning": True},
    }
    lookup = mc.build_aa_depth_lookup(
        [
            {
                "id": "1",
                "name": "Model A (High)",
                "slug": "model-a-high",
                "index": 46.8,
                "cost_per_task": 0.88,
                "as_of": "t1",
            },
            {
                "id": "2",
                "name": "Model A (Max)",
                "slug": "model-a",
                "index": 56.0,
                "cost_per_task": 5.46,
                "as_of": "t1",
            },
        ]
    )
    quality, _source = mc.resolve_quality(
        [cand], {}, {}, {}, "AA API v2", depth_lookup=lookup
    )
    assert quality["acme/model-a"]["index"] == 56.0
    assert quality["acme/model-a"]["effort"] == "max"
    assert cand["reasoning_depths"] == [
        {
            "effort": "max",
            "index": 56.0,
            "cost_per_task_usd": 5.46,
            "source": "aa",
            "as_of": "t1",
        },
        {
            "effort": "high",
            "index": 46.8,
            "cost_per_task_usd": 0.88,
            "source": "aa",
            "as_of": "t1",
        },
    ]


def test_resolve_quality_catalog_default_uses_best_measured():
    cand = {
        "id": "acme/model-a",
        "name": "Acme: Model A",
        "default_effort": "high",
        "reasoning": {"is_reasoning": True},
    }
    catalog_depths = {
        "acme/model-a": {
            "default_effort": "max",
            "depths": {
                "max": {
                    "effort": "max",
                    "index": 56.0,
                    "as_of": "t1",
                    "source": "catalog",
                },
            },
        }
    }
    quality, source = mc.resolve_quality(
        [cand],
        {},
        {},
        {},
        None,
        catalog_depths=catalog_depths,
        requested_effort="default",
    )
    assert quality["acme/model-a"]["index"] == 56.0
    assert quality["acme/model-a"]["effort"] == "max"
    assert source == {"acme/model-a": "catalog"}
    assert cand["reasoning_depths"] == [
        {
            "effort": "max",
            "index": 56.0,
            "cost_per_task_usd": None,
            "source": "catalog",
            "as_of": "t1",
        }
    ]


def test_resolve_quality_unmatched_yields_none_and_zero_penalty_unchanged():
    # compute_scores: an unmatched candidate keeps quality None and scores 0
    # on the quality axis, while a partial match keeps the quality weight
    # (quality-blind reweighting only fires when NO candidate matches).
    candidates = [_cand("a/x", 1.0), _cand("a/y", 2.0)]
    weights = mc.compute_scores(
        candidates,
        make_args(),
        {
            "a/x": {
                "index": 70.0,
                "source": "aa",
                "as_of": None,
                "drift": None,
                "or_index": None,
                "or_as_of": None,
            }
        },
    )
    assert "quality" in weights
    by_id = {c["id"]: c for c in candidates}
    assert by_id["a/y"]["quality"] is None
    assert by_id["a/y"]["quality_score"] == 0.0
    assert by_id["a/x"]["quality"] == 70.0
    assert by_id["a/x"]["quality_score"] == pytest.approx(1.0)


def test_compute_scores_and_build_schedules_keep_quality_numeric():
    # Both consumers read record["index"], never the record dict itself:
    # cand["quality"] stays the float index the ranking and display paths
    # expect, and a schedule row's "quality" is a number or null.
    record = {
        "index": 60.0,
        "source": "aa",
        "as_of": None,
        "drift": None,
        "or_index": None,
        "or_as_of": None,
    }
    candidates = [_cand("a/x", 1.0)]
    mc.compute_scores(candidates, make_args(), {"a/x": record})
    assert candidates[0]["quality"] == 60.0

    model = make_model(
        id="a/x",
        pricing={
            "prompt": "0.000001",
            "completion": "0.000002",
            "overrides": [
                {
                    "utc_start": 0,
                    "utc_end": 600,
                    "prompt": "0.0000005",
                    "completion": "0.000001",
                },
                {
                    "utc_start": 600,
                    "utc_end": 0,
                    "prompt": "0.000001",
                    "completion": "0.000002",
                },
            ],
        },
    )
    entry = mc.build_schedules([model], make_args(), {"a/x": record}, {}, {})[0]
    assert entry["quality"] == 60.0
    # The winner's provenance rides along for the site's QUAL tooltip.
    assert entry["quality_source"] == "aa"
    assert entry["quality_as_of"] is None


def test_build_schedules_reasoning_no_default_depth_is_null():
    model = make_model(
        id="acme/sched", pricing=deepseek_pricing(), context_length=500000
    )
    aa_by_id = {"acme/sched": {"intelligence_index": 30.0, "as_of": "20261001"}}
    entries = mc.build_schedules(
        [model],
        make_args(),
        {},
        aa_by_id,
        {},
        depth_lookup={},
        reasoning_by_id={
            "acme/sched": {
                "is_reasoning": True,
                "default_effort": "high",
                "supported_efforts": ["high"],
            }
        },
    )
    # The OR trio exists but has no effort label; it must not supply the score.
    assert entries[0]["quality"] is None
    assert entries[0]["quality_effort"] is None


def test_build_schedules_non_reasoning_keeps_or_trio():
    model = make_model(
        id="acme/sched", pricing=deepseek_pricing(), context_length=500000
    )
    aa_by_id = {"acme/sched": {"intelligence_index": 30.0, "as_of": "20261001"}}
    entries = mc.build_schedules(
        [model],
        make_args(),
        {},
        aa_by_id,
        {},
        depth_lookup={},
        reasoning_by_id={
            "acme/sched": {
                "is_reasoning": False,
                "default_effort": "none",
                "supported_efforts": [],
            }
        },
    )
    assert entries[0]["quality"] == 30.0
    assert entries[0]["quality_effort"] == "none"
    assert entries[0]["quality_source"] == "openrouter"


def test_build_schedules_explicit_effort_does_not_use_or_trio():
    model = make_model(
        id="acme/sched", pricing=deepseek_pricing(), context_length=500000
    )
    entries = mc.build_schedules(
        [model],
        make_args(),
        {},
        {"acme/sched": {"intelligence_index": 30.0, "as_of": "20261001"}},
        {},
        depth_lookup={},
        reasoning_by_id={
            "acme/sched": {
                "is_reasoning": False,
                "default_effort": "none",
                "supported_efforts": [],
            }
        },
        requested_effort="low",
    )
    assert entries[0]["quality"] is None
    assert entries[0]["quality_effort"] is None


def test_build_schedules_keyed_uses_default_depth_record():
    model = make_model(
        id="acme/sched", pricing=deepseek_pricing(), context_length=500000
    )
    depth_lookup = {
        "sched": {
            "max": {
                "effort": "max",
                "index": 56.0,
                "cost_per_task_usd": 5.0,
                "source": "aa",
                "as_of": "t-max",
            },
            "high": {
                "effort": "high",
                "index": 44.0,
                "cost_per_task_usd": 0.5,
                "source": "aa",
                "as_of": "t1",
            },
        }
    }
    entries = mc.build_schedules(
        [model],
        make_args(),
        {},
        {},
        {},
        depth_lookup=depth_lookup,
        reasoning_by_id={
            "acme/sched": {
                "is_reasoning": True,
                "default_effort": "high",
                "supported_efforts": ["max", "high"],
            }
        },
    )
    assert entries[0]["quality"] == 56.0
    assert entries[0]["quality_effort"] == "max"
    assert entries[0]["quality_source"] == "aa"
    assert entries[0]["quality_as_of"] == "t-max"


def test_build_schedules_unkeyed_carries_catalog_effort():
    model = make_model(
        id="acme/sched", pricing=deepseek_pricing(), context_length=500000
    )
    catalog_depths = {
        "acme/sched": {
            "default_effort": "high",
            "depths": {
                "max": {
                    "effort": "max",
                    "index": 56.0,
                    "as_of": "t-max",
                    "source": "catalog",
                },
                "high": {
                    "effort": "high",
                    "index": 44.0,
                    "as_of": "t1",
                    "source": "catalog",
                },
            },
        }
    }
    entries = mc.build_schedules(
        [model],
        make_args(),
        {},
        {},
        {},
        catalog_depths=catalog_depths,
        reasoning_by_id={
            "acme/sched": {
                "is_reasoning": True,
                "default_effort": "high",
                "supported_efforts": ["max", "high"],
            }
        },
    )
    assert entries[0]["quality"] == 56.0
    assert entries[0]["quality_effort"] == "max"
    assert entries[0]["quality_source"] == "catalog"
    assert entries[0]["quality_as_of"] == "t-max"


def test_build_schedules_unkeyed_catalog_honors_requested_effort():
    model = make_model(
        id="acme/sched", pricing=deepseek_pricing(), context_length=500000
    )
    catalog_depths = {
        "acme/sched": {
            "depths": {
                "max": {"effort": "max", "index": 56.0},
                "high": {"effort": "high", "index": 44.0},
            }
        }
    }
    entries = mc.build_schedules(
        [model],
        make_args(),
        {},
        {},
        {},
        catalog_depths=catalog_depths,
        reasoning_by_id={
            "acme/sched": {
                "is_reasoning": True,
                "default_effort": "high",
                "supported_efforts": ["max", "high"],
            }
        },
        requested_effort="high",
    )
    assert entries[0]["quality"] == 44.0
    assert entries[0]["quality_effort"] == "high"


def test_build_schedules_unkeyed_malformed_catalog_depths_falls_through():
    model = make_model(
        id="acme/sched", pricing=deepseek_pricing(), context_length=500000
    )
    entries = mc.build_schedules(
        [model],
        make_args(),
        {},
        {},
        {},
        catalog_quality={
            "acme/sched": {"index": 42.0, "effort": "high", "as_of": "t1"}
        },
        catalog_depths={"acme/sched": {"depths": "malformed"}},
        reasoning_by_id={
            "acme/sched": {
                "is_reasoning": True,
                "default_effort": "high",
                "supported_efforts": ["max", "high"],
            }
        },
    )
    assert entries[0]["quality"] == 42.0
    assert entries[0]["quality_effort"] == "high"
    assert entries[0]["quality_source"] == "catalog"


def test_build_schedules_keyed_honors_requested_effort():
    model = make_model(
        id="acme/sched", pricing=deepseek_pricing(), context_length=500000
    )
    depth_lookup = {
        "sched": {
            "max": {
                "effort": "max",
                "index": 56.0,
                "cost_per_task_usd": 5.0,
                "source": "aa",
                "as_of": "t-max",
            },
            "high": {
                "effort": "high",
                "index": 44.0,
                "cost_per_task_usd": 0.5,
                "source": "aa",
                "as_of": "t-high",
            },
        }
    }
    entries = mc.build_schedules(
        [model],
        make_args(),
        {},
        {},
        {},
        depth_lookup=depth_lookup,
        reasoning_by_id={
            "acme/sched": {
                "is_reasoning": True,
                "default_effort": "high",
                "supported_efforts": ["max", "high"],
            }
        },
        requested_effort="max",
    )
    assert entries[0]["quality"] == 56.0
    assert entries[0]["quality_effort"] == "max"
    assert entries[0]["quality_source"] == "aa"
    assert entries[0]["quality_as_of"] == "t-max"


def test_build_schedules_unkeyed_ignores_requested_effort_for_recorded_entry():
    model = make_model(
        id="acme/sched", pricing=deepseek_pricing(), context_length=500000
    )
    schedule_entries = {
        "acme/sched": {
            "quality": 44.0,
            "quality_effort": "high",
            "quality_as_of": "t1",
            "quality_source": "aa",
        }
    }
    entries = mc.build_schedules(
        [model],
        make_args(),
        {},
        {},
        {},
        schedule_entries=schedule_entries,
        reasoning_by_id={
            "acme/sched": {
                "is_reasoning": True,
                "default_effort": "high",
                "supported_efforts": ["max", "high"],
            }
        },
        requested_effort="max",
    )
    # Unkeyed schedules serve the published entry as-is: no depth list to
    # re-select from, so --reasoning-effort cannot change it.
    assert entries[0]["quality"] == 44.0
    assert entries[0]["quality_effort"] == "high"
    assert entries[0]["quality_source"] == "aa"
    assert entries[0]["quality_as_of"] == "t1"


def test_build_schedules_recorded_entry_without_effort_nulls_quality():
    # Spec 6.5: a recorded entry with no quality_effort is legacy/unknown,
    # so its quality is null. Tier 0 must not fall through to the OR trio
    # (which would republish a max-effort snapshot as the default).
    model = make_model(
        id="acme/sched", pricing=deepseek_pricing(), context_length=500000
    )
    aa_by_id = {"acme/sched": {"intelligence_index": 30.0, "as_of": "20261001"}}
    reasoning_by_id = {
        "acme/sched": {
            "is_reasoning": False,
            "default_effort": "none",
            "supported_efforts": [],
        }
    }
    for entry in (
        {
            "quality": 30.0,
            "quality_effort": None,
            "quality_as_of": "t1",
            "quality_source": "openrouter",
        },
        {  # quality_effort absent entirely
            "quality": 30.0,
            "quality_as_of": "t1",
            "quality_source": "openrouter",
        },
    ):
        entries = mc.build_schedules(
            [model],
            make_args(),
            {},
            aa_by_id,
            {},
            schedule_entries={"acme/sched": entry},
            reasoning_by_id=reasoning_by_id,
        )
        assert entries[0]["quality"] is None
        assert entries[0]["quality_effort"] is None
        assert entries[0]["quality_source"] is None


def test_print_json_rows_carry_quality_source_and_as_of(capsys):
    cand = _cand("a/x", 1.0)
    cand["quality"] = 55.0
    cand["quality_record"] = {
        "index": 55.0,
        "source": "aa",
        "as_of": "2026-10-23T12:00:00+00:00",
        "drift": False,
        "or_index": 57.5,
        "or_as_of": "20260915",
    }
    mc.print_json([cand])
    row = json.loads(capsys.readouterr().out)[0]
    assert row["quality_source"] == "aa"
    assert row["quality_as_of"] == "2026-10-23T12:00:00+00:00"
    assert row["quality_drift"] is False
    assert row["quality_or_index"] == 57.5
    assert row["quality_or_as_of"] == "20260915"


def test_print_json_pins_quality_to_best_selectable_depth(capsys):
    # The default is high but unmeasured; the best measured selectable depth
    # is max, and JSON must expose the same record and its index.
    cand = _cand("acme/model-a", 50.0)
    cand["quality"] = 56.0
    cand["quality_record"] = {
        "index": 56.0,
        "source": "aa",
        "as_of": "t-max",
        "effort": "max",
    }
    cand["reasoning"] = {
        "default_effort": "high",
        "supported_efforts": ["max", "high"],
        "is_reasoning": True,
    }
    cand["reasoning_depths"] = [
        {"effort": "max", "index": 56.0, "source": "aa", "as_of": "t-max"}
    ]
    mc.print_json([cand])
    row = json.loads(capsys.readouterr().out)[0]
    assert row["quality_effort"] == "max"
    assert row["quality_index"] == 56.0


def test_print_json_carries_depth_fields(capsys):
    cand = _cand("acme/model-a", 50.0)
    cand["quality"] = 46.8
    cand["quality_record"] = {
        "index": 46.8,
        "source": "aa",
        "as_of": "t1",
        "effort": "high",
    }
    cand["reasoning"] = {
        "default_effort": "high",
        "supported_efforts": ["max", "high"],
        "is_reasoning": True,
    }
    cand["reasoning_depths"] = [
        {"effort": "max", "index": 56.0, "source": "aa", "as_of": "t1"}
    ]
    mc.print_json([cand])
    row = json.loads(capsys.readouterr().out)[0]
    assert row["quality_effort"] == "high"
    assert row["default_effort"] == "high"
    assert row["reasoning"]["depths"][0]["effort"] == "max"
    # BUG-01: the internal flag never leaks into the published block.
    assert "is_reasoning" not in row["reasoning"]


def test_print_json_carries_unlabeled_aa_reference(capsys):
    cand = _cand("z-ai/glm-5.3-flash", 27.8)
    cand["quality"] = None
    cand["quality_record"] = None
    cand["reasoning"] = {
        "default_effort": "max",
        "supported_efforts": ["max", "high", "low"],
        "is_reasoning": True,
    }
    cand["reasoning_depths"] = []
    cand["quality_unlabeled"] = {
        "index": 27.8,
        "cost_per_task_usd": 0.58,
        "source": "aa",
        "as_of": "t1",
    }
    mc.print_json([cand])
    row = json.loads(capsys.readouterr().out)[0]
    assert row["quality_index"] is None
    assert row["reasoning"]["aa_unlabeled"] == {
        "index": 27.8,
        "cost_per_task_usd": 0.58,
        "source": "aa",
        "as_of": "t1",
    }


def test_print_json_unlabeled_aa_reference_is_null_when_absent(capsys):
    cand = _cand("acme/model-a", 50.0)
    cand["quality"] = None
    cand["quality_record"] = None
    cand["reasoning"] = {"default_effort": "high", "supported_efforts": ["high"]}
    cand["reasoning_depths"] = []
    mc.print_json([cand])
    row = json.loads(capsys.readouterr().out)[0]
    assert row["reasoning"]["aa_unlabeled"] is None


def test_print_json_non_reasoning_none_record(capsys):
    cand = _cand("acme/model-a", 50.0)
    cand["quality"] = 30.0
    cand["quality_record"] = {
        "index": 30.0,
        "source": "openrouter",
        "as_of": "t1",
        "effort": "none",
    }
    cand["reasoning"] = {
        "default_effort": "none",
        "supported_efforts": [],
        "is_reasoning": False,
    }
    cand["reasoning_depths"] = [
        {
            "effort": "none",
            "index": 30.0,
            "cost_per_task_usd": None,
            "source": "openrouter",
            "as_of": "t1",
        }
    ]
    mc.print_json([cand])
    row = json.loads(capsys.readouterr().out)[0]
    assert row["quality_effort"] == "none"
    assert row["reasoning"]["depths"][0]["cost_per_task_usd"] is None


def test_print_json_unmatched_effort_is_null(capsys):
    cand = _cand("acme/model-a", 50.0)
    cand["quality"] = None
    cand["quality_record"] = None
    cand["reasoning"] = {
        "default_effort": "high",
        "supported_efforts": ["high"],
        "is_reasoning": True,
    }
    cand["reasoning_depths"] = []
    mc.print_json([cand])
    row = json.loads(capsys.readouterr().out)[0]
    assert row["quality_effort"] is None
    assert row["default_effort"] == "high"


# ---------------------------------------------------------------------------
# published catalog hydration (pure parser + fetcher; Task 3)
# ---------------------------------------------------------------------------


def _catalog_doc(models, schema_version=6, generated_at="2026-10-23T13:15:07+00:00"):
    """Minimal published catalog.json: only the fields hydration reads."""
    return {
        "schema_version": schema_version,
        "generated_at": generated_at,
        "models": models,
    }


def test_catalog_quality_by_id_v6_passthrough(capsys):
    """v6 rows keep their recorded quality_as_of, source label and effort
    verbatim; unmatched rows (no value, no label) are skipped without a
    warning.

    Producer-faithful stamps: aa rows carry the CI run's ISO fetch stamp,
    OR rows its YYYYMMDD permaslug date (build_catalog never emits an ISO
    stamp on an openrouter row).
    """
    doc = _catalog_doc(
        [
            {
                "id": "acme/model-a",
                "quality": 45.1,
                "quality_match": "aa",
                "quality_as_of": "2026-10-23T13:15:07+00:00",
                "quality_effort": "high",
            },
            {
                "id": "acme/model-b",
                "quality": 39.5,
                "quality_match": "openrouter",
                "quality_as_of": "20261017",
                "quality_effort": "none",
            },
            {"id": "acme/model-c", "quality": None, "quality_match": None},
        ],
        schema_version=6,
    )
    assert mc.catalog_quality_by_id(doc) == {
        "acme/model-a": {
            "index": 45.1,
            "source": "aa",
            "as_of": "2026-10-23T13:15:07+00:00",
            "effort": "high",
        },
        "acme/model-b": {
            "index": 39.5,
            "source": "openrouter",
            "as_of": "20261017",
            "effort": "none",
        },
    }
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize("schema_version", [3, 4, 5])
def test_catalog_quality_by_id_old_schema_skipped(schema_version, capsys):
    """Only v6 hydrates: older normalization branches are retired, and
    mixing old semantics into a best-selectable-depth run is never allowed."""
    doc = _catalog_doc(
        [
            {
                "id": "acme/model-a",
                "quality": 45.1,
                "quality_match": "aa",
                "quality_as_of": None,
            },
            {"id": "acme/model-b", "quality": 39.5, "quality_match": "openrouter"},
        ],
        schema_version=schema_version,
    )
    assert mc.catalog_quality_by_id(doc) == {}
    err = capsys.readouterr().err
    assert f"published catalog schema v{schema_version}, expected 6" in err
    assert "skipping hydration" in err


def test_catalog_quality_by_id_v2_schema_skipped(capsys):
    """An old schema: warn + skip the whole artifact (empty by_id)."""
    doc = _catalog_doc(
        [{"id": "acme/model-a", "quality": 45.1, "quality_match": "api"}],
        schema_version=2,
    )
    assert mc.catalog_quality_by_id(doc) == {}
    err = capsys.readouterr().err
    assert "published catalog schema v2, expected 6" in err
    assert "skipping hydration" in err


def test_catalog_quality_by_id_malformed_never_raises(capsys):
    """Any shape surprise warns and yields an empty map -- never an exception,
    never a partial hydration (fail closed on the whole artifact)."""
    good_row = {"id": "acme/model-a", "quality": 45.1, "quality_match": "api"}
    bad_docs = [
        None,
        ["not", "a", "dict"],
        {"models": [good_row]},  # schema_version missing entirely
        {**_catalog_doc([good_row]), "schema_version": "3"},
        {**_catalog_doc([good_row]), "models": None},
        {**_catalog_doc([good_row]), "models": {"0": good_row}},
        _catalog_doc(["not a dict"]),
        _catalog_doc([{"quality": 45.1, "quality_match": "api"}]),  # id missing
        _catalog_doc([{"id": "", "quality": 45.1, "quality_match": "api"}]),
        _catalog_doc(
            [{"id": "acme/model-a", "quality": "45.1", "quality_match": "api"}]
        ),
        _catalog_doc([{"id": "acme/model-a", "quality": True, "quality_match": "api"}]),
        _catalog_doc([{"id": "acme/model-a", "quality": None, "quality_match": "api"}]),
        _catalog_doc([{"id": "acme/model-a", "quality": 45.1, "quality_match": None}]),
    ]
    for doc in bad_docs:
        assert mc.catalog_quality_by_id(doc) == {}
        assert "unexpected response shape" in capsys.readouterr().err


def test_catalog_quality_by_id_out_of_range_quality_degrades(capsys):
    """An out-of-range quality is poisoned data: hydration degrades the
    whole artifact instead of feeding a bogus index to resolve_quality's
    catalog tier, where a -50.0 would win and fabricate drift. The surface
    is unkeyed runs -- only they hydrate from the published catalog."""
    for bad in (-50.0, 200.0, 1e308):
        doc = _catalog_doc(
            [
                {"id": "acme/model-a", "quality": 45.1, "quality_match": "aa"},
                {"id": "acme/model-b", "quality": bad, "quality_match": "openrouter"},
            ],
            schema_version=6,
        )
        assert mc.catalog_quality_by_id(doc) == {}
        assert "unexpected response shape" in capsys.readouterr().err


def test_catalog_quality_by_id_malformed_depths_degrade(capsys):
    """A malformed depth block degrades the whole artifact: hydration never
    feeds a partial or guessed depth map into resolve_quality."""
    good = {
        "effort": "high",
        "index": 46.8,
        "cost_per_task_usd": 0.88,
        "source": "aa",
        "as_of": "t1",
    }
    bad_reasoning = [
        "not a dict",
        {"depths": "not a list"},
        {"depths": ["not a dict"]},
        {"depths": [{**good, "effort": ""}]},
        {"depths": [{**good, "index": "46.8"}]},
        {"depths": [{**good, "index": 200.0}]},
        # The published depth shape is `index`; the retired `quality_index`
        # spelling is malformed, never normalized.
        {
            "depths": [
                {
                    "effort": "high",
                    "quality_index": 46.8,
                    "source": "aa",
                    "as_of": "t1",
                }
            ]
        },
        {"depths": [{**good, "cost_per_task_usd": "cheap"}]},
        {"depths": [{**good, "source": None}]},
        {"depths": [{**good, "as_of": 5}]},
        {"depths": [good, good]},  # duplicate effort: ambiguous
        {"default_effort": 5},
    ]
    for reasoning in bad_reasoning:
        doc = _catalog_doc(
            [
                {
                    "id": "acme/model-a",
                    "quality": 46.8,
                    "quality_match": "aa",
                    "quality_effort": "high",
                    "reasoning": reasoning,
                }
            ],
            schema_version=6,
        )
        assert mc.catalog_quality_by_id(doc) == {}
        assert "unexpected response shape" in capsys.readouterr().err


def test_fetch_quality_catalog_parses_v6_depth_without_main_quality(monkeypatch):
    """A model can have measured depths without a main score: by_id skips it
    but depths_by_id still hydrates its depth map."""
    doc = _catalog_doc(
        [
            {
                "id": "acme/model-a",
                "quality": None,
                "quality_match": None,
                "reasoning": {
                    "default_effort": "high",
                    "depths": [
                        {
                            "effort": "high",
                            "index": 46.8,
                            "cost_per_task_usd": None,
                            "source": "aa",
                            "as_of": "t1",
                        }
                    ],
                },
            }
        ],
        schema_version=6,
    )
    monkeypatch.setattr(mc, "fetch_json", lambda *a, **k: doc)
    result = mc.fetch_quality_catalog(make_args())
    assert result["by_id"] == {}
    assert result["depths_by_id"]["acme/model-a"]["depths"]["high"]["index"] == 46.8


def test_fetch_quality_catalog_parses_v6_depths(monkeypatch):
    """The v6 per-model reasoning.depths list becomes an effort-keyed map;
    each depth record reuses the quality record's shape (`index`)."""
    stamp = "2026-10-23T13:15:07+00:00"
    doc = _catalog_doc(
        [
            {
                "id": "acme/model-a",
                "quality": 46.8,
                "quality_match": "aa",
                "quality_as_of": stamp,
                "quality_effort": "high",
                "reasoning": {
                    "default_effort": "high",
                    "supported_efforts": ["max", "high"],
                    "depths": [
                        {
                            "effort": "max",
                            "index": 56.0,
                            "cost_per_task_usd": 5.46,
                            "source": "aa",
                            "as_of": stamp,
                        },
                        {
                            "effort": "high",
                            "index": 46.8,
                            "cost_per_task_usd": 0.88,
                            "source": "aa",
                            "as_of": stamp,
                        },
                    ],
                },
            },
            {"id": "acme/model-b", "quality": None, "quality_match": None},
        ],
        schema_version=6,
    )
    monkeypatch.setattr(mc, "fetch_json", lambda *a, **k: doc)
    result = mc.fetch_quality_catalog(make_args())
    assert set(result) == {
        "by_id",
        "depths_by_id",
        "generated_at",
        "schedules_by_id",
    }
    assert result["by_id"]["acme/model-a"]["effort"] == "high"
    assert result["depths_by_id"] == {
        "acme/model-a": {
            "default_effort": "high",
            "depths": {
                "max": {
                    "effort": "max",
                    "index": 56.0,
                    "cost_per_task_usd": 5.46,
                    "source": "aa",
                    "as_of": stamp,
                },
                "high": {
                    "effort": "high",
                    "index": 46.8,
                    "cost_per_task_usd": 0.88,
                    "source": "aa",
                    "as_of": stamp,
                },
            },
            "unlabeled": None,
        }
    }


def test_fetch_quality_catalog_parses_v6_aa_unlabeled(monkeypatch):
    """An AA row with no effort label hydrates as a display-only reference:
    it never becomes the main quality, but the depth map carries it."""
    stamp = "2026-11-04T12:27:03+00:00"
    doc = _catalog_doc(
        [
            {
                "id": "z-ai/glm-5.3-flash",
                "quality": None,
                "quality_match": None,
                "reasoning": {
                    "default_effort": "max",
                    "supported_efforts": ["max", "high", "low"],
                    "depths": [],
                    "aa_unlabeled": {
                        "index": 27.8,
                        "cost_per_task_usd": 0.58,
                        "source": "aa",
                        "as_of": stamp,
                    },
                },
            }
        ],
        schema_version=6,
    )
    monkeypatch.setattr(mc, "fetch_json", lambda *a, **k: doc)
    result = mc.fetch_quality_catalog(make_args())
    assert result["by_id"] == {}
    assert result["depths_by_id"]["z-ai/glm-5.3-flash"]["unlabeled"] == {
        "index": 27.8,
        "cost_per_task_usd": 0.58,
        "source": "aa",
        "as_of": stamp,
    }


@pytest.mark.parametrize(
    "mutate",
    [
        lambda record: record.update(index=float("nan")),
        lambda record: record.update(index="27.8"),
        lambda record: record.update(index=True),
        lambda record: record.update(cost_per_task_usd="0.58"),
        lambda record: record.update(source=7),
        lambda record: record.update(source=""),
        lambda record: record.update(as_of=7),
        lambda record: record.pop("as_of"),
        lambda record: record.pop("index"),
        lambda record: record.pop("cost_per_task_usd"),
    ],
)
def test_catalog_quality_by_id_malformed_aa_unlabeled_degrades(mutate, capsys):
    record = {
        "index": 27.8,
        "cost_per_task_usd": 0.58,
        "source": "aa",
        "as_of": "t1",
    }
    mutate(record)
    doc = _catalog_doc(
        [
            {
                "id": "acme/model-a",
                "quality": None,
                "quality_match": None,
                "reasoning": {
                    "default_effort": "max",
                    "supported_efforts": [],
                    "depths": [],
                    "aa_unlabeled": record,
                },
            }
        ],
        schema_version=6,
    )
    # Fail closed for the whole artifact, like every other malformed row.
    assert mc.catalog_quality_by_id(doc) == {}
    assert "unexpected response shape" in capsys.readouterr().err


def test_fetch_quality_catalog_parses_schedule_entries(monkeypatch):
    """The published schedules array becomes an id -> recorded-provenance map.

    Schedule entries are the unkeyed rebuild's source of truth (spec 6.5):
    a schedule-only model never appears in the models array, so its deals
    quality can only come from here. Malformed rows are skipped, never
    guessed.
    """
    doc = _catalog_doc(
        [
            {
                "id": "acme/model-a",
                "quality": 45.1,
                "quality_match": "aa",
                "quality_as_of": "t1",
                "quality_effort": "high",
            }
        ]
    )
    doc["schedules"] = [
        {
            "id": "acme/sched",
            "quality": 47.5,
            "quality_effort": "high",
            "quality_as_of": "t9",
            "quality_source": "aa",
        },
        {
            "id": "acme/blank",
            "quality": None,
            "quality_effort": None,
            "quality_as_of": None,
            "quality_source": None,
        },
        "not a dict",
        {"quality": 1.0},  # no id
        {"id": "", "quality": 1.0},  # empty id
        {
            "id": "acme/bad-quality",
            "quality": "47.5",
            "quality_effort": "high",
            "quality_as_of": "t9",
            "quality_source": "aa",
        },
    ]
    monkeypatch.setattr(mc, "fetch_json", lambda *a, **k: doc)
    result = mc.fetch_quality_catalog(make_args())
    assert result["schedules_by_id"] == {
        "acme/sched": {
            "quality": 47.5,
            "quality_effort": "high",
            "quality_as_of": "t9",
            "quality_source": "aa",
        },
        "acme/blank": {
            "quality": None,
            "quality_effort": None,
            "quality_as_of": None,
            "quality_source": None,
        },
    }


def test_fetch_quality_catalog_hydrates_v6_and_rejects_v5(monkeypatch, capsys):
    entry = {
        "id": "acme/model-a",
        "quality": 56.0,
        "quality_match": "aa",
        "quality_as_of": "t-max",
        "quality_effort": "max",
        "reasoning": {
            "default_effort": "high",
            "depths": [
                {
                    "effort": "max",
                    "index": 56.0,
                    "cost_per_task_usd": 5.46,
                    "source": "aa",
                    "as_of": "t-max",
                }
            ],
        },
    }
    schedule = {
        "id": "acme/model-a",
        "quality": 56.0,
        "quality_effort": "max",
        "quality_as_of": "t-max",
        "quality_source": "aa",
    }
    doc = _catalog_doc([entry])
    doc["schedules"] = [schedule]
    monkeypatch.setattr(mc, "fetch_json", lambda *a, **k: doc)

    hydrated = mc.fetch_quality_catalog(make_args())
    assert hydrated["by_id"]["acme/model-a"]["index"] == 56.0
    assert hydrated["depths_by_id"]["acme/model-a"]["depths"]["max"]["index"] == 56.0
    assert hydrated["schedules_by_id"]["acme/model-a"]["quality_effort"] == "max"

    legacy = {**doc, "schema_version": 5}
    monkeypatch.setattr(mc, "fetch_json", lambda *a, **k: legacy)
    rejected = mc.fetch_quality_catalog(make_args())
    assert rejected["by_id"] == {}
    assert rejected["depths_by_id"] == {}
    assert rejected["schedules_by_id"] == {}
    assert "published catalog schema v5, expected 6" in capsys.readouterr().err


def test_fetch_quality_catalog_stale_generated_at_warns_but_uses(monkeypatch, capsys):
    """>14 days old: warn but use -- staleness stays visible via as_of, and
    the hydration is not thrown away over display-only provenance."""
    stale = (datetime.now(timezone.utc) - timedelta(days=20)).isoformat()
    doc = _catalog_doc(
        [
            {
                "id": "acme/model-a",
                "quality": 45.1,
                "quality_match": "aa",
                "quality_as_of": stale,
                "quality_effort": "high",
            }
        ],
        generated_at=stale,
    )
    monkeypatch.setattr(mc, "fetch_json", lambda *a, **k: doc)
    result = mc.fetch_quality_catalog(make_args())
    assert result["by_id"] == {
        "acme/model-a": {
            "index": 45.1,
            "source": "aa",
            "as_of": stale,
            "effort": "high",
        }
    }
    assert result["depths_by_id"] == {}
    assert "stale" in capsys.readouterr().err
    # A fresh catalog hydrates without the staleness warning.
    fresh = _catalog_doc(
        [
            {
                "id": "acme/model-a",
                "quality": 45.1,
                "quality_match": "aa",
                "quality_as_of": datetime.now(timezone.utc).isoformat(),
                "quality_effort": "high",
            }
        ],
        generated_at=datetime.now(timezone.utc).isoformat(),
    )
    monkeypatch.setattr(mc, "fetch_json", lambda *a, **k: fresh)
    result = mc.fetch_quality_catalog(make_args())
    assert result["by_id"]
    assert "stale" not in capsys.readouterr().err


def test_fetch_quality_catalog_fetch_failure_returns_none(monkeypatch, capsys):
    """Transport failure -> None (the caller's "catalog unavailable" signal);
    a fetched-but-unusable payload is NOT None -- it comes back as an
    envelope with empty maps. The None/empty distinction is pinned here."""

    def boom(*a, **k):
        raise RuntimeError("network down")

    monkeypatch.setattr(mc, "fetch_json", boom)
    assert mc.fetch_quality_catalog(make_args()) is None
    assert "published catalog request failed" in capsys.readouterr().err
    # Fetched but unusable (schema gate fail): an envelope, never None.
    monkeypatch.setattr(
        mc, "fetch_json", lambda *a, **k: _catalog_doc([], schema_version=2)
    )
    result = mc.fetch_quality_catalog(make_args())
    assert result is not None
    assert result["by_id"] == {}
    assert result["depths_by_id"] == {}
    # A payload that is not even an object: same envelope, unknown schema.
    monkeypatch.setattr(mc, "fetch_json", lambda *a, **k: ["not", "a", "dict"])
    result = mc.fetch_quality_catalog(make_args())
    assert result == {
        "by_id": {},
        "depths_by_id": {},
        "generated_at": None,
        "schedules_by_id": {},
    }


def test_fetch_quality_catalog_serves_schema_stamped_cache_without_refetch(monkeypatch):
    cached = {
        "schema_version": mc.CATALOG_QUALITY_SCHEMA_VERSION,
        "by_id": {"acme/cached": {"index": 42.0}},
        "depths_by_id": {},
        "generated_at": "2026-10-23T13:15:07+00:00",
        "schedules_by_id": {},
    }
    monkeypatch.setattr(mc, "load_cache", lambda *a, **k: cached)
    monkeypatch.setattr(
        mc,
        "fetch_json",
        lambda *a, **k: pytest.fail("schema-current cache should avoid refetch"),
    )
    assert mc.fetch_quality_catalog(make_args(no_cache=False)) == cached


@pytest.mark.parametrize("cached_schema", [None, 5], ids=["unstamped", "stale-version"])
def test_fetch_quality_catalog_refetches_unstamped_or_stale_cache(
    monkeypatch, cached_schema
):
    cached = {
        "by_id": {"acme/old": {"index": 40.0}},
        "depths_by_id": {},
        "generated_at": "2026-10-23T13:15:07+00:00",
        "schedules_by_id": {},
    }
    if cached_schema is not None:
        cached["schema_version"] = cached_schema
    doc = _catalog_doc(
        [
            {
                "id": "acme/fresh",
                "quality": 45.1,
                "quality_match": "aa",
                "quality_effort": "high",
            }
        ],
        schema_version=mc.CATALOG_SCHEMA_VERSION,
    )
    calls = []
    saved = []
    monkeypatch.setattr(mc, "load_cache", lambda *a, **k: cached)
    monkeypatch.setattr(mc, "fetch_json", lambda *a, **k: calls.append(a[0]) or doc)
    monkeypatch.setattr(mc, "save_cache", lambda *a, **k: saved.append(a[1]))

    result = mc.fetch_quality_catalog(make_args(no_cache=False))

    assert calls == [mc.QUALITY_CATALOG_URL]
    assert result["by_id"]["acme/fresh"]["index"] == 45.1
    assert result["schema_version"] == mc.CATALOG_QUALITY_SCHEMA_VERSION
    assert saved == [result]


def test_fetch_quality_catalog_respects_no_cache_and_ttl(monkeypatch, tmp_path):
    """Cache write/read roundtrip under QUALITY_CATALOG_CACHE; --no-cache
    neither reads nor writes; an expired entry is fetched again; only a
    usable (non-empty by_id) result is ever written."""
    calls = []
    doc = _catalog_doc(
        [
            {
                "id": "acme/model-a",
                "quality": 45.1,
                "quality_match": "aa",
                "quality_as_of": "2026-10-23T13:15:07+00:00",
                "quality_effort": "high",
            }
        ]
    )

    def fake_fetch(url, headers=None, timeout=30, retries=1):
        calls.append(url)
        return doc

    monkeypatch.setattr(mc, "fetch_json", fake_fetch)
    args = make_args(no_cache=False, cache_ttl=3600)
    first = mc.fetch_quality_catalog(args)
    assert calls == [mc.QUALITY_CATALOG_URL]
    cache_file = tmp_path / "model-compare" / f"{mc.QUALITY_CATALOG_CACHE}.json"
    assert cache_file.exists()
    second = mc.fetch_quality_catalog(args)
    assert len(calls) == 1  # served from cache, no second request
    assert second == first
    # An entry past its TTL is fetched again.
    blob = json.loads(cache_file.read_text())
    blob["fetched_at"] = time.time() - 7200
    cache_file.write_text(json.dumps(blob))
    mc.fetch_quality_catalog(args)
    assert len(calls) == 2
    # A fetched-but-unusable payload is never cached (self-heals next run):
    # with no cached envelope present the v2 doc is fetched, degrades to an
    # empty by_id, and leaves no cache file behind.
    cache_file.unlink()
    monkeypatch.setattr(
        mc, "fetch_json", lambda *a, **k: _catalog_doc([], schema_version=2)
    )
    assert mc.fetch_quality_catalog(args)["by_id"] == {}
    assert not cache_file.exists()
    # --no-cache neither reads nor writes the cache.
    monkeypatch.setattr(mc, "fetch_json", fake_fetch)
    calls.clear()
    result = mc.fetch_quality_catalog(make_args(no_cache=True))
    assert len(calls) == 1  # fresh fetch despite nothing cached
    assert not cache_file.exists()
    assert result["by_id"] == {
        "acme/model-a": {
            "index": 45.1,
            "source": "aa",
            "as_of": "2026-10-23T13:15:07+00:00",
            "effort": "high",
        }
    }
    assert result["depths_by_id"] == {}


def test_run_unkeyed_v5_catalog_cache_prevents_refetch(monkeypatch, capsys):
    # A still-live v5 catalog is a stable version-skew state, not a
    # transient malformed payload: its degraded (empty) envelope is cached
    # so a second unkeyed run does not re-fetch it. Older schemas remain
    # uncached/self-healing (pinned by the test above).
    models = [
        make_model(id="acme/candidate", name="Acme: Candidate"),
        make_model(
            id="acme/sched-only",
            name="Acme: Sched only",
            pricing=deepseek_pricing(),
            context_length=262144,
            reasoning={"default_effort": "high", "supported_efforts": ["high"]},
        ),
    ]
    v5_doc = _catalog_doc([], schema_version=5)
    v5_doc["schedules"] = [
        {
            "id": "acme/sched-only",
            "quality": 30.0,
            "quality_as_of": "20261017",
            "quality_source": "openrouter",
        }
    ]
    calls = []

    def fake_fetch(*a, **k):
        calls.append(1)
        return v5_doc

    monkeypatch.setattr(mc, "fetch_json", fake_fetch)
    zdr = {model["id"] for model in models}
    monkeypatch.setattr(
        mc,
        "fetch_openrouter_frontend",
        lambda a: (
            {},
            zdr,
            {"acme/sched-only": {"intelligence_index": 30.0}},
            set(),
            [],
        ),
    )
    monkeypatch.setattr(mc, "fetch_model_endpoints", lambda mid, a: [_endpoint()])
    monkeypatch.delenv("AA_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    args = make_args(
        catalog=True,
        min_context=1_000_000,
        no_cache=False,
        cache_ttl=3600,
    )
    assert mc.run(args, models, False) == 0
    capsys.readouterr()
    assert mc.run(args, models, False) == 0
    published = json.loads(capsys.readouterr().out)
    assert len(calls) == 1  # second run served from the cache
    assert published["sources"]["aa"]["mode"] == "none"
    (sched,) = published["schedules"]
    assert sched["quality"] is None


# ---------------------------------------------------------------------------
# formatting helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [(0.0, "0"), (0.123, "0.123"), (1.5, "1.50"), (12.3, "12.3"), (150.0, "150")],
)
def test_fmt_price(value, expected):
    assert mc.fmt_price(value) == expected


@pytest.mark.parametrize(
    "value,expected",
    [(500, "500"), (128_000, "128K"), (2_000_000, "2.00M")],
)
def test_fmt_context(value, expected):
    assert mc.fmt_context(value) == expected


def test_fmt_age():
    assert mc.fmt_age(None) == "-"
    assert mc.fmt_age(10) == "10d"
    assert mc.fmt_age(90).endswith("mo")
    assert mc.fmt_age(400).endswith("y")


# ---------------------------------------------------------------------------
# parse helpers
# ---------------------------------------------------------------------------


def test_parse_price():
    assert mc.parse_price("0.5") == 0.5
    assert mc.parse_price(None) is None
    assert mc.parse_price("x") is None


def test_parse_iso_datetime_naive_gets_utc():
    dt = mc.parse_iso_datetime("2024-01-01T00:00:00")
    assert dt is not None
    assert dt.tzinfo is not None


def test_parse_iso_datetime_bad():
    assert mc.parse_iso_datetime("not-a-date") is None
    assert mc.parse_iso_datetime(None) is None


# ---------------------------------------------------------------------------
# catalog document
# ---------------------------------------------------------------------------


# Test-file mirror of the producer/validator contract so the key set lives
# in exactly two places: the producer (build_catalog) and the validator it
# feeds. bsd.CATALOG_ENTRY_KEYS carries the "aa" trio plus Task 9's four
# schema-v3 additive blocks. Catalog v5 added quality_effort/reasoning;
# the current v6 producer and validator share those keys.
CATALOG_ENTRY_KEYS = set(bsd.CATALOG_ENTRY_KEYS) | {"quality_effort", "reasoning"}


def catalog_pool(**overrides):
    """Run the real pipeline over a small fixed pool and hand back everything
    build_catalog needs."""
    aa_by_id = overrides.pop(
        "aa_by_id",
        {
            "acme/model-b": {
                "intelligence_index": 68.4,
                "coding_index": 74.8,
                "agentic_index": 59.1,
            }
        },
    )
    args = make_args(min_context=0, **overrides)
    models = [
        make_model(
            id="acme/model-a",
            name="Acme: Model A",
            created=1_700_000_000,
        ),
        make_model(
            id="acme/model-b",
            name="B corp: Model B",
            pricing={"prompt": "0.000002", "completion": "0.000004"},
            context_length=4_000_000,
            created=1_750_000_000,
        ),
        # context_length is negative on purpose: the envelope test pins
        # min_context to 0 (the make_args default), so the "context" drop
        # reason -- which fires only for context < min_context -- needs a
        # negative context to land on acme/small instead of "not ZDR".
        make_model(
            id="acme/small",
            name="Small",
            context_length=-5,
        ),
    ]
    filtered = []
    candidates, dropped = mc.build_candidates(
        models, args, {"acme/model-a": 0.5}, {"acme/model-a", "acme/model-b"}, filtered
    )
    quality_by_id, quality_source_by_id = mc.resolve_quality(
        candidates, aa_by_id, {}, {}, overrides.get("aa_source", "AA API v2")
    )
    mc.compute_scores(candidates, args, quality_by_id)
    return (
        args,
        models,
        candidates,
        dropped,
        filtered,
        {"acme/model-a": 0.5},
        quality_by_id,
        quality_source_by_id,
        aa_by_id,
    )


def build_doc(**overrides):
    aa_source = overrides.pop("aa_source", "AA API v2")
    (
        args,
        models,
        candidates,
        dropped,
        filtered,
        discounts,
        quality_by_id,
        quality_source_by_id,
        aa_by_id,
    ) = catalog_pool(**overrides)
    return mc.build_catalog(
        args,
        models,
        candidates,
        dropped,
        filtered,
        discounts,
        quality_by_id,
        aa_source,
        aa_by_id,
        quality_source_by_id,
    )


def test_catalog_envelope():
    doc = build_doc()
    assert doc["schema_version"] == 6
    assert doc["tool"] == "model-compare"
    datetime.fromisoformat(doc["generated_at"])  # ISO with offset, raises if not
    p = doc["parameters"]
    assert p["input_share"] == 0.75
    assert p["quality_ref"] == 70.0
    assert p["min_context"] == 0
    assert p["recency_half_life"] == 120.0
    assert p["max_age_days"] == 0.0
    assert p["zdr_required"] is True
    assert p["require_tools"] is False  # make_args default
    assert p["exclude_free"] is False
    assert p["include_batch"] is False
    assert p["reasoning_effort"] == "default"
    assert set(p["weights"]) == {"balanced", "price", "quality"}
    assert p["weights"]["balanced"] == mc.PRIORITY_WEIGHTS["balanced"]
    assert doc["sources"] == {
        "openrouter": "ok",
        # The keyed chain (the fixture passes an AA source); model-b matched
        # through OpenRouter's benchmarks, so its bucket holds the count.
        "aa": {
            "mode": "aa",
            "matched_aa": 0,
            "matched_catalog": 0,
            "matched_openrouter": 1,
            # Depth coverage is candidate-scoped and separate from the
            # disjoint label buckets. The legacy fixture path measured no
            # depths.
            "depth_rows": 0,
            "matched_depth": 0,
            "matched_default": 1,
            "catalog_generated_at": None,
            "intelligence_index_version": None,
        },
        "zdr": "ok",
        "discounts": "ok",
        # Task 8's additive source states: the endpoint stage always ran
        # before build_catalog, and direct build_catalog calls (this
        # fixture) take the default keyless mode.
        "endpoints": "ok",
        "session_cost": "skipped",
    }
    # Default parameters carry no envelope: the block is omitted, not null.
    assert "session_cost" not in doc
    assert doc["pool"]["listed"] == 3
    assert doc["pool"]["candidates"] == 2
    assert doc["pool"]["dropped"]["context"] == 1
    assert set(doc["pool"]["dropped"]) == set(mc.CATALOG_DROP_REASONS)
    assert sum(doc["pool"]["dropped"].values()) == 3 - doc["pool"]["candidates"]


def test_catalog_entry_shape():
    doc = build_doc()
    assert {e["id"] for e in doc["models"]} == {"acme/model-a", "acme/model-b"}
    for entry in doc["models"]:
        # Task 12b's steering is an extra entry key NOT in CATALOG_ENTRY_KEYS:
        # the producer emits it on every entry, but validate_catalog must
        # keep accepting pre-endpoint-shaped documents that lack it, so it
        # stays optional (outside the required set).
        assert set(entry) == CATALOG_ENTRY_KEYS | {"steering"}
        assert set(entry["pricing"]) == {
            "input_per_1m",
            "output_per_1m",
            "blended_per_1m",
            "base",
            "tiers",
            "schedule",
        }
        assert set(entry["pricing"]["base"]) == {
            "input_per_1m",
            "output_per_1m",
            "blended_per_1m",
        }
        assert set(entry["scores"]) == {"price", "quality", "context", "age", "overall"}
        assert set(entry["scores"]["overall"]) == {"balanced", "price", "quality"}
    a = next(e for e in doc["models"] if e["id"] == "acme/model-a")
    assert a["name"] == "Model A"  # vendor prefix stripped
    assert a["provider"] == "acme"
    assert a["family"] == "model"
    assert a["pricing"] == {
        "input_per_1m": 1.0,
        "output_per_1m": 2.0,
        "blended_per_1m": 1.25,
        "base": {
            "input_per_1m": 1.0,
            "output_per_1m": 2.0,
            "blended_per_1m": 1.25,
        },
        "tiers": [],
        "schedule": None,
    }
    assert a["listed_at"] == "2023-11-14"  # utc date of 1_700_000_000
    assert isinstance(a["age_days"], int) and a["age_days"] >= 0
    assert a["tool_calling"] is True
    assert a["zdr"] is True
    assert a["discount"] == 0.5
    assert a["expired"] is False
    assert a["quality"] is None
    assert a["quality_match"] is None  # matched nothing
    b = next(e for e in doc["models"] if e["id"] == "acme/model-b")
    assert b["quality"] == 68.4
    assert b["quality_match"] == "openrouter"
    # Provenance (schema v4): the fixture's OR benchmark carries no as_of,
    # and an OpenRouter-sourced record has no drift pair to report.
    assert b["quality_as_of"] is None
    assert b["quality_drift"] is None
    assert b["discount"] is None


def test_build_catalog_v6_carries_reasoning_block():
    cand = _cand("acme/model-a", 50.0)
    cand.update(created=0.0, tool_calling=True, zdr=True, expired=False)
    cand["quality_record"] = {
        "index": 46.8,
        "source": "aa",
        "as_of": "t1",
        "effort": "high",
    }
    cand["reasoning"] = {
        "default_effort": "high",
        "supported_efforts": ["max", "high"],
        "is_reasoning": True,
    }
    cand["reasoning_depths"] = [
        {"effort": "max", "index": 56.0, "source": "aa", "as_of": "t1"}
    ]
    cand["quality_unlabeled"] = {
        "index": 27.8,
        "cost_per_task_usd": 0.58,
        "source": "aa",
        "as_of": "t1",
    }
    mc.compute_scores(
        [cand],
        make_args(),
        {
            "acme/model-a": {
                "index": 46.8,
                "source": "aa",
                "as_of": "t1",
                "effort": "high",
            }
        },
    )
    doc = mc.build_catalog(
        make_args(),
        [make_model()],
        [cand],
        {},
        [],
        {},
        {"acme/model-a": {"index": 46.8}},
        "AA API v2",
        {},
        {"acme/model-a": "aa"},
    )
    entry = doc["models"][0]
    assert doc["schema_version"] == 6
    assert doc["parameters"]["reasoning_effort"] == "default"
    assert entry["quality_effort"] == "high"
    assert entry["reasoning"]["default_effort"] == "high"
    assert entry["reasoning"]["supported_efforts"] == ["max", "high"]
    # No supports_max_tokens in the fixture: unknown publishes as null, not
    # the old bool() coercion's False.
    assert entry["reasoning"]["supports_max_tokens"] is None
    assert entry["reasoning"]["depths"][0]["effort"] == "max"
    # The published depth record reuses the quality record's shape: `index`,
    # never `quality_index`.
    assert entry["reasoning"]["depths"][0]["index"] == 56.0
    # The display-only unlabeled AA reference rides in the reasoning block;
    # it never becomes the entry's main quality.
    assert entry["reasoning"]["aa_unlabeled"] == {
        "index": 27.8,
        "cost_per_task_usd": 0.58,
        "source": "aa",
        "as_of": "t1",
    }
    assert entry["quality"] == 46.8
    assert doc["sources"]["aa"]["depth_rows"] == 1
    assert doc["sources"]["aa"]["matched_depth"] == 1
    assert doc["sources"]["aa"]["matched_default"] == 1


def _doc_with(models, min_context=1_000_000, zdr_ids=None, aa_by_id=None):
    args = make_args(min_context=min_context)
    filtered = []
    zdr = {m["id"] for m in models} if zdr_ids is None else zdr_ids
    candidates, dropped = mc.build_candidates(models, args, {}, zdr, filtered)
    mc.compute_scores(candidates, args, {})
    return mc.build_catalog(
        args, models, candidates, dropped, filtered, {}, {}, None, aa_by_id or {}, {}
    )


def test_catalog_tiered_pricing_block():
    doc = _doc_with([make_model(id="acme/haiku", pricing=haiku_pricing())])
    (entry,) = doc["models"]
    pricing = entry["pricing"]
    assert pricing["input_per_1m"] == 0.5
    assert pricing["output_per_1m"] == 2.5
    assert pricing["base"] == {
        "input_per_1m": pytest.approx(0.1),
        "output_per_1m": pytest.approx(0.5),
        "blended_per_1m": pytest.approx(0.2),
    }
    assert pricing["tiers"] == [
        {
            "min_prompt_tokens": 100000,
            "input_per_1m": 0.5,
            "output_per_1m": 2.5,
            "blended_per_1m": 1.0,
        }
    ]
    assert pricing["schedule"] is None


def test_catalog_schedule_pricing_block():
    doc = _doc_with([make_model(id="acme/hy4", pricing=hy4_pricing())])
    (entry,) = doc["models"]
    pricing = entry["pricing"]
    assert pricing["input_per_1m"] == pytest.approx(0.834)
    assert pricing["output_per_1m"] == pytest.approx(2.501)
    assert pricing["base"]["input_per_1m"] == pytest.approx(0.834)  # frozen peak
    assert pricing["base"]["output_per_1m"] == pytest.approx(2.501)
    schedule = pricing["schedule"]
    assert [(w["utc_start"], w["utc_end"], w["coverage"]) for w in schedule] == [
        (0, 1600, 0.6667),
        (1600, 0, 0.3333),
    ]
    assert schedule[0]["input_per_1m"] == pytest.approx(0.834)
    assert schedule[0]["blended_per_1m"] == pytest.approx(1.25075)
    assert pricing["tiers"] == []


def test_catalog_schedules_top_level():
    models = [
        make_model(id="acme/hy3", pricing=deepseek_pricing(), context_length=262144),
        make_model(id="acme/candidate-hy4", pricing=hy4_pricing()),
        make_model(id="~acme/alias", pricing=hy4_pricing()),
        make_model(id="acme/batch-hy4:batch", pricing=hy4_pricing()),
        make_model(id="acme/nonzdr-hy4", pricing=hy4_pricing()),
        make_model(id="acme/plain"),
    ]
    zdr = {m["id"] for m in models} - {"acme/nonzdr-hy4"}
    doc = _doc_with(models, zdr_ids=zdr)
    ids = [s["id"] for s in doc["schedules"]]
    assert ids == sorted(ids)
    assert "acme/hy3" in ids  # sub-floor context: schedules are unfiltered
    assert "acme/batch-hy4:batch" in ids  # :batch variants are distinct deals
    assert "acme/nonzdr-hy4" in ids  # non-ZDR: schedules stay unfiltered
    assert "~acme/alias" not in ids  # router aliases excluded
    assert "acme/plain" not in ids  # no schedule
    by_id = {s["id"]: s for s in doc["schedules"]}
    candidate = by_id["acme/candidate-hy4"]
    assert candidate["score"] is not None  # balanced overall for candidates
    assert candidate["quality"] is None  # no AA data in this document
    noncand = by_id["acme/hy3"]
    assert noncand["score"] is None
    assert noncand["quality"] is None
    assert noncand["max_discount"] == 0.5
    assert noncand["sched_note"] == "-50%"
    assert noncand["sched_detail"] == (
        "weekdays 00:00-01:00, 04:00-06:00, 10:00-00:00; weekends all day UTC"
    )
    assert noncand["context"] == 262144
    assert noncand["offpeak"]["input_per_1m"] == pytest.approx(0.66)
    assert noncand["peak"]["input_per_1m"] == pytest.approx(1.32)


def test_catalog_schedules_quality_from_openrouter_aa():
    doc = _doc_with(
        [
            make_model(id="acme/model-a"),  # keeps the candidate pool non-empty
            make_model(
                id="acme/hy3", pricing=deepseek_pricing(), context_length=262144
            ),
        ],
        aa_by_id={"acme/hy3": {"intelligence_index": 40.0}},
    )
    by_id = {entry["id"]: entry for entry in doc["schedules"]}
    assert by_id["acme/hy3"]["quality"] == 40.0  # OR-published AA for non-candidates
    assert by_id["acme/hy3"]["score"] is None


def test_catalog_determinism_with_frozen_clock(monkeypatch):
    class FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 10, 9, 6, 0, 0, tzinfo=timezone.utc)

    monkeypatch.setattr(mc, "datetime", FrozenDatetime)
    monkeypatch.setattr(mc.time, "time", lambda: 1_700_000_000.0)

    def build(top_in, top_out):
        pricing = dict(hy4_pricing())
        pricing["prompt"] = top_in
        pricing["completion"] = top_out
        doc = _doc_with([make_model(id="acme/hy4", pricing=pricing)])
        return json.dumps(doc, sort_keys=True)

    # Two fetch times differ only in the windowed model's top-level prices
    # (each equals one of its windows); the frozen peak base must make the
    # documents byte-identical.
    first = build("0.000000834", "0.000002501")
    second = build("0.0000007506", "0.0000022509")
    assert first == second


def test_catalog_future_created_age_days_clamped():
    # created in the future must not produce a negative age_days
    created = time.time() + 86400
    args = make_args(min_context=0)
    models = [make_model(id="acme/model-a", created=created)]
    candidates, dropped = mc.build_candidates(models, args, {}, {"acme/model-a"}, [])
    mc.compute_scores(candidates, args, {})
    doc = mc.build_catalog(args, models, candidates, dropped, [], {}, {}, None, {}, {})
    (entry,) = doc["models"]
    expected = datetime.fromtimestamp(created, tz=timezone.utc).date().isoformat()
    assert entry["listed_at"] == expected
    assert entry["age_days"] == 0


def test_catalog_family_null_without_prefix():
    # no model in the default pool exercises the None path; check directly
    assert mc.model_family("kimi/k2") is None


def test_catalog_overall_covers_all_priorities():
    args, _, candidates, _, _, _, quality_by_id, _, _ = catalog_pool()
    doc = mc.build_catalog(
        args, [], candidates, {}, [], {}, quality_by_id, "AA API v2", {}, {}
    )
    weights = mc.catalog_weights(candidates, quality_by_id)
    for entry in doc["models"]:
        s = entry["scores"]
        for priority, w in weights.items():
            expected = round(
                w.get("quality", 0.0) * s["quality"]
                + w.get("price", 0.0) * s["price"]
                + w.get("context", 0.0) * s["context"]
                + w.get("age", 0.0) * s["age"],
                4,
            )
            assert s["overall"][priority] == expected


def test_catalog_overall_matches_compute_scores_for_current_priority():
    args, _, candidates, _, _, _, quality_by_id, _, _ = catalog_pool(priority="price")
    doc = mc.build_catalog(
        args, [], candidates, {}, [], {}, quality_by_id, "AA API v2", {}, {}
    )
    by_id = {c["id"]: c for c in candidates}
    for entry in doc["models"]:
        assert entry["scores"]["overall"]["price"] == pytest.approx(
            round(by_id[entry["id"]]["score"], 4), abs=2e-4
        )


def test_catalog_scores_in_unit_range_and_no_nan():
    doc = build_doc()
    for entry in doc["models"]:
        for key in ("price", "quality", "context", "age"):
            value = entry["scores"][key]
            assert isinstance(value, (int, float)) and 0.0 <= value <= 1.0
        for value in entry["scores"]["overall"].values():
            assert 0.0 <= value <= 1.0


def test_catalog_filtered_entries_and_sorting():
    doc = build_doc()
    assert doc["filtered"] == [
        {"id": "acme/small", "name": "Small", "reasons": ["context"]}
    ]
    overalls = [e["scores"]["overall"]["balanced"] for e in doc["models"]]
    assert overalls == sorted(overalls, reverse=True)
    ids = [e["id"] for e in doc["models"]]
    assert ids == [
        e["id"]
        for e in sorted(
            doc["models"], key=lambda e: (-e["scores"]["overall"]["balanced"], e["id"])
        )
    ]


def inverting_pool():
    """Three candidates whose order inverts between the price and quality
    priorities: a cheap low-quality model, an expensive high-quality one,
    and a middle one. catalog_pool() cannot catch a wrong-weight-vector bug
    because one of its two models dominates every priority."""
    args = make_args(min_context=0)
    models = [
        make_model(
            id="acme/cheap",
            name="Cheap",
            pricing={"prompt": "0.0000001", "completion": "0.0000002"},
        ),
        make_model(
            id="acme/pricey",
            name="Pricey",
            pricing={"prompt": "0.00001", "completion": "0.00004"},
        ),
        make_model(
            id="acme/middle",
            name="Middle",
            pricing={"prompt": "0.000001", "completion": "0.000003"},
        ),
    ]
    zdr = {m["id"] for m in models}
    quality_by_id = {"acme/cheap": 20.0, "acme/pricey": 69.0, "acme/middle": 45.0}
    return args, models, zdr, quality_by_id


def _cli_order(priority):
    args, models, zdr, quality_by_id = inverting_pool()
    args.priority = priority
    candidates, _ = mc.build_candidates(models, args, {}, zdr, [])
    mc.compute_scores(candidates, args, quality_by_id)
    return [c["id"] for c in candidates]


def _inverting_catalog():
    args, models, zdr, quality_by_id = inverting_pool()
    filtered = []
    candidates, dropped = mc.build_candidates(models, args, {}, zdr, filtered)
    mc.compute_scores(candidates, args, quality_by_id)
    return mc.build_catalog(
        args, models, candidates, dropped, filtered, {}, quality_by_id, None, {}, {}
    )


def test_catalog_rankings_match_compute_scores_per_priority():
    # Single-key fence: the catalog ranking for P is exactly the order a
    # `--priority P` CLI run produces on the same data.
    doc = _inverting_catalog()
    assert set(doc["rankings"]) == set(mc.PRIORITY_WEIGHTS)
    orders = {p: _cli_order(p) for p in mc.PRIORITY_WEIGHTS}
    # the fixture must actually invert, or a wrong weight vector could pass
    assert orders["price"][0] == "acme/cheap"
    assert orders["quality"][0] == "acme/pricey"
    assert orders["price"] != orders["quality"]
    for priority in mc.PRIORITY_WEIGHTS:
        assert doc["rankings"][priority] == orders[priority], priority


def test_catalog_rankings_are_full_permutations_of_models():
    doc = build_doc()
    ids = {e["id"] for e in doc["models"]}
    for ranking in doc["rankings"].values():
        assert len(ranking) == len(doc["models"])
        assert set(ranking) == ids


def test_catalog_rankings_builder_is_pure():
    args, models, zdr, quality_by_id = inverting_pool()
    candidates, _ = mc.build_candidates(models, args, {}, zdr, [])
    mc.compute_scores(candidates, args, quality_by_id)
    weights = mc.catalog_weights(candidates, quality_by_id)
    before = json.dumps(candidates, sort_keys=True)
    order_before = [c["id"] for c in candidates]
    rankings = mc.catalog_rankings(candidates, weights)
    assert json.dumps(candidates, sort_keys=True) == before
    assert [c["id"] for c in candidates] == order_before
    assert rankings["balanced"] == order_before  # compute_scores ran balanced


def test_ranking_key_breaks_ties_like_the_cli():
    a = {"id": "b/x", "quality": 50.0, "blended": 1.0}
    b = {"id": "a/x", "quality": 50.0, "blended": 1.0}
    c = {"id": "c/x", "quality": None, "blended": 0.5}
    d = {"id": "d/x", "quality": 50.0, "blended": 0.5}
    ordered = sorted([a, b, c, d], key=lambda cand: mc.ranking_key(0.5, cand))
    assert [x["id"] for x in ordered] == ["d/x", "a/x", "b/x", "c/x"]


def test_weighted_score_operand_order():
    # Pins the operand ORDER (quality, price, context, age): float addition
    # is non-associative, so a reordered sum can drift by one ULP. The shared
    # weighted_score function is the structural guarantee that the CLI and
    # the catalog agree; this test guards against someone reordering it. No
    # single vector distinguishes all 23 wrong orders (many sums collide
    # exactly), hence several: the two added vectors each diverge on 22 of
    # 23. The 23rd (swapping the first two terms) is undetectable and
    # harmless, since IEEE-754 a + b == b + a exactly.
    cand = {
        "quality_score": 0.1,
        "price_score": 0.2,
        "context_score": 0.3,
        "age_score": 0.7,
    }
    w = {"quality": 0.3, "price": 0.3, "context": 0.2, "age": 0.2}
    expected = 0.3 * 0.1 + 0.3 * 0.2 + 0.2 * 0.3 + 0.2 * 0.7
    assert mc.weighted_score(cand, w) == expected  # exact, not approx
    assert mc.weighted_score(cand, {"price": 1.0}) == 1.0 * 0.2

    cand = {
        "quality_score": 0.89,
        "price_score": 0.66,
        "context_score": 0.8,
        "age_score": 0.84,
    }
    w = {"quality": 0.17, "price": 0.24, "context": 0.32, "age": 0.27}
    expected = 0.17 * 0.89 + 0.24 * 0.66 + 0.32 * 0.8 + 0.27 * 0.84
    assert mc.weighted_score(cand, w) == expected

    cand = {
        "quality_score": 0.47,
        "price_score": 0.35,
        "context_score": 0.43,
        "age_score": 0.79,
    }
    w = {"quality": 0.09, "price": 0.35, "context": 0.34, "age": 0.22}
    expected = 0.09 * 0.47 + 0.35 * 0.35 + 0.34 * 0.43 + 0.22 * 0.79
    assert mc.weighted_score(cand, w) == expected


def test_catalog_rankings_deterministic_across_builds_and_input_order():
    # Same synthetic candidates, built twice -> identical rankings; and since
    # the key is total (id is the last tiebreak), the caller's list order
    # cannot leak into them either.
    first = _inverting_catalog()["rankings"]
    second = _inverting_catalog()["rankings"]
    assert first == second
    args, models, zdr, quality_by_id = inverting_pool()
    candidates, _ = mc.build_candidates(models, args, {}, zdr, [])
    mc.compute_scores(candidates, args, quality_by_id)
    weights = mc.catalog_weights(candidates, quality_by_id)
    assert mc.catalog_rankings(list(reversed(candidates)), weights) == first


def test_cli_json_rows_project_cleanly_onto_catalog_rankings(capsys):
    # Cross-program check: real --json rows (top 2) per priority and the real
    # catalog pass the site's order_rows / build_snapshot.
    doc = _inverting_catalog()
    # The site validator must use the producer's catalog contract; the
    # unconditional lockstep test pins that requirement.
    if bsd.CATALOG_SCHEMA_VERSION == mc.CATALOG_SCHEMA_VERSION:
        bsd.validate_catalog(doc)
    priorities = {}
    for priority in mc.PRIORITY_WEIGHTS:
        args, models, zdr, quality_by_id = inverting_pool()
        args.priority = priority
        candidates, _ = mc.build_candidates(models, args, {}, zdr, [])
        mc.compute_scores(candidates, args, quality_by_id)
        mc.print_json(candidates[:2])
        priorities[priority] = json.loads(capsys.readouterr().out)
    data = bsd.build_data(
        mc.opencode_model_id(doc["rankings"]["balanced"][0]), priorities, catalog=doc
    )
    snap = bsd.build_snapshot(doc)
    for priority in mc.PRIORITY_WEIGHTS:
        table = [row["model"] for row in data["priorities"][priority]]
        assert table == doc["rankings"][priority][:2]
        assert [row["id"] for row in snap["tabs"][priority]][:2] == table


def test_catalog_deterministic_modulo_generated_at():
    doc1 = build_doc()
    doc2 = build_doc()
    doc1.pop("generated_at")
    doc2.pop("generated_at")
    assert doc1 == doc2


def test_catalog_document_passes_site_validator():
    # Drift between the producer and the site validator must fail the suite
    # here, not the publish; the unconditional schema lockstep test pins the
    # version contract too.
    # mixed provenance (OR benchmarks + AA api fallback) must survive too
    args, models, candidates, dropped, filtered, discounts, _, _, aa_by_id = (
        catalog_pool()
    )
    exact, fuzzy = mc.build_aa_lookup(
        [{"key": "openai/model-a", "name": "Model A", "index": 44.0}]
    )
    quality_by_id, quality_source_by_id = mc.resolve_quality(
        candidates, aa_by_id, exact, fuzzy, "AA API v2"
    )
    doc = mc.build_catalog(
        args,
        models,
        candidates,
        dropped,
        filtered,
        discounts,
        quality_by_id,
        "AA API v2",
        aa_by_id,
        quality_source_by_id,
    )
    assert {e["quality_match"] for e in doc["models"]} == {"aa", "openrouter"}
    assert doc["sources"]["aa"] == {
        "mode": "aa",
        "matched_aa": 1,
        "matched_catalog": 0,
        "matched_openrouter": 1,
        "depth_rows": 0,
        "matched_depth": 0,
        "matched_default": 2,
        "catalog_generated_at": None,
        "intelligence_index_version": None,
    }
    # The site validator must use the producer's catalog contract; the
    # unconditional lockstep test pins that requirement.
    if bsd.CATALOG_SCHEMA_VERSION == mc.CATALOG_SCHEMA_VERSION:
        bsd.validate_catalog(doc)


def test_catalog_aa_block_carries_or_trio():
    doc = build_doc()
    b = next(e for e in doc["models"] if e["id"] == "acme/model-b")
    assert b["aa"] == {
        "intelligence_index": 68.4,
        "coding_index": 74.8,
        "agentic_index": 59.1,
    }
    a = next(e for e in doc["models"] if e["id"] == "acme/model-a")
    assert a["aa"] == {
        "intelligence_index": None,
        "coding_index": None,
        "agentic_index": None,
    }


def test_catalog_quality_match_and_sources_counts():
    doc = build_doc()
    b = next(e for e in doc["models"] if e["id"] == "acme/model-b")
    assert b["quality"] == 68.4
    assert b["quality_match"] == "openrouter"
    a = next(e for e in doc["models"] if e["id"] == "acme/model-a")
    assert a["quality"] is None
    assert a["quality_match"] is None
    assert doc["sources"]["aa"] == {
        "mode": "aa",
        "matched_aa": 0,
        "matched_catalog": 0,
        "matched_openrouter": 1,
        "depth_rows": 0,
        "matched_depth": 0,
        "matched_default": 1,
        "catalog_generated_at": None,
        "intelligence_index_version": None,
    }


def test_catalog_quality_published_even_without_any_aa_source():
    doc = build_doc(aa_source=None)
    b = next(e for e in doc["models"] if e["id"] == "acme/model-b")
    assert b["quality"] == 68.4
    assert b["quality_match"] == "openrouter"
    # Without a key and without catalog meta the chain is "none"; the OR
    # tier is the chain's ambient last resort and shows up only in its own
    # bucket count.
    assert doc["sources"]["aa"]["mode"] == "none"


def test_catalog_aa_mode_when_or_empty():
    doc = build_doc(aa_source="AA API v2", aa_by_id={})
    assert doc["sources"]["aa"] == {
        "mode": "aa",
        "matched_aa": 0,
        "matched_catalog": 0,
        "matched_openrouter": 0,
        "depth_rows": 0,
        "matched_depth": 0,
        "matched_default": 0,
        "catalog_generated_at": None,
        "intelligence_index_version": None,
    }


def test_catalog_unknown_aa_source_fails_loudly():
    # a new AA source must update the mode map -- never publish a document
    # claiming mode "none" while entries carry matched quality
    with pytest.raises(ValueError, match="unknown AA source"):
        build_doc(aa_source="AA carrier pigeon")


def test_catalog_rejects_removed_scrape_source():
    # the page scrape is gone; its source string must not map to a mode
    with pytest.raises(ValueError, match="unknown AA source"):
        build_doc(aa_source="AA page scrape")


def _provenance_pool():
    """Four candidates covering every provenance shape the catalog carries.

    Returns (args, models, candidates, dropped, filtered, or_by_id,
    aa_api_entries, catalog_quality) -- the raw inputs; each test resolves
    quality the way its scenario needs, then scores and builds.

    acme/a: only a hydrated published-catalog record (parser shape: the
        row carries the keyed CI run's own "aa" label; resolve_quality's
        catalog tier relabels the record to the chain label "catalog").
    acme/d: AA API record whose OR check value differs by >= 3.0 (drift).
    acme/e: AA API record within the drift threshold of its OR check value.
    acme/f: AA API record with no OpenRouter check value at all.
    """
    args = make_args(min_context=0)
    models = [
        make_model(id="acme/a", name="Acme: A"),
        make_model(id="acme/d", name="Acme: D"),
        make_model(id="acme/e", name="Acme: E"),
        make_model(id="acme/f", name="Acme: F"),
    ]
    filtered = []
    zdr = {m["id"] for m in models}
    candidates, dropped = mc.build_candidates(models, args, {}, zdr, filtered)
    aa_api_entries = [
        {"key": "acme/d", "name": "D", "index": 45.1, "as_of": "2026-10-23T13:15:07Z"},
        {"key": "acme/e", "name": "E", "index": 42.4, "as_of": "2026-10-23T13:15:07Z"},
        {"key": "acme/f", "name": "F", "index": 45.1, "as_of": "2026-10-23T13:15:07Z"},
    ]
    or_by_id = {
        "acme/d": {
            "intelligence_index": 39.5,
            "coding_index": 50.0,
            "agentic_index": 40.0,
            "as_of": "20261017",
        },
        "acme/e": {
            "intelligence_index": 39.5,
            "coding_index": 50.0,
            "agentic_index": 40.0,
            "as_of": "20261017",
        },
    }
    catalog_quality = {
        "acme/a": {
            "index": 40.0,
            "source": "aa",
            "as_of": "2026-10-20T09:00:00Z",
        },
    }
    return (
        args,
        models,
        candidates,
        dropped,
        filtered,
        or_by_id,
        aa_api_entries,
        catalog_quality,
    )


def test_catalog_entries_carry_quality_as_of_and_source():
    (
        args,
        models,
        candidates,
        dropped,
        filtered,
        or_by_id,
        aa_api_entries,
        catalog_quality,
    ) = _provenance_pool()
    exact, fuzzy = mc.build_aa_lookup(aa_api_entries)
    quality_by_id, quality_source_by_id = mc.resolve_quality(
        candidates, or_by_id, exact, fuzzy, "AA API v2", catalog_quality
    )
    mc.compute_scores(candidates, args, quality_by_id)
    doc = mc.build_catalog(
        args,
        models,
        candidates,
        dropped,
        filtered,
        {},
        quality_by_id,
        "AA API v2",
        or_by_id,
        quality_source_by_id,
    )
    entries = {e["id"]: e for e in doc["models"]}
    # Every matched entry carries its record's as_of verbatim -- the AA
    # fetch stamp, the hydrated catalog row's stamp -- and the label of the
    # winning tier.
    assert entries["acme/a"]["quality_match"] == "catalog"
    assert entries["acme/a"]["quality_as_of"] == "2026-10-20T09:00:00Z"
    assert entries["acme/d"]["quality_match"] == "aa"
    assert entries["acme/d"]["quality_as_of"] == "2026-10-23T13:15:07Z"
    assert entries["acme/e"]["quality_as_of"] == "2026-10-23T13:15:07Z"
    assert entries["acme/f"]["quality_as_of"] == "2026-10-23T13:15:07Z"


def test_catalog_flags_quality_drift_verbatim_from_record():
    (
        args,
        models,
        candidates,
        dropped,
        filtered,
        or_by_id,
        aa_api_entries,
        catalog_quality,
    ) = _provenance_pool()
    exact, fuzzy = mc.build_aa_lookup(aa_api_entries)
    quality_by_id, quality_source_by_id = mc.resolve_quality(
        candidates, or_by_id, exact, fuzzy, "AA API v2", catalog_quality
    )
    mc.compute_scores(candidates, args, quality_by_id)
    doc = mc.build_catalog(
        args,
        models,
        candidates,
        dropped,
        filtered,
        {},
        quality_by_id,
        "AA API v2",
        or_by_id,
        quality_source_by_id,
    )
    entries = {e["id"]: e for e in doc["models"]}
    # The drift flag is the record's own verdict, verbatim: 45.1 vs 39.5
    # (5.6 points) flags, 42.4 vs 39.5 (2.9 points) does not, and a record
    # with no OpenRouter check value carries no verdict at all.
    assert entries["acme/d"]["quality_drift"] is True
    assert entries["acme/e"]["quality_drift"] is False
    assert entries["acme/f"]["quality_drift"] is None
    assert entries["acme/a"]["quality_drift"] is None


def test_catalog_sources_aa_block_reshaped_with_counts_version_and_meta():
    (
        args,
        models,
        candidates,
        dropped,
        filtered,
        or_by_id,
        aa_api_entries,
        catalog_quality,
    ) = _provenance_pool()
    exact, fuzzy = mc.build_aa_lookup(aa_api_entries)
    quality_by_id, quality_source_by_id = mc.resolve_quality(
        candidates, or_by_id, exact, fuzzy, "AA API v2", catalog_quality
    )
    mc.compute_scores(candidates, args, quality_by_id)
    doc = mc.build_catalog(
        args,
        models,
        candidates,
        dropped,
        filtered,
        {},
        quality_by_id,
        "AA API v2",
        or_by_id,
        quality_source_by_id,
        index_version=4.3,
    )
    # Keyed run: mode is the AA chain; the three counts are disjoint label
    # buckets summing to the matched entries; the AA envelope's index
    # version rides along; no catalog meta (keyed runs never hydrate).
    assert doc["sources"]["aa"] == {
        "mode": "aa",
        "matched_aa": 3,
        "matched_catalog": 1,
        "matched_openrouter": 0,
        "depth_rows": 0,
        "matched_depth": 0,
        "matched_default": 4,
        "catalog_generated_at": None,
        "intelligence_index_version": 4.3,
    }

    # Unkeyed run with a hydrated catalog: mode flips to the catalog chain
    # and the published catalog's generated_at is the block's meta. The
    # run-level label is the source chain, not the row's own: tier 2
    # relabels every hydrated record "catalog" (real catalogs carry the
    # keyed CI run's row-level "aa"/"openrouter" labels), so an unkeyed
    # run never reports matched_aa -- and OpenRouter covers the ids the
    # catalog lacks.
    hydrated = {
        "acme/a": {"index": 40.0, "source": "aa", "as_of": "2026-10-20T09:00:00Z"},
        "acme/d": {
            "index": 44.0,
            "source": "openrouter",
            "as_of": "2026-10-20T09:00:00Z",
        },
    }
    quality_by_id, quality_source_by_id = mc.resolve_quality(
        candidates, or_by_id, {}, {}, None, hydrated
    )
    mc.compute_scores(candidates, args, quality_by_id)
    doc = mc.build_catalog(
        args,
        models,
        candidates,
        dropped,
        filtered,
        {},
        quality_by_id,
        None,
        or_by_id,
        quality_source_by_id,
        catalog_generated_at="2026-10-23T13:15:07+00:00",
    )
    assert doc["sources"]["aa"] == {
        "mode": "catalog",
        "matched_aa": 0,
        "matched_catalog": 2,
        "matched_openrouter": 1,
        "depth_rows": 0,
        "matched_depth": 0,
        "matched_default": 3,
        "catalog_generated_at": "2026-10-23T13:15:07+00:00",
        "intelligence_index_version": None,
    }
    # The site validator must use the producer's catalog contract; the
    # unconditional lockstep test pins that requirement.
    if bsd.CATALOG_SCHEMA_VERSION == mc.CATALOG_SCHEMA_VERSION:
        bsd.validate_catalog(doc)


def test_catalog_no_zdr_marks_skipped(monkeypatch, capsys):
    # run() now fetches session cost once per invocation (Task 8); pin the
    # keyless path so the suite never touches the network for it.
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(
        mc, "fetch_openrouter_frontend", lambda args: ({}, set(), {}, set(), [])
    )
    monkeypatch.setattr(mc, "fetch_aa_entries", lambda args: ([], None, False, None))
    # Unkeyed runs hydrate quality from the published catalog (Task 4);
    # this test is about the zdr source state, so the catalog stays down.
    monkeypatch.setattr(mc, "fetch_quality_catalog", lambda args: None)
    # The endpoint stage fetches per candidate: one eligible endpoint keeps
    # the models in the pool (offline stub, like every other fetcher here).
    monkeypatch.setattr(mc, "fetch_model_endpoints", lambda mid, a: [_endpoint()])
    models = [
        make_model(id="acme/model-a", created=1_700_000_000),
        make_model(
            id="acme/model-b", pricing={"prompt": "0.000002", "completion": "0.000004"}
        ),
    ]
    args = make_args(min_context=0, no_zdr=True, catalog=True)
    assert mc.run(args, models, False) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["sources"]["zdr"] == "skipped"
    assert doc["parameters"]["zdr_required"] is False
    assert all(e["zdr"] is None for e in doc["models"])


def test_catalog_fails_closed_without_zdr(monkeypatch, capsys):
    monkeypatch.setattr(
        mc, "fetch_openrouter_frontend", lambda args: ({}, set(), {}, set(), [])
    )
    monkeypatch.setattr(mc, "fetch_aa_entries", lambda args: ([], None, False, None))
    models = [make_model(id="acme/model-a")]
    args = make_args(min_context=0, catalog=True)
    assert mc.run(args, models, False) == 1
    captured = capsys.readouterr()
    assert captured.out == ""  # no document
    assert "ZDR" in captured.err


def test_catalog_discounts_unavailable_source():
    args, models, candidates, dropped, filtered, _discounts, quality_by_id, _, _ = (
        catalog_pool()
    )
    doc = mc.build_catalog(
        args, models, candidates, dropped, filtered, {}, quality_by_id, None, {}, {}
    )
    assert doc["sources"]["discounts"] == "unavailable"


def test_parse_args_rejects_catalog_with_best(capsys):
    with pytest.raises(SystemExit) as exc:
        mc.parse_args(["--catalog", "--best"])
    assert exc.value.code == 2
    assert "cannot be combined" in capsys.readouterr().err


def test_parse_args_rejects_catalog_with_json(capsys):
    with pytest.raises(SystemExit) as exc:
        mc.parse_args(["--catalog", "--json"])
    assert exc.value.code == 2
    assert "cannot be combined" in capsys.readouterr().err


def test_parse_args_accepts_catalog_alone():
    args = mc.parse_args(["--catalog"])
    assert args.catalog is True


def test_parse_args_no_pager_flag():
    assert mc.parse_args(["--no-pager"]).no_pager is True
    assert mc.parse_args([]).no_pager is False


MC_SCRIPT = Path(__file__).resolve().parent / "model_compare.py"


def test_version_flag():
    proc = subprocess.run(
        [sys.executable, str(MC_SCRIPT), "--version"],
        capture_output=True,
        text=True,
        check=True,
    )
    # Literal pin: a release that forgets to bump VERSION fails here.
    assert proc.stdout.strip() == "model-compare 0.7.4"


# ---------------------------------------------------------------------------
# fetch_json retry policy
# ---------------------------------------------------------------------------


def _counting_http_get(monkeypatch, exc):
    attempts = []

    def fake_http_get(url, headers=None, timeout=30):
        attempts.append(url)
        raise exc

    monkeypatch.setattr(mc, "http_get", fake_http_get)
    monkeypatch.setattr(mc.time, "sleep", lambda s: None)
    return attempts


def _http_error(code):
    import urllib.error

    return urllib.error.HTTPError("https://example.test/", code, "x", {}, None)


def test_fetch_json_does_not_retry_4xx(monkeypatch):
    import urllib.error

    attempts = _counting_http_get(monkeypatch, _http_error(401))
    with pytest.raises(urllib.error.HTTPError):
        mc.fetch_json("https://example.test/")
    assert len(attempts) == 1


def test_fetch_json_retries_5xx_once(monkeypatch):
    import urllib.error

    attempts = _counting_http_get(monkeypatch, _http_error(503))
    with pytest.raises(urllib.error.HTTPError):
        mc.fetch_json("https://example.test/")
    assert len(attempts) == 2


def test_fetch_json_retries_urlerror_once(monkeypatch):
    import urllib.error

    attempts = _counting_http_get(monkeypatch, urllib.error.URLError("down"))
    with pytest.raises(urllib.error.URLError):
        mc.fetch_json("https://example.test/")
    assert len(attempts) == 2


# ---------------------------------------------------------------------------
# AA cache key versioning
# ---------------------------------------------------------------------------


def test_fetch_aa_entries_ignores_legacy_cache(monkeypatch, tmp_path):
    # Entries cached under the pre-v3 keys came from earlier endpoint
    # generations and must never be read: their payloads lack the per-entry
    # as_of stamp and the envelope's intelligence_index_version.
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.delenv("AA_API_KEY", raising=False)
    legacy = tmp_path / "model-compare" / "aa-intelligence-v2.json"
    legacy.parent.mkdir(parents=True)
    legacy.write_text(
        json.dumps(
            {
                "fetched_at": time.time(),
                "payload": {
                    "entries": [{"key": "legacy", "name": "L", "index": 1.0}],
                    "source": "AA API v2",
                },
            }
        )
    )
    entries, source, cached, version = mc.fetch_aa_entries(
        make_args(no_cache=False, cache_ttl=3600)
    )
    assert (entries, source, cached, version) == ([], None, False, None)


def test_fetch_aa_entries_writes_v4_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.delenv("AA_API_KEY", raising=False)
    monkeypatch.setattr(mc, "fetch_json", lambda *a, **k: aa_page([aa_item()]))
    mc.fetch_aa_entries(
        make_args(no_cache=False, cache_ttl=3600, aa_api_key="dummy-key")
    )
    cache_dir = tmp_path / "model-compare"
    assert (cache_dir / "aa-intelligence-v4.json").exists()
    assert not (cache_dir / "aa-intelligence-v3.json").exists()


def _dir_snapshot(path):
    if not path.is_dir():
        return None
    return {p.name: (p.stat().st_mtime_ns, p.stat().st_size) for p in path.iterdir()}


def test_cache_writes_land_in_isolated_dir_not_real_cache(monkeypatch, tmp_path):
    # No explicit XDG_CACHE_HOME here: the autouse isolation fixture alone
    # must redirect the save that fetch_aa_entries makes on success.
    real = Path(os.path.expanduser("~")) / ".cache" / "model-compare"
    before = _dir_snapshot(real)
    path = Path(mc.cache_path(mc.AA_ENTRIES_CACHE))
    # checked before any write, so a missing fixture fails without polluting
    assert path.is_relative_to(tmp_path), path
    assert not path.is_relative_to(real), path
    monkeypatch.delenv("AA_API_KEY", raising=False)
    monkeypatch.setattr(mc, "fetch_json", lambda *a, **k: aa_page([aa_item()]))
    mc.fetch_aa_entries(make_args(aa_api_key="dummy-key"))
    assert path.exists()
    assert _dir_snapshot(real) == before


# ---------------------------------------------------------------------------
# redirect handling: the AA key never leaves its origin
# ---------------------------------------------------------------------------


def _redirect(old_url, new_url):
    import urllib.request

    req = urllib.request.Request(
        old_url,
        headers={"x-api-key": "secret", "Authorization": "Bearer session-key"},
    )
    handler = mc.StripApiKeyRedirectHandler()
    return handler.redirect_request(req, None, 302, "Found", {}, new_url)


def test_redirect_same_host_keeps_api_key():
    new = _redirect("https://api.example.test/a", "https://api.example.test/b")
    assert new.get_header("X-api-key") == "secret"


def test_redirect_same_host_keeps_bearer_token():
    new = _redirect("https://api.example.test/a", "https://api.example.test/b")
    assert new.get_header("Authorization") == "Bearer session-key"


def test_redirect_cross_host_strips_api_key():
    new = _redirect("https://api.example.test/a", "https://evil.example.test/b")
    assert new is not None
    assert not new.has_header("X-api-key")
    assert new.full_url == "https://evil.example.test/b"


def test_redirect_cross_host_strips_bearer_token():
    # fetch_session_cost authenticates with "Authorization: Bearer <key>",
    # and urllib copies headers across hosts on redirect like any other.
    new = _redirect("https://api.example.test/a", "https://evil.example.test/b")
    assert new is not None
    assert not new.has_header("Authorization")
    assert new.full_url == "https://evil.example.test/b"


def test_redirect_scheme_downgrade_strips_api_key():
    new = _redirect("https://api.example.test/a", "http://api.example.test/b")
    assert not new.has_header("X-api-key")


def test_redirect_scheme_downgrade_strips_bearer_token():
    new = _redirect("https://api.example.test/a", "http://api.example.test/b")
    assert not new.has_header("Authorization")


def test_redirect_explicit_default_port_keeps_credentials():
    # An explicit default port is the scheme's default, not a different
    # origin: https://host/a -> https://host:443/b keeps credentials.
    new = _redirect("https://api.example.test/a", "https://api.example.test:443/b")
    assert new is not None
    assert new.get_header("X-api-key") == "secret"
    assert new.get_header("Authorization") == "Bearer session-key"
    # The http/80 equivalent.
    new = _redirect("http://api.example.test/a", "http://api.example.test:80/b")
    assert new is not None
    assert new.get_header("X-api-key") == "secret"
    assert new.get_header("Authorization") == "Bearer session-key"


def test_redirect_default_port_to_explicit_keeps_credentials():
    # The reverse direction: https://host:443/a -> https://host/b is the
    # same origin too.
    new = _redirect("https://api.example.test:443/a", "https://api.example.test/b")
    assert new is not None
    assert new.get_header("X-api-key") == "secret"
    assert new.get_header("Authorization") == "Bearer session-key"


def test_redirect_nondefault_port_strips_credentials():
    # An explicit non-default port is a different origin: fail closed.
    new = _redirect("https://api.example.test/a", "https://api.example.test:8443/b")
    assert new is not None
    assert not new.has_header("X-api-key")
    assert not new.has_header("Authorization")


def test_http_get_uses_redirect_stripping_opener(monkeypatch):
    seen = {}

    class FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b"{}"

    class FakeOpener:
        def open(self, req, timeout=None):
            seen["req"] = req
            seen["timeout"] = timeout
            return FakeResp()

    def fake_build_opener(*handlers):
        seen["handlers"] = handlers
        return FakeOpener()

    monkeypatch.setattr(mc.urllib.request, "build_opener", fake_build_opener)
    assert mc.http_get("https://x.test/", {"x-api-key": "k"}, timeout=7) == b"{}"
    assert any(
        isinstance(h, mc.StripApiKeyRedirectHandler)
        or h is mc.StripApiKeyRedirectHandler
        for h in seen["handlers"]
    )
    assert seen["timeout"] == 7
    assert seen["req"].get_header("X-api-key") == "k"


# ---------------------------------------------------------------------------
# ZDR endpoint status: only status == 0 counts (fail closed)
# ---------------------------------------------------------------------------


def _zdr_ids_for(monkeypatch, tmp_path, entries):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    stub_frontend(monkeypatch, [], frontend_payload(), {"data": entries})
    return mc.fetch_openrouter_frontend(make_args(no_cache=True))[1]


def _with_status(model_id, status):
    entry = zdr_endpoint(model_id)
    if status is _MISSING:
        del entry["status"]
    else:
        entry["status"] = status
    return entry


_MISSING = object()


@pytest.mark.parametrize("status", [-2, -5, _MISSING, False, None, "0", 0.0])
def test_fetch_openrouter_frontend_zdr_skips_non_zero_status(
    monkeypatch, tmp_path, status
):
    ids = _zdr_ids_for(
        monkeypatch,
        tmp_path,
        [zdr_endpoint("acme/ok"), _with_status("acme/down", status)],
    )
    assert ids == {"acme/ok"}


def test_fetch_openrouter_frontend_zdr_counts_status_zero(monkeypatch, tmp_path):
    ids = _zdr_ids_for(monkeypatch, tmp_path, [_with_status("acme/ok", 0)])
    assert ids == {"acme/ok"}


def test_fetch_openrouter_frontend_zdr_one_live_endpoint_suffices(
    monkeypatch, tmp_path
):
    ids = _zdr_ids_for(
        monkeypatch,
        tmp_path,
        [_with_status("acme/a", 0), _with_status("acme/a", -5)],
    )
    assert ids == {"acme/a"}


# ---------------------------------------------------------------------------
# sources.aa: the run's quality chain, unmasked by match outcomes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "aa_source, expected_mode",
    [("AA API v2", "aa"), (None, "none")],
)
def test_catalog_mode_tracks_quality_chain(aa_source, expected_mode):
    # mode is the chain, not a match outcome: a keyed run stays "aa" even
    # when every match came from OpenRouter's benchmarks, and an unkeyed
    # run without catalog meta stays "none" for the same reason.
    doc = build_doc(aa_source=aa_source)
    assert doc["sources"]["aa"]["mode"] == expected_mode
    # ...with or without OpenRouter coverage.
    doc = build_doc(aa_source=aa_source, aa_by_id={})
    assert doc["sources"]["aa"]["mode"] == expected_mode


def test_catalog_mode_not_masked_by_openrouter_matches():
    # Regression (the v3 mode was a match outcome): OpenRouter matches
    # never upgrade the chain -- an unkeyed run without catalog meta stays
    # "none" no matter how many models OpenRouter's benchmarks cover.
    doc = build_doc(aa_source=None)
    assert doc["sources"]["aa"]["matched_openrouter"] > 0
    assert doc["sources"]["aa"]["mode"] == "none"
    # The site validator must use the producer's catalog contract; the
    # unconditional lockstep test pins that requirement.
    if bsd.CATALOG_SCHEMA_VERSION == mc.CATALOG_SCHEMA_VERSION:
        bsd.validate_catalog(doc)


# ---------------------------------------------------------------------------
# OpenRouter session-cost dataset (per-task cost, outside the score)
# ---------------------------------------------------------------------------


def _cost_cell(permaslug, turn_range, median):
    """One session-cost dataset cell (brainstorm section 2.6 shape)."""
    return {
        "app_name": "Hermes Agent",
        "app_slug": "hermes-agent",
        "median_session_cost_usd": median,
        "model_permaslug": permaslug,
        "turn_range": turn_range,
    }


def _cost_payload(cells, as_of="2026-10-15T00:00:00Z"):
    return {
        "data": cells,
        "meta": {
            "as_of": as_of,
            "version": "1",
            "window_days": 30,
            "window_end_date": "2026-10-12",
        },
    }


def test_session_cost_pairs_permaslugs_via_models(monkeypatch):
    """The #22 shape: the base listing and its dated snapshot are distinct
    catalog slugs, and the dataset publishes a cell for each permaslug.
    Pairing through the catalog's own slug/permaslug map gives each slug
    its own values -- under the old blind strip with latest-date-wins the
    base listing borrowed the snapshot's cell and the snapshot got none."""
    payload = _cost_payload(
        [
            _cost_cell("deepseek/deepseek-v4.1-flash-20260910", "10-49-turns", 0.50),
            _cost_cell("deepseek/deepseek-v4.1-flash-20261017", "10-49-turns", 0.25),
        ]
    )
    # models-API shape: id is the exact catalog slug, canonical_slug the
    # dated permaslug (the snapshot listing's id carries its own -1017).
    models = [
        make_model(
            id="deepseek/deepseek-v4.1-flash",
            canonical_slug="deepseek/deepseek-v4.1-flash-20260910",
        ),
        make_model(
            id="deepseek/deepseek-v4.1-flash-1017",
            canonical_slug="deepseek/deepseek-v4.1-flash-20261017",
        ),
    ]
    monkeypatch.setattr(mc, "fetch_json", lambda *a, **k: payload)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or")
    result = mc.fetch_session_cost(make_args(openrouter_api_key=None), models=models)
    assert result is not None  # no usable cells means the pairing never engaged
    assert result["medians"] == {
        "deepseek/deepseek-v4.1-flash": {"10-49-turns": 0.50},
        "deepseek/deepseek-v4.1-flash-1017": {"10-49-turns": 0.25},
    }
    # The strip helper: dated -> bare, undated unchanged.
    assert (
        mc.strip_permaslug_date("z-ai/glm-5.3-flash-20260826") == "z-ai/glm-5.3-flash"
    )
    assert mc.strip_permaslug_date("z-ai/glm-5.3-flash") == "z-ai/glm-5.3-flash"


def test_run_catalog_session_cost_pairs_models_api_shape(monkeypatch, capsys):
    """End to end: run() hands the raw models-API listing to
    fetch_session_cost; the pairing resolves dated permaslugs to exact
    slugs and the catalog block carries each slug's own medians."""
    models = [
        make_model(
            id="deepseek/deepseek-v4.1-flash",
            name="DeepSeek: DeepSeek V4.1 Flash",
            canonical_slug="deepseek/deepseek-v4.1-flash-20260910",
        ),
        make_model(
            id="deepseek/deepseek-v4.1-flash-1017",
            name="DeepSeek: DeepSeek V4.1 Flash (1017)",
            canonical_slug="deepseek/deepseek-v4.1-flash-20261017",
        ),
    ]
    _stub_endpoint_pipeline(
        monkeypatch, models, {m["id"]: _endpoint_stage_eps() for m in models}
    )
    payload = _cost_payload(
        [
            _cost_cell("deepseek/deepseek-v4.1-flash-20260910", "10-49-turns", 0.50),
            _cost_cell("deepseek/deepseek-v4.1-flash-20261017", "10-49-turns", 0.25),
        ]
    )
    monkeypatch.setattr(mc, "fetch_json", lambda *a, **k: payload)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or")
    args = make_args(min_context=0, catalog=True, openrouter_api_key="sk-or")
    assert mc.run(args, models, False) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["sources"]["session_cost"] == "ok"
    assert doc["session_cost"]["medians"] == {
        "deepseek/deepseek-v4.1-flash": {"10-49-turns": 0.50},
        "deepseek/deepseek-v4.1-flash-1017": {"10-49-turns": 0.25},
    }


def test_session_cost_merges_disjoint_turn_ranges_across_permaslugs(monkeypatch):
    """One slug, several dated permaslugs with disjoint turn-range coverage:
    the merge unions the range maps -- each range resolving to the latest
    date's cell (first wins on an equal date) -- instead of pass 1
    replacing the whole payload with the latest listing, which silently
    dropped the older permaslug's ranges."""
    payload = _cost_payload(
        [
            _cost_cell("acme/m-20260826", "1-turn", 0.10),
            _cost_cell("acme/m-20260826", "10-49-turns", 0.30),
            _cost_cell("acme/m-20260901", "10-49-turns", 0.50),
        ]
    )
    # The same id listed twice with different canonical slugs (variant
    # listings repeat an identical id): both permaslugs pair to one slug.
    models = [
        make_model(id="acme/m", canonical_slug="acme/m-20260826"),
        make_model(id="acme/m", canonical_slug="acme/m-20260901"),
    ]
    monkeypatch.setattr(mc, "fetch_json", lambda *a, **k: payload)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or")
    result = mc.fetch_session_cost(make_args(openrouter_api_key=None), models=models)
    assert result is not None
    assert result["medians"] == {
        "acme/m": {"1-turn": 0.10, "10-49-turns": 0.50},
    }


def test_session_cost_ambiguous_strip_dropped(monkeypatch, capsys):
    """The #22 shape with no catalog pairing to disambiguate: two dated
    permaslugs strip to one id, so neither may claim it (no borrowing) --
    the medians degrade to None rather than a merged block."""
    payload = _cost_payload(
        [
            _cost_cell("deepseek/deepseek-v4.1-flash-20261017", "10-49-turns", 0.25),
            _cost_cell("deepseek/deepseek-v4.1-flash-20260910", "10-49-turns", 0.50),
        ]
    )
    monkeypatch.setattr(mc, "fetch_json", lambda *a, **k: payload)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or")
    assert mc.fetch_session_cost(make_args(openrouter_api_key=None)) is None
    assert "no usable cells" in capsys.readouterr().err


def test_session_cost_undated_and_unambiguous_strip(monkeypatch):
    """An undated key self-maps; a dated key the catalog does not pair
    still strips when the stripped id is unclaimed and unambiguous
    (mirrors build_aa_benchmarks passes 2 and 3)."""
    payload = _cost_payload(
        [
            _cost_cell("acme/model-a", "1-turn", 0.10),
            _cost_cell("acme/legacy-20250101", "10-49-turns", 0.20),
        ]
    )
    monkeypatch.setattr(mc, "fetch_json", lambda *a, **k: payload)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or")
    result = mc.fetch_session_cost(make_args(openrouter_api_key=None), models=[])
    assert result["medians"] == {
        "acme/model-a": {"1-turn": 0.10},
        "acme/legacy": {"10-49-turns": 0.20},
    }


def test_session_cost_cache_key_is_v3_and_never_reads_v2(monkeypatch, tmp_path):
    """The cache key is session-cost-v3: a payload cached under v2 was
    written by the flawed pre-fix build and must never be served; fresh
    results land under v3. (v2 was already the bump from v1; v3 is the
    merge-strategy fix's own bump, unreleased as of 0.7.3.)"""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or")
    args = make_args(openrouter_api_key=None, no_cache=False, cache_ttl=10_000)
    # A stale v2 payload: if load_cache("session-cost-v2") ever runs, this
    # wrong-shaped envelope would be served instead of the fetch.
    v2_path = Path(mc.cache_path("session-cost-v2"))
    v2_path.parent.mkdir(parents=True, exist_ok=True)
    v2_path.write_text(
        json.dumps(
            {
                "fetched_at": time.time(),
                "payload": {
                    "attribution": "stale",
                    "medians": {"deepseek/deepseek-v4.1-flash": {"10-49-turns": 99.0}},
                },
            }
        )
    )
    payload = _cost_payload([_cost_cell("acme/model-a", "1-turn", 0.10)])
    calls = []
    monkeypatch.setattr(mc, "fetch_json", lambda *a, **k: calls.append(k) or payload)
    result = mc.fetch_session_cost(args)
    # The fetch ran (the v2 payload was not served) and the result is fresh.
    assert len(calls) == 1
    assert result["medians"] == {"acme/model-a": {"1-turn": 0.10}}
    # The fresh result is cached under v3 and served from there next time.
    v3_path = Path(mc.cache_path("session-cost-v3"))
    assert v3_path.exists()
    cached = mc.fetch_session_cost(args)
    assert len(calls) == 1  # no second fetch: the v3 payload was served
    assert cached["medians"] == {"acme/model-a": {"1-turn": 0.10}}


def test_session_cost_keyless_returns_none(monkeypatch):
    """No flag and no env: None, and the network is never touched."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    def no_network(*a, **k):
        raise AssertionError("fetch_json must not be called without a key")

    monkeypatch.setattr(mc, "fetch_json", no_network)
    assert mc.fetch_session_cost(make_args(openrouter_api_key=None)) is None


def test_session_cost_fetch_failure_returns_none(monkeypatch, capsys):
    """A fetch failing after fetch_json's retries warns once and yields None."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or")

    def boom(*a, **k):
        raise RuntimeError("network down")

    monkeypatch.setattr(mc, "fetch_json", boom)
    assert mc.fetch_session_cost(make_args(openrouter_api_key=None)) is None
    assert "session-cost dataset request failed" in capsys.readouterr().err
    # Same None contract for a response with an unexpected shape: a
    # malformed payload must never crash the run.
    monkeypatch.setattr(mc, "fetch_json", lambda *a, **k: ["not", "a", "dict"])
    assert mc.fetch_session_cost(make_args(openrouter_api_key=None)) is None
    assert "unexpected response shape" in capsys.readouterr().err


def test_session_cost_pagination_stops_on_short_page(monkeypatch):
    """Pagination requests pages until one is shorter than the limit."""
    calls = []

    def fake_fetch_json(url, headers=None, timeout=30, retries=1):
        calls.append(url)
        return _cost_payload([_cost_cell("acme/model-a", "1-turn", 0.10)])

    monkeypatch.setattr(mc, "fetch_json", fake_fetch_json)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or")
    result = mc.fetch_session_cost(make_args(openrouter_api_key=None))
    # A short first page ends pagination: exactly one request.
    assert len(calls) == 1
    assert calls[0].endswith("offset=0")
    assert result["medians"] == {"acme/model-a": {"1-turn": 0.10}}
    # A full 500-cell page keeps going; the next short page stops it.
    calls.clear()
    full_page = _cost_payload(
        [_cost_cell(f"acme/model-{i}", "1-turn", 0.10) for i in range(500)]
    )

    def paged_fetch_json(url, headers=None, timeout=30, retries=1):
        calls.append(url)
        return full_page if url.endswith("offset=0") else _cost_payload([])

    monkeypatch.setattr(mc, "fetch_json", paged_fetch_json)
    result = mc.fetch_session_cost(make_args(openrouter_api_key=None))
    assert len(calls) == 2
    assert calls[0].endswith("offset=0") and calls[1].endswith("offset=500")
    assert result["medians"] == {
        f"acme/model-{i}": {"1-turn": 0.10} for i in range(500)
    }


def test_session_cost_shape_and_attribution(monkeypatch):
    """Envelope: attribution + harness + meta; every turn range survives."""
    payload = _cost_payload(
        [
            _cost_cell("acme/model-a", "1-turn", 0.05),
            _cost_cell("acme/model-a", "2-9-turns", 0.10),
            _cost_cell("acme/model-a", "10-49-turns", 1.74),
            _cost_cell("acme/model-a", "50-plus-turns", 9.99),
        ]
    )
    seen = {}

    def fake_fetch_json(url, headers=None, timeout=30, retries=1):
        seen["url"] = url
        seen["headers"] = headers
        return payload

    monkeypatch.setattr(mc, "fetch_json", fake_fetch_json)
    # The flag wins over the env for the key (same resolution as AA).
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-env")
    result = mc.fetch_session_cost(make_args(openrouter_api_key="sk-flag"))
    assert result == {
        "attribution": "Data © OpenRouter, CC BY 4.0",
        "app_slug": "hermes-agent",
        "as_of": "2026-10-15T00:00:00Z",
        "window_days": 30,
        "window_end_date": "2026-10-12",
        "medians": {
            "acme/model-a": {
                "1-turn": 0.05,
                "2-9-turns": 0.10,
                "10-49-turns": 1.74,
                "50-plus-turns": 9.99,
            }
        },
    }
    assert seen["headers"] == {"Authorization": "Bearer sk-flag"}
    assert seen["url"] == (
        f"{mc.SESSION_COST_URL}?app_slug=hermes-agent&limit=500&offset=0"
    )
    assert mc.SESSION_COST_DISPLAY_TURN_RANGE == "10-49-turns"


def test_session_cost_meta_malformed_returns_none(monkeypatch, capsys):
    """Dataset meta drift degrades to None -- never a block the catalog
    validator would reject (design section 4.8: session cost must never
    fail the publish)."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or")
    base = _cost_payload([_cost_cell("acme/model-a", "1-turn", 0.10)])
    for bad_meta in (
        {**base["meta"], "as_of": None},
        {**base["meta"], "window_end_date": ""},
        {**base["meta"], "window_days": "30"},
        {**base["meta"], "window_days": 30.0},
        {**base["meta"], "window_days": 0},
        {"as_of": "2026-10-15T00:00:00Z"},  # window fields missing entirely
    ):
        payload = {**base, "meta": bad_meta}
        monkeypatch.setattr(mc, "fetch_json", lambda *a, **k: payload)
        assert mc.fetch_session_cost(make_args(openrouter_api_key=None)) is None
        assert "malformed meta" in capsys.readouterr().err


def test_session_cost_drops_unknown_turn_ranges(monkeypatch):
    """A dataset-side new turn range is dropped, never published -- the
    catalog contract's range set is closed (the validator would otherwise
    reject the whole document over a session-cost cell)."""
    payload = _cost_payload(
        [
            _cost_cell("acme/model-a", "10-49-turns", 0.25),
            _cost_cell("acme/model-a", "100-plus-turns", 9.99),
        ]
    )
    monkeypatch.setattr(mc, "fetch_json", lambda *a, **k: payload)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or")
    result = mc.fetch_session_cost(make_args(openrouter_api_key=None))
    assert result["medians"] == {"acme/model-a": {"10-49-turns": 0.25}}
    # Everything kept is inside the pinned set the validator accepts.
    assert set(mc.SESSION_COST_TURN_RANGES) == {
        "1-turn",
        "2-9-turns",
        "10-49-turns",
        "50-plus-turns",
    }


# ---------------------------------------------------------------------------
# Provider-aware eval: catalog v6 (schema bump + additive blocks)
# ---------------------------------------------------------------------------


def _offline_catalog_doc(
    monkeypatch, capsys, models, endpoints_by_id, *, api_key=None, session_cost=None
):
    """Run --catalog through the stubbed offline pipeline; return the document.

    Same pattern as _stub_endpoint_pipeline (Task 6): every fetcher is
    stubbed, so the run touches no network. The session-cost fetch is
    stubbed only when a key is configured -- keyless runs take
    fetch_session_cost's own fast no-key path (None, no network), which is
    exactly the guard the --best-cheap pin relies on.
    """
    _stub_endpoint_pipeline(monkeypatch, models, endpoints_by_id)
    if api_key is None:
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    else:
        monkeypatch.setattr(
            mc, "fetch_session_cost", lambda args, models=None: session_cost
        )
    args = make_args(min_context=0, catalog=True, openrouter_api_key=api_key)
    assert mc.run(args, models, False) == 0
    return json.loads(capsys.readouterr().out)


def test_catalog_schema_version_is_6(monkeypatch, capsys):
    models = [make_model(id="z-ai/glm", name="Z.AI: GLM")]
    doc = _offline_catalog_doc(
        monkeypatch, capsys, models, {"z-ai/glm": _endpoint_stage_eps()}
    )
    assert doc["schema_version"] == 6
    # run()'s unkeyed wiring, end to end: no AA key (no index version) and
    # the published catalog unavailable (no hydration meta) -- the chain is
    # "none" and every bucket is empty.
    assert doc["sources"]["aa"] == {
        "mode": "none",
        "matched_aa": 0,
        "matched_catalog": 0,
        "matched_openrouter": 0,
        "depth_rows": 0,
        "matched_depth": 0,
        "matched_default": 0,
        "catalog_generated_at": None,
        "intelligence_index_version": None,
    }


def test_catalog_entry_carries_endpoint_blocks(monkeypatch, capsys):
    eps = _endpoint_stage_eps()
    models = [make_model(id="z-ai/glm", name="Z.AI: GLM")]
    doc = _offline_catalog_doc(monkeypatch, capsys, models, {"z-ai/glm": eps})
    (entry,) = doc["models"]
    assert entry["id"] == "z-ai/glm"
    # The parsed endpoint list, verbatim from the stage.
    assert entry["endpoints"] == eps
    assert entry["default_precision"] == "fp8"
    # The annotations dict (design section 4.6), same values the --json row
    # surfaces (Task 6 pinned risk_flag/price_spread for this fixture).
    assert set(entry["annotations"]) == {
        "price_spread",
        "cheapest_quantized",
        "risk_flag",
        "baseline_context_below_catalog",
    }
    assert entry["annotations"]["risk_flag"] is True
    assert entry["annotations"]["price_spread"] == pytest.approx(0.5)
    # Fourth block of the brief's pinned quartet; its exact shape is test
    # test_catalog_pricing_block_is_baseline_endpoint's job.
    assert entry["advertised_pricing"]["source"] == "models-api"
    # The endpoint stage always ran before build_catalog is reached, so the
    # source state is deterministic "ok" (failures surface as pool.dropped
    # counts, not source state).
    assert doc["sources"]["endpoints"] == "ok"


def test_catalog_entry_carries_steering_block(monkeypatch, capsys):
    eps = _endpoint_stage_eps()
    models = [make_model(id="z-ai/glm", name="Z.AI: GLM")]
    doc = _offline_catalog_doc(monkeypatch, capsys, models, {"z-ai/glm": eps})
    (entry,) = doc["models"]
    # The build_steering output shape (design section 4.7), the same dict
    # the --json row carries.
    assert entry["steering"] == {
        "model": "z-ai/glm",
        "provider": {
            "order": ["z-ai/fp8"],
            "ignore": ["deepinfra/fp4"],
            "quantizations": ["fp8"],
            "allow_fallbacks": True,
        },
    }
    # Pre-endpoint-shaped fixtures (candidates that never went through the
    # endpoint stage) degrade to null -- the same .get() pattern as the
    # other additive entry keys. _scored_candidates() yields two, both
    # pre-endpoint.
    args = make_args(min_context=0, catalog=True)
    doc = mc.build_catalog(
        args, models, _scored_candidates(), {}, [], {}, {}, None, {}, {}
    )
    assert all(entry["steering"] is None for entry in doc["models"])


def test_catalog_session_cost_block_omitted_keyless(monkeypatch, capsys):
    models = [make_model(id="z-ai/glm", name="Z.AI: GLM")]
    doc = _offline_catalog_doc(
        monkeypatch, capsys, models, {"z-ai/glm": _endpoint_stage_eps()}
    )
    # Omitted entirely -- consumers branch on key presence, not null shape.
    assert "session_cost" not in doc
    assert doc["sources"]["session_cost"] == "skipped"


def test_catalog_session_cost_block_present_with_key(monkeypatch, capsys):
    envelope = {
        "attribution": "Data © OpenRouter, CC BY 4.0",
        "app_slug": "hermes-agent",
        "as_of": "2026-10-15T00:00:00Z",
        "window_days": 30,
        "window_end_date": "2026-10-12",
        "medians": {"z-ai/glm": {"10-49-turns": 0.42}},
    }
    models = [make_model(id="z-ai/glm", name="Z.AI: GLM")]
    doc = _offline_catalog_doc(
        monkeypatch,
        capsys,
        models,
        {"z-ai/glm": _endpoint_stage_eps()},
        api_key="sk-or",
        session_cost=envelope,
    )
    # The envelope verbatim, medians duplication included.
    assert doc["session_cost"] == envelope
    assert doc["sources"]["session_cost"] == "ok"


def test_catalog_session_cost_unavailable_with_failed_fetch(monkeypatch, capsys):
    models = [make_model(id="z-ai/glm", name="Z.AI: GLM")]
    doc = _offline_catalog_doc(
        monkeypatch,
        capsys,
        models,
        {"z-ai/glm": _endpoint_stage_eps()},
        api_key="sk-or",
        session_cost=None,  # a configured key whose fetch failed
    )
    assert "session_cost" not in doc
    assert doc["sources"]["session_cost"] == "unavailable"


def test_catalog_pricing_block_is_baseline_endpoint(monkeypatch, capsys):
    models = [make_model(id="z-ai/glm", name="Z.AI: GLM")]
    doc = _offline_catalog_doc(
        monkeypatch, capsys, models, {"z-ai/glm": _endpoint_stage_eps()}
    )
    (entry,) = doc["models"]
    pricing = entry["pricing"]
    # Baseline endpoint (z-ai/fp8) effective prices per-1M -- NOT the
    # models-API scalars the candidate carried before the stage (1.0/2.0).
    assert pricing["input_per_1m"] == pytest.approx(0.2)
    assert pricing["output_per_1m"] == pytest.approx(0.4)
    assert pricing["blended_per_1m"] == pytest.approx(0.75 * 0.2 + 0.25 * 0.4)
    assert pricing["base"]["input_per_1m"] == pytest.approx(0.2)
    assert pricing["base"]["output_per_1m"] == pytest.approx(0.4)
    # The advertised block preserves exactly those models-API scalars.
    # Entry-level key (sibling of pricing), per the brief's pinned shapes.
    advertised = entry["advertised_pricing"]
    assert advertised["source"] == "models-api"
    assert advertised["input_per_1m"] == pytest.approx(1.0)
    assert advertised["output_per_1m"] == pytest.approx(2.0)
    assert advertised["blended_per_1m"] == pytest.approx(1.25)


def test_json_rows_carry_session_cost_usd(monkeypatch, capsys):
    envelope = {
        "attribution": "Data © OpenRouter, CC BY 4.0",
        "app_slug": "hermes-agent",
        "as_of": "2026-10-15T00:00:00Z",
        "window_days": 30,
        "window_end_date": "2026-10-12",
        "medians": {"z-ai/glm": {"10-49-turns": 0.42}},
    }
    models = [
        make_model(id="z-ai/glm", name="Z.AI: GLM"),
        make_model(id="acme/plain", name="Acme: Plain"),
    ]
    endpoints = _endpoint_stage_eps()
    _stub_endpoint_pipeline(
        monkeypatch,
        models,
        {"z-ai/glm": endpoints, "acme/plain": _endpoint_stage_eps()},
    )
    monkeypatch.setattr(mc, "fetch_session_cost", lambda args, models=None: envelope)
    args = make_args(min_context=0, json=True, openrouter_api_key="sk-or")
    assert mc.run(args, models, False) == 0
    rows = {row["model"]: row for row in json.loads(capsys.readouterr().out)}
    # The display turn range cell, via the pinned constant's value.
    assert rows["z-ai/glm"]["session_cost_usd"] == pytest.approx(0.42)
    # A model the dataset does not cover degrades to None.
    assert rows["acme/plain"]["session_cost_usd"] is None


def test_best_run_skips_session_cost_fetch(monkeypatch, capsys):
    # The session-cost medians surface only in --catalog/--json output, so
    # a keyed --best run must not pay the paginated fetch (Task 9's run()
    # gate). The stub raises: any fetch attempt fails the test loudly.
    models = [make_model(id="z-ai/glm", name="Z.AI: GLM")]
    _stub_endpoint_pipeline(monkeypatch, models, {"z-ai/glm": _endpoint_stage_eps()})

    def boom(args):
        raise AssertionError("session-cost fetch must be skipped for --best")

    monkeypatch.setattr(mc, "fetch_session_cost", boom)
    args = make_args(min_context=0, best=True, openrouter_api_key="sk-or")
    assert mc.run(args, models, False) == 0
    assert capsys.readouterr().out.strip() == "openrouter/z-ai/glm"


# ---------------------------------------------------------------------------
# Task 4: run() wiring -- mode-scoped chains + footer rework
# ---------------------------------------------------------------------------


def _stub_quality_run(
    monkeypatch,
    models,
    *,
    aa_entries=(),
    aa_version=None,
    catalog=None,
    or_bench=None,
    aa_raises=False,
    catalog_raises=False,
    frontend_models=(),
):
    """Stub every fetcher so mc.run() exercises the quality chain offline.

    The three quality fetchers are controlled independently so tests can pin
    the mode-scoped wiring: unkeyed runs must never touch the AA API
    (aa_raises), keyed runs must never touch the published catalog
    (catalog_raises). The endpoint stage returns one eligible endpoint per
    candidate so no model drops fail-closed.
    """
    monkeypatch.delenv("AA_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    zdr = {m["id"] for m in models}
    monkeypatch.setattr(
        mc,
        "fetch_openrouter_frontend",
        lambda a: ({}, zdr, or_bench or {}, set(), list(frontend_models)),
    )
    if aa_raises:

        def fake_aa(args):
            raise AssertionError("fetch_aa_entries must not be called")

    else:

        def fake_aa(args):
            return (
                list(aa_entries),
                "AA API v2" if aa_entries else None,
                False,
                aa_version,
            )

    monkeypatch.setattr(mc, "fetch_aa_entries", fake_aa)
    if catalog_raises:

        def fake_catalog(args):
            raise AssertionError("fetch_quality_catalog must not be called")

    else:

        def fake_catalog(args):
            return catalog

    monkeypatch.setattr(mc, "fetch_quality_catalog", fake_catalog)
    monkeypatch.setattr(mc, "fetch_model_endpoints", lambda mid, a: [_endpoint()])


def _non_reasoning_frontend(models):
    """Frontend reasoning rows marking models known non-reasoning."""
    return [
        {"slug": model["id"], "supports_reasoning": False, "reasoning_config": {}}
        for model in models
    ]


def test_run_unkeyed_uses_catalog_not_openrouter_for_covered_models(
    monkeypatch, capsys
):
    # Unkeyed: the published catalog's depth record wins over the OR trio
    # for the model it covers. Tier 2 labels the record with the source
    # chain ("catalog" -- read from our published catalog) while as_of
    # stays the row's own stamp; the row's "aa" label is the keyed CI
    # run's row-level truth inside the artifact, never the run's.
    models = [
        make_model(
            id="deepseek/deepseek-chat",
            name="DeepSeek: Chat",
            reasoning={"supported_efforts": ["high"], "default_effort": "high"},
        )
    ]
    stamp = "2026-10-23T13:15:00Z"
    catalog = {
        "by_id": {
            "deepseek/deepseek-chat": {
                "index": 42.0,
                "source": "aa",
                "as_of": stamp,
                "effort": "high",
            }
        },
        "depths_by_id": {
            "deepseek/deepseek-chat": {
                "default_effort": "high",
                "depths": {
                    "high": {
                        "effort": "high",
                        "index": 42.0,
                        "cost_per_task_usd": None,
                        "source": "aa",
                        "as_of": stamp,
                    }
                },
            }
        },
        "generated_at": stamp,
    }
    or_bench = {
        "deepseek/deepseek-chat": {"intelligence_index": 39.5, "as_of": "20261017"}
    }
    _stub_quality_run(
        monkeypatch, models, catalog=catalog, or_bench=or_bench, aa_raises=True
    )
    args = make_args(json=True)
    assert mc.run(args, models, False) == 0
    (row,) = json.loads(capsys.readouterr().out)
    assert row["quality_source"] == "catalog"
    assert row["quality_index"] == 42.0
    assert row["quality_as_of"] == "2026-10-23T13:15:00Z"


def test_run_unkeyed_uses_openrouter_only_for_models_catalog_lacks(monkeypatch, capsys):
    # Unkeyed: the catalog covers one model; the model it lacks falls through
    # to the OR trio -- the last-resort tier.
    models = [
        make_model(
            id="acme/covered",
            name="Acme: Covered",
            reasoning={"supported_efforts": ["high"], "default_effort": "high"},
        ),
        make_model(id="acme/uncovered", name="Acme: Uncovered"),
    ]
    stamp = "2026-10-23T13:15:00Z"
    catalog = {
        "by_id": {
            "acme/covered": {
                "index": 42.0,
                "source": "aa",
                "as_of": stamp,
                "effort": "high",
            }
        },
        "depths_by_id": {
            "acme/covered": {
                "default_effort": "high",
                "depths": {
                    "high": {
                        "effort": "high",
                        "index": 42.0,
                        "cost_per_task_usd": None,
                        "source": "aa",
                        "as_of": stamp,
                    }
                },
            }
        },
        "generated_at": stamp,
    }
    or_bench = {
        "acme/covered": {"intelligence_index": 41.0, "as_of": "20261017"},
        "acme/uncovered": {"intelligence_index": 30.0, "as_of": "20261010"},
    }
    _stub_quality_run(
        monkeypatch,
        models,
        catalog=catalog,
        or_bench=or_bench,
        aa_raises=True,
        frontend_models=_non_reasoning_frontend([models[1]]),
    )
    args = make_args(json=True)
    assert mc.run(args, models, False) == 0
    rows = {row["model"]: row for row in json.loads(capsys.readouterr().out)}
    assert rows["acme/covered"]["quality_source"] == "catalog"
    assert rows["acme/uncovered"]["quality_source"] == "openrouter"
    assert rows["acme/uncovered"]["quality_index"] == 30.0


def test_run_keyed_aa_down_falls_to_openrouter_not_catalog(monkeypatch, capsys):
    # Keyed: an AA failure must degrade to the OR trio, and the catalog
    # fetcher must NEVER run -- a keyed CI/site build hydrating from the
    # published catalog would consume its own output (self-reference loop).
    models = [make_model(id="acme/model-a", name="Acme: A")]
    or_bench = {"acme/model-a": {"intelligence_index": 35.0, "as_of": "20261017"}}
    _stub_quality_run(
        monkeypatch,
        models,
        or_bench=or_bench,
        catalog_raises=True,
        frontend_models=_non_reasoning_frontend(models),
    )
    args = make_args(aa_api_key="dummy-key", json=True)
    assert mc.run(args, models, False) == 0
    (row,) = json.loads(capsys.readouterr().out)
    assert row["quality_source"] == "openrouter"


def test_run_unkeyed_catalog_down_falls_to_openrouter(monkeypatch, capsys):
    # Unkeyed: a transport failure (None) and a fetched-but-unusable artifact
    # (empty by_id) both degrade to the OR trio -- the footer treats them
    # identically as "catalog unavailable".
    models = [make_model(id="acme/model-a", name="Acme: A")]
    or_bench = {"acme/model-a": {"intelligence_index": 35.0, "as_of": "20261017"}}
    args = make_args(json=True)
    for catalog in (
        None,
        {"by_id": {}, "depths_by_id": {}, "generated_at": None},
    ):
        _stub_quality_run(
            monkeypatch,
            models,
            catalog=catalog,
            or_bench=or_bench,
            aa_raises=True,
            frontend_models=_non_reasoning_frontend(models),
        )
        assert mc.run(args, models, False) == 0
        (row,) = json.loads(capsys.readouterr().out)
        assert row["quality_source"] == "openrouter"


def test_footer_as_of_date_only_stamp_any_source():
    # A YYYYMMDD stamp renders as a plain date for ANY source -- catalog
    # rows hydrated from OR rows carry the same date-only stamp and must
    # not fall through parse_iso_datetime to a fabricated midnight time.
    assert mc._footer_as_of("openrouter", "20261017") == "2026-10-17"
    assert mc._footer_as_of("catalog", "20261017") == "2026-10-17"
    assert mc._footer_as_of("aa", "2026-10-23T13:15:07+00:00") == "13:15Z"
    assert mc._footer_as_of("catalog", "2026-10-23T13:15:07+00:00") == (
        "2026-10-23 13:15Z"
    )


def test_run_footer_note_lists_sources_counts_as_of_and_version(monkeypatch, capsys):
    # Keyed: the footer lists the index version, per-source counts and each
    # source's own as_of, plus the drift footnote for the AA-vs-OR gap.
    models = [
        make_model(id="acme/model-a", name="Acme: A"),
        make_model(id="acme/model-b", name="Acme: B"),
        make_model(id="acme/model-c", name="Acme: C"),
        make_model(id="acme/model-d", name="Acme: D"),
    ]
    aa_entries = [
        {
            "key": "acme/model-a",
            "name": "A (Non-reasoning)",
            "slug": "a-non-reasoning",
            "index": 45.1,
            "as_of": "2026-10-23T13:15:00Z",
        },
        {
            "key": "acme/model-b",
            "name": "B (Non-reasoning)",
            "slug": "b-non-reasoning",
            "index": 50.0,
            "as_of": "2026-10-23T13:15:00Z",
        },
    ]
    or_bench = {
        "acme/model-a": {"intelligence_index": 39.5, "as_of": "20261017"},
        "acme/model-b": {"intelligence_index": 49.0, "as_of": "20261017"},
        "acme/model-c": {"intelligence_index": 30.0, "as_of": "20261017"},
    }
    _stub_quality_run(
        monkeypatch,
        models,
        aa_entries=aa_entries,
        aa_version=4.3,
        or_bench=or_bench,
        catalog_raises=True,
        frontend_models=_non_reasoning_frontend([models[0], models[1], models[2]]),
    )
    # Keyed: the AA API version prefix only exists on keyed runs, so the
    # args carry a key (the brief's footer example shows "AA index v4.3").
    args = make_args(aa_api_key="dummy-key")
    assert mc.run(args, models, False) == 0
    out = capsys.readouterr().out
    assert "pool:" not in out
    assert "Quality source: AA index v4.3 - live AA (2 models, checked 13:15Z) |" in out
    assert "                OpenRouter snapshot (1 model, as of 2026-10-17);" in out
    assert "                1 model unmatched (no quality data; quality score 0)" in out
    assert (
        "                drift vs OpenRouter snapshot: 1 comparable (non-reasoning) row"
        in out
    )

    # Unkeyed: the catalog block shows the hydrated row's as_of (the v4
    # per-row stamp; the tier labels the record "catalog" regardless of
    # the row's own label) and the OR block shows the permaslug date; no
    # index version (the AA API is not consulted).
    models = [
        make_model(
            id="acme/covered",
            name="Acme: Covered",
            reasoning={"supported_efforts": ["high"], "default_effort": "high"},
        ),
        make_model(id="acme/or-only", name="Acme: OR only"),
    ]
    stamp = "2026-10-23T13:15:00Z"
    catalog = {
        "by_id": {
            "acme/covered": {
                "index": 42.0,
                "source": "aa",
                "as_of": stamp,
                "effort": "high",
            }
        },
        "depths_by_id": {
            "acme/covered": {
                "default_effort": "high",
                "depths": {
                    "high": {
                        "effort": "high",
                        "index": 42.0,
                        "cost_per_task_usd": None,
                        "source": "aa",
                        "as_of": stamp,
                    }
                },
            }
        },
        "generated_at": stamp,
    }
    or_bench = {
        "acme/or-only": {"intelligence_index": 30.0, "as_of": "20261017"},
    }
    _stub_quality_run(
        monkeypatch,
        models,
        catalog=catalog,
        or_bench=or_bench,
        aa_raises=True,
        frontend_models=_non_reasoning_frontend([models[1]]),
    )
    args = make_args()
    assert mc.run(args, models, False) == 0
    out = capsys.readouterr().out
    assert (
        "Quality source: our published catalog (1 model, as of 2026-10-23 13:15Z) |"
        in out
    )
    assert "                OpenRouter snapshot (1 model, as of 2026-10-17)" in out


def test_run_footer_mixed_catalog_stamp_shapes(monkeypatch, capsys):
    # Unkeyed: the hydrated catalog mixes stamp shapes exactly like the
    # published artifact does -- OR rows carry the YYYYMMDD permaslug date,
    # aa rows the CI run's ISO stamp, and both land in the one "our
    # catalog" block. The footer picks the truly latest stamp per source
    # (parsed instants, never string order: "20261017" sorts above every
    # "2026-..." ISO string yet is not the fresher stamp) and renders a
    # date-only stamp as a plain date -- never a fabricated midnight time.
    models = [
        make_model(id="acme/or-row", name="Acme: OR row"),
        make_model(id="acme/aa-row", name="Acme: AA row"),
    ]

    def catalog_with(aa_stamp):
        doc = _catalog_doc(
            [
                {
                    "id": "acme/or-row",
                    "quality": 30.0,
                    "quality_match": "openrouter",
                    "quality_as_of": "20261017",
                    "quality_effort": "none",
                    "reasoning": {
                        "default_effort": "none",
                        "depths": [
                            {
                                "effort": "none",
                                "index": 30.0,
                                "cost_per_task_usd": None,
                                "source": "openrouter",
                                "as_of": "20261017",
                            }
                        ],
                    },
                },
                {
                    "id": "acme/aa-row",
                    "quality": 45.1,
                    "quality_match": "aa",
                    "quality_as_of": aa_stamp,
                    "quality_effort": "none",
                    "reasoning": {
                        "default_effort": "none",
                        "depths": [
                            {
                                "effort": "none",
                                "index": 45.1,
                                "cost_per_task_usd": None,
                                "source": "aa",
                                "as_of": aa_stamp,
                            }
                        ],
                    },
                },
            ],
            schema_version=6,
        )
        return {
            "by_id": mc.catalog_quality_by_id(doc),
            "depths_by_id": {
                "acme/or-row": {
                    "default_effort": "none",
                    "depths": {
                        "none": {
                            "effort": "none",
                            "index": 30.0,
                            "cost_per_task_usd": None,
                            "source": "openrouter",
                            "as_of": "20261017",
                        }
                    },
                },
                "acme/aa-row": {
                    "default_effort": "none",
                    "depths": {
                        "none": {
                            "effort": "none",
                            "index": 45.1,
                            "cost_per_task_usd": None,
                            "source": "aa",
                            "as_of": aa_stamp,
                        }
                    },
                },
            },
            "generated_at": "2026-10-23T13:15:07+00:00",
        }

    # The aa row's ISO stamp is the fresher one: it wins despite the
    # date-only stamp's lexicographic head start.
    _stub_quality_run(
        monkeypatch,
        models,
        catalog=catalog_with("2026-10-23T13:15:07+00:00"),
        aa_raises=True,
        frontend_models=_non_reasoning_frontend(models),
    )
    assert mc.run(make_args(), models, False) == 0
    out = capsys.readouterr().out
    assert (
        "Quality source: our published catalog (2 models, as of 2026-10-23 13:15Z)"
        in out
    )

    # The date-only stamp is the fresher one: it wins and renders as a
    # plain date (the block ends at "2026-10-17)", never "00:00Z").
    _stub_quality_run(
        monkeypatch,
        models,
        catalog=catalog_with("2026-10-16T13:15:07+00:00"),
        aa_raises=True,
        frontend_models=_non_reasoning_frontend(models),
    )
    assert mc.run(make_args(), models, False) == 0
    out = capsys.readouterr().out
    assert "Quality source: our published catalog (2 models, as of 2026-10-17)" in out


def test_run_unkeyed_catalog_schedules_quality_stays_numeric(monkeypatch, capsys):
    # End-to-end: build_schedules consumes the quality record through
    # record["index"] -- a hydrated catalog row must never leak the dict
    # into the catalog document's schedules[].quality (number-or-null).
    models = [make_model(id="acme/model-a", pricing=deepseek_pricing())]
    catalog = {
        "by_id": {
            "acme/model-a": {
                "index": 42.0,
                "source": "aa",
                "as_of": "2026-10-23T13:15:00Z",
                "effort": "high",
            }
        },
        "generated_at": "2026-10-23T13:15:00Z",
        "depths_by_id": {},
    }
    _stub_quality_run(monkeypatch, models, catalog=catalog, aa_raises=True)
    args = make_args(catalog=True)
    assert mc.run(args, models, False) == 0
    doc = json.loads(capsys.readouterr().out)
    (sched,) = doc["schedules"]
    assert sched["id"] == "acme/model-a"
    assert isinstance(sched["quality"], float)
    assert sched["quality"] == 42.0
    assert sched["quality_effort"] == "high"


def test_run_unkeyed_catalog_schedules_quality_for_non_candidates(monkeypatch, capsys):
    # build_schedules walks the same fallback chain resolve_quality walks,
    # minus its AA tier: a schedules-only model (filtered out of the
    # candidate pool by the context floor) that the hydrated catalog
    # covers must get a numeric deals QUAL with catalog provenance -- the
    # run-level chain label "catalog" and the row's own stamp, exactly
    # like resolve_quality's catalog tier records it.
    models = [
        make_model(id="acme/candidate", name="Acme: Candidate"),
        make_model(
            id="acme/sched-only",
            name="Acme: Sched only",
            pricing=deepseek_pricing(),
            context_length=262144,
        ),
    ]
    catalog_doc = _catalog_doc(
        [
            {
                "id": "acme/sched-only",
                "quality": 30.0,
                "quality_match": "openrouter",
                "quality_as_of": "20261017",
                "quality_effort": "none",
            },
        ],
        schema_version=6,
    )
    catalog = {
        "by_id": mc.catalog_quality_by_id(catalog_doc),
        "generated_at": "2026-10-23T13:15:07+00:00",
    }
    _stub_quality_run(monkeypatch, models, catalog=catalog, aa_raises=True)
    args = make_args(catalog=True, min_context=1_000_000)
    assert mc.run(args, models, False) == 0
    published = json.loads(capsys.readouterr().out)
    assert [entry["id"] for entry in published["models"]] == ["acme/candidate"]
    (sched,) = published["schedules"]
    assert sched["id"] == "acme/sched-only"
    assert isinstance(sched["quality"], float)
    assert sched["quality"] == 30.0
    assert sched["quality_source"] == "catalog"
    assert sched["quality_as_of"] == "20261017"


def test_run_unkeyed_catalog_schedule_entry_from_schedules_verbatim(
    monkeypatch, capsys
):
    # Spec 6.5: unkeyed schedules hydrate from the published catalog's
    # schedules array, not its models array. A schedule-only model (sub-1M
    # context) is absent from models by design, so this is the only place
    # its recorded deals quality can come from. The entry's four
    # provenance fields are copied verbatim -- no run-level relabeling.
    models = [
        make_model(id="acme/candidate", name="Acme: Candidate"),
        make_model(
            id="acme/sched-only",
            name="Acme: Sched only",
            pricing=deepseek_pricing(),
            context_length=262144,
            reasoning={"default_effort": "high", "supported_efforts": ["high"]},
        ),
    ]
    catalog = {
        "by_id": {},
        "depths_by_id": {},
        "generated_at": "2026-10-23T13:15:07+00:00",
        "schedules_by_id": {
            "acme/sched-only": {
                "quality": 47.5,
                "quality_effort": "high",
                "quality_as_of": "t9",
                "quality_source": "aa",
            }
        },
    }
    _stub_quality_run(
        monkeypatch,
        models,
        catalog=catalog,
        aa_raises=True,
        or_bench={"acme/sched-only": {"intelligence_index": 30.0}},
    )
    args = make_args(catalog=True, min_context=1_000_000)
    assert mc.run(args, models, False) == 0
    published = json.loads(capsys.readouterr().out)
    (sched,) = published["schedules"]
    assert sched["id"] == "acme/sched-only"
    assert sched["quality"] == 47.5
    assert sched["quality_effort"] == "high"
    assert sched["quality_as_of"] == "t9"
    # Verbatim, never relabeled to the run-level "catalog" chain.
    assert sched["quality_source"] == "aa"


def test_run_unkeyed_catalog_schedule_null_quality_reasoning_stays_null(
    monkeypatch, capsys
):
    # A recorded entry with null quality falls through the remaining tiers.
    # The model is reasoning, so the live OR trio must not supply a score.
    models = [
        make_model(id="acme/candidate", name="Acme: Candidate"),
        make_model(
            id="acme/sched-only",
            name="Acme: Sched only",
            pricing=deepseek_pricing(),
            context_length=262144,
            reasoning={"default_effort": "high", "supported_efforts": ["high"]},
        ),
    ]
    catalog = {
        "by_id": {},
        "depths_by_id": {},
        "generated_at": "2026-10-23T13:15:07+00:00",
        "schedules_by_id": {
            "acme/sched-only": {
                "quality": None,
                "quality_effort": None,
                "quality_as_of": None,
                "quality_source": None,
            }
        },
    }
    _stub_quality_run(
        monkeypatch,
        models,
        catalog=catalog,
        aa_raises=True,
        or_bench={"acme/sched-only": {"intelligence_index": 30.0}},
    )
    args = make_args(catalog=True, min_context=1_000_000)
    assert mc.run(args, models, False) == 0
    published = json.loads(capsys.readouterr().out)
    (sched,) = published["schedules"]
    assert sched["id"] == "acme/sched-only"
    assert sched["quality"] is None
    assert sched["quality_effort"] is None
    assert sched["quality_source"] is None


def test_run_unkeyed_v4_catalog_never_hydrates_schedules(monkeypatch, capsys):
    # An older catalog hydrates no quality rather than mixing old quality
    # semantics into a best-selectable-depth run. Its schedules carry no
    # quality_effort, so the parser stays behind the v6 gate.
    models = [
        make_model(id="acme/candidate", name="Acme: Candidate"),
        make_model(
            id="acme/sched-only",
            name="Acme: Sched only",
            pricing=deepseek_pricing(),
            context_length=262144,
            reasoning={"default_effort": "high", "supported_efforts": ["high"]},
        ),
    ]
    v4_doc = _catalog_doc([], schema_version=4)
    v4_doc["schedules"] = [
        {
            "id": "acme/sched-only",
            "quality": 30.0,
            "quality_as_of": "20261017",
            "quality_source": "openrouter",
        }
    ]
    monkeypatch.setattr(mc, "fetch_json", lambda *a, **k: v4_doc)
    envelope = mc.fetch_quality_catalog(make_args(no_cache=True))
    assert envelope["schedules_by_id"] == {}

    _stub_quality_run(
        monkeypatch,
        models,
        catalog=envelope,
        aa_raises=True,
        or_bench={"acme/sched-only": {"intelligence_index": 30.0}},
    )
    args = make_args(catalog=True, min_context=1_000_000)
    assert mc.run(args, models, False) == 0
    published = json.loads(capsys.readouterr().out)
    assert published["sources"]["aa"]["mode"] == "none"
    (sched,) = published["schedules"]
    assert sched["id"] == "acme/sched-only"
    # No tier-0 hydration and no OR fallback for a reasoning model.
    assert sched["quality"] is None
    assert sched["quality_effort"] is None
    assert sched["quality_source"] is None


def test_run_no_network_quality_blind_ranking_unchanged(monkeypatch, capsys):
    # All sources empty (no key, catalog down, OR silent): the existing
    # quality-blind path must be unchanged -- no fetch, no catalog, and the
    # fallback footer note.
    models = [make_model(id="acme/model-a", name="Acme: A")]
    _stub_quality_run(monkeypatch, models, aa_raises=True)
    args = make_args()
    assert mc.run(args, models, False) == 0
    out = capsys.readouterr().out
    assert "pool:" not in out
    assert "Quality source: unavailable; ranked on price/context/age" in out


# ---------------------------------------------------------------------------
# Task 4: CLI depth column, depth block, and --best
# ---------------------------------------------------------------------------


def test_print_table_includes_depth_column(capsys):
    cand = _cand("acme/model-a", 50.0)
    cand["reasoning"] = {"default_effort": "high"}
    cand["quality_record"] = {"index": 46.8, "effort": "high"}
    mc.print_table(
        [cand],
        {"quality": 0.4, "price": 0.4, "context": 0.1, "age": 0.1},
        None,
    )
    out = capsys.readouterr().out
    assert "DEPTH" in out
    assert "high*" in out


def test_depth_detail_marks_default_and_unmeasured(capsys):
    cand = _cand("acme/model-a", 50.0)
    cand["reasoning"] = {
        "default_effort": "high",
        "supported_efforts": ["max", "high", "none"],
    }
    cand["reasoning_depths"] = [
        {"effort": "max", "index": 56.0, "cost_per_task_usd": 5.46, "as_of": "t1"},
        {"effort": "none", "index": None, "cost_per_task_usd": None, "as_of": None},
    ]
    mc.print_depth_detail([cand])
    out = capsys.readouterr().out
    assert "high*" in out and "max" in out and "#max" in out
    # Supported but unmeasured: still selectable, with "-" in the AA columns.
    assert "#none" in out


def test_depth_detail_marks_aa_only_effort_not_selectable(capsys):
    cand = _cand("acme/model-a", 50.0)
    cand["reasoning"] = {"default_effort": "high", "supported_efforts": ["high"]}
    cand["reasoning_depths"] = [
        {"effort": "max", "index": 56.0, "cost_per_task_usd": 5.46, "as_of": "t1"},
    ]
    mc.print_depth_detail([cand])
    out = capsys.readouterr().out
    assert "not selectable" in out
    assert "#max" not in out


def test_best_prints_none_variant_when_selected(capsys):
    cand = _cand("acme/model-a", 50.0)
    cand["quality_record"] = {"index": 20.0, "effort": "none"}
    cand["reasoning"] = {
        "default_effort": "high",
        "supported_efforts": ["high", "none"],
    }
    top = [cand]
    args = make_args(best=True, reasoning_effort="none")
    mc.print_best(top, args)
    assert capsys.readouterr().out.strip() == "openrouter/acme/model-a#none"


def test_best_prints_bare_for_default(capsys):
    cand = _cand("acme/model-a", 50.0)
    cand["quality_record"] = {"index": 46.8, "effort": "high"}
    cand["reasoning"] = {"default_effort": "high", "supported_efforts": ["high", "max"]}
    mc.print_best([cand], make_args(best=True))
    assert capsys.readouterr().out.strip() == "openrouter/acme/model-a"


def test_parse_args_accepts_depth_flags():
    args = mc.parse_args(["--reasoning-effort", "max", "--no-depths"])
    assert args.reasoning_effort == "max"
    assert args.no_depths is True


def test_depth_detail_empty_prints_nothing(capsys):
    mc.print_depth_detail([])
    assert capsys.readouterr().out == ""


def test_run_prints_depth_detail_by_default(monkeypatch, capsys):
    models = [make_model(id="acme/model-a", name="Acme: A")]
    _stub_quality_run(monkeypatch, models, aa_raises=True)
    assert mc.run(make_args(), models, False) == 0
    out = capsys.readouterr().out
    assert "depth detail" in out
    assert "no AA depth data" in out


def test_run_summary_moves_below_table(monkeypatch, capsys):
    models = [make_model(id="acme/model-a", name="Acme: A")]
    _stub_quality_run(monkeypatch, models, aa_raises=True)
    assert mc.run(make_args(), models, False) == 0
    out = capsys.readouterr().out
    assert "pool:" not in out
    assert "Weights: " in out
    assert "Quality source: unavailable; ranked on price/context/age" in out
    assert out.index("RANK") < out.index("Weights:")


def test_run_no_depths_suppresses_depth_detail(monkeypatch, capsys):
    models = [make_model(id="acme/model-a", name="Acme: A")]
    _stub_quality_run(monkeypatch, models, aa_raises=True)
    assert mc.run(make_args(no_depths=True), models, False) == 0
    out = capsys.readouterr().out
    assert "depth detail  (AA index" not in out
    assert "no AA depth data" not in out
    assert "DEPTH" in out  # the column stays


def test_best_single_state_none_variant_and_non_reasoning(capsys):
    single = _cand("acme/single", 1.0)
    single["reasoning"] = {
        "default_effort": "reasoning",
        "supported_efforts": [],
        "is_reasoning": True,
        "mandatory": False,
    }
    mc.print_best([single], make_args(best=True, reasoning_effort="none"))
    assert capsys.readouterr().out.strip() == "openrouter/acme/single#none"

    plain = _cand("acme/plain", 1.0)
    plain["reasoning"] = {
        "default_effort": "none",
        "supported_efforts": [],
        "is_reasoning": False,
    }
    mc.print_best([plain], make_args(best=True, reasoning_effort="none"))
    assert capsys.readouterr().out.strip() == "openrouter/acme/plain"


def test_best_bare_when_effort_not_exposed(capsys):
    cand = _cand("acme/model-a", 1.0)
    cand["reasoning"] = {
        "default_effort": "high",
        "supported_efforts": ["high"],
        "is_reasoning": True,
    }
    mc.print_best([cand], make_args(best=True, reasoning_effort="max"))
    assert capsys.readouterr().out.strip() == "openrouter/acme/model-a"


def test_depth_detail_select_rules_for_single_state_and_non_reasoning(capsys):
    single = _cand("acme/single", 1.0)
    single["reasoning"] = {
        "default_effort": "reasoning",
        "supported_efforts": [],
        "is_reasoning": True,
        "mandatory": False,
    }
    single["reasoning_depths"] = [
        {"effort": "reasoning", "index": 40.0, "cost_per_task_usd": 0.2},
        {"effort": "none", "index": 30.0, "cost_per_task_usd": 0.1},
    ]
    plain = _cand("acme/plain", 1.0)
    plain["reasoning"] = {
        "default_effort": "none",
        "supported_efforts": [],
        "is_reasoning": False,
    }
    plain["reasoning_depths"] = [
        {"effort": "none", "index": 30.0, "cost_per_task_usd": 0.1}
    ]
    mc.print_depth_detail([single, plain])
    out = capsys.readouterr().out
    # Single-state reasoning row: not selectable; its none toggle is #none.
    reasoning_row = next(
        line for line in out.splitlines() if line.strip().startswith("reasoning")
    )
    assert reasoning_row.rstrip().endswith("-")
    assert "#none" in out
    # A non-reasoning model's none row stays bare even as a measured row.
    plain_block = out.split("#2 acme/plain")[1]
    none_row = next(
        line for line in plain_block.splitlines() if line.strip().startswith("none")
    )
    assert none_row.rstrip().endswith("-")
    assert "#none" not in plain_block


def test_depth_detail_vs_default_ratio_and_zero(capsys):
    cand = _cand("acme/model-a", 1.0)
    cand["reasoning"] = {
        "default_effort": "high",
        "supported_efforts": ["max", "high", "low"],
        "is_reasoning": True,
    }
    cand["reasoning_depths"] = [
        {"effort": "max", "index": 50.0, "cost_per_task_usd": 0.0},
        {"effort": "high", "index": 40.0, "cost_per_task_usd": 0.5},
        {"effort": "low", "index": 30.0, "cost_per_task_usd": 1.0},
    ]
    zero_default = _cand("acme/zero-default", 1.0)
    zero_default["reasoning"] = {
        "default_effort": "high",
        "supported_efforts": ["high"],
        "is_reasoning": True,
    }
    zero_default["reasoning_depths"] = [
        {"effort": "high", "index": 40.0, "cost_per_task_usd": 0.0},
    ]
    mc.print_depth_detail([cand, zero_default])
    out = capsys.readouterr().out
    first_block = out.split("#2 acme/zero-default")[0]
    rows = {
        line.split()[0]: line.split()
        for line in first_block.splitlines()
        if line.split() and line.split()[0] in {"max", "high*", "low"}
    }
    assert rows["max"][3] == "-"  # zero measured cost
    assert rows["high*"][3] == "1.0x"
    assert rows["low"][3] == "2.0x"
    zero_block = out.split("#2 acme/zero-default")[1]
    zero_row = next(
        line for line in zero_block.splitlines() if line.strip().startswith("high*")
    )
    assert zero_row.split()[3] == "-"  # zero default cost


def test_run_prints_depth_detail_before_table_legend(monkeypatch, capsys):
    models = [make_model(id="acme/model-a", name="Acme: A")]
    _stub_quality_run(monkeypatch, models, aa_raises=True)
    assert mc.run(make_args(), models, False) == 0
    out = capsys.readouterr().out
    depth_at = out.index("depth detail  (AA index")
    legend_at = out.index("best measured selectable depth")
    assert depth_at < legend_at


def test_depth_detail_reports_missing_default_record(capsys):
    cand = _cand("acme/model-a", 50.0)
    cand["reasoning"] = {
        "default_effort": "high",
        "supported_efforts": ["max", "high", "low", "none"],
        "is_reasoning": True,
    }
    cand["reasoning_depths"] = [
        {"effort": "max", "index": 45.1, "cost_per_task_usd": 0.26, "as_of": "t1"},
    ]
    mc.print_depth_detail([cand])
    out = capsys.readouterr().out
    assert "AA has no high record for this model." in out
    assert "high*" in out


def test_depth_detail_shows_unlabeled_aa_reference(capsys):
    cand = _cand("z-ai/glm-5.3-flash", 27.8)
    cand["reasoning"] = {
        "default_effort": "max",
        "supported_efforts": ["max", "high", "low"],
        "is_reasoning": True,
    }
    cand["reasoning_depths"] = []
    cand["quality_unlabeled"] = {
        "index": 27.8,
        "cost_per_task_usd": 0.58,
        "source": "aa",
        "as_of": "t1",
    }
    mc.print_depth_detail([cand])
    out = capsys.readouterr().out
    assert "AA index 27.8 without an effort label" in out
    assert "AA $0.58 per task" in out
    assert "reference only" in out
    assert "may not be the quality you get at a selectable effort" in out
    # The per-effort table is still absent, but the blanket "no data" line
    # would contradict the reference above it.
    assert "no AA depth data" not in out


def test_depth_detail_unlabeled_reference_without_cost(capsys):
    cand = _cand("z-ai/glm-5.3-flash", 27.8)
    cand["reasoning"] = {"default_effort": "max", "supported_efforts": ["max"]}
    cand["reasoning_depths"] = []
    cand["quality_unlabeled"] = {
        "index": 27.8,
        "cost_per_task_usd": None,
        "source": "aa",
        "as_of": "t1",
    }
    mc.print_depth_detail([cand])
    out = capsys.readouterr().out
    note_line = next(line for line in out.splitlines() if "AA index 27.8" in line)
    assert "without an effort label" in note_line
    assert "$" not in note_line


def test_best_and_json_print_no_depth_detail(monkeypatch, capsys):
    models = [make_model(id="acme/model-a", name="Acme: A")]
    _stub_quality_run(monkeypatch, models, aa_raises=True)
    assert mc.run(make_args(best=True), models, False) == 0
    assert "depth detail" not in capsys.readouterr().out

    _stub_quality_run(monkeypatch, models, aa_raises=True)
    assert mc.run(make_args(json=True), models, False) == 0
    out = capsys.readouterr().out
    assert "depth detail" not in out
    assert json.loads(out)  # the JSON document is still the only output


def test_run_stderr_reports_ranked_and_dropped(monkeypatch, capsys):
    models = [
        make_model(id="acme/model-a", name="Acme: A"),
        make_model(id="acme/short", name="Acme: Short", context_length=1000),
    ]
    _stub_quality_run(monkeypatch, models, aa_raises=True)
    args = make_args(min_context=1_000_000)
    assert mc.run(args, models, False) == 0
    err = capsys.readouterr().err
    assert "1 of 2 listed models ranked" in err
    assert "(dropped: 1 context)" in err


def test_run_bare_prints_table_only(monkeypatch, capsys):
    models = [make_model(id="acme/model-a", name="Acme: A")]
    _stub_quality_run(monkeypatch, models, aa_raises=True)
    assert mc.run(make_args(bare=True), models, False) == 0
    captured = capsys.readouterr()
    assert captured.out.startswith("RANK")
    for fragment in ("Weights:", "Quality source:", "QUALITY", "SCORE     weighted"):
        assert fragment not in captured.out
    assert captured.err == ""


def test_run_bare_keeps_failure_warnings(monkeypatch, capsys):
    models = [make_model(id="acme/short", name="Acme: Short", context_length=1000)]
    _stub_quality_run(monkeypatch, models, aa_raises=True)
    args = make_args(min_context=1_000_000, bare=True)
    assert mc.run(args, models, False) == 2
    assert "no models satisfy the filters" in capsys.readouterr().err


def test_run_auto_color_is_plain_under_capture(monkeypatch, capsys):
    models = [make_model(id="acme/model-a", name="Acme: A")]
    _stub_quality_run(monkeypatch, models, aa_raises=True)
    assert mc.run(make_args(color="auto"), models, False) == 0
    captured = capsys.readouterr()
    assert "\x1b[" not in captured.out
    assert "\x1b[" not in captured.err


def test_run_color_on_emits_sgr_even_when_piped(monkeypatch, capsys):
    models = [make_model(id="acme/model-a", name="Acme: A")]
    _stub_quality_run(monkeypatch, models, aa_raises=True)
    assert mc.run(make_args(color="on"), models, False) == 0
    assert "\x1b[1m" in capsys.readouterr().out


def test_run_json_ignores_output_flags(monkeypatch, capsys):
    models = [make_model(id="acme/model-a", name="Acme: A")]
    _stub_quality_run(monkeypatch, models, aa_raises=True)
    args = make_args(json=True, color="on", bare=True, detail="simple")
    assert mc.run(args, models, False) == 0
    out = capsys.readouterr().out
    assert "\x1b[" not in out
    json.loads(out)  # still valid JSON, unchanged shape


def test_parse_args_output_flag_defaults():
    args = mc.parse_args([])
    assert args.color == "auto"
    assert args.detail == "expert"
    assert args.off_peak is False
    assert args.bare is False


@pytest.mark.parametrize(
    "argv,flag,value",
    [
        (["--detail", "fancy"], "--detail", "fancy"),
        (["--color", "sometimes"], "--color", "sometimes"),
    ],
)
def test_parse_args_rejects_unknown_detail_and_color(argv, flag, value, capsys):
    with pytest.raises(SystemExit) as exc:
        mc.parse_args(argv)
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert f"argument {flag}: invalid choice: '{value}'" in err


def test_help_groups_output_options(capsys):
    with pytest.raises(SystemExit):
        mc.parse_args(["--help"])
    out = capsys.readouterr().out
    headings = {
        name: f"\n{name}:\n"
        for name in ("output", "ranking", "filters", "data sources & cache")
    }
    for heading in headings.values():
        assert heading in out
    # Search after each heading: the usage line lists every option up front.
    at = {name: out.index(heading) for name, heading in headings.items()}
    assert at["output"] < out.index("--bare", at["output"]) < at["ranking"]
    assert at["ranking"] < out.index("--priority", at["ranking"]) < at["filters"]
    assert (
        at["filters"]
        < out.index("--min-context", at["filters"])
        < at["data sources & cache"]
    )
    assert at["data sources & cache"] < out.index(
        "--cache-ttl", at["data sources & cache"]
    )


def test_quality_footer_note_structured_sources_and_clauses():
    records = {
        "acme/a": {"source": "aa", "as_of": "2026-10-23T13:15:00Z"},
        "acme/b": {"source": "catalog", "as_of": "2026-10-22T09:00:00Z"},
        "acme/c": {"source": "openrouter", "as_of": "20261017"},
    }
    note = mc.quality_footer_note(records, 4.3, unmatched=1)
    assert note["warn"] is True
    assert note["sources"] == [
        "AA index v4.3 - live AA (1 model, checked 13:15Z)",
        "our published catalog (1 model, as of 2026-10-22 09:00Z)",
        "OpenRouter snapshot (1 model, as of 2026-10-17)",
    ]
    assert note["clauses"] == ["1 model unmatched (no quality data; quality score 0)"]


def test_quality_footer_note_none_when_no_records():
    assert mc.quality_footer_note({}, None, unmatched=2) is None


def test_quality_source_lines_wrap_at_source_boundaries():
    note = {
        "sources": [
            "live AA (2 models, checked 13:15Z)",
            "our published catalog (1 model, as of 2026-10-23 13:15Z)",
        ],
        "clauses": ["1 model unmatched (no quality data; quality score 0)"],
        "warn": True,
    }
    assert mc.quality_source_lines(note) == [
        "Quality source: live AA (2 models, checked 13:15Z) |",
        "                our published catalog (1 model, as of 2026-10-23 13:15Z);",
        "                1 model unmatched (no quality data; quality score 0)",
    ]


def test_quality_source_lines_last_source_unpunctuated_without_clauses():
    note = {"sources": ["live AA (1 model)"], "clauses": [], "warn": False}
    assert mc.quality_source_lines(note) == ["Quality source: live AA (1 model)"]


def test_quality_source_lines_fallback():
    assert mc.quality_source_lines(None) == [
        "Quality source: unavailable; ranked on price/context/age"
    ]


# ---------------------------------------------------------------------------
# Task 6: --off-peak deals table
# ---------------------------------------------------------------------------


def test_overall_scores_matches_build_catalog():
    models = [make_model(id="acme/hy4", pricing=hy4_pricing())]
    args = make_args(min_context=1_000_000)
    candidates, dropped = mc.build_candidates(
        models, args, {}, {m["id"] for m in models}, []
    )
    mc.compute_scores(candidates, args, {})
    doc = mc.build_catalog(args, models, candidates, dropped, [], {}, {}, None, {}, {})
    (entry,) = doc["models"]
    weights = mc.catalog_weights(candidates, {})
    assert (
        mc.overall_scores(candidates[0], weights)["balanced"]
        == entry["scores"]["overall"]["balanced"]
    )
    assert mc.overall_scores(candidates[0], weights) == entry["scores"]["overall"]


def test_off_peak_deals_sorts_caps_and_drops_zero_rounding():
    schedules = [
        {"id": "acme/a", "max_discount": 0.10},
        {"id": "acme/b", "max_discount": 0.50},
        {"id": "acme/c", "max_discount": 0.004},
        {"id": "acme/d", "max_discount": 0.50},
    ]
    ids = [row["id"] for row in mc.off_peak_deals(schedules)]
    assert ids == ["acme/b", "acme/d", "acme/a"]


def test_off_peak_deals_caps_at_ten():
    schedules = [{"id": f"acme/m{i}", "max_discount": 0.1 + i / 100} for i in range(15)]
    assert len(mc.off_peak_deals(schedules)) == 10


def test_print_deals_table_and_empty_state(capsys):
    rows = [
        {
            "id": "acme/deal",
            "sched_note": "-45%",
            "sched_detail": "daily 01:00-05:00 UTC",
            "quality": 38.9,
            "offpeak": {"input_per_1m": 0.017, "output_per_1m": 0.066},
            "context": 256000,
            "score": 0.661,
        }
    ]
    mc.print_deals(rows)
    out = capsys.readouterr().out
    assert "Off-peak deals" in out
    assert "daily 01:00-05:00 UTC" in out
    assert "-45%" in out
    assert "OFF-PEAK  largest scheduled discount" in out
    mc.print_deals([])
    assert (
        "No scheduled off-peak discounts in the current data."
        in capsys.readouterr().out
    )


def test_print_deals_dashes_out_of_pool_score_and_quality(capsys):
    rows = [
        {
            "id": "acme/deal",
            "sched_note": "-45%",
            "sched_detail": "daily 01:00-05:00 UTC",
            "quality": None,
            "offpeak": {"input_per_1m": 0.017, "output_per_1m": 0.066},
            "context": 256000,
            "score": None,
        }
    ]
    mc.print_deals(rows)
    lines = capsys.readouterr().out.splitlines()
    header = next(line for line in lines if line.startswith("MODEL"))
    row = next(line for line in lines if "acme/deal" in line)
    for column in ("QUAL", "SCORE"):
        end = header.index(column) + len(column)
        assert row[:end].endswith(" -")  # right-aligned cell under its header
    assert row.endswith("-")  # SCORE is the last column


def _deal_row(**overrides):
    row = {
        "id": "acme/deal",
        "sched_note": "-45%",
        "sched_detail": "daily 01:00-05:00 UTC",
        "quality": 38.9,
        "offpeak": {"input_per_1m": 0.017, "output_per_1m": 0.066},
        "context": 256000,
        "score": 0.661,
    }
    row.update(overrides)
    return row


def test_print_deals_color_header_and_placeholders(capsys):
    mc.print_deals(
        [_deal_row(), _deal_row(id="acme/unknown", quality=None, score=None)],
        style=mc.Style(True),
    )
    lines = capsys.readouterr().out.splitlines()
    header = next(line for line in lines if "SCORE" in line and "MODEL" in line)
    assert re.search(r"\x1b\[1m\s*OFF-PEAK\s*\x1b\[0m", header)
    assert re.search(r"\x1b\[1m\s*QUAL\s*\x1b\[0m", header)
    assert not re.search(r"\x1b\[1m\s*SCORE", header)
    known = next(line for line in lines if "acme/deal" in line)
    assert re.search(r"\x1b\[1m\s*-45%\x1b\[0m", known)
    assert "\x1b[2m" not in known
    unknown = next(line for line in lines if "acme/unknown" in line)
    assert re.search(r"\x1b\[2m\s*-\x1b\[0m", unknown)  # QUAL placeholder
    assert not re.search(r"\x1b\[1m[^\x1b]*SCORE", unknown)
    assert unknown.endswith("-")  # SCORE placeholder stays plain
    assert not unknown.endswith("\x1b[0m")


def test_run_off_peak_appends_deals(monkeypatch, capsys):
    models = [
        # Below the context floor: the id can only come from the deals table.
        make_model(id="acme/deal", pricing=hy4_pricing(), context_length=262144),
        make_model(id="acme/plain"),
    ]
    _stub_quality_run(monkeypatch, models, aa_raises=True)
    assert mc.run(make_args(off_peak=True, min_context=1_000_000), models, False) == 0
    out = capsys.readouterr().out
    lines = out.splitlines()
    assert "Off-peak deals" in out
    assert any(
        line.startswith("MODEL") and "OFF-PEAK" in line and "SCHEDULE" in line
        for line in lines
    )
    deal = next(line for line in lines if line.startswith("acme/deal"))
    assert "-10%" in deal
    assert "daily 16:00-00:00 UTC" in deal
    assert out.count("acme/deal") == 1  # deals row, not a main-table row


def test_run_off_peak_empty_state(monkeypatch, capsys):
    models = [make_model(id="acme/plain")]
    _stub_quality_run(monkeypatch, models, aa_raises=True)
    assert mc.run(make_args(off_peak=True), models, False) == 0
    assert (
        "No scheduled off-peak discounts in the current data."
        in capsys.readouterr().out
    )


def test_print_deals_empty_state_separator(capsys):
    mc.print_deals([])
    non_bare = capsys.readouterr().out
    assert non_bare == "\nNo scheduled off-peak discounts in the current data.\n"
    mc.print_deals([], bare=True)
    bare = capsys.readouterr().out
    assert bare == "No scheduled off-peak discounts in the current data.\n"


def test_run_bare_off_peak_suppresses_title_and_legend(monkeypatch, capsys):
    models = [
        # Below the context floor: absent from print_table's output, so the
        # deals header and body prove the second table is still rendered.
        make_model(id="acme/deal", pricing=hy4_pricing(), context_length=262144),
        make_model(id="acme/plain"),
    ]
    _stub_quality_run(monkeypatch, models, aa_raises=True)
    assert (
        mc.run(
            make_args(off_peak=True, bare=True, min_context=1_000_000), models, False
        )
        == 0
    )
    out = capsys.readouterr().out
    lines = out.splitlines()
    assert "Off-peak deals" not in out
    assert "largest scheduled discount" not in out
    header = next(
        line for line in lines if line.startswith("MODEL") and "OFF-PEAK" in line
    )
    assert "SCHEDULE" in header
    deal = next(line for line in lines if line.startswith("acme/deal"))
    assert "-10%" in deal
    assert out.count("acme/deal") == 1  # not also in the main table


def test_run_off_peak_non_candidate_deal_keeps_resolved_quality(monkeypatch, capsys):
    # The web page shows the resolved quality for out-of-pool schedule rows
    # and build_schedules resolves it, so QUAL keeps that number; only SCORE
    # stays "-" (no balanced score exists outside the ranked pool).
    models = [
        make_model(id="acme/ranked", name="Acme: Ranked", pricing=deepseek_pricing()),
        make_model(
            id="acme/sched-only",
            name="Acme: Sched only",
            pricing=deepseek_pricing(),
            context_length=262144,
        ),
    ]
    ranked_depth = {
        "effort": "none",
        "index": 41.0,
        "source": "catalog",
        "as_of": "2026-10-23T13:15:07+00:00",
    }
    catalog = {
        "by_id": {
            "acme/ranked": {
                "index": 41.0,
                "source": "aa",
                "as_of": "2026-10-23T13:15:07+00:00",
            },
            "acme/sched-only": {
                "index": 30.0,
                "source": "openrouter",
                "as_of": "20261017",
            },
        },
        "depths_by_id": {
            "acme/ranked": {"default_effort": "none", "depths": {"none": ranked_depth}},
        },
        "schedules_by_id": {
            "acme/sched-only": {
                "quality": 30.0,
                "quality_effort": "none",
                "quality_as_of": "20261017",
                "quality_source": "openrouter",
            }
        },
        "generated_at": "2026-10-23T13:15:07+00:00",
        "schema_version": 6,
    }
    _stub_quality_run(
        monkeypatch,
        models,
        catalog=catalog,
        aa_raises=True,
        frontend_models=_non_reasoning_frontend(models),
    )
    args = make_args(off_peak=True, bare=True, min_context=1_000_000)
    assert mc.run(args, models, False) == 0
    lines = capsys.readouterr().out.splitlines()
    ranked = next(line for line in lines if line.startswith("acme/ranked"))
    sched_only = next(line for line in lines if line.startswith("acme/sched-only"))
    assert "41.0" in ranked
    assert "30.0" in sched_only


def test_run_balanced_score_matches_between_main_and_deals(monkeypatch, capsys):
    # quality 10 at 1.5M context, listed 20 days ago: the unrounded score is
    # 0.57548..., while weighting the 4-dp-rounded components gives 0.5755
    # (prints 0.576). Both tables must print the same SCORE.
    monkeypatch.setattr(mc.time, "time", lambda: 1_700_000_000.0)
    models = [
        make_model(
            id="acme/sched",
            pricing=hy4_pricing(),
            context_length=1_500_000,
            created=1_700_000_000 - 20 * 86400,
        )
    ]
    depth = {
        "effort": "none",
        "index": 10.0,
        "source": "catalog",
        "as_of": "2026-10-23T13:15:00Z",
    }
    catalog = {
        "by_id": {
            "acme/sched": {
                "index": 10.0,
                "source": "aa",
                "as_of": "2026-10-23T13:15:00Z",
            }
        },
        "depths_by_id": {
            "acme/sched": {"default_effort": "none", "depths": {"none": depth}}
        },
        "schedules_by_id": {
            "acme/sched": {
                "quality": 10.0,
                "quality_effort": "none",
                "quality_as_of": "2026-10-23T13:15:00Z",
                "quality_source": "catalog",
            }
        },
        "generated_at": "2026-10-23T13:15:00Z",
        "schema_version": 6,
    }
    _stub_quality_run(
        monkeypatch,
        models,
        catalog=catalog,
        aa_raises=True,
        frontend_models=_non_reasoning_frontend(models),
    )
    assert mc.run(make_args(off_peak=True, min_context=1_000_000), models, False) == 0
    rows = [
        line
        for line in capsys.readouterr().out.splitlines()
        if "acme/sched" in line and not line.startswith("#")
    ]
    assert len(rows) == 2  # main table + deals table
    scores = [line.split()[-1] for line in rows]
    assert scores[0] == scores[1] == "0.575"


def test_footer_aa_version_prefix_only_on_aa_block():
    note = mc.quality_footer_note(
        {"acme/x": {"source": "openrouter", "as_of": "20261017"}}, 4.3, 0
    )
    lines = mc.quality_source_lines(note)
    assert lines[0] == (
        "Quality source: OpenRouter snapshot (1 model, as of 2026-10-17)"
    )
    assert "AA index" not in lines[0]


def test_weights_percentages_sum_to_100_quality_blind(capsys):
    models = [make_model(id="acme/a"), make_model(id="acme/b")]
    args = make_args(min_context=1_000_000)
    candidates, _ = mc.build_candidates(models, args, {}, {"acme/a", "acme/b"}, [])
    weights = mc.compute_scores(candidates, args, {})  # quality-blind pool
    mc.print_table(candidates, weights, None)
    line = next(
        line
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("Weights:")
    )
    percentages = [
        int(token.rstrip(",%"))
        for token in line.split()
        if token.rstrip(",").endswith("%")
    ]
    assert sum(percentages) == 100


def test_print_deals_headers_left_aligned(capsys):
    mc.print_deals([_deal_row()])
    lines = capsys.readouterr().out.splitlines()
    header = next(line for line in lines if line.startswith("MODEL"))
    rule = next(line for line in lines if set(line.replace(" ", "")) == {"-"})
    widths = [len(segment) for segment in rule.split("  ")]
    start = 0
    for col, width in zip(mc.DEALS_COLUMNS, widths):
        assert header[start : start + width].rstrip() == col
        start += width + 2


def test_print_deals_bolds_title(capsys):
    mc.print_deals([_deal_row()], style=mc.Style(True))
    title = next(
        line
        for line in capsys.readouterr().out.splitlines()
        if "Off-peak deals" in line
    )
    assert title.startswith("\x1b[1mOff-peak deals")
    assert title.endswith("\x1b[0m")


@pytest.mark.parametrize("mode", ["json", "best"])
def test_off_peak_ignored_under_machine_outputs(mode, monkeypatch, capsys):
    models = [
        make_model(id="acme/deal", pricing=hy4_pricing()),
        make_model(id="acme/plain"),
    ]
    _stub_quality_run(monkeypatch, models, aa_raises=True)
    assert mc.run(make_args(**{mode: True}), models, False) == 0
    without_off_peak = capsys.readouterr().out
    assert mc.run(make_args(off_peak=True, **{mode: True}), models, False) == 0
    assert capsys.readouterr().out == without_off_peak


def test_run_off_peak_catalog_output_keeps_non_candidate_quality(monkeypatch, capsys):
    models = [
        make_model(id="acme/candidate", name="Acme: Candidate"),
        make_model(
            id="acme/sched-only",
            name="Acme: Sched only",
            pricing=deepseek_pricing(),
            context_length=262144,
        ),
    ]
    catalog = {
        "by_id": {
            "acme/sched-only": {
                "index": 30.0,
                "source": "openrouter",
                "as_of": "20261017",
            }
        },
        "depths_by_id": {},
        "schedules_by_id": {
            "acme/sched-only": {
                "quality": 30.0,
                "quality_effort": "none",
                "quality_as_of": "20261017",
                "quality_source": "openrouter",
            }
        },
        "generated_at": "2026-10-23T13:15:07+00:00",
        "schema_version": 6,
    }
    _stub_quality_run(
        monkeypatch,
        models,
        catalog=catalog,
        aa_raises=True,
        frontend_models=_non_reasoning_frontend(models),
    )
    args = make_args(catalog=True, off_peak=True, min_context=1_000_000)
    assert mc.run(args, models, False) == 0
    (sched,) = json.loads(capsys.readouterr().out)["schedules"]
    assert sched["quality"] == 30.0


@pytest.mark.parametrize("detail", ["expert", "simple"])
def test_print_table_legend_shows_actual_input_share(detail, capsys):
    mc.print_table(
        _scored_candidates(),
        mc.PRIORITY_WEIGHTS["balanced"],
        None,
        detail=detail,
        input_share=0.5,
    )
    out = capsys.readouterr().out
    assert "50% input / 50% output (--input-share)" in out
    assert "75% input" not in out


def test_print_table_legend_defaults_to_75_25(capsys):
    mc.print_table(_scored_candidates(), mc.PRIORITY_WEIGHTS["balanced"], None)
    assert "75% input / 25% output (--input-share)" in capsys.readouterr().out


def test_run_legend_uses_args_input_share(monkeypatch, capsys):
    models = [make_model(id="acme/model-a", name="Acme: A")]
    _stub_quality_run(monkeypatch, models, aa_raises=True)
    assert mc.run(make_args(input_share=0.9), models, False) == 0
    assert "90% input / 10% output (--input-share)" in capsys.readouterr().out


def test_print_table_simple_model_legend_matches_printed_ids(capsys):
    top = _scored_candidates()
    mc.print_table(top, mc.PRIORITY_WEIGHTS["balanced"], None, detail="simple")
    out = capsys.readouterr().out
    assert "openrouter/<provider>/<model>" not in out
    assert "MODEL       provider/model id as listed; --best prefixes openrouter/" in out
    assert top[0]["id"] in out  # the table prints the bare id


def test_run_bare_suppresses_cached_note(monkeypatch, capsys):
    models = [make_model(id="acme/model-a", name="Acme: A")]
    _stub_quality_run(monkeypatch, models, aa_raises=True)
    assert mc.run(make_args(bare=True), models, True) == 0
    assert capsys.readouterr().err == ""


def test_run_cached_note_printed_when_not_bare(monkeypatch, capsys):
    models = [make_model(id="acme/model-a", name="Acme: A")]
    _stub_quality_run(monkeypatch, models, aa_raises=True)
    assert mc.run(make_args(), models, True) == 0
    assert "note: catalog cached (use --no-cache to refresh)" in capsys.readouterr().err


def test_run_bare_keeps_zdr_refusal_warning(monkeypatch, capsys):
    models = [make_model(id="acme/model-a", name="Acme: A")]
    _stub_quality_run(monkeypatch, models, aa_raises=True)
    monkeypatch.setattr(
        mc, "fetch_openrouter_frontend", lambda a: ({}, set(), {}, set(), [])
    )
    assert mc.run(make_args(bare=True), models, False) == 1
    captured = capsys.readouterr()
    assert "ZDR data unavailable; refusing to rank" in captured.err
    assert captured.out == ""


def test_run_color_on_dims_dropped_line_on_stderr(monkeypatch, capsys):
    models = [
        make_model(id="acme/model-a", name="Acme: A"),
        make_model(id="acme/short", name="Acme: Short", context_length=1000),
    ]
    _stub_quality_run(monkeypatch, models, aa_raises=True)
    args = make_args(color="on", min_context=1_000_000)
    assert mc.run(args, models, False) == 0
    err = capsys.readouterr().err
    assert "\x1b[2m(dropped: 1 context)\x1b[0m" in err
    assert "1 of 2 listed models ranked" in err
    assert "\x1b[" not in err.replace("\x1b[2m(dropped: 1 context)\x1b[0m", "")
