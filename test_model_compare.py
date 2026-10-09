"""Unit tests for model_compare scoring, matching, and filtering logic.

These cover pure functions only -- no network access is performed. Run with:

    pytest -q
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / "web"))
import build_site_data as bsd  # noqa: E402  (web/ module, contract validator)
import model_compare as mc


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
    mc.print_table(_scored_candidates(), 2, mc.PRIORITY_WEIGHTS["balanced"], "note")
    out = capsys.readouterr().out
    lines = out.splitlines()
    header = lines[0]
    assert "DISC" in header and "TIER" in header and "CTX" in header
    assert header.index("DISC") < header.index("TIER") < header.index("CTX")
    tiered_row = next(line for line in lines if "acme/tiered" in line)
    assert ">100k" in tiered_row
    plain_row = next(line for line in lines if "acme/plain" in line)
    assert ">100k" not in plain_row


def test_footer_legend_lines(capsys):
    mc.print_table(_scored_candidates(), 2, mc.PRIORITY_WEIGHTS["balanced"], "note")
    out = capsys.readouterr().out
    assert (
        "Prices shown are what a long session pays: several models bill at a "
        "higher rate once the prompt outgrows their cheap short-context tier."
    ) in out
    assert (
        "TIER marks how prices vary: a token threshold is the prompt size "
        "above which the higher rate applies; a percentage is a scheduled "
        "off-peak discount (price shown: the standard rate); SCHED marks a "
        "schedule whose discount is under 1%. --json lists each model's "
        "schedule."
    ) in out
    legend_lines = [
        line
        for line in out.splitlines()
        if line.startswith(("Prices shown", "TIER marks"))
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
    """Base-URL payload with discounts and benchmarks in one response."""
    return {
        "data": {
            "models": [
                {
                    "slug": "acme/a",
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
    discounts, zdr_ids, aa_by_id, cache_hits = mc.fetch_openrouter_frontend(
        make_args(no_cache=True)
    )
    assert discounts == {"acme/a": 0.5, "acme/b:free": 0.0, "acme/c:batch": 0.75}
    assert zdr_ids == {"acme/a", "acme/b:batch"}
    assert aa_by_id == {
        "acme/a": {
            "intelligence_index": 57.5,
            "coding_index": 71.5,
            "agentic_index": 58.2,
        }
    }
    assert cache_hits == set()
    assert mc.OPENROUTER_DISCOUNTS_URL in calls
    assert mc.OPENROUTER_ZDR_URL in calls


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
    discounts, zdr_ids, aa_by_id, cache_hits = mc.fetch_openrouter_frontend(
        make_args(no_cache=True)
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
    discounts, zdr_ids, aa_by_id, cache_hits = mc.fetch_openrouter_frontend(
        make_args(no_cache=False, cache_ttl=3600)
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
    discounts, zdr_ids, aa_by_id, cache_hits = mc.fetch_openrouter_frontend(
        make_args(no_cache=False, cache_ttl=3600)
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
    discounts, zdr_ids, aa_by_id, cache_hits = mc.fetch_openrouter_frontend(
        make_args(no_cache=True)
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
    assert not (cache_dir / "openrouter-frontend-aa.json").exists()
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
    (cache_dir / "openrouter-frontend-aa.json").write_text(
        json.dumps({"fetched_at": now, "payload": {}})
    )
    (cache_dir / "openrouter-zdr-v3.json").write_text(
        json.dumps({"fetched_at": now, "payload": []})
    )
    discounts, zdr_ids, aa_by_id, cache_hits = mc.fetch_openrouter_frontend(
        make_args(no_cache=False, cache_ttl=3600)
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
    discounts, zdr_ids, aa_by_id, cache_hits = mc.fetch_openrouter_frontend(
        make_args(no_cache=False, cache_ttl=3600)
    )
    assert len(calls) == 2  # aa not cached, so the base URL is hit anyway
    assert discounts == {"acme/a": 0.5, "acme/b:free": 0.0, "acme/c:batch": 0.75}
    assert aa_by_id == {
        "acme/a": {
            "intelligence_index": 57.5,
            "coding_index": 71.5,
            "agentic_index": 58.2,
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
    discounts, zdr_ids, aa_by_id, cache_hits = mc.fetch_openrouter_frontend(
        make_args(no_cache=True, no_zdr=True)
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
    (cache_dir / "openrouter-frontend-aa.json").write_text(
        json.dumps(
            {
                "fetched_at": time.time(),
                "payload": {"acme/a": {"intelligence_index": 57.5}},
            }
        )
    )
    discounts, zdr_ids, aa_by_id, cache_hits = mc.fetch_openrouter_frontend(
        make_args(no_cache=False, cache_ttl=3600)
    )
    assert len(calls) == 1  # only the ZDR URL
    assert discounts == {"acme/a": 0.5}
    assert aa_by_id == {"acme/a": {"intelligence_index": 57.5}}
    assert cache_hits == {"discounts", "aa"}


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
    mc.print_table(top, 1, {"price": 1.0}, "note")
    out = capsys.readouterr().out
    assert "DISC" in out
    assert "50%" in out


def test_print_table_dash_without_discount(capsys):
    top = [_cand("a/x", 1.0)]
    top[0].update({"score": 0.9, "quality": None, "discount": None})
    mc.print_table(top, 1, {"price": 1.0}, "note")
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
        mc, "fetch_openrouter_frontend", lambda a: ({}, {"acme/model-a"}, {}, set())
    )
    monkeypatch.setattr(mc, "fetch_aa_entries", lambda a: ([], None, False))
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
        ),
    )
    monkeypatch.setattr(mc, "fetch_aa_entries", lambda a: ([], None, False))
    argv = ["--top", "1", "--no-cache", "--min-context", "0", "--no-require-tools"]
    assert mc.main(argv) == 0
    captured = capsys.readouterr()
    assert (
        "quality via OpenRouter benchmarks (1): matched 1/2 candidates; "
        "unmatched candidates score 0 on quality" in captured.out
    )
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


def test_build_aa_lookup_skips_non_numeric_index():
    entries = [
        {"key": "acme/model-a", "name": "Model A", "index": "not-a-number"},
        {"key": "acme/model-b", "name": "Model B", "index": 55.0},
    ]
    exact, fuzzy = mc.build_aa_lookup(entries)
    # only the numeric entry survives into the fuzzy list
    assert len(fuzzy) == 1
    assert (
        mc.match_quality({"id": "acme/model-b", "name": "Model B"}, exact, fuzzy)
        == 55.0
    )


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
    assert (
        mc.match_quality({"id": "acme/model-b", "name": "Model B"}, exact, fuzzy)
        == 55.0
    )


def test_match_quality_exact_by_id():
    entries = [{"key": "acme/model-a", "name": "Model A", "index": 60.0}]
    exact, fuzzy = mc.build_aa_lookup(entries)
    assert (
        mc.match_quality({"id": "acme/model-a", "name": "Acme: Model A"}, exact, fuzzy)
        == 60.0
    )


def test_match_quality_fuzzy_overlap():
    entries = [{"key": "model-a", "name": "Model A", "index": 55.0}]
    exact, fuzzy = mc.build_aa_lookup(entries)
    result = mc.match_quality(
        {"id": "acme/model-a-extra", "name": "Model A Extra"},
        exact,
        fuzzy,
        allow_fuzzy=True,
    )
    assert result == 55.0


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
    assert mc.match_quality(model, exact, fuzzy, allow_fuzzy=True) == 59.5


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
    aa = mc.build_aa_benchmarks(benchmarks)
    assert aa["z-ai/glm-5.3-flash"] == {
        "intelligence_index": 57.5,
        "coding_index": 71.5,
        "agentic_index": 58.2,
    }
    assert aa["acme/plain"] == {"intelligence_index": 40.0}


def test_build_aa_benchmarks_latest_date_wins():
    benchmarks = {
        "acme/m-20260801": {"aa": {"intelligence_index": 10}},
        "acme/m-20260820": {"aa": {"intelligence_index": 20}},
        "acme/m": {"aa": {"intelligence_index": 30}},
    }
    assert mc.build_aa_benchmarks(benchmarks)["acme/m"] == {"intelligence_index": 20.0}


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
    assert aa == {"acme/partial": {"coding_index": 70.0}}


def test_build_aa_benchmarks_empty_inputs():
    assert mc.build_aa_benchmarks({}) == {}
    assert mc.build_aa_benchmarks(None) == {}


def test_base_model_id_strips_variant():
    assert mc.base_model_id("z-ai/glm-5.3-flash:free") == "z-ai/glm-5.3-flash"
    assert mc.base_model_id("z-ai/glm-5.3") == "z-ai/glm-5.3"


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


def aa_item(slug="acme/m", name="M", index=42.0):
    item = {"name": name}
    if slug is not None:
        item["slug"] = slug
    if index is not None:
        item["evaluations"] = {"artificial_analysis_intelligence_index": index}
    return item


def test_aa_api_sends_auth_header_and_timeout(monkeypatch):
    seen = {}

    def fake_fetch(url, headers=None, timeout=None, **k):
        seen.update(headers=headers, timeout=timeout)
        return aa_page([aa_item()])

    monkeypatch.setattr(mc, "fetch_json", fake_fetch)
    mc.aa_api_entries("dummy-key")
    assert seen["headers"] == {"x-api-key": "dummy-key"}
    assert seen["timeout"] == 30


def test_aa_api_parses_documented_shape(monkeypatch):
    payload = aa_page([aa_item("acme/a", "A", 42.5), aa_item("acme/b", "B", None)])
    monkeypatch.setattr(mc, "fetch_json", lambda *a, **k: payload)
    entries = mc.aa_api_entries("dummy-key")
    assert entries == [{"key": "acme/a", "name": "A", "index": 42.5}]


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
    assert mc.aa_api_entries("dummy-key") == [
        {"key": "acme/ok", "name": "OK", "index": 5.0}
    ]


def test_aa_api_skips_index_too_large_for_float(monkeypatch):
    # A huge JSON integer is a valid int but overflows float conversion; it
    # must be skipped like any unmeasurable value, not abort the dataset.
    payload = aa_page(
        [aa_item("acme/huge", "Huge", 10**400), aa_item("acme/ok", "OK", 5.0)]
    )
    monkeypatch.setattr(mc, "fetch_json", lambda *a, **k: payload)
    assert mc.aa_api_entries("dummy-key") == [
        {"key": "acme/ok", "name": "OK", "index": 5.0}
    ]


def test_aa_api_non_list_data_on_page_one_yields_nothing(monkeypatch):
    fake, calls = aa_url_stub({1: aa_page(5)})
    monkeypatch.setattr(mc, "fetch_json", fake)
    assert mc.aa_api_entries("dummy-key") == []
    assert calls == [f"{mc.AA_API_URL}?page=1"]


def test_aa_api_non_list_data_mid_pagination_keeps_earlier_pages(monkeypatch):
    fake, calls = aa_url_stub(
        {
            1: aa_page([aa_item("acme/a", "A", 40.0)], 1, 2, True),
            2: aa_page(5, 2, 2, False),
        }
    )
    monkeypatch.setattr(mc, "fetch_json", fake)
    assert mc.aa_api_entries("dummy-key") == [
        {"key": "acme/a", "name": "A", "index": 40.0}
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
    entries = mc.aa_api_entries("dummy-key")
    assert entries == [{"key": "Some Model", "name": "Some Model", "index": 7.0}]


def test_aa_api_ignores_non_dict_data_items(monkeypatch):
    payload = aa_page(["junk", 3, aa_item()])
    monkeypatch.setattr(mc, "fetch_json", lambda *a, **k: payload)
    assert len(mc.aa_api_entries("dummy-key")) == 1


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
    entries = mc.aa_api_entries("dummy-key")
    assert calls == [f"{mc.AA_API_URL}?page=1", f"{mc.AA_API_URL}?page=2"]
    assert entries == [
        {"key": "acme/a", "name": "A", "index": 40.0},  # first page wins
        {"key": "acme/b", "name": "B", "index": 9.0},
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
    entries = mc.aa_api_entries("dummy-key")
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
    entries = mc.aa_api_entries("dummy-key")
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
    entries = mc.aa_api_entries("dummy-key")
    assert entries == [{"key": "acme/a", "name": "A", "index": 40.0}]
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
    assert result == ([], None, False)
    assert pages == []  # no page-scrape fallback
    err = capsys.readouterr().err
    assert "AA API request failed (401)" in err
    assert "scrape" not in err


def test_fetch_aa_entries_api_empty_returns_empty(monkeypatch, capsys):
    monkeypatch.delenv("AA_API_KEY", raising=False)
    pages = _no_page_fetch(monkeypatch)
    monkeypatch.setattr(mc, "fetch_json", lambda *a, **k: aa_page([]))
    result = mc.fetch_aa_entries(make_args(aa_api_key="dummy-key"))
    assert result == ([], None, False)
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
    assert result == ([], None, False)
    assert calls == [] and pages == []
    assert capsys.readouterr().err == ""


def test_fetch_aa_entries_ignores_cached_scrape_source(monkeypatch, tmp_path):
    # A cache entry written by the removed page scrape carries a source
    # build_catalog no longer knows (it would raise); it must never be read.
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.delenv("AA_API_KEY", raising=False)
    _no_page_fetch(monkeypatch)
    mc.save_cache(
        "aa-intelligence-v2",
        {
            "entries": [{"key": "s", "name": "S", "index": 1.0, "estimated": False}],
            "source": "AA page scrape",
        },
    )
    result = mc.fetch_aa_entries(make_args(no_cache=False, cache_ttl=3600))
    assert result == ([], None, False)


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
    discounts, zdr_ids, aa_by_id, cache_hits = mc.fetch_openrouter_frontend(
        make_args(no_cache=True)
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
# resolve_quality (OR benchmarks first, exact AA fallback)
# ---------------------------------------------------------------------------


def test_resolve_quality_prefers_openrouter_benchmarks():
    candidates = [{"id": "acme/model-a", "name": "Acme: Model A"}]
    aa_by_id = {"acme/model-a": {"intelligence_index": 57.5}}
    exact, fuzzy = mc.build_aa_lookup(
        [{"key": "model-a", "name": "Model A", "index": 55.0}]
    )
    quality, source = mc.resolve_quality(
        candidates, aa_by_id, exact, fuzzy, "AA API v2"
    )
    assert quality == {"acme/model-a": 57.5}
    assert source == {"acme/model-a": "openrouter"}


def test_resolve_quality_falls_back_to_exact_aa_and_records_provenance():
    candidates = [{"id": "acme/model-b", "name": "Acme: Model B"}]
    exact, fuzzy = mc.build_aa_lookup(
        [{"key": "openai/model-b", "name": "Model B", "index": 51.2}]
    )
    quality, source = mc.resolve_quality(candidates, {}, exact, fuzzy, "AA API v2")
    assert quality == {"acme/model-b": 51.2}
    assert source == {"acme/model-b": "api"}


def test_resolve_quality_variant_inherits_base_trio():
    candidates = [{"id": "acme/model-a:free", "name": "Acme: Model A (free)"}]
    aa_by_id = {"acme/model-a": {"intelligence_index": 57.5}}
    quality, source = mc.resolve_quality(candidates, aa_by_id, {}, {}, None)
    assert quality == {"acme/model-a:free": 57.5}
    assert source == {"acme/model-a:free": "openrouter"}


def test_resolve_quality_trio_without_intelligence_falls_through():
    candidates = [{"id": "acme/model-a", "name": "Acme: Model A"}]
    aa_by_id = {"acme/model-a": {"coding_index": 70.0}}
    exact, fuzzy = mc.build_aa_lookup(
        [{"key": "acme/model-a", "name": "Model A", "index": 44.0}]
    )
    quality, source = mc.resolve_quality(
        candidates, aa_by_id, exact, fuzzy, "AA API v2"
    )
    assert quality == {"acme/model-a": 44.0}
    assert source == {"acme/model-a": "api"}


def test_resolve_quality_unmatched_is_none():
    candidates = [{"id": "acme/model-a", "name": "Acme: Model A"}]
    quality, source = mc.resolve_quality(candidates, {}, {}, {}, None)
    assert quality == {}
    assert source == {}


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
# feeds. bsd.CATALOG_ENTRY_KEYS already carries the "aa" trio.
CATALOG_ENTRY_KEYS = set(bsd.CATALOG_ENTRY_KEYS)


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
    assert doc["schema_version"] == 2
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
    assert set(p["weights"]) == {"balanced", "price", "quality"}
    assert p["weights"]["balanced"] == mc.PRIORITY_WEIGHTS["balanced"]
    assert doc["sources"] == {
        "openrouter": "ok",
        "aa": {
            "mode": "openrouter",
            "fallback": "api",
            "matched": 1,
            "matched_openrouter": 1,
        },
        "zdr": "ok",
        "discounts": "ok",
    }
    assert doc["pool"]["listed"] == 3
    assert doc["pool"]["candidates"] == 2
    assert doc["pool"]["dropped"]["context"] == 1
    assert set(doc["pool"]["dropped"]) == set(mc.CATALOG_DROP_REASONS)
    assert sum(doc["pool"]["dropped"].values()) == 3 - doc["pool"]["candidates"]


def test_catalog_entry_shape():
    doc = build_doc()
    assert {e["id"] for e in doc["models"]} == {"acme/model-a", "acme/model-b"}
    for entry in doc["models"]:
        assert set(entry) == CATALOG_ENTRY_KEYS
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
    assert b["discount"] is None


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
    # catalog pass the site's order_rows / validate_catalog / build_snapshot.
    doc = _inverting_catalog()
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
    # drift between the producer and the site validator must fail the
    # suite here, not the publish
    bsd.validate_catalog(build_doc())  # must not raise

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
    assert {e["quality_match"] for e in doc["models"]} == {"openrouter", "api"}
    assert doc["sources"]["aa"] == {
        "mode": "openrouter",
        "fallback": "api",
        "matched": 2,
        "matched_openrouter": 1,
    }
    bsd.validate_catalog(doc)  # must not raise


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
        "mode": "openrouter",
        "fallback": "api",
        "matched": 1,
        "matched_openrouter": 1,
    }


def test_catalog_quality_published_even_without_any_aa_source():
    doc = build_doc(aa_source=None)
    b = next(e for e in doc["models"] if e["id"] == "acme/model-b")
    assert b["quality"] == 68.4
    assert b["quality_match"] == "openrouter"
    assert doc["sources"]["aa"]["mode"] == "openrouter"


def test_catalog_aa_mode_reflects_fallback_when_or_empty():
    doc = build_doc(aa_source="AA API v2", aa_by_id={})
    assert doc["sources"]["aa"] == {
        "mode": "api",
        "fallback": "api",
        "matched": 0,
        "matched_openrouter": 0,
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


def test_catalog_no_zdr_marks_skipped(monkeypatch, capsys):
    monkeypatch.setattr(
        mc, "fetch_openrouter_frontend", lambda args: ({}, set(), {}, set())
    )
    monkeypatch.setattr(mc, "fetch_aa_entries", lambda args: ([], None, False))
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
        mc, "fetch_openrouter_frontend", lambda args: ({}, set(), {}, set())
    )
    monkeypatch.setattr(mc, "fetch_aa_entries", lambda args: ([], None, False))
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


MC_SCRIPT = Path(__file__).resolve().parent / "model_compare.py"


def test_version_flag():
    proc = subprocess.run(
        [sys.executable, str(MC_SCRIPT), "--version"],
        capture_output=True,
        text=True,
        check=True,
    )
    # Literal pin: a release that forgets to bump VERSION fails here.
    assert proc.stdout.strip() == "model-compare 0.2.7"


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
    # Entries cached under the pre-V2 key came from the retired endpoint
    # and must never be read.
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.delenv("AA_API_KEY", raising=False)
    legacy = tmp_path / "model-compare" / "aa-intelligence.json"
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
    entries, source, cached = mc.fetch_aa_entries(
        make_args(no_cache=False, cache_ttl=3600)
    )
    assert (entries, source, cached) == ([], None, False)


def test_fetch_aa_entries_writes_v2_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.delenv("AA_API_KEY", raising=False)
    monkeypatch.setattr(mc, "fetch_json", lambda *a, **k: aa_page([aa_item()]))
    mc.fetch_aa_entries(
        make_args(no_cache=False, cache_ttl=3600, aa_api_key="dummy-key")
    )
    cache_dir = tmp_path / "model-compare"
    assert (cache_dir / "aa-intelligence-v2.json").exists()
    assert not (cache_dir / "aa-intelligence.json").exists()


def _dir_snapshot(path):
    if not path.is_dir():
        return None
    return {p.name: (p.stat().st_mtime_ns, p.stat().st_size) for p in path.iterdir()}


def test_cache_writes_land_in_isolated_dir_not_real_cache(monkeypatch, tmp_path):
    # No explicit XDG_CACHE_HOME here: the autouse isolation fixture alone
    # must redirect the save that fetch_aa_entries makes on success.
    real = Path(os.path.expanduser("~")) / ".cache" / "model-compare"
    before = _dir_snapshot(real)
    path = Path(mc.cache_path("aa-intelligence-v2"))
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

    req = urllib.request.Request(old_url, headers={"x-api-key": "secret"})
    handler = mc.StripApiKeyRedirectHandler()
    return handler.redirect_request(req, None, 302, "Found", {}, new_url)


def test_redirect_same_host_keeps_api_key():
    new = _redirect("https://api.example.test/a", "https://api.example.test/b")
    assert new.get_header("X-api-key") == "secret"


def test_redirect_cross_host_strips_api_key():
    new = _redirect("https://api.example.test/a", "https://evil.example.test/b")
    assert new is not None
    assert not new.has_header("X-api-key")
    assert new.full_url == "https://evil.example.test/b"


def test_redirect_scheme_downgrade_strips_api_key():
    new = _redirect("https://api.example.test/a", "http://api.example.test/b")
    assert not new.has_header("X-api-key")


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
# sources.aa.fallback: what the AA API fallback yielded, unmasked
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "aa_source, fallback",
    [("AA API v2", "api"), (None, "none")],
)
def test_catalog_aa_fallback_tracks_aa_source(aa_source, fallback):
    # with OpenRouter benchmarks matched, mode stays "openrouter"
    doc = build_doc(aa_source=aa_source)
    assert doc["sources"]["aa"]["mode"] == "openrouter"
    assert doc["sources"]["aa"]["fallback"] == fallback
    # without them, mode and fallback agree
    doc = build_doc(aa_source=aa_source, aa_by_id={})
    assert doc["sources"]["aa"]["mode"] == fallback
    assert doc["sources"]["aa"]["fallback"] == fallback


def test_catalog_aa_fallback_not_masked_by_openrouter_matches():
    # Regression: the publish gate read mode alone, and any OpenRouter
    # match turned a failed AA API into mode "openrouter".
    doc = build_doc(aa_source=None)
    assert doc["sources"]["aa"]["matched_openrouter"] > 0
    assert doc["sources"]["aa"]["mode"] == "openrouter"
    assert doc["sources"]["aa"]["fallback"] == "none"
    bsd.validate_catalog(doc)  # must not raise
