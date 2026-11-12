"""Real-LiteLLM compatibility checks for litellm.json and steering.

These tests are skipped when LiteLLM is not installed. The pinned CI job
installs `web/requirements-litellm.txt` and runs them on the Python version
the workflow pins. The loader test uses a subprocess because LiteLLM reads
LITELLM_MODEL_COST_MAP_URL at import time.
"""

from __future__ import annotations

import asyncio
import functools
import http.server
import os
import subprocess
import sys
import threading
from contextlib import contextmanager

import pytest

litellm = pytest.importorskip("litellm")
from litellm.integrations.custom_logger import CustomLogger

import build_litellm_artifacts as bla
from test_build_litellm_artifacts import _basic_catalog, _upstream_bytes
from test_litellm_contract import apply_steering, resolve_key

KEY = "openrouter/z-ai/glm-5.3"
HAIKU = "openrouter/anthropic/claude-haiku-5.5"
OFF_PEAK = "openrouter/tencent/hy3"


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


@contextmanager
def _serve(directory):
    handler = functools.partial(_QuietHandler, directory=str(directory))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/model_prices.json"
    finally:
        server.shutdown()
        thread.join(timeout=5)


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory):
    catalog = _basic_catalog()
    card, warnings = bla.build_rate_card(catalog)
    card_bytes = bla.serialize_readable(card)
    steering = bla.build_steering(catalog, card_bytes, "deadbee", warnings)
    merged, status = bla.build_merge_status(
        catalog, "deadbee", card_bytes, "ok", _upstream_bytes(), None, None
    )
    assert status["merged"]["status"] == "ok"
    root = tmp_path_factory.mktemp("litellm-compat")
    path = root / "model_prices.json"
    path.write_bytes(merged)
    return {"path": path, "steering": steering}


def _run_loader(url: str) -> subprocess.CompletedProcess:
    script = f"""
import math
import litellm

key = {KEY!r}
assert key in litellm.model_cost, "model key missing from loaded map"
info = litellm.get_model_info(key)
assert math.isclose(info["input_cost_per_token"], 0.022 / 1e6, rel_tol=1e-12)
assert math.isclose(info["output_cost_per_token"], 0.4 / 1e6, rel_tol=1e-12)
assert info["max_input_tokens"] == 1_048_576
assert info["max_output_tokens"] == 64_000

low = litellm.cost_per_token(model=key, prompt_tokens=100_000, completion_tokens=0)[0]
high = litellm.cost_per_token(model=key, prompt_tokens=100_001, completion_tokens=0)[0]
assert math.isclose(low, 0.0022, rel_tol=1e-9), low
assert math.isclose(high, 0.0500005, rel_tol=1e-9), high

off_peak = litellm.model_cost[{OFF_PEAK!r}]["off_peak_pricing"]
assert off_peak["hours_utc"] == ["16:00-00:00"]
print("ok")
"""
    env = os.environ.copy()
    env["LITELLM_MODEL_COST_MAP_URL"] = url
    # The fixture map is intentionally tiny; relax LiteLLM's bundled-map
    # integrity checks the same way its own integration tests do.
    env["MODEL_COST_MAP_MIN_MODEL_COUNT"] = "1"
    env["MODEL_COST_MAP_MAX_SHRINK_RATIO"] = "0"
    env.pop("LITELLM_LOCAL_MODEL_COST_MAP", None)
    return subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_litellm_loads_merged_map_and_prices_it(artifacts):
    with _serve(artifacts["path"].parent) as url:
        result = _run_loader(url)
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


class SteeringHook(CustomLogger):
    """Test-only consumer: the shape the documented proxy hook implements."""

    def __init__(self, steering, strict_models=()):
        self.steering = steering
        self.strict_models = set(strict_models)

    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        if call_type != "completion":
            return data
        model = data.get("model", "")
        key = resolve_key(model, self.steering)
        if key is None:
            return data
        return apply_steering(
            data, self.steering, model, strict=key in self.strict_models
        )


def _hook(artifacts, strict=False):
    return SteeringHook(artifacts["steering"], {KEY} if strict else set())


def test_steering_hook_injects_preference(artifacts):
    data = {"model": KEY}
    out = asyncio.run(
        _hook(artifacts).async_pre_call_hook(None, None, data, "completion")
    )
    entry = artifacts["steering"][KEY]
    assert out["extra_body"]["provider"] == entry["provider"]
    assert out["extra_body"]["reasoning"] == entry["reasoning"]["preferred"]


def test_steering_hook_injects_strict_profile(artifacts):
    data = {"model": KEY}
    out = asyncio.run(
        _hook(artifacts, strict=True).async_pre_call_hook(
            None, None, data, "completion"
        )
    )
    entry = artifacts["steering"][KEY]
    assert out["extra_body"]["provider"] == entry["provider_strict"]
    assert out["extra_body"]["provider"]["allow_fallbacks"] is False


def test_steering_hook_keeps_caller_values(artifacts):
    data = {
        "model": KEY,
        "extra_body": {
            "provider": {"order": ["mine"]},
            "reasoning": {"effort": "low"},
        },
    }
    out = asyncio.run(
        _hook(artifacts).async_pre_call_hook(None, None, data, "completion")
    )
    assert out["extra_body"]["provider"] == {"order": ["mine"]}
    assert out["extra_body"]["reasoning"] == {"effort": "low"}


def test_steering_hook_ignores_unknown_models(artifacts):
    data = {"model": "unknown/model"}
    out = asyncio.run(
        _hook(artifacts).async_pre_call_hook(None, None, data, "completion")
    )
    assert out == {"model": "unknown/model"}


def test_steering_hook_matches_contract_helper(artifacts):
    steering = artifacts["steering"]
    for strict in (False, True):
        via_hook = asyncio.run(
            _hook(artifacts, strict=strict).async_pre_call_hook(
                None, None, {"model": KEY}, "completion"
            )
        )
        via_helper = apply_steering({"model": KEY}, steering, KEY, strict=strict)
        assert via_hook["extra_body"] == via_helper["extra_body"]


def test_steering_hook_resolves_bare_alias(artifacts):
    via_bare = asyncio.run(
        _hook(artifacts).async_pre_call_hook(
            None, None, {"model": "z-ai/glm-5.3"}, "completion"
        )
    )
    via_full = asyncio.run(
        _hook(artifacts).async_pre_call_hook(None, None, {"model": KEY}, "completion")
    )
    assert via_bare["extra_body"] == via_full["extra_body"]
    assert via_bare["extra_body"]["provider"] == artifacts["steering"][KEY]["provider"]
    # Strict membership is keyed on the resolved key, so the bare alias
    # must pick up the strict profile too.
    strict_bare = asyncio.run(
        _hook(artifacts, strict=True).async_pre_call_hook(
            None, None, {"model": "z-ai/glm-5.3"}, "completion"
        )
    )
    assert (
        strict_bare["extra_body"]["provider"]
        == artifacts["steering"][KEY]["provider_strict"]
    )


def test_steering_hook_skips_reasoning_for_mandatory_none(artifacts):
    base = artifacts["steering"][KEY]
    steering = {
        KEY: {
            "provider": base["provider"],
            "provider_strict": base["provider_strict"],
            "reasoning": {"preferred": {"effort": "none"}, "mandatory": True},
        }
    }
    out = asyncio.run(
        SteeringHook(steering).async_pre_call_hook(
            None, None, {"model": KEY}, "completion"
        )
    )
    assert out["extra_body"]["provider"] == base["provider"]
    assert "reasoning" not in out["extra_body"]
