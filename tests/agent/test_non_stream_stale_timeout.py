"""Tests for the non-stream stale-call detector context estimator.

Covers:
- ``estimate_request_context_tokens`` for Chat Completions, Responses API,
  bare lists, and mixed-shape dicts.
- ``AIAgent._compute_non_stream_stale_timeout`` with both legacy ``messages``
  list and full ``api_kwargs`` dicts.
- The May 2026 default-base change (300s -> 90s) and the lowered
  context-tier ceilings (450/600 -> 150/240).
"""

from __future__ import annotations

from pathlib import Path



def _write_config(tmp_path: Path, body: str) -> None:
    hermes_home = tmp_path
    (hermes_home / "config.yaml").write_text(body or "{}\n", encoding="utf-8")


def _make_agent(tmp_path: Path, **overrides):
    from run_agent import AIAgent
    kwargs = dict(
        model="gpt-5.5",
        provider="openai-codex",
        api_key="sk-dummy",
        base_url="https://chatgpt.com/backend-api/codex",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        platform="cli",
    )
    kwargs.update(overrides)
    return AIAgent(**kwargs)


# ── estimator ──────────────────────────────────────────────────────────────




def test_estimator_responses_api_input():
    from agent.chat_completion_helpers import estimate_request_context_tokens
    payload = {
        "model": "gpt-5.5",
        "instructions": "i" * 1000,
        "input": "x" * 4000,
        "tools": [{"name": "t", "description": "d" * 200}],
    }
    # input(4000) + instructions(1000) + tools (~stringified) -> well over 1000 tokens
    tokens = estimate_request_context_tokens(payload)
    assert tokens >= 1200, f"Responses API estimator returned {tokens}"






def test_estimator_empty_inputs():
    from agent.chat_completion_helpers import estimate_request_context_tokens
    assert estimate_request_context_tokens({}) == 0
    assert estimate_request_context_tokens([]) == 0
    assert estimate_request_context_tokens(None) == 0




# ── default base + tier scaling ────────────────────────────────────────────


def test_default_base_is_90s(monkeypatch, tmp_path):
    """Default base stale timeout dropped from 300s to 90s (May 2026)."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("", encoding="utf-8")
    monkeypatch.delenv("HERMES_API_CALL_STALE_TIMEOUT", raising=False)
    _write_config(tmp_path, "")

    agent = _make_agent(tmp_path)
    base, implicit = agent._resolved_api_call_stale_timeout_base()
    assert base == 90.0
    assert implicit is True










def test_explicit_user_config_overrides_default(monkeypatch, tmp_path):
    """If the user explicitly sets a stale_timeout, the new defaults don't apply."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("", encoding="utf-8")
    _write_config(tmp_path, """\
providers:
  openai-codex:
    stale_timeout_seconds: 1800
""")
    monkeypatch.delenv("HERMES_API_CALL_STALE_TIMEOUT", raising=False)

    import importlib
    from hermes_cli import timeouts as to_mod
    importlib.reload(to_mod)

    agent = _make_agent(tmp_path)
    assert agent._compute_non_stream_stale_timeout({"input": "hi"}) == 1800.0


# ── openai-codex gateway-scale stale floor ────────────────────────────────




def test_openai_codex_stale_floor_tiers():
    from agent.chat_completion_helpers import openai_codex_stale_timeout_floor

    assert openai_codex_stale_timeout_floor(55_000) == 900.0
    assert openai_codex_stale_timeout_floor(120_000) == 1200.0


# ── local-endpoint reasoning-floor disarm (local LLM stale-kill) ──────────




def _make_reasoning_agent(tmp_path: Path, base_url: str):
    """Agent whose model carries a reasoning stale floor (deepseek-v4-flash -> 600s)."""
    return _make_agent(
        tmp_path,
        model="deepseek-v4-flash",
        provider="openai-codex",
        base_url=base_url,
    )


def test_local_reasoning_endpoint_disarms_stale_detector(monkeypatch, tmp_path):
    """A reasoning model on a genuinely-local endpoint must not inherit the hosted-cloud
    reasoning floor (that floor exists for cloud gateways that idle-kill mid-think); the
    local-endpoint short-circuit disables the stale detector instead of killing a slow
    self-hosted prefill. Regression for the local-LLM stale-kill (#104402)."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("", encoding="utf-8")
    monkeypatch.delenv("HERMES_API_CALL_STALE_TIMEOUT", raising=False)
    _write_config(tmp_path, "")

    agent = _make_reasoning_agent(tmp_path, "http://192.168.1.99:8080/v1")
    # Reasoning floor still resolves (and stays non-explicit -> yields to run budget) ...
    base, implicit = agent._resolved_api_call_stale_timeout_base()
    assert base == 600.0 and implicit is False
    # ... but on a local endpoint with no explicit config the detector is disarmed.
    assert agent._compute_non_stream_stale_timeout({"input": "hi"}) == float("inf")


def test_cloud_reasoning_endpoint_keeps_reasoning_floor(monkeypatch, tmp_path):
    """A reasoning model on a hosted endpoint keeps the reasoning stale floor (no disarm)."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("", encoding="utf-8")
    monkeypatch.delenv("HERMES_API_CALL_STALE_TIMEOUT", raising=False)
    _write_config(tmp_path, "")

    agent = _make_reasoning_agent(tmp_path, "https://api.deepseek.com/v1")
    assert agent._compute_non_stream_stale_timeout({"input": "hi"}) == 600.0


def test_local_reasoning_endpoint_explicit_config_still_wins(monkeypatch, tmp_path):
    """An explicitly-configured stale_timeout_seconds is NOT disarmed on a local endpoint."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("", encoding="utf-8")
    monkeypatch.delenv("HERMES_API_CALL_STALE_TIMEOUT", raising=False)
    # Simulate a provider-level explicit config (as the runtime resolves it for the test agent).
    monkeypatch.setenv("HERMES_API_CALL_STALE_TIMEOUT", "360000")

    agent = _make_reasoning_agent(tmp_path, "http://192.168.1.99:8080/v1")
    assert agent._compute_non_stream_stale_timeout({"input": "hi"}) == 360000.0
