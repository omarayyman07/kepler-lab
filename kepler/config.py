"""Model IDs and API clients for Meta Model API."""

from __future__ import annotations

import os

BASE_URL = os.environ.get("META_BASE_URL", "https://api.meta.ai/v1")
API_KEY_ENV = "MODEL_API_KEY"

SAM_MODEL = os.environ.get("KEPLER_SAM_MODEL", "sam-3.1")
# Fallback if muse-wonder-1.0 fails on tool calling or is too slow in the loop:
# KEPLER_SCIENTIST_MODEL=muse-spark-1.3
SCIENTIST_MODEL = os.environ.get("KEPLER_SCIENTIST_MODEL", "muse-wonder-1.0")
# A/B on synthetic clips with known verdicts (scripts/ab_technician.py, 5 runs each):
# muse-wonder-1.0 10/10 correct; muse-spark-1.3 7/10 (it rejected the clean clip 3/5,
# citing a second bob that is not there).
QC_MODEL = os.environ.get("KEPLER_QC_MODEL", "muse-wonder-1.0")

# USD per 1M tokens (input, cached input, output), dev.meta.ai/docs/pricing-rate-limits.
# SAM 3.1 is $0.20 per 1,000 frames (50-frame minimum).
PRICES = {
    "muse-spark-1.3": (1.25, 0.15, 4.25),
    "muse-wonder-1.0": (4.00, 0.20, 20.00),
}


SAM_USD_PER_FRAME = 0.20 / 1000
SAM_MIN_FRAMES = 50

# Live spend in this process, by role. Cached replays add nothing. The scientist's
# spending cap checks the total, because in a real session SAM frames dominate.
SPEND = {"scientist": 0.0, "technician": 0.0, "sam": 0.0}


def record_spend(role: str, usd: float) -> None:
    SPEND[role] = SPEND.get(role, 0.0) + usd


def total_spend() -> float:
    return sum(SPEND.values())


def sam_cost(n_frames: int) -> float:
    """One SAM video request, billed on every frame of the clip sent (conservative)."""
    return max(n_frames, SAM_MIN_FRAMES) * SAM_USD_PER_FRAME


def response_cost(model: str, usage: dict) -> float:
    """Dollar cost of one Responses call from {"input_tokens", "output_tokens"}."""
    if not usage or model not in PRICES:
        return 0.0
    p_in, _, p_out = PRICES[model]
    return (usage.get("input_tokens", 0) * p_in + usage.get("output_tokens", 0) * p_out) / 1e6


def chat_cost(model: str, usage) -> float:
    """Dollar cost of one Chat Completions call from its `usage` (0 if unpriced)."""
    if usage is None or model not in PRICES:
        return 0.0
    p_in, p_cached, p_out = PRICES[model]
    details = getattr(usage, "prompt_tokens_details", None)
    cached = (getattr(details, "cached_tokens", 0) or 0) if details is not None else 0
    return ((usage.prompt_tokens - cached) * p_in + cached * p_cached
            + usage.completion_tokens * p_out) / 1e6


def _api_key() -> str:
    key = os.environ.get(API_KEY_ENV)
    if not key:
        raise RuntimeError(
            f"Set {API_KEY_ENV} to your Meta Model API key "
            "(dev.meta.ai -> Dashboard -> API keys)."
        )
    return key


def client():
    from openai import OpenAI

    return OpenAI(base_url=BASE_URL, api_key=_api_key())


def async_client():
    from openai import AsyncOpenAI

    return AsyncOpenAI(base_url=BASE_URL, api_key=_api_key())
