"""
llm_dispatch.py — thin facade that routes call_llm(model, ...) to the
right provider client based on model.provider.

Adds provider dispatch without renaming ollama_client (used by dozens of
imports) or duplicating call_llm signatures.

Usage anywhere a call was previously `ollama_client.call_llm(model, ...)`:
    import llm_dispatch
    llm_dispatch.call_llm(model, ...)

Providers:
    "ollama"     → ollama_client (default; local Ollama runtime)
    "openrouter" → openrouter_client (OpenRouter unified API for Claude,
                                       GPT-4o-mini, Llama-70B, etc.)
"""
from __future__ import annotations

import logging
from typing import Optional

from config import ModelSpec
from ollama_client import LLMResponse
import ollama_client
import openrouter_client

logger = logging.getLogger(__name__)


def _provider_of(model: ModelSpec) -> str:
    """Read model.provider; default to 'ollama' for backward compat."""
    return getattr(model, "provider", "ollama")


def call_llm(model: ModelSpec, prompt: str, **kwargs) -> LLMResponse:
    """Route to the right client based on model.provider."""
    prov = _provider_of(model)
    if prov == "openrouter":
        return openrouter_client.call_llm(model, prompt, **kwargs)
    if prov == "ollama":
        return ollama_client.call_llm(model, prompt, **kwargs)
    raise ValueError(
        f"Unknown provider {prov!r} for model {model.ollama_tag}. "
        f"Expected one of: 'ollama', 'openrouter'."
    )


def ensure_model_available(model: ModelSpec) -> bool:
    prov = _provider_of(model)
    if prov == "openrouter":
        return openrouter_client.ensure_model_available(model)
    if prov == "ollama":
        return ollama_client.ensure_model_available(model)
    raise ValueError(
        f"Unknown provider {prov!r} for model {model.ollama_tag}."
    )
