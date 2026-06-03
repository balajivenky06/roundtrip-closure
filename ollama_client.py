"""
ollama_client.py — thin wrapper around the Ollama Python client.

Responsibilities:
    1. Translate (ModelSpec, prompt, generation parameters) -> Ollama API call.
    2. Apply retry + back-off on transient failures (timeouts, model swap).
    3. Hook into closure_cache so identical (model, prompt) calls are cached.
    4. Log call latency, token counts, and cache-hit flag for later analysis.

Stub status: function signatures + module docstring; implementation TBD.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

from config import ModelSpec, TEMPERATURE, TOP_P, TOP_K, REPEAT_PENALTY, NUM_CTX, MAX_OUTPUT_TOKENS


@dataclass
class LLMResponse:
    """Structured result of one Ollama call."""
    text: str
    model_tag: str
    prompt_tokens: int
    completion_tokens: int
    elapsed_s: float
    cache_hit: bool
    finish_reason: str  # "stop", "length", "error"
    error: Optional[str] = None


def call_llm(
    model: ModelSpec,
    prompt: str,
    *,
    system_prompt: Optional[str] = None,
    temperature: float = TEMPERATURE,
    top_p: float = TOP_P,
    top_k: int = TOP_K,
    repeat_penalty: float = REPEAT_PENALTY,
    num_ctx: int = NUM_CTX,
    max_tokens: int = MAX_OUTPUT_TOKENS,
    use_cache: bool = True,
) -> LLMResponse:
    """
    Single Ollama chat-completion call with retry + caching.

    Args:
        model:           ModelSpec (from config.py)
        prompt:          user message
        system_prompt:   optional system message (default: None)
        temperature/top_p/top_k/repeat_penalty: generation params
        num_ctx:         context window in tokens
        max_tokens:      max output tokens
        use_cache:       if True, look up (model, prompt) in closure_cache
                         before calling; if hit, return cached LLMResponse
                         with cache_hit=True.

    Returns:
        LLMResponse — structured result (success or error captured in fields)

    Implementation notes:
        - Use tenacity for retry: 3 attempts, exponential back-off,
          retry on (TimeoutError, ConnectionError, ollama.ResponseError).
        - Compute cache key via closure_cache.make_key(model.ollama_tag,
          prompt, system_prompt, temperature, ...).
        - Cache stores the serialised LLMResponse (text + metadata).
    """
    raise NotImplementedError("Stub — implementation pending.")


def ensure_model_available(model: ModelSpec) -> bool:
    """
    Verify that `model.ollama_tag` is pulled and ready in the local Ollama
    instance. If not present, optionally trigger `ollama pull`.

    Returns True if the model is ready; raises RuntimeError otherwise.
    """
    raise NotImplementedError("Stub — implementation pending.")


def list_available_models() -> list[str]:
    """Return the list of Ollama tags currently pulled. Wraps `ollama list`."""
    raise NotImplementedError("Stub — implementation pending.")
