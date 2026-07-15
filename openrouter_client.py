"""
openrouter_client.py — OpenRouter API client mirroring ollama_client.call_llm.

Responsibilities:
    1. Translate (ModelSpec, prompt, generation parameters) → OpenRouter API call.
    2. Apply retry + back-off on 429 / 5xx.
    3. Hook into closure_cache so identical (model, role, prompt) calls are
       served from disk instead of re-hitting the API.
    4. Log call latency, token counts, and cache-hit flag.

Public API:
    - call_llm(model, prompt, ...) → LLMResponse
      Signature-compatible with ollama_client.call_llm so llm_dispatch.py
      can route between them via model.provider.
    - ensure_model_available(model) → bool
      For OpenRouter, this is a network reachability probe.

The LLMResponse dataclass is imported from ollama_client so both clients
produce identical downstream types.
"""
from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from typing import Optional

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

import closure_cache
from config import (
    ModelSpec,
    TEMPERATURE,
    TOP_P,
    TOP_K,
    REPEAT_PENALTY,
    NUM_CTX,
    MAX_OUTPUT_TOKENS,
)
from ollama_client import LLMResponse


logger = logging.getLogger(__name__)


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def _api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Export it before running "
            "any cell that references an OpenRouter-provider model."
        )
    return key


class _OpenRouterHTTPError(Exception):
    """Wrapper so tenacity retries only on retryable HTTP codes."""
    def __init__(self, code: int, body: str):
        super().__init__(f"HTTP {code}: {body[:200]}")
        self.code = code
        self.body = body


class _OpenRouterRateLimit(_OpenRouterHTTPError):
    pass


class _OpenRouterServerError(_OpenRouterHTTPError):
    pass


_RETRYABLE = (_OpenRouterRateLimit, _OpenRouterServerError,
              urllib.error.URLError, TimeoutError, ConnectionError)


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=2, max=60),
    retry=retry_if_exception_type(_RETRYABLE),
    reraise=True,
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
def _post_chat(payload: dict) -> dict:
    """Single OpenRouter chat call with retry on 429/5xx/network errors."""
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        OPENROUTER_URL,
        data=body,
        headers={
            "Authorization": f"Bearer {_api_key()}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/balajivenky06/roundtrip-closure",
            "X-Title": "roundtrip-closure pipeline",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        if e.code == 429:
            raise _OpenRouterRateLimit(e.code, body) from e
        if 500 <= e.code < 600:
            raise _OpenRouterServerError(e.code, body) from e
        # 4xx other than 429 = permanent, no retry
        raise _OpenRouterHTTPError(e.code, body) from e


def call_llm(
    model: ModelSpec,
    prompt: str,
    *,
    system_prompt: Optional[str] = None,
    role_hint: str = "generic",
    temperature: float = TEMPERATURE,
    top_p: float = TOP_P,
    top_k: int = TOP_K,
    repeat_penalty: float = REPEAT_PENALTY,
    num_ctx: int = NUM_CTX,
    max_tokens: int = MAX_OUTPUT_TOKENS,
    use_cache: bool = True,
) -> LLMResponse:
    """
    Signature-compatible with ollama_client.call_llm. See its docstring.

    Cache-key parity: uses the same closure_cache.make_key function with
    model.ollama_tag as the model identifier, so cache namespaces are
    disjoint between providers (an OpenRouter model tag will never collide
    with an Ollama model tag).
    """
    cache_key = closure_cache.make_key(
        model.ollama_tag,
        role_hint,
        prompt,
        system_prompt=system_prompt,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_ctx=num_ctx,
        max_tokens=max_tokens,
    )

    if use_cache:
        cached = closure_cache.get(cache_key)
        if cached is not None:
            allowed = {f for f in LLMResponse.__dataclass_fields__}
            filtered = {k: v for k, v in cached.items() if k in allowed}
            filtered["cache_hit"] = True
            try:
                return LLMResponse(**filtered)
            except TypeError as e:
                logger.warning(
                    f"Cached entry {cache_key[:8]}… has incompatible schema, "
                    f"falling through to live call: {e}"
                )

    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model.ollama_tag,   # slug like "anthropic/claude-sonnet-4.5"
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    # top_k / repeat_penalty are provider-specific; OpenRouter accepts them
    # but silently ignores for providers that don't support them.
    if top_k > 0:
        payload["top_k"] = top_k
    if repeat_penalty != 1.0:
        payload["frequency_penalty"] = repeat_penalty - 1.0  # rough mapping

    t0 = time.perf_counter()
    try:
        resp = _post_chat(payload)
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        logger.error(
            f"OpenRouter call failed: model={model.ollama_tag}, "
            f"role={role_hint}, elapsed={elapsed:.2f}s, error={exc}"
        )
        return LLMResponse(
            text="", model_tag=model.ollama_tag,
            prompt_tokens=0, completion_tokens=0,
            elapsed_s=elapsed, cache_hit=False,
            finish_reason="error", error=str(exc),
        )
    elapsed = time.perf_counter() - t0

    try:
        choice = resp["choices"][0]
        text = choice["message"]["content"] or ""
        finish = choice.get("finish_reason", "stop")
        usage = resp.get("usage", {}) or {}
        response = LLMResponse(
            text=text,
            model_tag=model.ollama_tag,
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            elapsed_s=elapsed,
            cache_hit=False,
            finish_reason=finish,
            error=None,
        )
    except (KeyError, IndexError, TypeError) as exc:
        return LLMResponse(
            text="", model_tag=model.ollama_tag,
            prompt_tokens=0, completion_tokens=0,
            elapsed_s=elapsed, cache_hit=False,
            finish_reason="error",
            error=f"unparseable response: {exc}; raw={str(resp)[:200]}",
        )

    if use_cache and response.finish_reason != "error":
        cache_payload = {
            "text": response.text,
            "model_tag": response.model_tag,
            "prompt_tokens": response.prompt_tokens,
            "completion_tokens": response.completion_tokens,
            "elapsed_s": response.elapsed_s,
            "finish_reason": response.finish_reason,
        }
        try:
            closure_cache.put(cache_key, cache_payload)
        except Exception as e:                                       # pragma: no cover
            logger.warning(f"Cache write failed for {cache_key[:8]}…: {e}")

    return response


def ensure_model_available(model: ModelSpec) -> bool:
    """
    Verify the OpenRouter API is reachable and the model slug resolves.
    A single tiny call ("Reply: OK") is used as a probe. Costs ~$0.00005.
    """
    logger.info(f"Probing OpenRouter for {model.ollama_tag} …")
    resp = call_llm(
        model,
        "Reply with exactly: OK",
        role_hint="probe",
        max_tokens=5,
        use_cache=False,
    )
    if resp.finish_reason == "error":
        raise RuntimeError(
            f"OpenRouter model {model.ollama_tag} probe failed: {resp.error}"
        )
    return True
