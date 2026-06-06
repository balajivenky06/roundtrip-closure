"""
ollama_client.py — thin wrapper around the Ollama Python client.

Responsibilities:
    1. Translate (ModelSpec, prompt, generation parameters) → Ollama API call.
    2. Apply retry + back-off on transient failures (timeouts, model swap,
       network blips).
    3. Hook into closure_cache so identical (model, prompt) calls are
       served from disk instead of re-running inference.
    4. Log call latency, token counts, and cache-hit flag.

Public API:
    - LLMResponse — dataclass capturing one call's output + metadata
    - call_llm(model, prompt, ...) → LLMResponse
    - ensure_model_available(model) → bool (raises if not pulled)
    - list_available_models() → list[str]
"""

from __future__ import annotations
import logging
import os
import re
import time
from dataclasses import asdict, dataclass
from typing import Optional

try:
    import ollama
except ImportError as e:                                           # pragma: no cover
    raise ImportError(
        "The 'ollama' Python package is required. Install with: pip install ollama"
    ) from e

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


logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Module-level Ollama client
# ──────────────────────────────────────────────────────────────────────
_OLLAMA_HOST: str = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
_client = ollama.Client(host=_OLLAMA_HOST)


# Errors we want to retry on (network / transient server hiccups)
_RETRYABLE_ERRORS: tuple[type[Exception], ...] = (
    ConnectionError,
    TimeoutError,
    OSError,
)
# Ollama-specific transient errors live under ollama.ResponseError;
# include conditionally because the symbol moved between minor versions.
if hasattr(ollama, "ResponseError"):
    _RETRYABLE_ERRORS = _RETRYABLE_ERRORS + (ollama.ResponseError,)


# ──────────────────────────────────────────────────────────────────────
# Public types
# ──────────────────────────────────────────────────────────────────────
@dataclass
class LLMResponse:
    """Structured result of one Ollama call."""
    text: str
    model_tag: str
    prompt_tokens: int
    completion_tokens: int
    elapsed_s: float
    cache_hit: bool
    finish_reason: str               # "stop" | "length" | "error"
    error: Optional[str] = None


# ──────────────────────────────────────────────────────────────────────
# Internal: retry-wrapped chat call
# ──────────────────────────────────────────────────────────────────────
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type(_RETRYABLE_ERRORS),
    reraise=True,
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
def _ollama_chat(model_tag: str, messages: list[dict], options: dict) -> dict:
    """Single Ollama chat call with retry. Raises on permanent failure."""
    return _client.chat(model=model_tag, messages=messages, options=options)


# ──────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────
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
    Single Ollama call with retry + caching.

    Args:
        model:         ModelSpec from config.py
        prompt:        user-role content
        system_prompt: optional system-role content
        role_hint:     stage identifier — "L_spec" / "L_test" / "L_code" /
                       "judge" / "paraphrase" / etc. Influences the cache
                       key (so different stages with identical prompts
                       cache independently) and appears in logs.
        temperature ... max_tokens: generation parameters (defaults come
                       from config.py).
        use_cache:     if True, check the disk cache first; on hit return
                       the cached LLMResponse with cache_hit=True.

    Returns:
        LLMResponse — success populates `text`; error sets `finish_reason="error"`
        and captures the message in `error`.
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

    # ── Cache lookup ──────────────────────────────────────────────────
    if use_cache:
        cached = closure_cache.get(cache_key)
        if cached is not None:
            # Guard against schema drift: only pass fields LLMResponse knows about
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

    # ── Build the request ─────────────────────────────────────────────
    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    options = {
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "repeat_penalty": repeat_penalty,
        "num_ctx": num_ctx,
        "num_predict": max_tokens,
    }

    # ── Execute ───────────────────────────────────────────────────────
    t0 = time.perf_counter()
    try:
        result = _ollama_chat(model.ollama_tag, messages, options)
    except Exception as exc:                                       # pragma: no cover
        elapsed = time.perf_counter() - t0
        logger.error(
            f"LLM call failed: model={model.ollama_tag}, "
            f"role={role_hint}, elapsed={elapsed:.2f}s, error={exc}"
        )
        return LLMResponse(
            text="",
            model_tag=model.ollama_tag,
            prompt_tokens=0,
            completion_tokens=0,
            elapsed_s=elapsed,
            cache_hit=False,
            finish_reason="error",
            error=str(exc),
        )
    elapsed = time.perf_counter() - t0

    # ── Parse the response ────────────────────────────────────────────
    text = _extract_text(result)
    prompt_tokens = int(_get_field(result, "prompt_eval_count", 0) or 0)
    completion_tokens = int(_get_field(result, "eval_count", 0) or 0)
    finish_reason = _determine_finish_reason(result, completion_tokens, max_tokens)

    response = LLMResponse(
        text=text,
        model_tag=model.ollama_tag,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        elapsed_s=elapsed,
        cache_hit=False,
        finish_reason=finish_reason,
        error=None,
    )

    # ── Persist on success ────────────────────────────────────────────
    # Skip the cache write for responses that are almost certainly garbage:
    #   - "error"  : the call itself failed
    #   - "length" : output was truncated mid-reasoning; usually unparseable
    #   - empty text: model returned nothing useful (e.g. <think>-only output
    #                 that got stripped to "")
    # Caching these poisons the disk store for every future identical call
    # and was responsible for ~150 of the pilot's judge parse-failures.
    if (use_cache
            and finish_reason == "stop"
            and text.strip()):
        closure_cache.put(cache_key, asdict(response))

    logger.debug(
        f"LLM call OK: model={model.ollama_tag}, role={role_hint}, "
        f"elapsed={elapsed:.2f}s, in={prompt_tokens}, out={completion_tokens}, "
        f"finish={finish_reason}"
    )
    return response


def ensure_model_available(model: ModelSpec) -> bool:
    """
    Verify `model.ollama_tag` is pulled and ready. Raises RuntimeError
    with the `ollama pull` command if missing.
    """
    try:
        available = list_available_models()
    except Exception as exc:                                       # pragma: no cover
        raise RuntimeError(
            f"Cannot reach Ollama at {_OLLAMA_HOST}. "
            f"Start the server with: ollama serve\n"
            f"Underlying error: {exc}"
        )

    if model.ollama_tag in available:
        return True

    # Accept a base-name match (e.g. "qwen3.6:27b" vs "qwen3.6:latest")
    base = model.ollama_tag.split(":", 1)[0]
    if any(m.split(":", 1)[0] == base for m in available):
        logger.info(
            f"Model {model.ollama_tag} found via base-name match against "
            f"{', '.join(m for m in available if m.startswith(base))}"
        )
        return True

    raise RuntimeError(
        f"Model {model.ollama_tag!r} not pulled. Run:\n"
        f"    ollama pull {model.ollama_tag}\n"
        f"Currently available: {available}"
    )


def list_available_models() -> list[str]:
    """Return the list of Ollama tags currently pulled.

    Handles both response shapes the Python client has used:
      - v0.4+: Pydantic ListResponse with .models attribute, each Model has .model
      - v0.3 : dict {"models": [{"model" / "name": ...}, ...]}
    """
    try:
        result = _client.list()
    except Exception as exc:                                       # pragma: no cover
        logger.error(f"Failed to list Ollama models: {exc}")
        raise

    # New-style: Pydantic object with .models attribute
    if hasattr(result, "models"):
        return [m.model for m in result.models if getattr(m, "model", "")]

    # Old-style: dict with "models" key
    if isinstance(result, dict):
        tags: list[str] = []
        for m in result.get("models", []):
            if isinstance(m, dict):
                tag = m.get("model") or m.get("name") or ""
                if tag:
                    tags.append(tag)
        return tags

    return []


# ──────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────
def _get_field(obj, name: str, default=None):
    """Get a field from either a Pydantic model (attribute) or a dict (key)."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _extract_text(result) -> str:
    """Pull assistant text from Ollama response (Pydantic v0.4+ or dict v0.3-).

    Strips inline <think>...</think> tags (older reasoning models like
    DeepSeek-R1 v1 use these inside `content`). For newer reasoning SLMs
    (qwen3.6:27b), Ollama splits reasoning into `message.thinking` and the
    answer into `message.content` — we return content only, but warn when
    content is empty while thinking is populated (signals max_tokens was
    too small; reasoning consumed the budget). See pilot post-mortem.
    """
    msg = _get_field(result, "message", None)
    content = ""
    thinking = ""
    if msg is not None:
        content = _get_field(msg, "content", "") or ""
        thinking = _get_field(msg, "thinking", "") or ""
    if not content:
        # Generate API (legacy) uses 'response' field at the top level
        content = _get_field(result, "response", "") or ""

    cleaned = _THINK_BLOCK_RE.sub("", content).strip()

    # Diagnostic: budget-exhaustion fingerprint
    if not cleaned and thinking.strip():
        logger.warning(
            f"Response had empty content but {len(thinking)} chars of "
            f"`message.thinking` — likely max_tokens budget exhausted "
            f"during reasoning. Bump max_tokens for this stage."
        )
    return cleaned


def _determine_finish_reason(result, completion_tokens: int, max_tokens: int) -> str:
    """Pick a reason: prefer Ollama's done_reason, else infer from token count."""
    explicit = _get_field(result, "done_reason", None)
    if explicit in {"stop", "length", "error"}:
        return explicit
    if max_tokens > 0 and completion_tokens >= max_tokens:
        return "length"
    return "stop"


# ──────────────────────────────────────────────────────────────────────
# CLI sanity check — runs only if Ollama is up + one model is pulled
# ──────────────────────────────────────────────────────────────────────
def _self_test() -> None:                                          # pragma: no cover
    """Smoke-test the live Ollama connection. Skip gracefully if unavailable."""
    print(f"Connecting to Ollama at {_OLLAMA_HOST}…")
    try:
        models = list_available_models()
    except Exception as exc:
        print(f"  ✗ Cannot reach Ollama: {exc}")
        print("    To start: `ollama serve` in another terminal.")
        return

    if not models:
        print("  ✗ No models pulled. Try: `ollama pull llama3.2:3b`")
        return

    print(f"  ✓ Ollama reachable; {len(models)} model(s) pulled.")
    for m in models[:5]:
        print(f"      - {m}")

    # Round-trip a tiny call on the smallest pulled model
    from config import MODELS_BY_OLLAMA_TAG, LLAMA_3_2_3B
    target = LLAMA_3_2_3B if LLAMA_3_2_3B.ollama_tag in models else None
    if target is None:
        # Pick any pulled model that's also in our lineup
        for tag in models:
            if tag in MODELS_BY_OLLAMA_TAG:
                target = MODELS_BY_OLLAMA_TAG[tag]
                break
    if target is None:
        print("  ! None of the lineup models is pulled yet. Skipping round-trip test.")
        return

    print(f"\nRound-trip test on {target.ollama_tag}…")
    closure_cache.clear()  # Ensure first call is a real LLM invocation

    import time
    wall_start = time.perf_counter()
    resp = call_llm(target, "Say 'hi' in one word.",
                    role_hint="self_test", max_tokens=16, use_cache=True)
    wall_first = time.perf_counter() - wall_start
    print(f"  text={resp.text.strip()[:60]!r}")
    print(f"  cache_hit={resp.cache_hit}, finish={resp.finish_reason}, "
          f"wall_clock={wall_first:.3f}s (inference={resp.elapsed_s:.3f}s)")
    print(f"  prompt_tokens={resp.prompt_tokens}, "
          f"completion_tokens={resp.completion_tokens}")

    if not resp.text.strip():
        print("\n✗ FAIL: response text is empty (parsing bug?)")
        return
    if resp.cache_hit:
        print("\n✗ FAIL: first call should not be a cache hit")
        return

    # Second call: same args → should hit the disk cache (wall-clock ≈ ms)
    wall_start = time.perf_counter()
    resp2 = call_llm(target, "Say 'hi' in one word.",
                     role_hint="self_test", max_tokens=16, use_cache=True)
    wall_second = time.perf_counter() - wall_start
    print(f"\nSecond call: cache_hit={resp2.cache_hit}, "
          f"wall_clock={wall_second*1000:.2f}ms")
    print(f"  speed-up: {wall_first / max(wall_second, 0.0001):.0f}x")

    assert resp2.cache_hit is True, "Expected second call to be a cache hit"
    assert resp2.text == resp.text, "Cached text differs from original"
    assert wall_second < 1.0, f"Cache lookup too slow: {wall_second*1000:.0f}ms"
    print("\n✓ ollama_client self-test passed.")


if __name__ == "__main__":
    _self_test()
