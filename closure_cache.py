"""
closure_cache.py — disk-backed cache for (model, prompt) -> LLMResponse.

Justification: across the 20-cell DOE x 150 functions x 3 paths x 2 LLM
calls = ~18,000 LLM invocations, many of them duplicates because the same
(model, prompt) tuple appears across different cells (e.g., every cell
that uses qwen3.6:27b as L_spec on function f produces the same docstring).

A simple disk-backed dict keyed on SHA256 of the canonicalised request
cuts ~30 % of redundant compute, dropping ~120 GPU-hours to ~80–90.

Cache key:
    SHA256(
        model_tag || "\\x00" ||
        role_hint || "\\x00" ||    # e.g. "L_spec" / "L_test" / "L_code" / "judge"
        prompt    || "\\x00" ||
        system_prompt_or_empty || "\\x00" ||
        f"{temperature:.4f}|{top_p:.4f}|{top_k}|{num_ctx}|{max_tokens}"
    )

Cache value:
    JSON-serialised LLMResponse (text + metadata).

Storage layout:
    checkpoints/cache/<first-2-hex-chars>/<full-hash>.json

Stub status: function signatures only.
"""

from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Optional

from config import CACHE_DIR


def make_key(
    model_tag: str,
    role_hint: str,
    prompt: str,
    *,
    system_prompt: Optional[str] = None,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 0,
    num_ctx: int = 0,
    max_tokens: int = 0,
) -> str:
    """
    Canonicalise + SHA256 the request parameters into a 64-char hex key.

    The key is deterministic — the same arguments always produce the same key,
    so cache lookups are order-independent.
    """
    raise NotImplementedError("Stub — implementation pending.")


def cache_path(key: str) -> Path:
    """Resolve a cache key to its on-disk location."""
    raise NotImplementedError("Stub — implementation pending.")


def get(key: str) -> Optional[dict]:
    """
    Read the cached LLMResponse for `key`. Returns the deserialised dict
    or None if the key is not present.

    The dict's shape matches LLMResponse.__dict__.
    """
    raise NotImplementedError("Stub — implementation pending.")


def put(key: str, response_dict: dict) -> None:
    """
    Write the serialised LLMResponse to disk under `key`.
    Idempotent — overwrites if the key already exists.
    """
    raise NotImplementedError("Stub — implementation pending.")


def stats() -> dict:
    """
    Return {"hits": int, "misses": int, "size_mb": float, "entry_count": int}
    over the lifetime of the current process.

    Used for the §4.7 "cache hit rate" reporting in the paper.
    """
    raise NotImplementedError("Stub — implementation pending.")


def clear() -> None:
    """Delete all cache entries. Used by tests and `--no-cache` runs."""
    raise NotImplementedError("Stub — implementation pending.")
