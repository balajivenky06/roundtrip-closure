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
        model_tag       ||  "\\x00"  ||
        role_hint       ||  "\\x00"  ||  # "L_spec" / "L_test" / "L_code" / "judge"
        prompt          ||  "\\x00"  ||
        system_prompt   ||  "\\x00"  ||  # empty string if None
        f"{temperature:.4f}|{top_p:.4f}|{top_k}|{num_ctx}|{max_tokens}"
    )

Cache value:
    JSON-serialised LLMResponse dict (from ollama_client.LLMResponse).

Storage layout:
    checkpoints/cache/<first-2-hex-chars-of-key>/<full-hash>.json

    Sharding by 2-hex-char prefix → 256 shard directories, keeps
    per-directory file counts manageable for ~20 k entries.
"""

from __future__ import annotations
import hashlib
import json
import logging
import os
import shutil
from pathlib import Path
from threading import Lock
from typing import Optional

from config import CACHE_DIR

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Process-lifetime statistics (used by §4.7 cache-hit-rate reporting)
# ──────────────────────────────────────────────────────────────────────
_stats_lock = Lock()
_hits = 0
_misses = 0


def _hex_safe(key: str) -> bool:
    """SHA-256 keys are 64 lowercase hex characters."""
    return len(key) == 64 and all(c in "0123456789abcdef" for c in key)


# ──────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────
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

    Deterministic: same arguments always produce the same key, so cache
    lookups are independent of construction order.
    """
    sp = system_prompt if system_prompt is not None else ""
    params_str = f"{temperature:.4f}|{top_p:.4f}|{top_k}|{num_ctx}|{max_tokens}"
    components = [model_tag, role_hint, prompt, sp, params_str]
    canonical = "\x00".join(components).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def cache_path(key: str) -> Path:
    """Resolve a cache key to its on-disk location (sharded by first 2 hex chars)."""
    if not _hex_safe(key):
        raise ValueError(f"Invalid cache key (must be 64-char hex): {key!r}")
    return CACHE_DIR / key[:2] / f"{key}.json"


def get(key: str) -> Optional[dict]:
    """
    Read the cached LLMResponse dict for `key`. Returns None on miss.

    Counts hits/misses for stats(); JSON-decode errors are treated as
    cache misses (the stale entry is left in place for manual inspection).
    """
    global _hits, _misses
    path = cache_path(key)
    if not path.exists():
        with _stats_lock:
            _misses += 1
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        with _stats_lock:
            _hits += 1
        return data
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Failed to read cache entry {key[:8]}…: {e}")
        with _stats_lock:
            _misses += 1
        return None


def put(key: str, response_dict: dict) -> None:
    """
    Atomically write the serialised LLMResponse to disk under `key`.

    Uses a `.tmp` rename to avoid partial writes on crashes.
    """
    path = cache_path(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".json.tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(response_dict, f, ensure_ascii=False)
        os.replace(tmp_path, path)
    except OSError as e:
        logger.error(f"Failed to write cache entry {key[:8]}…: {e}")
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def stats() -> dict:
    """
    Return {"hits", "misses", "hit_rate", "size_mb", "entry_count"}
    over the lifetime of this process + the on-disk store.
    """
    with _stats_lock:
        h, m = _hits, _misses
    total_files = 0
    total_size = 0
    if CACHE_DIR.exists():
        for shard_dir in CACHE_DIR.iterdir():
            if shard_dir.is_dir():
                for entry in shard_dir.iterdir():
                    if entry.suffix == ".json":
                        total_files += 1
                        try:
                            total_size += entry.stat().st_size
                        except OSError:
                            pass
    return {
        "hits": h,
        "misses": m,
        "hit_rate": h / max(h + m, 1),
        "size_mb": round(total_size / (1024 * 1024), 3),
        "entry_count": total_files,
    }


def clear() -> None:
    """Delete all on-disk cache entries and reset in-process counters."""
    global _hits, _misses
    if CACHE_DIR.exists():
        for shard_dir in CACHE_DIR.iterdir():
            if shard_dir.is_dir():
                shutil.rmtree(shard_dir, ignore_errors=True)
    with _stats_lock:
        _hits = 0
        _misses = 0
    logger.info("Cache cleared.")


# ──────────────────────────────────────────────────────────────────────
# Sanity self-test (run as `python closure_cache.py`)
# ──────────────────────────────────────────────────────────────────────
def _self_test() -> None:
    """Quick correctness check — not a substitute for the real test suite."""
    import tempfile

    # Test 1: make_key is deterministic
    k1 = make_key("foo:1b", "L_spec", "hello", temperature=0.2)
    k2 = make_key("foo:1b", "L_spec", "hello", temperature=0.2)
    assert k1 == k2, "make_key not deterministic"
    print(f"  ✓ make_key deterministic — {k1[:16]}…")

    # Test 2: different inputs produce different keys
    k3 = make_key("foo:1b", "L_test", "hello", temperature=0.2)
    assert k1 != k3, "make_key collision on role_hint"
    print("  ✓ make_key sensitive to role_hint")

    k4 = make_key("foo:1b", "L_spec", "hello", temperature=0.3)
    assert k1 != k4, "make_key collision on temperature"
    print("  ✓ make_key sensitive to temperature")

    # Test 3: round-trip put/get
    test_key = make_key("test:0b", "test", "round-trip", temperature=0.0)
    test_value = {"text": "echo", "model_tag": "test:0b",
                  "prompt_tokens": 5, "completion_tokens": 1,
                  "elapsed_s": 0.001, "cache_hit": False,
                  "finish_reason": "stop", "error": None}
    put(test_key, test_value)
    retrieved = get(test_key)
    assert retrieved == test_value, "Cache round-trip failed"
    print(f"  ✓ put/get round-trip — {len(json.dumps(test_value))} bytes")

    # Test 4: missing key returns None
    assert get(make_key("missing:0b", "x", "y")) is None
    print("  ✓ missing key returns None")

    # Test 5: stats() reports correctly
    s = stats()
    assert s["hits"] >= 1, f"hits not counted: {s}"
    assert s["misses"] >= 1, f"misses not counted: {s}"
    print(f"  ✓ stats() = hits={s['hits']}, misses={s['misses']}, "
          f"entries={s['entry_count']}, size={s['size_mb']} MB")

    # Cleanup the test entry
    cache_path(test_key).unlink(missing_ok=True)
    print("\n✓ closure_cache self-test passed.")


if __name__ == "__main__":
    _self_test()
