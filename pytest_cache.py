"""
pytest_cache.py — disk-backed cache for pytest subprocess results.

Why this exists:
    `mutation_testing.evaluate_mutants` runs ~15 pytest subprocesses per
    function (one per mutant), and `_filter_passing_tests` runs one per
    test in the suite. Each call takes 1-2 seconds. Across the full
    20-cell × 150-function sweep that's ~50,000 pytest invocations.

    If Colab disconnects mid-mutation-evaluation, we lose every mutant
    result accumulated in memory — on restart we re-run all 15 mutants
    from scratch.

    This module gives every (test_code, function_code) pytest call the
    same disk-cache treatment that closure_cache.py gives LLM calls:
    SHA-256 keyed, atomic writes with fsync, sharded directory layout.

Cache key:
    SHA256(
        "pytest-v1\\0" ||
        test_code     ||  "\\0" ||
        function_code ||  "\\0" ||
        str(timeout)
    )

Cache value:
    {"result": "pass" | "fail" | "error" | "timeout", "ts": float}

Storage layout:
    checkpoints/pytest_cache/<first-2-hex-chars>/<full-hash>.json
"""

from __future__ import annotations
import hashlib
import json
import logging
import os
import shutil
import time
from pathlib import Path
from threading import Lock
from typing import Optional

from config import CHECKPOINTS_DIR


logger = logging.getLogger(__name__)

PYTEST_CACHE_DIR: Path = CHECKPOINTS_DIR / "pytest_cache"
PYTEST_CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────
# Process-lifetime stats
# ──────────────────────────────────────────────────────────────────────
_stats_lock = Lock()
_hits = 0
_misses = 0


# ──────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────
def make_key(test_code: str, function_code: str, timeout: int = 30) -> str:
    """Deterministic SHA-256 key over (test_code, function_code, timeout)."""
    parts = ["pytest-v1", test_code, function_code, str(timeout)]
    canonical = "\x00".join(parts).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def cache_path(key: str) -> Path:
    """Resolve a key to its on-disk location (sharded by first 2 hex chars)."""
    if len(key) != 64:
        raise ValueError(f"Invalid pytest cache key: {key!r}")
    return PYTEST_CACHE_DIR / key[:2] / f"{key}.json"


def get(key: str) -> Optional[str]:
    """Return the cached result string ('pass' / 'fail' / 'error' / 'timeout'),
    or None on miss."""
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
        return data.get("result")
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"pytest_cache read error for {key[:8]}…: {e}")
        with _stats_lock:
            _misses += 1
        return None


def put(key: str, result: str) -> None:
    """Atomically persist a pytest result. Flushes + fsyncs before rename so
    Drive lag can't lose the entry."""
    if result not in {"pass", "fail", "error", "timeout"}:
        logger.warning(f"pytest_cache.put: unusual result {result!r}; storing anyway")
    path = cache_path(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".json.tmp")
    payload = {"result": result, "ts": time.time()}
    try:
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())     # critical — defends against Drive sync lag
        os.replace(tmp_path, path)
    except OSError as e:
        logger.error(f"pytest_cache.put failed for {key[:8]}…: {e}")
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def stats() -> dict:
    """{"hits", "misses", "hit_rate", "size_mb", "entry_count"}."""
    with _stats_lock:
        h, m = _hits, _misses
    total_files = 0
    total_size = 0
    if PYTEST_CACHE_DIR.exists():
        for shard in PYTEST_CACHE_DIR.iterdir():
            if shard.is_dir():
                for entry in shard.iterdir():
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
    """Wipe all pytest cache entries + reset counters."""
    global _hits, _misses
    if PYTEST_CACHE_DIR.exists():
        for shard in PYTEST_CACHE_DIR.iterdir():
            if shard.is_dir():
                shutil.rmtree(shard, ignore_errors=True)
    with _stats_lock:
        _hits = 0
        _misses = 0
    logger.info("pytest_cache cleared.")


# ──────────────────────────────────────────────────────────────────────
# Self-test
# ──────────────────────────────────────────────────────────────────────
def _self_test() -> None:                                         # pragma: no cover
    print("=== pytest_cache self-test ===\n")

    k1 = make_key("def test(): pass", "def f(): return 1")
    k2 = make_key("def test(): pass", "def f(): return 1")
    assert k1 == k2, "make_key not deterministic"
    print(f"  ✓ make_key deterministic ({k1[:16]}…)")

    k3 = make_key("def test(): pass", "def f(): return 2")
    assert k1 != k3, "make_key collision on function_code"
    print("  ✓ make_key sensitive to function_code")

    put(k1, "pass")
    assert get(k1) == "pass"
    print("  ✓ put/get round-trip")

    assert get(make_key("missing", "missing")) is None
    print("  ✓ missing key returns None")

    s = stats()
    assert s["hits"] >= 1 and s["misses"] >= 1
    print(f"  ✓ stats() = hits={s['hits']}, misses={s['misses']}, "
          f"entries={s['entry_count']}, size={s['size_mb']} MB")

    cache_path(k1).unlink(missing_ok=True)
    print("\n✓ pytest_cache self-test passed.")


if __name__ == "__main__":
    _self_test()
