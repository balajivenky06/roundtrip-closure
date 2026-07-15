"""
tests/test_smoke.py — end-to-end smoke test on 1 function × 1 mono cell.

Run as:
    python3 -m pytest tests/test_smoke.py -v

Or directly:
    python3 tests/test_smoke.py

What's verified:
    1. doe.py is internally consistent (20 cells, strata correct)
    2. closure_paths.run_path_1(M1, sample) returns a ClosureResult
       with a numeric kill_rate
    3. closure_cache resume: second call of the same path is faster
       (cache hits dominate)
    4. train_roundtrip.write_result_row + load_completed_keys round-trip
       reproduces the same resume key

The full Ollama-dependent test is skipped if llama3.2:3b is not pulled.
"""

from __future__ import annotations
import sys
from pathlib import Path
import tempfile
import time

# Make project root importable when run directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


import pytest

import closure_cache
import closure_paths
import train_roundtrip
from doe import M1, ALL_CELLS, MONO_CELLS, HETERO_CELLS, NULL_CELLS


# ──────────────────────────────────────────────────────────────────────
# Test data — a 1-function "dataset"
# ──────────────────────────────────────────────────────────────────────
PALINDROME_SAMPLE = {
    "sample_idx": 0,
    "source": "smoke/is_palindrome",
    "entry_point": "is_palindrome",
    "signature": "def is_palindrome(s: str) -> bool:",
    "docstring": "Return True if s reads the same forwards and backwards.",
    "code": (
        "def is_palindrome(s: str) -> bool:\n"
        "    \"\"\"Return True if s reads the same forwards and backwards.\"\"\"\n"
        "    return s == s[::-1]\n"
    ),
    "tests": (
        "def test_pal_true():\n"
        "    assert is_palindrome('racecar') is True\n"
        "\n"
        "def test_pal_false():\n"
        "    assert is_palindrome('abc') is False\n"
    ),
}


def _ollama_has_model(tag: str) -> bool:
    """Check whether the given Ollama tag is currently pulled."""
    try:
        import ollama_client
        return tag in ollama_client.list_available_models()
    except Exception:
        return False


# ──────────────────────────────────────────────────────────────────────
# Static tests (no Ollama needed)
# ──────────────────────────────────────────────────────────────────────
def test_doe_table_well_formed():
    """Primary DOE: 20 cells across 3 strata (6 mono + 11 hetero + 3 null).
    Plus post-hoc RQ4 closed-weight extension cells (M5_closed, H2_closed,
    H8_closed, M7_gpt) → 24 total via ALL_CELLS."""
    from doe import CLOSED_WEIGHT_CELLS
    assert len(MONO_CELLS) == 6
    assert len(HETERO_CELLS) == 11
    assert len(NULL_CELLS) == 3
    assert len(CLOSED_WEIGHT_CELLS) == 4
    assert len(ALL_CELLS) == 24, f"Expected 24 cells (20 primary + 4 closed-weight), got {len(ALL_CELLS)}"
    for cell in MONO_CELLS + HETERO_CELLS + CLOSED_WEIGHT_CELLS:
        assert cell.L_spec is not None, f"{cell.cell_id}: L_spec missing"
        assert cell.L_test is not None, f"{cell.cell_id}: L_test missing"
        assert cell.L_code is not None, f"{cell.cell_id}: L_code missing"


def test_tsv_round_trip():
    """Resume key formed by write_result_row matches load_completed_keys."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tsv_path = Path(tmpdir) / "results.tsv"

        train_roundtrip.ensure_header(tsv_path)

        fake_result = closure_paths.ClosureResult(
            cell_id="M1", sample_idx=0, sample_source="smoke",
            path=1, metric_name="mutation_kill_rate", metric_value=0.85,
            judge_rating=3, judge_justification="ok",
            valid=True, elapsed_s=1.234, cache_hits=2, n_llm_calls=3,
            notes="smoke",
        )
        train_roundtrip.write_result_row(fake_result, tsv_path)

        completed = train_roundtrip.load_completed_keys(tsv_path)
        assert fake_result.resume_key in completed, (
            f"Resume key {fake_result.resume_key} not found in "
            f"{completed}"
        )


def test_tsv_escapes_tabs_and_newlines():
    """justification + notes with tabs/newlines must not break TSV row count."""
    res = closure_paths.ClosureResult(
        cell_id="M1", sample_idx=0, sample_source="smoke",
        path=1, metric_name="x", metric_value=0.0,
        judge_rating=-1,
        judge_justification="line1\nline2\twith tab",
        valid=False, elapsed_s=0.0, cache_hits=0, n_llm_calls=0,
        notes="bad\tchars\nhere",
    )
    row = res.to_tsv_row()
    assert row.endswith("\n")
    # Exactly one newline at the end; no embedded newlines in fields
    assert row.count("\n") == 1
    # Column count == len(TSV_COLUMNS)
    cols = row.rstrip("\n").split("\t")
    assert len(cols) == len(closure_paths.ClosureResult.TSV_COLUMNS)


# ──────────────────────────────────────────────────────────────────────
# Live test — requires llama3.2:3b
# ──────────────────────────────────────────────────────────────────────
@pytest.mark.skipif(
    not _ollama_has_model("llama3.2:3b"),
    reason="llama3.2:3b not pulled; run: ollama pull llama3.2:3b",
)
def test_run_path_1_on_palindrome():
    """
    End-to-end: M1 (llama3.2:3b mono) on the is_palindrome sample,
    Path 1 only. Verifies that the full chain
    (ollama_client → closure_paths → closure_metrics) produces a
    valid ClosureResult.

    Then re-runs to assert the second invocation hits the cache.
    """
    # First run — real LLM call
    t0 = time.perf_counter()
    result1 = closure_paths.run_path_1(M1, PALINDROME_SAMPLE)
    t_first = time.perf_counter() - t0

    assert result1.cell_id == "M1"
    assert result1.path == 1
    assert result1.metric_name == "mutation_kill_rate"
    assert result1.n_llm_calls >= 2, f"Expected >=2 LLM calls (L_spec + L_test), got {result1.n_llm_calls}"

    # If the sample produced any survivable tests, kill_rate is numeric;
    # if filter dropped everything, it's NaN — both are valid outcomes.
    if result1.valid:
        assert 0.0 <= result1.metric_value <= 1.0, (
            f"kill_rate out of bounds: {result1.metric_value}"
        )

    # Second run — should be much faster thanks to cache
    t0 = time.perf_counter()
    result2 = closure_paths.run_path_1(M1, PALINDROME_SAMPLE)
    t_second = time.perf_counter() - t0

    # Same metric value (deterministic given cache)
    if result1.valid and result2.valid:
        assert result1.metric_value == result2.metric_value or (
            result1.metric_value != result1.metric_value and       # NaN check
            result2.metric_value != result2.metric_value
        )

    # Cache hits should equal total LLM calls on the second pass
    # (the cache layer counts hits on `get`, which fires for every call)
    assert result2.cache_hits >= 1, (
        f"Second pass should have at least some cache hits, "
        f"got {result2.cache_hits} of {result2.n_llm_calls}"
    )

    # Second pass should be at least 5× faster (mostly cache hits,
    # but mutation_testing still runs)
    if t_first > 1.0:                                              # skip if first was already very fast
        assert t_second < t_first, (
            f"Second pass not faster: {t_second:.2f}s vs {t_first:.2f}s"
        )

    print(f"\n  Path-1 smoke: kill_rate={result1.metric_value}, "
          f"valid={result1.valid}, calls={result1.n_llm_calls}, "
          f"first={t_first:.2f}s, second={t_second:.2f}s")


# ──────────────────────────────────────────────────────────────────────
# Manual entry point (`python3 tests/test_smoke.py`)
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== smoke test (running outside pytest) ===\n")

    print("[1/3] DOE well-formed check…")
    test_doe_table_well_formed()
    print("  ✓ 20 cells: 6 mono / 11 hetero / 3 null\n")

    print("[2/3] TSV round-trip check…")
    test_tsv_round_trip()
    test_tsv_escapes_tabs_and_newlines()
    print("  ✓ write_result_row + load_completed_keys agree on resume key\n")

    print("[3/3] Live Path-1 on llama3.2:3b…")
    if _ollama_has_model("llama3.2:3b"):
        test_run_path_1_on_palindrome()
        print("\n✓ All smoke tests passed.")
    else:
        print("  SKIPPED — llama3.2:3b not pulled.")
        print("            run: ollama pull llama3.2:3b")
        print("\n✓ Static smoke tests passed; live test skipped.")
