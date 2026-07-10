"""
closure_metrics.py — numeric closure metrics for the three paths.

Wraps the Chapter-2 mutation_testing module (copied verbatim into this
repo) and adds two new metrics specific to Chapter 3:

    mutation_kill_rate(tests, code)               — Path 1 metric
    reference_test_pass_rate(orig_tests, code')   — Path 2 metric
    bert_similarity(docstring_a, docstring_b)     — Path 3 metric
    test_filter(tests, original_code)             — reused; drops bad tests

The judge-LLM equivalence rating lives in judge_llm.py, not here.

All functions are pure — no I/O state, no Ollama calls. The wrapper
shape lets the closure_paths.py drivers compose them freely.
"""

from __future__ import annotations
import logging
import re
from typing import Optional

from config import MAX_MUTANTS_PER_FUNCTION

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Lazy import of mutation_testing
#
# `mutation_testing.py` imports pandas at module-level. On macOS+Anaconda
# this triggers a NumPy 1.x/2.x ABI clash (the dev environment has
# numpy 2.4 but pandas/numexpr were compiled against 1.x). On Colab
# the env is clean and pandas imports fine.
#
# By lazy-importing here, the closure_metrics module is usable on macOS
# for BERTScore + parser smoke tests; the mutation_testing-dependent
# functions raise the original ImportError only when actually invoked.
# ──────────────────────────────────────────────────────────────────────
_mutation_testing = None


def _get_mt():
    global _mutation_testing
    if _mutation_testing is None:
        import mutation_testing as _mt
        _mutation_testing = _mt
    return _mutation_testing


# ──────────────────────────────────────────────────────────────────────
# Path 1 — Mutation kill rate
# ──────────────────────────────────────────────────────────────────────
def mutation_kill_rate(
    tests: str,
    code: str,
    *,
    ground_truth_tests: str = "",
    max_mutants: int = MAX_MUTANTS_PER_FUNCTION,    # noqa: ARG001  (kept for API symmetry; cap enforced inside mutation_testing)
) -> tuple[float, dict]:
    """
    Run the Chapter-2 mutation-testing pipeline.

    Args:
        tests:              candidate test suite (the one being evaluated)
        code:               function under test (the "ground truth" code)
        ground_truth_tests: optional reference tests used to detect
                            equivalent mutants (passes-on-mutant means
                            mutant is equivalent, exclude from denominator)

    Returns:
        (kill_rate, breakdown) where breakdown is the full per-operator
        dict returned by mutation_testing.evaluate_mutants.

    kill_rate is NaN if zero tests survive the test-filter step
    (i.e. the LLM produced a useless suite). Caller should treat
    NaN as "sample excluded".
    """
    if not tests.strip() or not code.strip():
        return float("nan"), {"reason": "empty_input"}

    breakdown = _get_mt().evaluate_mutants(
        function_code=code,
        test_code=tests,
        ground_truth_tests=ground_truth_tests,
    )
    rate = breakdown.get("kill_rate", float("nan"))
    return float(rate), breakdown


# ──────────────────────────────────────────────────────────────────────
# Path 2 — Reference-test pass rate
# ──────────────────────────────────────────────────────────────────────
def reference_test_pass_rate(reference_tests: str, candidate_code: str) -> float:
    """
    Run the dataset's reference tests against the candidate (reconstructed)
    code in a pytest subprocess.

    Reuses mutation_testing.run_tests_against_code which returns
    "pass" | "fail" | "error".

    Returns the fraction in [0.0, 1.0]:
        - 1.0 if the full suite passes
        - 0.0 if any test fails or a runtime error stops collection

    For finer-grained reporting (e.g., 5 of 8 tests pass), see the
    `_per_test_pass_rate` helper below — currently unused by the main
    pipeline but kept for ablations.
    """
    if not reference_tests.strip() or not candidate_code.strip():
        return 0.0

    status = _get_mt().run_tests_against_code(reference_tests, candidate_code)
    if status == "pass":
        return 1.0
    return 0.0


def _per_test_pass_rate(reference_tests: str, candidate_code: str) -> float:
    """
    Run each `def test_*` separately and return the pass fraction.

    Useful for diagnostics — most of the time reference_test_pass_rate
    above gives the right binary signal.
    """
    if not reference_tests.strip() or not candidate_code.strip():
        return 0.0
    test_fns = _split_test_functions(reference_tests)
    if not test_fns:
        return 0.0
    passed = 0
    for fn_body in test_fns:
        status = _get_mt().run_tests_against_code(fn_body, candidate_code)
        if status == "pass":
            passed += 1
    return passed / len(test_fns)


_DEF_TEST_RE = re.compile(r"(?m)^def (test_[A-Za-z_0-9]+)\s*\(")


def _split_test_functions(test_code: str) -> list[str]:
    """Split a test file into individual test-function source snippets.
    Each chunk preserves the imports prefix (lines before the first def)
    so each function still runs in isolation."""
    matches = list(_DEF_TEST_RE.finditer(test_code))
    if not matches:
        return []
    # Everything before the first def is shared (imports, fixtures).
    prefix = test_code[: matches[0].start()]
    chunks = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(test_code)
        chunks.append(prefix + test_code[m.start():end])
    return chunks


# ──────────────────────────────────────────────────────────────────────
# Path 3 — BERTScore semantic similarity (docstring round-trip)
# ──────────────────────────────────────────────────────────────────────
_bertscore_scorer = None     # lazy-loaded; expensive on first use


def _get_bertscore_scorer():
    """Initialise the BERTScore scorer once per process.

    We use `roberta-large` as the model — the BERTScore-paper default
    and the most widely-cited choice for SE-similarity tasks.
    """
    global _bertscore_scorer
    if _bertscore_scorer is None:
        try:
            from bert_score import BERTScorer
        except ImportError as e:                                  # pragma: no cover
            raise ImportError(
                "BERTScore not installed. Run: pip install bert-score"
            ) from e
        logger.info("Loading BERTScore scorer (roberta-large) — first call only…")
        _bertscore_scorer = BERTScorer(
            model_type="roberta-large",
            lang="en",
            rescale_with_baseline=True,
        )
    return _bertscore_scorer


def bert_similarity(text_a: str, text_b: str, *, use_cache: bool = True) -> tuple[float, bool]:
    """
    BERTScore F1 between two docstrings, rescaled with baseline.
    Returns (score, was_cache_hit). Score is roughly in [0.0, 1.0]
    (slight excursions possible because of the baseline rescaling).

    Cached on (text_a, text_b) via closure_cache so each unique pair is
    only scored once across the entire sweep — important on Colab where
    bert_similarity is the slowest non-LLM step (~1-2 s/call after the
    scorer is loaded).

    Returns (0.0, False) on empty/None inputs.
    """
    if not text_a or not text_b:
        return 0.0, False

    if use_cache:
        import closure_cache
        cache_key = closure_cache.make_key(
            model_tag="bertscore-roberta-large",
            role_hint="bert_similarity",
            prompt=text_a + "\x00" + text_b,
            temperature=0.0,
        )
        cached = closure_cache.get(cache_key)
        if cached is not None and "score" in cached:
            return float(cached["score"]), True

    scorer = _get_bertscore_scorer()
    _p, _r, f1 = scorer.score([text_b], [text_a])
    score = float(f1[0])

    if use_cache:
        closure_cache.put(cache_key, {"score": score})

    return score, False


# ──────────────────────────────────────────────────────────────────────
# Shared utility — drop tests that fail on the original code
# ──────────────────────────────────────────────────────────────────────
def test_filter(tests: str, original_code: str) -> str:
    """
    Drop tests from `tests` that fail on `original_code`. Returns the
    filtered test string (possibly empty if all tests fail).

    Reuses the Chapter-2 _filter_passing_tests helper directly so the
    behaviour is identical to the SQJ pipeline.
    """
    if not tests.strip() or not original_code.strip():
        return ""
    return _get_mt().filter_passing_tests(tests, original_code)


_DEF_TEST_ANY_INDENT = re.compile(r"(?m)^\s*def\s+test_[A-Za-z_0-9]+\s*\(")


def filter_tests_with_reason(tests: str, original_code: str) -> tuple[str, str]:
    """
    Algorithm 3: test-filter validity gate with a diagnostic label.

    Wraps `test_filter` but also returns a `filter_reason` explaining what
    happened. Used by future closure_paths runs to persist the reason
    directly to the TSV, replacing the current "0/N tests kept" heuristic
    log line with a machine-readable label.

    Filter reasons:
        empty_input           — inputs were empty strings
        no_test_functions     — no ``def test_...`` matched in the input
        all_dropped           — every test failed on the original code
        kept_K_of_N           — K tests kept out of N candidate test functions
    """
    if not tests.strip() or not original_code.strip():
        return "", "empty_input"

    n_before = len(_DEF_TEST_ANY_INDENT.findall(tests))
    if n_before == 0:
        # No detectable test functions in the input at all.
        return "", "no_test_functions"

    filtered = _get_mt().filter_passing_tests(tests, original_code)
    if not filtered.strip():
        return "", "all_dropped"

    n_after = len(_DEF_TEST_ANY_INDENT.findall(filtered))
    return filtered, f"kept_{n_after}_of_{n_before}"


# ──────────────────────────────────────────────────────────────────────
# Sanity self-test
# ──────────────────────────────────────────────────────────────────────
def _self_test() -> None:                                         # pragma: no cover
    """Self-test using a tiny is_palindrome function + its tests."""
    print("=== closure_metrics self-test ===\n")

    code = (
        "def is_palindrome(s: str) -> bool:\n"
        "    return s == s[::-1]\n"
    )
    correct_tests = (
        "def test_pal_true():\n"
        "    assert is_palindrome('racecar') is True\n"
        "\n"
        "def test_pal_false():\n"
        "    assert is_palindrome('abc') is False\n"
    )
    weak_tests = (
        # Doesn't catch boundary or comparison mutants
        "def test_pal_weak():\n"
        "    result = is_palindrome('a')\n"
        "    assert result is True or result is False\n"
    )

    # Mutation kill rate
    print("1) mutation_kill_rate on correct_tests…")
    rate, breakdown = mutation_kill_rate(correct_tests, code)
    print(f"   kill_rate = {rate:.3f}, total_mutants = {breakdown.get('total_mutants')}")
    assert rate == rate, "kill_rate is NaN — pipeline broken"
    assert 0.0 <= rate <= 1.0, f"kill_rate out of bounds: {rate}"

    print("\n2) mutation_kill_rate on weak_tests…")
    rate_weak, _ = mutation_kill_rate(weak_tests, code)
    print(f"   kill_rate = {rate_weak:.3f}")
    # Weak tests should kill fewer mutants than correct tests
    # (degeneracies aside)
    print(f"   weak < correct? {rate_weak < rate}")

    # Reference test pass rate
    print("\n3) reference_test_pass_rate (good code vs ref tests)…")
    pr = reference_test_pass_rate(correct_tests, code)
    print(f"   pass_rate = {pr:.3f}")
    assert pr == 1.0, "Good code should pass all reference tests"

    # Reference test pass rate on bad code
    bad_code = "def is_palindrome(s: str) -> bool:\n    return False\n"
    pr_bad = reference_test_pass_rate(correct_tests, bad_code)
    print(f"   pass_rate on bad code = {pr_bad:.3f}")
    assert pr_bad == 0.0, "Bad code should fail reference tests"

    # Test filter
    print("\n4) test_filter…")
    bad_tests = (
        "def test_good():\n"
        "    assert is_palindrome('racecar') is True\n"
        "\n"
        "def test_bad():\n"
        "    assert is_palindrome('abc') is True   # wrong\n"
    )
    filtered = test_filter(bad_tests, code)
    print(f"   filtered = {len(filtered)} chars (was {len(bad_tests)})")
    assert "test_good" in filtered, "Good test was dropped"
    assert "test_bad" not in filtered, "Bad test was kept"

    # BERTScore — only attempt if available
    print("\n5) bert_similarity…")
    try:
        s_self, _ = bert_similarity("Return True if s reads the same forwards and backwards.",
                                     "Return True if s reads the same forwards and backwards.")
        s_diff, _ = bert_similarity("Return True if s reads the same forwards and backwards.",
                                     "Compute the factorial of an integer.")
        print(f"   same-text  F1 = {s_self:.3f}")
        print(f"   diff-text  F1 = {s_diff:.3f}")
        assert s_self > s_diff, "Same text should score higher than different"
    except ImportError as e:
        print(f"   SKIPPED — {e}")

    print("\n✓ closure_metrics self-test passed.")


if __name__ == "__main__":
    _self_test()
