"""
closure_metrics.py — numeric closure metrics for the three paths.

This module wraps the existing mutation_testing module + adds two new
metrics specific to Chapter 3:

    - mutation_kill_rate(tests, code):    reused from Chapter 2
    - reference_test_pass_rate(orig_tests, code'):  new — Path 2 metric
    - bert_similarity(docstring_a, docstring_b):   new — Path 3 metric
    - test_filter(tests, original_code):           reused from Chapter 2

The judge-LLM equivalence rating lives in judge_llm.py, not here.

Stub status: function signatures; bert_score + pytest wrappers TBD.
"""

from __future__ import annotations


def mutation_kill_rate(tests: str, code: str,
                       max_mutants: int = 15) -> tuple[float, dict]:
    """
    Run the Chapter 2 mutation-testing pipeline. Returns (kill_rate, breakdown)
    where breakdown is a per-operator dict.
    """
    raise NotImplementedError("Stub — wrap mutation_testing.run_one_function.")


def reference_test_pass_rate(reference_tests: str, candidate_code: str) -> float:
    """
    Run the dataset's reference tests against the candidate code in a
    pytest subprocess. Returns the fraction of tests that pass.

    Used as the Path 2 closure metric.
    """
    raise NotImplementedError("Stub — pytest subprocess wrapper pending.")


def bert_similarity(text_a: str, text_b: str) -> float:
    """
    BERTScore F1 between two docstrings. Returns score in [0, 1].

    Used as the Path 3 closure metric.
    """
    raise NotImplementedError("Stub — bert_score wrapper pending.")


def test_filter(tests: str, original_code: str) -> str:
    """
    Drop tests from `tests` that fail on `original_code`.
    Returns the filtered test string.

    Reused from Chapter 2's mutation_testing module.
    """
    raise NotImplementedError("Stub — wrap mutation_testing.filter_passing_tests.")
