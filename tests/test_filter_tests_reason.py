"""
Unit tests for closure_metrics.filter_tests_with_reason (Algorithm 3).

Covers the four filter_reason categories:
    empty_input
    no_test_functions
    all_dropped
    kept_K_of_N

Uses a tiny `is_palindrome` function and a mix of correct + broken tests to
exercise each branch.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from closure_metrics import filter_tests_with_reason


PALIN_CODE = (
    "def is_palindrome(s: str) -> bool:\n"
    "    return s == s[::-1]\n"
)

CORRECT_TESTS = (
    "def test_pal_true():\n"
    "    assert is_palindrome('racecar') is True\n"
    "\n"
    "def test_pal_false():\n"
    "    assert is_palindrome('abc') is False\n"
)

MIXED_TESTS = (
    "def test_pal_true():\n"
    "    assert is_palindrome('racecar') is True\n"
    "\n"
    "def test_pal_wrong_expectation():\n"
    "    assert is_palindrome('abc') is True   # broken: should be False\n"
    "\n"
    "def test_pal_empty():\n"
    "    assert is_palindrome('') is True\n"
)

ALL_BROKEN_TESTS = (
    "def test_bad_a():\n"
    "    assert is_palindrome('abc') is True\n"
    "\n"
    "def test_bad_b():\n"
    "    assert is_palindrome('xyz') is True\n"
)


class TestEmptyInput:
    def test_empty_tests(self):
        filtered, reason = filter_tests_with_reason("", PALIN_CODE)
        assert filtered == ""
        assert reason == "empty_input"

    def test_whitespace_tests(self):
        filtered, reason = filter_tests_with_reason("   \n\n  ", PALIN_CODE)
        assert filtered == ""
        assert reason == "empty_input"

    def test_empty_code(self):
        filtered, reason = filter_tests_with_reason(CORRECT_TESTS, "")
        assert filtered == ""
        assert reason == "empty_input"


class TestNoTestFunctions:
    def test_only_imports_no_defs(self):
        blob = "import pytest\nx = 42\n"
        filtered, reason = filter_tests_with_reason(blob, PALIN_CODE)
        assert filtered == ""
        assert reason == "no_test_functions"

    def test_non_test_defs(self):
        blob = "def helper():\n    return 1\n"
        filtered, reason = filter_tests_with_reason(blob, PALIN_CODE)
        assert filtered == ""
        assert reason == "no_test_functions"


class TestAllDropped:
    def test_all_broken_tests_dropped(self):
        filtered, reason = filter_tests_with_reason(ALL_BROKEN_TESTS, PALIN_CODE)
        assert filtered.strip() == ""
        assert reason == "all_dropped"


class TestKeptOfN:
    def test_all_correct_kept(self):
        filtered, reason = filter_tests_with_reason(CORRECT_TESTS, PALIN_CODE)
        assert "test_pal_true" in filtered
        assert "test_pal_false" in filtered
        assert reason.startswith("kept_")
        # kept_2_of_2 expected
        assert reason == "kept_2_of_2"

    def test_mixed_tests_partial_kept(self):
        filtered, reason = filter_tests_with_reason(MIXED_TESTS, PALIN_CODE)
        assert "test_pal_true" in filtered
        # broken test dropped
        assert "test_pal_wrong_expectation" not in filtered
        # kept_2_of_3 expected (empty-string palindrome is edge; s == s[::-1]
        # is True for '' so test_pal_empty passes)
        assert reason.startswith("kept_")
        parts = reason.split("_")
        # kept_K_of_N form
        assert parts[0] == "kept" and parts[2] == "of"
        k, n = int(parts[1]), int(parts[3])
        assert n == 3
        assert 1 <= k < n  # at least the true one kept; broken dropped
