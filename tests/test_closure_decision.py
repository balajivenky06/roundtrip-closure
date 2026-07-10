"""
Unit tests for closure_decision.decide_validity (Algorithm 2).

Covers the five decision-reason categories, boundary values of τ and ρ,
and structural-NA edge cases (NaN metric, judge = -1, None).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from closure_decision import (
    DECISION_REASONS,
    RHO_DEFAULT,
    TAU_DEFAULT,
    decide_validity,
)


# ──────────────────────────────────────────────────────────────────────
# Both-agree-valid
# ──────────────────────────────────────────────────────────────────────
class TestBothAgreeValid:
    def test_metric_above_tau_and_judge_above_rho(self):
        v, r = decide_validity(0.9, 4, path=1)
        assert v is True
        assert r == "both_agree_valid"

    def test_judge_exactly_at_rho(self):
        v, r = decide_validity(0.5, 3, path=1)
        assert v is True
        assert r == "both_agree_valid"

    def test_metric_strictly_above_tau_zero(self):
        # τ_default[1] = 0. Metric = 0.001 (just above) with judge >= ρ.
        v, r = decide_validity(0.001, 3, path=1)
        assert v is True
        assert r == "both_agree_valid"


# ──────────────────────────────────────────────────────────────────────
# Both-agree-invalid
# ──────────────────────────────────────────────────────────────────────
class TestBothAgreeInvalid:
    def test_metric_at_tau_and_judge_below_rho(self):
        v, r = decide_validity(0.0, 1, path=1)
        assert v is False
        assert r == "both_agree_invalid"

    def test_metric_at_tau_and_judge_at_rho_minus_one(self):
        v, r = decide_validity(0.0, RHO_DEFAULT - 1, path=1)
        assert v is False
        assert r == "both_agree_invalid"


# ──────────────────────────────────────────────────────────────────────
# False-closure candidate — metric over-credits
# ──────────────────────────────────────────────────────────────────────
class TestFalseClosureCandidate:
    def test_high_metric_low_judge(self):
        v, r = decide_validity(1.0, 1, path=1)
        assert v is False
        assert r == "false_closure_candidate"

    def test_high_metric_judge_zero(self):
        v, r = decide_validity(0.5, 0, path=1)
        assert v is False
        assert r == "false_closure_candidate"


# ──────────────────────────────────────────────────────────────────────
# Metric-false-negative — judge accepts, metric rejects
# ──────────────────────────────────────────────────────────────────────
class TestMetricFalseNegative:
    def test_low_metric_high_judge(self):
        v, r = decide_validity(0.0, 4, path=1)
        assert v is False
        assert r == "metric_false_negative"

    def test_metric_at_tau_judge_at_rho(self):
        # metric == τ (not > τ) should NOT count as metric_ok.
        v, r = decide_validity(0.0, RHO_DEFAULT, path=1)
        assert v is False
        assert r == "metric_false_negative"


# ──────────────────────────────────────────────────────────────────────
# Structural NA
# ──────────────────────────────────────────────────────────────────────
class TestStructuralNA:
    def test_nan_metric(self):
        v, r = decide_validity(float("nan"), 3, path=1)
        assert v is False
        assert r == "structural_NA"

    def test_none_metric(self):
        v, r = decide_validity(None, 3, path=1)
        assert v is False
        assert r == "structural_NA"

    def test_none_judge(self):
        v, r = decide_validity(0.5, None, path=1)
        assert v is False
        assert r == "structural_NA"

    def test_judge_negative_one(self):
        # Judge parse-failure code from judge_llm.py
        v, r = decide_validity(0.5, -1, path=1)
        assert v is False
        assert r == "structural_NA"

    def test_nan_judge_float(self):
        v, r = decide_validity(0.5, float("nan"), path=1)
        assert v is False
        assert r == "structural_NA"


# ──────────────────────────────────────────────────────────────────────
# Path handling
# ──────────────────────────────────────────────────────────────────────
class TestPaths:
    def test_all_paths_supported(self):
        for path in (1, 2, 3):
            v, r = decide_validity(0.5, 3, path=path)
            assert r in DECISION_REASONS

    def test_unknown_path_raises(self):
        with pytest.raises(ValueError):
            decide_validity(0.5, 3, path=4)


# ──────────────────────────────────────────────────────────────────────
# Threshold overrides (τ, ρ)
# ──────────────────────────────────────────────────────────────────────
class TestThresholdOverrides:
    def test_tau_override_moves_boundary(self):
        # metric = 0.4, τ_default = 0 → metric_ok.
        # τ_override = 0.5 → metric NOT ok.
        v_default, _ = decide_validity(0.4, 3, path=1)
        assert v_default is True
        v_override, r_override = decide_validity(0.4, 3, path=1, tau=0.5)
        assert v_override is False
        assert r_override == "metric_false_negative"

    def test_rho_override_moves_boundary(self):
        # judge = 3, ρ_default = 3 → judge_ok.
        # ρ_override = 4 → judge NOT ok.
        v_default, _ = decide_validity(0.5, 3, path=1)
        assert v_default is True
        v_override, r_override = decide_validity(0.5, 3, path=1, rho=4)
        assert v_override is False
        assert r_override == "false_closure_candidate"

    def test_permissive_rho_makes_more_valid(self):
        v, r = decide_validity(0.5, 2, path=1, rho=2)
        assert v is True


# ──────────────────────────────────────────────────────────────────────
# Property-style test: reasons partition the input space
# ──────────────────────────────────────────────────────────────────────
class TestPartition:
    @pytest.mark.parametrize("m", [0.0, 0.1, 0.5, 0.9])
    @pytest.mark.parametrize("j", [-1, 0, 1, 2, 3, 4])
    def test_every_input_yields_known_reason(self, m, j):
        v, r = decide_validity(m, j, path=1)
        assert r in DECISION_REASONS
        # is_valid only when both_agree_valid
        assert v == (r == "both_agree_valid")
