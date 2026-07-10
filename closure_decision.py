"""
closure_decision.py — closure validity decision (Algorithm 2 from the paper).

Given an automated closure metric and a judge LLM rating, decide whether a
round-trip closure is valid. Combines two signals under a strict-AND policy
(both must agree) and labels each decision with a `decision_reason` category
that surfaces analytical structure — the same categories the paper uses in
§5 for the judge-metric disagreement decomposition.

Public API:
    - TAU_DEFAULT: per-path threshold defaults for the metric.
    - RHO_DEFAULT: judge-rating threshold default (3 = "equivalent or better").
    - DecisionReason: string literal type of the five decision-reason labels.
    - decide_validity(metric_value, judge_rating, path, *, tau=None, rho=RHO_DEFAULT)
      → (is_valid: bool, reason: DecisionReason)

The categories exhaust the two-signal decision space:
    structural_NA           — metric or judge signal is missing
    both_agree_valid        — metric > τ AND judge >= ρ
    both_agree_invalid      — metric <= τ AND judge < ρ
    false_closure_candidate — metric > τ AND judge < ρ  (metric over-credits)
    metric_false_negative   — metric <= τ AND judge >= ρ (metric under-credits)

Under the strict-AND policy, only `both_agree_valid` counts as valid closure.
The disagreement categories are labelled but flagged invalid, and become the
analytical hooks for the false-closure / metric-validity discussion in §5.
"""
from __future__ import annotations

import math
from typing import Literal

# Path-specific metric thresholds (τ). Paths use different metric scales:
#   Path 1: mutation kill rate ∈ [0, 1]; τ = 0 means "any kill counts"
#   Path 2: reference pass rate ∈ {0, 1}; τ = 0 keeps binary semantics
#   Path 3: BERTScore-F1 rescaled; τ = 0 gives closure to any positive score
TAU_DEFAULT: dict[int, float] = {
    1: 0.0,
    2: 0.0,
    3: 0.0,
}

# Judge rating threshold (ρ). Paper default = 3 ("equivalent or better" on the
# 0-4 rubric in judge_llm.py). ρ = 4 is a ceiling (identical only); ρ = 2 is
# permissive (approximately equivalent).
RHO_DEFAULT: int = 3


DecisionReason = Literal[
    "structural_NA",
    "both_agree_valid",
    "both_agree_invalid",
    "false_closure_candidate",
    "metric_false_negative",
]


DECISION_REASONS: tuple[DecisionReason, ...] = (
    "both_agree_valid",
    "both_agree_invalid",
    "false_closure_candidate",
    "metric_false_negative",
    "structural_NA",
)


def _is_missing(value) -> bool:
    """True if the value is None or a NaN float."""
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return False


def decide_validity(
    metric_value: float | None,
    judge_rating: int | float | None,
    path: int,
    *,
    tau: float | None = None,
    rho: int = RHO_DEFAULT,
) -> tuple[bool, DecisionReason]:
    """
    Decide closure validity for one row.

    Args:
        metric_value: the automated closure metric (path-dependent).
        judge_rating: judge LLM rating on 0-4 rubric, or -1 on judge failure.
        path:         1 (kill rate), 2 (pass rate), or 3 (BERTScore).
        tau:          metric threshold override; defaults to TAU_DEFAULT[path].
        rho:          judge threshold; defaults to RHO_DEFAULT (3).

    Returns:
        (is_valid, reason). is_valid is True only under both_agree_valid.

    Raises:
        ValueError if path is not in {1, 2, 3}.
    """
    if path not in TAU_DEFAULT:
        raise ValueError(f"Unknown path {path!r}; expected 1, 2, or 3.")

    tau_eff = TAU_DEFAULT[path] if tau is None else tau

    # Tier 0: structural NA
    if _is_missing(metric_value) or _is_missing(judge_rating):
        return False, "structural_NA"
    j = int(judge_rating)
    if j < 0:
        return False, "structural_NA"

    m = float(metric_value)
    metric_ok = m > tau_eff
    judge_ok = j >= rho

    # Tier 1: both signals agree
    if metric_ok and judge_ok:
        return True, "both_agree_valid"
    if not metric_ok and not judge_ok:
        return False, "both_agree_invalid"

    # Tier 2: signal disagreement (analytical categories)
    if metric_ok and not judge_ok:
        return False, "false_closure_candidate"
    # not metric_ok and judge_ok
    return False, "metric_false_negative"
