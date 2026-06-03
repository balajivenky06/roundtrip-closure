"""
judge_llm.py — external LLM-as-judge for closure validity.

The judge LLM (DeepSeek-R1 14B by default) is asked: "are these two
artefacts semantically equivalent?" on every closure check. Its rating
serves two purposes:
    1. A backup signal against false-closure (paired with the automated
       metric and the human study).
    2. An RQ2 validation: do the judge's ratings correlate with the
       automated closure metric, and with human-annotator ratings?

The judge must be a DIFFERENT family than any pipeline SLM — otherwise
it would be rating its own outputs. DeepSeek satisfies this constraint
(none of the pipeline LLMs are from DeepSeek).

Stub status: function signatures + prompt template TBD.
"""

from __future__ import annotations
from dataclasses import dataclass

from config import JUDGE_MODEL


@dataclass
class JudgeResult:
    """Structured judge output."""
    rating: int               # 0-4: 0=unrelated, 4=identical
    justification: str        # one-line natural-language reason
    raw_response: str         # full LLM response (for debugging / audit)


_RUBRIC_PROMPT_TEMPLATE = """\
You are a software-engineering equivalence judge. Given two functions
(or two docstrings, or two test suites), decide whether they are
semantically equivalent.

Rate on this 0-4 scale:
  4 — Identical: same output on every valid input
  3 — Equivalent: same observable behaviour, different implementation
  2 — Approximately equivalent: differs only on rare edge cases
  1 — Clearly different: different output on common inputs
  0 — Unrelated: solving different problems entirely

Respond in this exact format (two lines):
  RATING: <0|1|2|3|4>
  REASON: <one short sentence>

ARTEFACT A (original):
{artefact_a}

ARTEFACT B (reconstructed):
{artefact_b}
"""


def judge_equivalence(
    artefact_a: str,
    artefact_b: str,
    *,
    artefact_kind: str = "code",   # "code" | "docstring" | "tests"
) -> JudgeResult:
    """
    Single judge call. Returns a JudgeResult with rating in 0-4.

    Uses JUDGE_MODEL (DeepSeek-R1 14B by default) via the Ollama client.
    Cached via closure_cache.
    """
    raise NotImplementedError("Stub — implementation pending.")


def _parse_judge_response(text: str) -> tuple[int, str]:
    """
    Extract (rating, reason) from the model's response.

    Robust to minor formatting drift: case-insensitive, accepts
    'rating:', 'Rating ', '**RATING**' etc.
    """
    raise NotImplementedError("Stub — implementation pending.")
