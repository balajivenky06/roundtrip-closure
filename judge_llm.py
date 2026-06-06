"""
judge_llm.py — external SLM-as-judge for closure validity.

The judge SLM (`deepseek-r1:14b` by default, per concept-note §10 Q2)
is asked: "are these two artefacts semantically equivalent?" on every
closure check. Its rating serves two purposes:

    1. A backup signal against false closure (paired with the automated
       metric and the human study).
    2. An RQ2 validation: do the judge's ratings correlate with the
       automated closure metric and with human-annotator ratings?

The judge MUST be a different family than any pipeline SLM — otherwise
it would be rating its own outputs. DeepSeek-R1 satisfies this constraint
(none of the six pipeline SLMs are from DeepSeek).

Public API:
    - JudgeResult dataclass
    - judge_equivalence(a, b, artefact_kind) → JudgeResult
"""

from __future__ import annotations
import logging
import re
from dataclasses import dataclass
from typing import Literal

from config import JUDGE_MODEL
import ollama_client

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Public types
# ──────────────────────────────────────────────────────────────────────
ArtefactKind = Literal["code", "docstring", "tests"]


@dataclass
class JudgeResult:
    """Structured judge output."""
    rating: int                  # 0-4; -1 means parse failure
    justification: str           # one-line natural-language reason
    raw_response: str            # full LLM response (for debugging / audit)
    cache_hit: bool = False      # whether the underlying LLM call hit the cache


# ──────────────────────────────────────────────────────────────────────
# Rubric prompt (the bedrock of RQ2 validity)
# ──────────────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = (
    "You are a software-engineering equivalence judge. You compare two "
    "code/docstring/test artefacts and decide whether they are semantically "
    "equivalent. Be strict but fair. Always respond in the exact two-line "
    "format requested."
)

_RUBRIC = """\
Rate semantic equivalence on this 0-4 scale:
  4 — Identical: same output on every valid input
  3 — Equivalent: same observable behaviour, different implementation
  2 — Approximately equivalent: differs only on rare edge cases
  1 — Clearly different: different output on common inputs
  0 — Unrelated: solving different problems entirely

Respond in EXACTLY this format (two lines, nothing else):
RATING: <0|1|2|3|4>
REASON: <one short sentence>"""


_USER_TEMPLATE = """\
{rubric}

ARTEFACT KIND: {artefact_kind}

ARTEFACT A (original):
{artefact_a}

ARTEFACT B (reconstructed):
{artefact_b}
"""


# ──────────────────────────────────────────────────────────────────────
# Response parser — robust to minor formatting drift
# ──────────────────────────────────────────────────────────────────────
_RATING_RE = re.compile(
    r"(?im)^\s*\**\s*rating\s*\**\s*[:\-]?\s*\**\s*([0-4])"
)
_REASON_RE = re.compile(
    r"(?im)^\s*\**\s*reason\s*\**\s*[:\-]?\s*\**\s*(.+?)\s*$"
)


def _parse_judge_response(text: str) -> tuple[int, str]:
    """
    Extract (rating, reason) from the model's response.

    Returns (-1, raw_first_120_chars) on parse failure — callers should
    treat -1 as "judge could not be parsed; exclude from agreement
    analysis but keep raw_response for audit".
    """
    if not text:
        return -1, ""

    # DeepSeek-R1 emits <think>…</think> blocks; strip them before parsing
    text_no_think = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text_no_think = text_no_think.strip()

    rating_m = _RATING_RE.search(text_no_think)
    reason_m = _REASON_RE.search(text_no_think)

    rating = int(rating_m.group(1)) if rating_m else -1
    # Use first line if explicit REASON: tag missing
    reason = reason_m.group(1).strip() if reason_m else _first_meaningful_line(text_no_think)
    # Trim trailing punctuation noise
    reason = reason.rstrip(".,;: ")
    return rating, reason[:200]


def _first_meaningful_line(text: str) -> str:
    """Pick the first non-empty line that isn't 'RATING: n'."""
    for line in text.splitlines():
        stripped = line.strip().lstrip("*").strip()
        if not stripped:
            continue
        if stripped.lower().startswith("rating"):
            continue
        return stripped
    return ""


# ──────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────
def judge_equivalence(
    artefact_a: str,
    artefact_b: str,
    *,
    artefact_kind: ArtefactKind = "code",
    use_cache: bool = True,
) -> JudgeResult:
    """
    Single judge call. Returns a JudgeResult with rating in 0-4
    (or -1 on parse failure).

    Cached via closure_cache — same (a, b, kind) pair will not re-invoke
    the SLM. The cache key includes artefact_kind via role_hint so that
    judging the same pair on different kinds (code vs docstring vs tests)
    caches independently.
    """
    if not artefact_a or not artefact_b:
        return JudgeResult(rating=-1, justification="empty input", raw_response="")

    user_msg = _USER_TEMPLATE.format(
        rubric=_RUBRIC,
        artefact_kind=artefact_kind,
        artefact_a=artefact_a,
        artefact_b=artefact_b,
    )

    response = ollama_client.call_llm(
        JUDGE_MODEL,
        user_msg,
        system_prompt=_SYSTEM_PROMPT,
        role_hint=f"judge_{artefact_kind}",
        temperature=0.0,           # deterministic judging
        top_p=1.0,
        top_k=0,
        max_tokens=2048,           # DeepSeek-R1 uses ~200-300 tokens of
                                   # <think>...</think> reasoning before
                                   # emitting RATING/REASON; 256 truncated.
                                   # Measured ~248 tokens typical usage,
                                   # 2048 gives 8x headroom for harder pairs.
        use_cache=use_cache,
    )

    if response.finish_reason == "error":
        logger.error(f"Judge LLM call failed: {response.error}")
        return JudgeResult(
            rating=-1, justification="judge_error",
            raw_response=response.text, cache_hit=response.cache_hit,
        )

    rating, reason = _parse_judge_response(response.text)
    if rating < 0:
        logger.warning(
            f"Could not parse judge response — first 120 chars: "
            f"{response.text[:120]!r}"
        )

    return JudgeResult(
        rating=rating, justification=reason,
        raw_response=response.text, cache_hit=response.cache_hit,
    )


# ──────────────────────────────────────────────────────────────────────
# Sanity self-test (requires deepseek-r1:14b pulled)
# ──────────────────────────────────────────────────────────────────────
def _self_test() -> None:                                         # pragma: no cover
    print("=== judge_llm self-test ===\n")

    # Smoke check the parser FIRST (no Ollama needed)
    print("1) Parser smoke tests…")
    cases = [
        ("RATING: 3\nREASON: same behaviour, different implementation",
         (3, "same behaviour, different implementation")),
        ("rating: 0\nReason: clearly different.",
         (0, "clearly different")),
        ("**RATING**: 4\n**REASON**: identical.",
         (4, "identical")),
        ("<think>I think about it</think>\nRATING: 2\nREASON: edge cases differ",
         (2, "edge cases differ")),
        ("garbled response with no rating",
         (-1, "garbled response with no rating")),
    ]
    for raw, (want_rating, want_reason) in cases:
        r, reason = _parse_judge_response(raw)
        ok = (r == want_rating) and (reason.lower() == want_reason.lower())
        marker = "  ✓" if ok else "  ✗"
        print(f"{marker} parse({raw[:35]!r}) = (rating={r}, reason={reason[:50]!r})")
        assert ok, f"mismatch: want ({want_rating}, {want_reason}), got ({r}, {reason})"

    # Live test — needs deepseek-r1:14b
    print("\n2) Live judge call (requires deepseek-r1:14b)…")
    try:
        ollama_client.ensure_model_available(JUDGE_MODEL)
    except RuntimeError as exc:
        print(f"   SKIPPED — {exc}")
        print("\n✓ judge_llm self-test passed (parser only; live test skipped).")
        return

    # Equivalent pair
    a = "def is_palindrome(s): return s == s[::-1]"
    b = "def is_palindrome(s): return s == ''.join(reversed(s))"
    print(f"   judging equivalent pair…")
    result = judge_equivalence(a, b, artefact_kind="code", use_cache=False)
    print(f"     rating={result.rating}, reason={result.justification[:80]}")
    assert result.rating >= 3, f"Equivalent pair should rate ≥3, got {result.rating}"

    # Different pair
    c = "def factorial(n): return 1 if n <= 1 else n * factorial(n-1)"
    print(f"   judging different pair…")
    result_diff = judge_equivalence(a, c, artefact_kind="code", use_cache=False)
    print(f"     rating={result_diff.rating}, reason={result_diff.justification[:80]}")
    assert result_diff.rating <= 1, f"Different pair should rate ≤1, got {result_diff.rating}"

    print("\n✓ judge_llm self-test passed (parser + live).")


if __name__ == "__main__":
    _self_test()
