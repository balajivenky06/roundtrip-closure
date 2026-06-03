"""
decontaminate.py — build the HumanEval-Mutated subset for §3.X.

Purpose: defend against the data-contamination reviewer attack (Open-weight
SLMs trained on GitHub may have seen HumanEval/MBPP solutions and be
producing memorised code rather than genuinely reasoning).

Strategy: take each HumanEval problem and apply three semantics-preserving
transformations that BREAK surface-level memorisation while keeping the
semantics intact:
    1. Rename function (AST-based, deterministic)
    2. Permute parameter names (AST-based)
    3. Paraphrase docstring (one-shot LLM rewrite)

Then run a SANITY CHECK: the decontaminated tests should still pass on the
decontaminated code. If they don't, the problem is excluded.

Stub status: function signatures + flow comments; AST + LLM wrappers TBD.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DecontaminatedProblem:
    """One HumanEval problem after decontamination."""
    original_id: str                         # e.g. "humaneval/12"
    decontam_id: str                         # e.g. "humaneval-mutated/12"
    original_code: str
    decontam_code: str
    original_docstring: str
    decontam_docstring: str
    original_tests: str
    decontam_tests: str
    rename_mapping: dict[str, str] = field(default_factory=dict)
    sanity_check_passed: bool = False
    notes: str = ""


# ──────────────────────────────────────────────────────────────────────
# Stage 1: rename function (AST)
# ──────────────────────────────────────────────────────────────────────
def rename_function(code: str, new_name: str) -> tuple[str, str]:
    """
    Rename the top-level function (and all internal references to it)
    via AST. Returns (rewritten_code, original_name).
    """
    raise NotImplementedError("Stub — ast + astor implementation pending.")


# ──────────────────────────────────────────────────────────────────────
# Stage 2: permute parameter names (AST)
# ──────────────────────────────────────────────────────────────────────
def permute_params(code: str, mapping: dict[str, str]) -> str:
    """
    Apply parameter-rename mapping {old: new} throughout the function
    body. AST-based, deterministic.
    """
    raise NotImplementedError("Stub — ast walker implementation pending.")


# ──────────────────────────────────────────────────────────────────────
# Stage 3: paraphrase docstring (LLM)
# ──────────────────────────────────────────────────────────────────────
def paraphrase_docstring(docstring: str,
                         paraphraser_short_name: str = "qwen3.6"
                         ) -> str:
    """
    Single LLM call to rewrite a docstring with different wording but
    identical semantics. Returns the rewritten docstring string.

    Default paraphraser = qwen3.6 (free local Ollama call; deterministic
    if temperature=0.0).
    """
    raise NotImplementedError("Stub — LLM call pending.")


# ──────────────────────────────────────────────────────────────────────
# Orchestrator
# ──────────────────────────────────────────────────────────────────────
def decontaminate_problem(problem: dict,
                          paraphraser_short_name: str = "qwen3.6"
                          ) -> DecontaminatedProblem:
    """
    Run all three transformations + sanity check. Returns a fully
    populated DecontaminatedProblem.

    The sanity check executes the decontaminated tests against the
    decontaminated code in a pytest subprocess; sanity_check_passed
    captures whether all tests pass.
    """
    raise NotImplementedError("Stub — implementation pending.")


# ──────────────────────────────────────────────────────────────────────
# Build the full HumanEval-Mutated subset (50 functions)
# ──────────────────────────────────────────────────────────────────────
def build_humaneval_mutated_subset(
    humaneval_problems: list[dict],
    output_path: Path,
    n: int = 50,
    seed: int = 42,
    paraphraser_short_name: str = "qwen3.6",
) -> None:
    """
    Sample n HumanEval problems (stratified, deterministic seed),
    decontaminate each, exclude problems whose sanity check fails,
    write the resulting JSONL to `output_path`.

    Logs the inclusion / exclusion rate so we can report decontamination
    yield in the methodology section.
    """
    raise NotImplementedError("Stub — implementation pending.")
