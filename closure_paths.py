"""
closure_paths.py — drivers for the three round-trip closure paths.

For each (cell, function) pair, three traversals of the docstring-test-code
triangle are executed:

    Path 1:  C → D → T   close-check: mutation kill rate of T against C
    Path 2:  D → T → C   close-check: pass rate of ORIGINAL C's tests on C'
    Path 3:  C → T → D   close-check: BERTScore + judge-LLM equivalence(D, D')

Each driver returns a list of ClosureResult records (one per metric computed).

Stub status: function signatures + flow comments; implementation TBD.
"""

from __future__ import annotations
from dataclasses import dataclass

from config import ModelSpec
from doe import Cell


@dataclass
class ClosureResult:
    """One closure metric measurement."""
    cell_id: str
    sample_idx: int
    sample_source: str        # "humaneval/12" / "mbpp/77" / "livecodebench/3" etc.
    path: int                 # 1, 2, or 3
    metric_name: str          # "mutation_kill_rate" | "reference_pass_rate" | "bertscore"
    metric_value: float       # the closure metric in [0, 1]
    judge_rating: int         # 0-4 from the judge LLM
    judge_justification: str  # one-line explanation from the judge
    valid: bool               # False if test filter dropped this sample
    elapsed_s: float
    cache_hits: int
    notes: str = ""


# ──────────────────────────────────────────────────────────────────────
# Path 1:  C → D → T
# ──────────────────────────────────────────────────────────────────────
def run_path_1(cell: Cell, sample: dict) -> list[ClosureResult]:
    """
    Path 1 — Code-to-docstring-to-tests.

    Steps:
        1. D' = L_spec(C)                  generate docstring from original code
        2. T' = L_test(D')                 generate tests from generated docstring
        3. Apply test filter to T':        drop tests that fail on ORIGINAL C
        4. mutation_kill_rate(T', C):      run T' against C's mutants
        5. judge_LLM(D', original D):      equivalence rating on the docstring

    Returns one ClosureResult for the kill-rate measurement plus the judge
    rating (recorded on the same record).
    """
    raise NotImplementedError("Stub — implementation pending.")


# ──────────────────────────────────────────────────────────────────────
# Path 2:  D → T → C
# ──────────────────────────────────────────────────────────────────────
def run_path_2(cell: Cell, sample: dict) -> list[ClosureResult]:
    """
    Path 2 — Docstring-to-tests-to-code.

    Steps:
        1. T' = L_test(D)                   tests from ORIGINAL docstring
        2. C' = L_code(D, T')               code from D + T'
        3. Run ORIGINAL reference tests against C':  pass rate is the metric
        4. judge_LLM(original C, C'):       equivalence rating on the code

    Returns one ClosureResult for the reference-pass-rate measurement.
    """
    raise NotImplementedError("Stub — implementation pending.")


# ──────────────────────────────────────────────────────────────────────
# Path 3:  C → T → D
# ──────────────────────────────────────────────────────────────────────
def run_path_3(cell: Cell, sample: dict) -> list[ClosureResult]:
    """
    Path 3 — Code-to-tests-to-docstring.

    Steps:
        1. T' = L_test(C)                   tests from ORIGINAL code (skip D)
        2. D' = L_spec(T')                  docstring from the generated tests
        3. BERTScore(D, D'):                semantic similarity of docstrings
        4. judge_LLM(D, D'):                equivalence rating on docstrings

    Returns one ClosureResult combining BERTScore + judge rating.
    """
    raise NotImplementedError("Stub — implementation pending.")


# ──────────────────────────────────────────────────────────────────────
# Convenience: run all three paths in sequence
# ──────────────────────────────────────────────────────────────────────
def run_all_paths(cell: Cell, sample: dict,
                  paths_to_run: tuple[int, ...] = (1, 2, 3)
                  ) -> list[ClosureResult]:
    """Loop over `paths_to_run` and concatenate the results."""
    raise NotImplementedError("Stub — implementation pending.")
