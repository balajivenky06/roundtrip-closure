"""
tests/test_smoke.py — end-to-end smoke test on 1 function × 1 mono cell.

This is the absolute-minimum verification that the pipeline runs end-to-end
without crashing. Run as:

    pytest tests/test_smoke.py -v

Expected: under 60 seconds on Colab A100, longer locally without GPU.

The test is intentionally NOT a correctness test of closure metrics —
those are validated through the pilot (scripts/run_pilot.py).
"""

from __future__ import annotations
import sys
from pathlib import Path

# Make project root importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest


@pytest.mark.skip(reason="Implementation pending — scaffold-only at this point")
def test_one_function_one_mono_cell():
    """
    Smoke test: pick the smallest cell (M1 — llama3.2 3B mono) and run
    Path 1 (the cheapest closure check) on a single HumanEval problem.

    Asserts:
        - All 3 LLM stages return non-empty strings
        - mutation_kill_rate returns a float in [0.0, 1.0]
        - Judge LLM returns a JudgeResult with rating in 0-4
    """
    from doe import M1
    # from closure_paths import run_path_1
    # sample = load_one_humaneval_problem()
    # results = run_path_1(M1, sample)
    # assert len(results) >= 1
    # assert 0.0 <= results[0].metric_value <= 1.0
    # assert 0 <= results[0].judge_rating <= 4


@pytest.mark.skip(reason="Implementation pending — scaffold-only")
def test_doe_table_well_formed():
    """
    Sanity-check the doe.py registry: 20 cells, valid strata, no None
    stages outside null cells.
    """
    from doe import ALL_CELLS, MONO_CELLS, HETERO_CELLS, NULL_CELLS
    assert len(ALL_CELLS) == 20
    assert len(MONO_CELLS) == 6
    assert len(HETERO_CELLS) == 11
    assert len(NULL_CELLS) == 3
    for cell in MONO_CELLS + HETERO_CELLS:
        assert cell.L_spec is not None
        assert cell.L_test is not None
        assert cell.L_code is not None
