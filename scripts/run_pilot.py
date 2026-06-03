"""
scripts/run_pilot.py — 30-function pilot driver.

Per the §6 concept-note plan, the pilot:
    - 30 functions (10 HumanEval + 20 MBPP, stratified, seed=42)
    - 6 cells from doe.PILOT_CELLS: M1, M3, M6, H1, H4, N2
    - All 3 closure paths
    - Logs everything to logs/pilot_*.log
    - Writes to results/pilot_results.tsv
    - Validates the 6 go/no-go checks (see concept note §6)

Run as:
    python scripts/run_pilot.py
"""

from __future__ import annotations
import sys
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import PILOT_RESULTS_TSV, LOGS_DIR
from doe import PILOT_CELLS


def run_pilot() -> int:
    """Execute the 30-function pilot across all 6 PILOT_CELLS."""
    # 1. Load the pilot dataset (30 stratified samples)
    # 2. For each cell in PILOT_CELLS:
    #       for each sample in pilot_dataset:
    #           run all 3 closure paths
    #           write rows to PILOT_RESULTS_TSV
    # 3. Run the 6 go/no-go checks from concept note §6
    # 4. Produce pilot_summary.md
    raise NotImplementedError("Stub — implementation pending.")


if __name__ == "__main__":
    raise SystemExit(run_pilot())
