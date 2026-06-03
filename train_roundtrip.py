"""
train_roundtrip.py — main experiment driver.

For one DOE cell (set via CELL_ID below), iterate over the dataset and
execute all three closure paths per (cell, function) pair. Append one
row per ClosureResult to results/results_roundtrip.tsv.

To run a different cell, edit the CONFIG block at the top of this file
and re-run. The cell_id determines which (L_spec, L_test, L_code) triple
is used.

Stub status: configuration + main loop signature; per-cell execution TBD.
"""

from __future__ import annotations
import logging
from pathlib import Path

from config import (
    PIPELINE_MODELS,
    JUDGE_MODEL,
    CORE_SAMPLE_SIZE,
    DATA_DIR,
    LOGS_DIR,
    RESULTS_TSV,
)
from doe import get_cell, ALL_CELLS


# ─────────── CONFIGURATION (agent edits this block per cell) ────────────
CELL_ID: str = "M3"              # DOE cell to run; see doe.py for the list
DATASET: str = "core"            # "core" | "livecodebench" | "humaneval_mutated"
PATHS_TO_RUN: tuple[int, ...] = (1, 2, 3)
N_SAMPLES: int = CORE_SAMPLE_SIZE
SHUFFLE_SEED: int = 42
USE_CACHE: bool = True
APPEND_TO_TSV: bool = True       # if False, overwrite the TSV
RESUME_FROM_CHECKPOINT: bool = True  # skip (cell, sample_idx) pairs already in TSV
# ────────────────────────────────────────────────────────────────────────


logger = logging.getLogger("roundtrip")


def load_dataset(dataset_name: str, n: int, seed: int) -> list[dict]:
    """Load one of: core / livecodebench / humaneval_mutated."""
    raise NotImplementedError("Stub — implementation pending.")


def write_result_row(result: dict, tsv_path: Path, append: bool = True) -> None:
    """Append one ClosureResult dict as a TSV row. Creates header on first write."""
    raise NotImplementedError("Stub — implementation pending.")


def already_processed(cell_id: str, sample_idx: int, path: int,
                      tsv_path: Path) -> bool:
    """Resume support: check if (cell, sample, path) is already in the TSV."""
    raise NotImplementedError("Stub — implementation pending.")


def main() -> int:
    """Run one cell of the DOE end-to-end."""
    # 1. Set up logging
    log_path = LOGS_DIR / f"cell_{CELL_ID}_{DATASET}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )

    # 2. Look up the cell + load dataset
    cell = get_cell(CELL_ID)
    logger.info(f"Running cell {cell.cell_id} [{cell.stratum}]: {cell.hypothesis}")
    logger.info(f"  L_spec = {cell.L_spec.ollama_tag if cell.L_spec else 'SKIP'}")
    logger.info(f"  L_test = {cell.L_test.ollama_tag if cell.L_test else 'SKIP'}")
    logger.info(f"  L_code = {cell.L_code.ollama_tag if cell.L_code else 'SKIP'}")

    # 3. Iterate over (sample, path)
    #    for sample in dataset:
    #       for path in PATHS_TO_RUN:
    #          if RESUME_FROM_CHECKPOINT and already_processed(...): continue
    #          result = run_path(cell, sample, path)
    #          write_result_row(result, RESULTS_TSV)

    raise NotImplementedError("Stub — main loop implementation pending.")


if __name__ == "__main__":
    raise SystemExit(main())
