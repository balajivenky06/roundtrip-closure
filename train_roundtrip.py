"""
train_roundtrip.py — main experiment driver with Colab-disconnect-safe resume.

Runs one DOE cell (set CELL_ID in the CONFIG block below) over a dataset
and writes one TSV row per (cell, sample_idx, path). Resilient to mid-run
process kills because:

    1. The `closure_cache` (SHA256-keyed disk store) makes every identical
       LLM call free on re-execution.
    2. The results TSV is opened in append mode with fsync after each
       row, so partial state survives a Colab kill.
    3. Before processing each (cell, sample_idx, path) tuple we check
       whether its key is already in the TSV — if yes we skip.

Run as:
    python3 train_roundtrip.py             # uses CELL_ID below
    CELL_ID=H1 python3 train_roundtrip.py  # override via env
"""

from __future__ import annotations
import json
import logging
import os
import sys
import time
from pathlib import Path

from config import (
    CORE_SAMPLE_SIZE,
    DATA_DIR,
    LOGS_DIR,
    RESULTS_TSV,
    DATASET_SEED,
)
from doe import get_cell, ALL_CELLS, Cell
import closure_cache
import closure_paths


# ════════════════════════════════════════════════════════════════════════
# CONFIGURATION — edit per cell, or override via environment variables
# ════════════════════════════════════════════════════════════════════════
CELL_ID: str = os.environ.get("CELL_ID", "M1")
DATASET: str = os.environ.get("DATASET", "core")     # core | livecodebench | humaneval_mutated
PATHS_TO_RUN: tuple[int, ...] = tuple(
    int(p) for p in os.environ.get("PATHS_TO_RUN", "1,2,3").split(",")
)
N_SAMPLES: int = int(os.environ.get("N_SAMPLES", CORE_SAMPLE_SIZE))
USE_CACHE: bool = os.environ.get("USE_CACHE", "1") != "0"
RESULTS_PATH: Path = Path(os.environ.get("RESULTS_PATH", str(RESULTS_TSV)))
SHOW_PROGRESS_EVERY: int = int(os.environ.get("PROGRESS_EVERY", "1"))    # per-result by default
# ════════════════════════════════════════════════════════════════════════


logger = logging.getLogger("roundtrip")


# ──────────────────────────────────────────────────────────────────────
# TSV I/O with crash-safe writes
# ──────────────────────────────────────────────────────────────────────
def ensure_header(tsv_path: Path) -> None:
    """
    Ensure the TSV has an up-to-date header.

    Three cases:
        1. File doesn't exist / is empty  → write full header, done.
        2. File exists and header already matches ClosureResult.TSV_COLUMNS
           exactly  → nothing to do.
        3. File exists with an OLDER header (fewer columns; pre-Algorithm-2/3
           schema)  → migrate: rewrite the header to the current schema and
           append empty values for the missing columns to every existing row.
           Migration is atomic — writes to a .tmp file then os.replace.
    """
    expected_cols = list(closure_paths.ClosureResult.TSV_COLUMNS)
    expected_header = "\t".join(expected_cols) + "\n"

    if not tsv_path.exists() or tsv_path.stat().st_size == 0:
        with tsv_path.open("w", encoding="utf-8") as f:
            f.write(expected_header)
            f.flush()
            os.fsync(f.fileno())
        return

    with tsv_path.open("r", encoding="utf-8") as f:
        first_line = f.readline().rstrip("\n")
    existing_cols = first_line.split("\t") if first_line else []

    if existing_cols == expected_cols:
        return  # already current schema

    # Migrate: rewrite with new header + pad every existing row with empty
    # values for the new columns. Preserve the old column order and append
    # only the delta at the end.
    if not existing_cols:
        return  # empty / unreadable header — leave alone

    missing_cols = [c for c in expected_cols if c not in existing_cols]
    if not missing_cols:
        # Header has all expected columns but in a different order — leave
        # as-is (unusual; would break parsers if we reordered).
        logger.warning(
            f"TSV header columns match by set but not by order; "
            f"leaving unchanged. header = {existing_cols!r}"
        )
        return

    logger.info(
        f"Migrating TSV schema: adding empty columns {missing_cols} to "
        f"{tsv_path.name}"
    )

    tmp_path = tsv_path.with_suffix(tsv_path.suffix + ".migrating")
    n_padded = 0
    with tsv_path.open("r", encoding="utf-8") as src, \
         tmp_path.open("w", encoding="utf-8") as dst:
        src.readline()  # skip old header
        # Write new header with old cols first (preserving order) then new cols
        # at the end. This matches the append-at-end policy in ClosureResult.
        new_header_cols = existing_cols + missing_cols
        dst.write("\t".join(new_header_cols) + "\n")

        # Sanity-check that missing_cols are appended (not interleaved)
        # against expected_cols to detect drift.
        if new_header_cols != expected_cols:
            logger.warning(
                f"Migrated header does not exactly match ClosureResult "
                f"schema. Migrated: {new_header_cols}. Expected: "
                f"{expected_cols}. Loader must map by name."
            )

        pad = "\t".join("" for _ in missing_cols)
        for line in src:
            line = line.rstrip("\n")
            if not line:
                dst.write("\n")
                continue
            dst.write(line + "\t" + pad + "\n")
            n_padded += 1
        dst.flush()
        os.fsync(dst.fileno())

    os.replace(tmp_path, tsv_path)
    logger.info(f"TSV schema migration complete: {n_padded:,} rows padded.")


def load_completed_keys(tsv_path: Path) -> set[str]:
    """
    Scan the TSV once at startup and return the set of (cell|sample|path)
    keys that already have a row. The set is used to skip already-processed
    work on resume.
    """
    completed: set[str] = set()
    if not tsv_path.exists():
        return completed
    with tsv_path.open("r", encoding="utf-8") as f:
        header = f.readline()  # skip header
        if not header:
            return completed
        # Column indices for the three resume-key fields
        cols = header.rstrip("\n").split("\t")
        try:
            i_cell = cols.index("cell_id")
            i_sample = cols.index("sample_idx")
            i_path = cols.index("path")
        except ValueError:
            logger.warning("Results TSV has unexpected header; treating all rows as unknown.")
            return completed
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= max(i_cell, i_sample, i_path):
                continue
            try:
                key = closure_paths.ClosureResult.key(
                    parts[i_cell], int(parts[i_sample]), int(parts[i_path])
                )
                completed.add(key)
            except ValueError:
                continue
    return completed


def write_result_row(result: closure_paths.ClosureResult, tsv_path: Path) -> None:
    """Append one ClosureResult, flushed and fsynced for crash safety."""
    with tsv_path.open("a", encoding="utf-8") as f:
        f.write(result.to_tsv_row())
        f.flush()
        os.fsync(f.fileno())


# ──────────────────────────────────────────────────────────────────────
# Dataset loading
# ──────────────────────────────────────────────────────────────────────
def load_dataset(dataset_name: str, n: int) -> list[dict]:
    """
    Load one of: core / livecodebench / humaneval_mutated.

    Expects pre-built JSONL files in data/:
        data/core_sample_150.jsonl
        data/livecodebench_25.jsonl
        data/humaneval_mutated_50.jsonl

    These are generated by prepare_roundtrip.py (Batch 4).
    """
    filename_map = {
        "core": "core_sample_150.jsonl",
        "livecodebench": "livecodebench_25.jsonl",
        "humaneval_mutated": "humaneval_mutated_50.jsonl",
    }
    if dataset_name not in filename_map:
        raise ValueError(
            f"Unknown dataset {dataset_name!r}. "
            f"Choose from: {sorted(filename_map.keys())}"
        )

    path = DATA_DIR / filename_map[dataset_name]
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not built yet: {path}\n"
            f"Run: python3 prepare_roundtrip.py"
        )

    samples: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples[:n] if n > 0 else samples


# ──────────────────────────────────────────────────────────────────────
# Main loop
# ──────────────────────────────────────────────────────────────────────
def run_cell(cell: Cell,
             dataset: list[dict],
             paths: tuple[int, ...],
             results_path: Path,
             completed_keys: set[str]) -> dict:
    """
    Process the dataset for one DOE cell. Returns a small summary dict
    used by the caller for logging.
    """
    summary = {
        "cell_id": cell.cell_id,
        "n_samples_seen": 0,
        "n_results_written": 0,
        "n_results_skipped_resume": 0,
        "n_errors": 0,
        "elapsed_s": 0.0,
    }
    start = time.perf_counter()

    for sample in dataset:
        summary["n_samples_seen"] += 1
        sample_idx = sample.get("sample_idx", -1)

        for p in paths:
            key = closure_paths.ClosureResult.key(cell.cell_id, sample_idx, p)
            if key in completed_keys:
                summary["n_results_skipped_resume"] += 1
                continue

            try:
                # Each path returns ONE ClosureResult (see closure_paths.py)
                if p == 1:
                    result = closure_paths.run_path_1(cell, sample)
                elif p == 2:
                    result = closure_paths.run_path_2(cell, sample)
                elif p == 3:
                    result = closure_paths.run_path_3(cell, sample)
                else:
                    logger.warning(f"Unknown path id {p}; skipping.")
                    continue
            except Exception as exc:                              # pragma: no cover
                summary["n_errors"] += 1
                logger.exception(
                    f"Cell {cell.cell_id} sample {sample_idx} path {p} crashed: {exc}"
                )
                continue

            write_result_row(result, results_path)
            completed_keys.add(key)
            summary["n_results_written"] += 1

            if summary["n_results_written"] % SHOW_PROGRESS_EVERY == 0:
                cs = closure_cache.stats()
                logger.info(
                    f"  written={summary['n_results_written']} "
                    f"skipped(resume)={summary['n_results_skipped_resume']} "
                    f"errors={summary['n_errors']} "
                    f"cache_hit_rate={cs['hit_rate']:.2%} "
                    f"(hits={cs['hits']}, misses={cs['misses']})"
                )

    summary["elapsed_s"] = time.perf_counter() - start
    return summary


# ──────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────
def main() -> int:
    log_path = LOGS_DIR / f"cell_{CELL_ID}_{DATASET}.log"

    # Line-buffered stdout so Colab !python3 cells stream output in real
    # time instead of buffering until the script exits.
    sys.stdout.reconfigure(line_buffering=True)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.flush = sys.stdout.flush  # extra-defensive

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            stream_handler,
        ],
        force=True,
    )

    cell = get_cell(CELL_ID)
    logger.info(f"=== Running cell {cell.cell_id} [{cell.stratum}] on {DATASET} ===")
    logger.info(f"  Hypothesis: {cell.hypothesis}")
    logger.info(f"  L_spec = {cell.L_spec.ollama_tag if cell.L_spec else 'SKIP'}")
    logger.info(f"  L_test = {cell.L_test.ollama_tag if cell.L_test else 'SKIP'}")
    logger.info(f"  L_code = {cell.L_code.ollama_tag if cell.L_code else 'SKIP'}")
    logger.info(f"  paths = {PATHS_TO_RUN}, n_samples = {N_SAMPLES}")
    logger.info(f"  results -> {RESULTS_PATH}")
    logger.info(f"  log     -> {log_path}")

    # 1. Header + completed-key index (resume support)
    ensure_header(RESULTS_PATH)
    completed = load_completed_keys(RESULTS_PATH)
    logger.info(f"  found {len(completed)} previously-completed (cell, sample, path) tuples")

    # 2. Load dataset
    try:
        dataset = load_dataset(DATASET, N_SAMPLES)
    except FileNotFoundError as e:
        logger.error(str(e))
        return 1
    logger.info(f"  loaded {len(dataset)} samples from {DATASET}")

    # 3. Run the cell
    try:
        summary = run_cell(cell, dataset, PATHS_TO_RUN, RESULTS_PATH, completed)
    except KeyboardInterrupt:                                     # pragma: no cover
        logger.warning("Interrupted by user. State preserved in TSV + cache; re-run to resume.")
        return 130

    # 4. Final summary
    cs = closure_cache.stats()
    logger.info(
        f"\n=== Cell {cell.cell_id} complete ===\n"
        f"  samples_seen          = {summary['n_samples_seen']}\n"
        f"  results_written       = {summary['n_results_written']}\n"
        f"  results_skipped_resume = {summary['n_results_skipped_resume']}\n"
        f"  errors                 = {summary['n_errors']}\n"
        f"  elapsed                = {summary['elapsed_s']:.1f} s\n"
        f"  cache_hits             = {cs['hits']}\n"
        f"  cache_misses           = {cs['misses']}\n"
        f"  cache_hit_rate         = {cs['hit_rate']:.2%}\n"
        f"  cache_size             = {cs['size_mb']} MB ({cs['entry_count']} entries)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
