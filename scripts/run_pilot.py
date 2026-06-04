"""
scripts/run_pilot.py — 30-function pilot driver.

Per the concept-note §6 plan:
    - 30 functions sampled from the core dataset (deterministic seed=42)
    - 6 cells (doe.PILOT_CELLS: M1, M3, M6, H1, H4, N2)
    - All 3 closure paths per (cell, function)
    - Results written to results/pilot_results.tsv
    - 6 go/no-go checks evaluated at the end
    - Pilot summary written to results/pilot_summary.md

This script is the gate between the engineering phase and the full
sweep. If the 6 go/no-go checks pass, proceed to the 150-function
full sweep (train_roundtrip.py for each cell). If any fail, fix the
underlying issue before scaling up.

Resumability: same three-layer strategy as train_roundtrip.py
(LLM cache + result-level resume + atomic writes), so a Colab
disconnect mid-pilot just means re-running picks up where it left off.

Run as:
    python3 scripts/run_pilot.py
"""

from __future__ import annotations
import logging
import sys
import time
from pathlib import Path

# Add project root to path so imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    DATA_DIR,
    LOGS_DIR,
    RESULTS_DIR,
    PILOT_RESULTS_TSV,
)
from doe import PILOT_CELLS
import closure_cache
import closure_paths
import train_roundtrip


logger = logging.getLogger("pilot")


# ════════════════════════════════════════════════════════════════════════
# Pilot configuration
# ════════════════════════════════════════════════════════════════════════
PILOT_N_SAMPLES = 30
PILOT_PATHS = (1, 2, 3)
PILOT_DATASET = "core"          # core_sample_150.jsonl
PILOT_SAMPLE_INDICES = range(PILOT_N_SAMPLES)   # first 30 of the core sample
# ════════════════════════════════════════════════════════════════════════


# ──────────────────────────────────────────────────────────────────────
# Pilot dataset loader
# ──────────────────────────────────────────────────────────────────────
def load_pilot_dataset() -> list[dict]:
    """Take the first 30 samples of the core dataset."""
    full = train_roundtrip.load_dataset(PILOT_DATASET, n=PILOT_N_SAMPLES)
    if len(full) < PILOT_N_SAMPLES:
        logger.warning(
            f"  Core dataset has only {len(full)} samples (wanted {PILOT_N_SAMPLES})."
            f" Continuing with what's available."
        )
    return full


# ──────────────────────────────────────────────────────────────────────
# Pilot sweep
# ──────────────────────────────────────────────────────────────────────
def run_pilot() -> dict:
    """Execute all 6 pilot cells. Returns aggregate summary."""
    pilot_log = LOGS_DIR / "pilot.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(pilot_log),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )

    logger.info("════════════════════════════════════════════════════════")
    logger.info("Pilot run: 30 functions × 6 cells × 3 paths = 540 round-trips")
    logger.info("════════════════════════════════════════════════════════")
    logger.info(f"  Results TSV: {PILOT_RESULTS_TSV}")
    logger.info(f"  Pilot cells: {[c.cell_id for c in PILOT_CELLS]}")

    # Initialise the TSV (writes header if absent)
    train_roundtrip.ensure_header(PILOT_RESULTS_TSV)
    completed = train_roundtrip.load_completed_keys(PILOT_RESULTS_TSV)
    logger.info(f"  Found {len(completed)} previously-completed tuples (resume).")

    try:
        dataset = load_pilot_dataset()
    except FileNotFoundError as e:
        logger.error(str(e))
        logger.error("  Run prepare_roundtrip.py first to build the core sample.")
        return {"status": "no_dataset"}

    logger.info(f"  Loaded {len(dataset)} samples for pilot.")
    print()

    overall = {
        "cells_processed": 0,
        "n_results_written": 0,
        "n_results_skipped_resume": 0,
        "n_errors": 0,
        "elapsed_s": 0.0,
        "per_cell": {},
    }
    t_start = time.perf_counter()

    for cell in PILOT_CELLS:
        logger.info(f"───  cell {cell.cell_id} [{cell.stratum}]  ───")
        logger.info(f"   {cell.hypothesis}")
        cell_summary = train_roundtrip.run_cell(
            cell, dataset, PILOT_PATHS, PILOT_RESULTS_TSV, completed
        )
        overall["cells_processed"] += 1
        overall["n_results_written"] += cell_summary["n_results_written"]
        overall["n_results_skipped_resume"] += cell_summary["n_results_skipped_resume"]
        overall["n_errors"] += cell_summary["n_errors"]
        overall["per_cell"][cell.cell_id] = cell_summary
        logger.info(
            f"   {cell.cell_id}: written={cell_summary['n_results_written']}, "
            f"skipped={cell_summary['n_results_skipped_resume']}, "
            f"errors={cell_summary['n_errors']}, "
            f"elapsed={cell_summary['elapsed_s']:.1f}s"
        )
        print()

    overall["elapsed_s"] = time.perf_counter() - t_start
    overall["cache_stats"] = closure_cache.stats()
    return overall


# ──────────────────────────────────────────────────────────────────────
# Go/No-Go checks (concept note §6)
# ──────────────────────────────────────────────────────────────────────
def go_no_go_checks(tsv_path: Path) -> dict:
    """
    Evaluate the 6 go/no-go checks on the pilot output. Each returns
    PASS / FAIL / SKIP and a one-line explanation.
    """
    import csv
    import math

    checks: list[dict] = []

    if not tsv_path.exists() or tsv_path.stat().st_size == 0:
        return {"checks": [], "verdict": "NO_DATA", "message": "TSV is empty"}

    rows: list[dict] = []
    with tsv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)

    if not rows:
        return {"checks": [], "verdict": "NO_DATA", "message": "no rows in TSV"}

    # Check 1: No path-execution errors
    error_rows = [r for r in rows if r.get("metric_name") == "error"]
    checks.append({
        "name": "1. No path-execution errors",
        "result": "PASS" if not error_rows else "FAIL",
        "detail": f"{len(error_rows)} error rows out of {len(rows)}",
    })

    # Check 2: Cache deterministic — measured as cache_hits > 0 on
    #          some path of some cell (if pilot already ran once)
    total_hits = sum(int(r.get("cache_hits", 0)) for r in rows)
    total_calls = sum(int(r.get("n_llm_calls", 0)) for r in rows)
    hit_rate = total_hits / max(total_calls, 1)
    checks.append({
        "name": "2. Cache layer is active",
        "result": "PASS" if total_hits > 0 else "SKIP (1st run)",
        "detail": f"hits={total_hits}, calls={total_calls}, hit_rate={hit_rate:.2%}",
    })

    # Check 3: <5% NaN in mutation_kill_rate
    kill_rows = [r for r in rows if r.get("metric_name") == "mutation_kill_rate"]
    nan_kill = sum(1 for r in kill_rows
                   if _is_nan(r.get("metric_value", "nan")))
    nan_ratio = nan_kill / max(len(kill_rows), 1)
    checks.append({
        "name": "3. <5% NaN in mutation_kill_rate (Path 1)",
        "result": "PASS" if nan_ratio < 0.05 else "WARN",
        "detail": f"{nan_kill}/{len(kill_rows)} = {nan_ratio:.1%} NaN",
    })

    # Check 4: Judge LLM produces valid ratings (0-4)
    judged = [int(r["judge_rating"]) for r in rows
              if r.get("judge_rating", "").lstrip("-").isdigit()]
    valid_judged = [r for r in judged if 0 <= r <= 4]
    invalid_judged = [r for r in judged if r == -1]
    if judged:
        valid_frac = len(valid_judged) / len(judged)
    else:
        valid_frac = 0.0
    if not judged:
        check4_result = "SKIP (no judged rows)"
    elif valid_frac >= 0.95:
        check4_result = "PASS"
    elif valid_frac >= 0.50:
        check4_result = "WARN"
    else:
        check4_result = "FAIL (judge model likely not pulled)"
    checks.append({
        "name": "4. Judge LLM produces valid 0-4 ratings",
        "result": check4_result,
        "detail": f"valid={len(valid_judged)}, parse_fail={len(invalid_judged)}, "
                  f"valid_frac={valid_frac:.1%}",
    })

    # Check 5: All required columns present and parseable (structural)
    required_cols = closure_paths.ClosureResult.TSV_COLUMNS
    cols_in_tsv = set(rows[0].keys()) if rows else set()
    missing_cols = [c for c in required_cols if c not in cols_in_tsv]
    checks.append({
        "name": "5. TSV schema is well-formed",
        "result": "PASS" if not missing_cols else "FAIL",
        "detail": f"missing columns: {missing_cols}" if missing_cols else f"{len(required_cols)} columns present",
    })

    # Check 6: Per-cell valid sample count ≥ 20
    per_cell_valid = {}
    for r in rows:
        if r.get("valid") == "True":
            per_cell_valid[r["cell_id"]] = per_cell_valid.get(r["cell_id"], 0) + 1
    n_min = min(per_cell_valid.values()) if per_cell_valid else 0
    weakest = min(per_cell_valid, key=per_cell_valid.get) if per_cell_valid else None
    checks.append({
        "name": "6. Per-cell valid count ≥ 20",
        "result": "PASS" if n_min >= 20 else "WARN",
        "detail": f"weakest cell: {weakest} with n={n_min}; per-cell: {per_cell_valid}",
    })

    # Verdict
    has_fail = any(c["result"].startswith("FAIL") for c in checks)
    has_warn = any(c["result"].startswith("WARN") for c in checks)
    if has_fail:
        verdict = "NO_GO"
    elif has_warn:
        verdict = "GO_WITH_NOTES"
    else:
        verdict = "GO"

    return {"checks": checks, "verdict": verdict,
            "message": f"{sum(c['result'] == 'PASS' for c in checks)}/{len(checks)} PASS"}


def _is_nan(s: str) -> bool:
    s = str(s).strip().lower()
    return s in {"nan", "", "none"}


# ──────────────────────────────────────────────────────────────────────
# Summary writer
# ──────────────────────────────────────────────────────────────────────
def write_pilot_summary(summary: dict, checks_result: dict,
                        output_path: Path) -> None:
    """Write a human-readable summary to results/pilot_summary.md."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# Pilot run summary\n")
    lines.append(f"**Verdict:** **{checks_result.get('verdict', 'UNKNOWN')}**  "
                 f"({checks_result.get('message', '')})\n\n")

    lines.append("## Cells run\n")
    lines.append("| Cell | Stratum | Written | Skipped (resume) | Errors | Elapsed |")
    lines.append("|---|---|---|---|---|---|")
    for cell_id, cs in summary.get("per_cell", {}).items():
        # Look up stratum from doe.py
        stratum = next((c.stratum for c in PILOT_CELLS if c.cell_id == cell_id), "?")
        lines.append(
            f"| {cell_id} | {stratum} | "
            f"{cs.get('n_results_written', 0)} | "
            f"{cs.get('n_results_skipped_resume', 0)} | "
            f"{cs.get('n_errors', 0)} | "
            f"{cs.get('elapsed_s', 0):.1f}s |"
        )
    lines.append("")

    cache = summary.get("cache_stats", {})
    lines.append("## Cache\n")
    lines.append(f"- hits: {cache.get('hits', 0)}")
    lines.append(f"- misses: {cache.get('misses', 0)}")
    lines.append(f"- hit_rate: {cache.get('hit_rate', 0):.2%}")
    lines.append(f"- entries: {cache.get('entry_count', 0)}")
    lines.append(f"- size: {cache.get('size_mb', 0)} MB\n")

    lines.append("## Go/No-Go checks\n")
    lines.append("| # | Check | Result | Detail |")
    lines.append("|---|---|---|---|")
    for c in checks_result.get("checks", []):
        lines.append(f"| {c['name'][0]} | {c['name'][3:]} | **{c['result']}** | {c['detail']} |")
    lines.append("")

    lines.append("## Aggregate\n")
    lines.append(f"- Cells processed: {summary.get('cells_processed', 0)}")
    lines.append(f"- Total results written: {summary.get('n_results_written', 0)}")
    lines.append(f"- Skipped on resume: {summary.get('n_results_skipped_resume', 0)}")
    lines.append(f"- Errors: {summary.get('n_errors', 0)}")
    lines.append(f"- Total elapsed: {summary.get('elapsed_s', 0):.1f}s\n")

    if checks_result.get("verdict") == "GO":
        lines.append("\n**→ Pilot passes. Proceed to the full 150-function sweep.**\n")
    elif checks_result.get("verdict") == "GO_WITH_NOTES":
        lines.append("\n**→ Pilot passes with warnings. Review warnings before scaling up.**\n")
    elif checks_result.get("verdict") == "NO_GO":
        lines.append("\n**→ Pilot FAILED. Fix issues before running the full sweep.**\n")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print()
    print(f"Pilot summary written: {output_path}")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def main() -> int:
    summary = run_pilot()
    if summary.get("status") == "no_dataset":
        return 2
    checks = go_no_go_checks(PILOT_RESULTS_TSV)

    print()
    print("════════════════════════════════════════════════════════")
    print(f"VERDICT: {checks.get('verdict', '?')}")
    print(f"  {checks.get('message', '')}")
    print("════════════════════════════════════════════════════════\n")
    for c in checks.get("checks", []):
        marker = {"PASS": "✓", "FAIL": "✗", "WARN": "⚠", "SKIP": "—"}.get(
            c["result"].split()[0], "?"
        )
        print(f"  {marker} {c['name']}: {c['result']}")
        print(f"       {c['detail']}")

    summary_md = RESULTS_DIR / "pilot_summary.md"
    write_pilot_summary(summary, checks, summary_md)

    return 0 if checks.get("verdict") in ("GO", "GO_WITH_NOTES") else 1


if __name__ == "__main__":
    raise SystemExit(main())
