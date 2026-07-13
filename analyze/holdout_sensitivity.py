"""
analyze/holdout_sensitivity.py

W3 contamination sensitivity — reviewer response.

Compares per-cell per-path closure rates between the core HumanEval + MBPP
sweep and the held-out datasets:
    - humaneval_mutated_50: same algorithms, renamed functions and
      paraphrased docstrings, isolates surface-form memorisation
    - livecodebench_25: post-training-cutoff (>= 2024-12-01),
      uncontaminated by construction

For each (cell, path) triple present in both the core and holdout sweeps
we report the closure rate on each and the delta. Large negative deltas
on humaneval_mutated indicate surface-form memorisation; positive or
small deltas on livecodebench indicate genuine capability generalisation.

Reads:
    results/results_roundtrip.tsv                         (core sweep)
    results/results_holdout_humaneval_mutated.tsv          (may be absent)
    results/results_holdout_livecodebench.tsv              (may be absent)

Writes:
    results/holdout_sensitivity.json    — machine-readable summary
    tables/tab_holdout_sensitivity.tex   — paper-ready LaTeX table

Run:
    python3 analyze/holdout_sensitivity.py
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORE_TSV = PROJECT_ROOT / "results" / "results_roundtrip.tsv"
HEM_TSV = PROJECT_ROOT / "results" / "results_holdout_humaneval_mutated.tsv"
LCB_TSV = PROJECT_ROOT / "results" / "results_holdout_livecodebench.tsv"
OUT_JSON = PROJECT_ROOT / "results" / "holdout_sensitivity.json"
OUT_TEX = PROJECT_ROOT / "tables" / "tab_holdout_sensitivity.tex"

# Ground-truth policy for "valid closure":
#   metric_value > TAU_METRIC and judge_rating >= RHO_JUDGE and valid == True
# Matches the strict-AND policy documented in closure_decision.py
# (paper default: τ=0, ρ=3).
TAU_METRIC = 0.0
RHO_JUDGE = 3


def load_tsv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def is_valid_closure(row: dict) -> bool | None:
    """True/False = closure decision; None = structural NA (skip in rate)."""
    try:
        mv = float(row["metric_value"])
    except (KeyError, ValueError):
        return None
    if math.isnan(mv):
        return None
    try:
        jr = int(round(float(row["judge_rating"])))
    except (KeyError, ValueError):
        return None
    if jr < 0:
        return None
    if row.get("valid", "").strip().lower() != "true":
        return None
    return (mv > TAU_METRIC) and (jr >= RHO_JUDGE)


def closure_rate(rows: list[dict], cell_id: str, path: int) -> tuple[float | None, int, int]:
    """Return (rate, n_valid_decisions, n_rows). Rate is None on empty."""
    cell_rows = [r for r in rows if r["cell_id"] == cell_id and int(r["path"]) == path]
    decisions = [is_valid_closure(r) for r in cell_rows]
    kept = [d for d in decisions if d is not None]
    if not kept:
        return (None, 0, len(cell_rows))
    rate = sum(kept) / len(kept)
    return (rate, len(kept), len(cell_rows))


def wilson_ci(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% CI on a binomial proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    halfwidth = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - halfwidth), min(1.0, centre + halfwidth))


def main() -> None:
    core = load_tsv(CORE_TSV)
    hem = load_tsv(HEM_TSV)
    lcb = load_tsv(LCB_TSV)

    print(f"Core sweep     rows: {len(core)}")
    print(f"HumanEval-Mut  rows: {len(hem)}")
    print(f"LiveCodeBench  rows: {len(lcb)}")

    if not hem and not lcb:
        print("\n! No holdout results found — run scripts/w3_holdout_sweep.sh on Colab first.")
        return

    cells = sorted({r["cell_id"] for r in (core + hem + lcb)})
    paths = (1, 2, 3)

    report = {"cells": {}}
    for cell in cells:
        report["cells"][cell] = {}
        for path in paths:
            core_rate, core_n, _ = closure_rate(core, cell, path)
            hem_rate,  hem_n,  _ = closure_rate(hem,  cell, path)
            lcb_rate,  lcb_n,  _ = closure_rate(lcb,  cell, path)

            entry = {
                "core":          {"rate": core_rate, "n": core_n},
                "humaneval_mut": {"rate": hem_rate,  "n": hem_n},
                "livecodebench": {"rate": lcb_rate,  "n": lcb_n},
            }
            if core_rate is not None and hem_rate is not None:
                entry["hem_delta"] = hem_rate - core_rate
            if core_rate is not None and lcb_rate is not None:
                entry["lcb_delta"] = lcb_rate - core_rate
            report["cells"][cell][f"path_{path}"] = entry

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {OUT_JSON}")

    # ── LaTeX table ─────────────────────────────────────────────────────
    def fmt_pct(x: float | None) -> str:
        if x is None or math.isnan(x):
            return "n/a"
        return f"{100*x:.1f}\\%"

    def fmt_delta(x: float | None) -> str:
        if x is None or math.isnan(x):
            return "n/a"
        return f"{100*x:+.1f}~pp"

    priority_cells = [c for c in ("H1", "M3", "M4") if c in cells]
    other_cells = [c for c in cells if c not in priority_cells]
    ordered_cells = priority_cells + other_cells

    lines = [
        r"% Auto-generated by analyze/holdout_sensitivity.py",
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{Contamination sensitivity: strict-AND closure rate on"
        r" the core sweep (HumanEval + MBPP) vs.\ two held-out subsets,"
        r" per cell and closure path. \emph{HEM-50} is the 50-problem"
        r" HumanEval-Mutated subset (function-rename + docstring-paraphrase);"
        r" \emph{LCB-25} is the 25-problem LiveCodeBench post-2024-12-01"
        r" subset (post-training-cutoff for all pipeline SLMs). Large"
        r" negative $\Delta$ on HEM-50 would signal surface-form"
        r" memorisation; small or positive $\Delta$ on LCB-25 signals"
        r" capability generalisation.}",
        r"  \label{tab:holdout_sensitivity}",
        r"  \begin{tabular}{llrrrrr}",
        r"    \toprule",
        r"    Cell & Path & Core & HEM-50 & $\Delta_{\textsf{HEM}}$"
        r" & LCB-25 & $\Delta_{\textsf{LCB}}$ \\",
        r"    \midrule",
    ]
    for cell in ordered_cells:
        for path in paths:
            e = report["cells"][cell][f"path_{path}"]
            hem_delta = e.get("hem_delta")
            lcb_delta = e.get("lcb_delta")
            lines.append(
                f"    {cell} & {path} & "
                f"{fmt_pct(e['core']['rate'])} & "
                f"{fmt_pct(e['humaneval_mut']['rate'])} & "
                f"{fmt_delta(hem_delta)} & "
                f"{fmt_pct(e['livecodebench']['rate'])} & "
                f"{fmt_delta(lcb_delta)} \\\\"
            )
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]
    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    with OUT_TEX.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Wrote {OUT_TEX}")

    # Summary print
    print("\n── Summary ─────────────────────────────────────────────────")
    print(f"{'cell':<4} {'path':<4} {'core':>10} {'HEM':>10} {'ΔHEM':>10} "
          f"{'LCB':>10} {'ΔLCB':>10}")
    for cell in ordered_cells:
        for path in paths:
            e = report["cells"][cell][f"path_{path}"]
            print(f"{cell:<4} {path:<4} "
                  f"{fmt_pct(e['core']['rate']):>10} "
                  f"{fmt_pct(e['humaneval_mut']['rate']):>10} "
                  f"{fmt_delta(e.get('hem_delta')):>10} "
                  f"{fmt_pct(e['livecodebench']['rate']):>10} "
                  f"{fmt_delta(e.get('lcb_delta')):>10}")


if __name__ == "__main__":
    main()
