"""
analyze/rq4_closed_vs_open.py

RQ4 analysis — closed-weight ceiling vs open-weight closure.

Reads:
    results/results_roundtrip.tsv                (primary open-weight sweep)
    results/results_rq4_closed_weight.tsv        (Claude + GPT-4o-mini sweep)

Computes per-cell per-path strict-AND closure rates and the pairwise
comparisons that matter for RQ4:

    - M5_closed (Claude mono) vs M4 (best open-weight mono, qwen-coder)
    - M5_closed vs M3 (2nd best open-weight mono, qwen3.6)
    - M7_gpt (GPT-4o-mini mono) vs M4 and M3
    - H2_closed (Claude spec + open test/code) vs H1 (best open-weight hetero)
    - H8_closed (Claude spec + Claude test + open code) vs H1, H4

Emits:
    results/rq4_closed_vs_open.json          — machine-readable
    tables/tab_rq4_closed_vs_open.tex        — paper-ready LaTeX
    plots/output/fig_rq4_closed_vs_open.png  — closure-rate bar chart (optional)

Run:
    python3 analyze/rq4_closed_vs_open.py
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OPEN_TSV = PROJECT_ROOT / "results" / "results_roundtrip.tsv"
CLOSED_TSV = PROJECT_ROOT / "results" / "results_rq4_closed_weight.tsv"
OUT_JSON = PROJECT_ROOT / "results" / "rq4_closed_vs_open.json"
OUT_TEX = PROJECT_ROOT / "tables" / "tab_rq4_closed_vs_open.tex"


def load_tsv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def is_valid_closure(row: dict, tau: float = 0.0, rho: int = 3) -> bool | None:
    """Strict-AND validity per Algorithm 2. Returns None on structural NA."""
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
    return (mv > tau) and (jr >= rho)


def closure_rate(rows: list[dict], cell_id: str, path: int) -> tuple[float | None, int]:
    """Return (rate, n_valid_decisions) for a given cell × path."""
    cell_rows = [r for r in rows if r["cell_id"] == cell_id and int(r["path"]) == path]
    decisions = [is_valid_closure(r) for r in cell_rows]
    kept = [d for d in decisions if d is not None]
    if not kept:
        return (None, 0)
    return (sum(kept) / len(kept), len(kept))


def wilson_ci(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    hw = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - hw), min(1.0, centre + hw))


def main() -> None:
    open_rows = load_tsv(OPEN_TSV)
    closed_rows = load_tsv(CLOSED_TSV)
    print(f"Open-weight sweep  rows: {len(open_rows)}")
    print(f"RQ4 closed sweep   rows: {len(closed_rows)}")

    if not closed_rows:
        print("\n! No RQ4 closed-weight results found — run scripts/rq4_closed_weight_sweep.sh first.")
        return

    all_rows = open_rows + closed_rows

    # Cells to report; each row is (closed_cell, open_baseline_cell, comparison_note)
    comparisons = [
        ("M5_closed", "M4", "Claude Sonnet 4.5 mono vs qwen-coder mono (best open Path 1/3)"),
        ("M5_closed", "M3", "Claude Sonnet 4.5 mono vs qwen3.6 mono (best open Path 2)"),
        ("M7_gpt",    "M4", "GPT-4o-mini mono vs qwen-coder mono"),
        ("M7_gpt",    "M3", "GPT-4o-mini mono vs qwen3.6 mono"),
        ("H2_closed", "H1", "Closed spec + open test/code vs best open hetero (H1)"),
        ("H8_closed", "H1", "Closed spec + closed test + open code vs H1"),
        ("H8_closed", "H4", "H8_closed vs best-Path-1 open hetero (H4)"),
    ]

    report: dict = {"comparisons": [], "raw_closure": {}}

    print(f"\n{'cmp':<50} {'path':>4} {'closed':>10} {'open':>10} {'delta':>8}")
    for closed, open_c, note in comparisons:
        entry = {"closed": closed, "open": open_c, "note": note, "paths": {}}
        for path in (1, 2, 3):
            cr, cn = closure_rate(all_rows, closed, path)
            oor, on = closure_rate(all_rows, open_c, path)
            if cr is None or oor is None:
                delta = None
            else:
                delta = cr - oor
            entry["paths"][str(path)] = {
                "closed_rate": cr, "closed_n": cn,
                "open_rate": oor, "open_n": on,
                "delta": delta,
            }
            def fmt(x):
                return "n/a" if x is None else f"{100*x:.1f}%"
            def fmtd(x):
                return "n/a" if x is None else f"{100*x:+.1f}pp"
            print(f"  {note[:48]:<50} {path:>4} "
                  f"{fmt(cr):>10} {fmt(oor):>10} {fmtd(delta):>8}")
        report["comparisons"].append(entry)
        print()

    # Raw closure rates for all cells (for reference table)
    all_cells = sorted({r["cell_id"] for r in all_rows})
    for c in all_cells:
        report["raw_closure"][c] = {}
        for path in (1, 2, 3):
            rate, n = closure_rate(all_rows, c, path)
            report["raw_closure"][c][f"path_{path}"] = {"rate": rate, "n": n}

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {OUT_JSON}")

    # LaTeX table
    def fmt(x):
        return "n/a" if x is None else f"{100*x:.1f}\\%"
    def fmtd(x):
        return "n/a" if x is None else f"{100*x:+.1f}~pp"

    lines = [
        r"% Auto-generated by analyze/rq4_closed_vs_open.py",
        r"\begin{table}[t]",
        r"  \centering",
        r"  \small",
        r"  \caption{RQ4: closed-weight closure rate versus open-weight baseline,"
        r" per cell and closure path. Closed-weight cells are Claude Sonnet 4.5"
        r" (\emph{M5\_closed}, \emph{H2\_closed}, \emph{H8\_closed}) and"
        r" GPT-4o-mini (\emph{M7\_gpt}) via OpenRouter. Open baselines: M4"
        r" (qwen-coder mono), M3 (qwen3.6 mono), H1 (best open hetero)."
        r" Positive $\Delta$ = closed-weight ceiling premium.}",
        r"  \label{tab:rq4_closed_vs_open}",
        r"  \begin{tabular}{lllrrr}",
        r"    \toprule",
        r"    Closed cell & Open baseline & Path & Closed rate & Open rate & $\Delta$ \\",
        r"    \midrule",
    ]
    for e in report["comparisons"]:
        for path in (1, 2, 3):
            p = e["paths"][str(path)]
            lines.append(
                f"    {e['closed'].replace('_', r'\_')} & "
                f"{e['open']} & {path} & "
                f"{fmt(p['closed_rate'])} & "
                f"{fmt(p['open_rate'])} & "
                f"{fmtd(p['delta'])} \\\\"
            )
        lines.append(r"    \midrule")
    # Drop trailing midrule
    if lines[-1].strip() == r"\midrule":
        lines.pop()
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]

    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    with OUT_TEX.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Wrote {OUT_TEX}")


if __name__ == "__main__":
    main()
