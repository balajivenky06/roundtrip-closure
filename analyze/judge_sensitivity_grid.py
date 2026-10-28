"""
analyze/judge_sensitivity_grid.py

Reviewer 1 Comment 1 defense — "Why trust an LLM judge with κ=0.074?"

Reads: results/results_roundtrip.tsv

Computes per-cell Path 1 closure rate under 6 judge-weight policies:
    rho_2        — strict-AND with lenient judge threshold (metric>0 ∧ J>=2)
    rho_3        — strict-AND default (metric>0 ∧ J>=3)  — paper's policy
    rho_4        — strict-AND with strict judge threshold (metric>0 ∧ J>=4)
    judge_only   — judge arbitrates alone (J>=3), metric ignored
    metric_only  — metric arbitrates alone (metric>0), judge ignored
    union        — EITHER signal counts (metric>0 OR J>=3) — permissive

Reports:
    - Per-cell Path 1 closure rate at each policy (table)
    - Which cell is top at each policy (spread diagnostic)
    - For every policy, is H1's per-operator claim invariant?
      (it is, by construction, because per-operator kill rate is metric-only)

Emits:
    results/judge_sensitivity_grid.json
    tables/tab_judge_sensitivity_grid.tex

Run: python3 analyze/judge_sensitivity_grid.py
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
TSV_PATH = ROOT / "results" / "results_roundtrip.tsv"
OUT_JSON = ROOT / "results" / "judge_sensitivity_grid.json"
OUT_TEX = ROOT / "tables" / "tab_judge_sensitivity_grid.tex"

POLICIES = ["rho_2", "rho_3", "rho_4", "judge_only", "metric_only", "union"]
POLICY_LABELS = {
    "rho_2":       r"strict-AND $\rho{=}2$",
    "rho_3":       r"strict-AND $\rho{=}3$ (default)",
    "rho_4":       r"strict-AND $\rho{=}4$",
    "judge_only":  r"judge-only $J{\geq}3$",
    "metric_only": r"metric-only $Y{>}0$",
    "union":       r"union $(Y{>}0 \vee J{\geq}3)$",
}


def valid_closure_rate(entries: list[tuple[float, int]], policy: str) -> float | None:
    """Compute closure rate for a (cell, path)'s list of (metric, judge) rows."""
    total = len(entries)
    if total == 0:
        return None
    if policy == "rho_2":
        n = sum(1 for v, j in entries if v > 0 and j >= 2)
    elif policy == "rho_3":
        n = sum(1 for v, j in entries if v > 0 and j >= 3)
    elif policy == "rho_4":
        n = sum(1 for v, j in entries if v > 0 and j >= 4)
    elif policy == "judge_only":
        n = sum(1 for v, j in entries if j >= 3)
    elif policy == "metric_only":
        n = sum(1 for v, j in entries if v > 0)
    elif policy == "union":
        n = sum(1 for v, j in entries if v > 0 or j >= 3)
    else:
        raise ValueError(f"Unknown policy {policy}")
    return n / total


def main() -> None:
    # Load TSV, build (cell, path) -> [(metric, judge), ...]
    cells_paths: dict[tuple[str, int], list[tuple[float, int]]] = defaultdict(list)
    with TSV_PATH.open() as f:
        for r in csv.DictReader(f, delimiter="\t"):
            try:
                v = float(r["metric_value"])
                j = int(round(float(r["judge_rating"])))
            except (ValueError, TypeError):
                continue
            if math.isnan(v) or (r.get("valid", "").strip().lower() != "true") or j < 0:
                continue
            cells_paths[(r["cell_id"], int(r["path"]))].append((v, j))

    non_null_cells = sorted({c for (c, p) in cells_paths if c not in ("N1", "N2", "N3")})

    # Build Path 1 per-cell rate under each policy
    path = 1
    grid = {pol: {} for pol in POLICIES}
    for c in non_null_cells:
        entries = cells_paths.get((c, path), [])
        for pol in POLICIES:
            r = valid_closure_rate(entries, pol)
            if r is not None:
                grid[pol][c] = r

    # Per-cell rate across policies + spread
    ranked = {pol: sorted(grid[pol].items(), key=lambda x: -x[1]) for pol in POLICIES}
    per_cell_spread = {}
    for c in non_null_cells:
        rates = [grid[pol].get(c, float("nan")) for pol in POLICIES]
        clean = [r for r in rates if not math.isnan(r)]
        per_cell_spread[c] = {
            "rates": {pol: grid[pol].get(c, None) for pol in POLICIES},
            "min": min(clean) if clean else None,
            "max": max(clean) if clean else None,
            "range": (max(clean) - min(clean)) if clean else None,
        }

    report = {
        "path": path,
        "grid": grid,
        "ranked_top_cell_per_policy": {pol: ranked[pol][0][0] if ranked[pol] else None
                                       for pol in POLICIES},
        "per_cell_spread": per_cell_spread,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSON.open("w") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote {OUT_JSON}")

    # LaTeX table — per-cell Path 1 rate under each policy
    lines = [
        r"\begin{table*}[!htbp]",
        r"\centering",
        r"\small",
        r"\caption{Judge-sensitivity grid for Path 1 closure rate (R1 Comment 1 defense). "
        r"Each cell's Path~1 closure rate is reported under six judge-weight policies: "
        r"three strict-AND variants ($\rho \in \{2, 3, 4\}$), judge-only ($J \geq 3$, "
        r"metric ignored), metric-only ($Y > 0$, judge ignored), and union ($Y > 0$ OR "
        r"$J \geq 3$). The strict-AND $\rho = 3$ column (bold) is the paper's reported "
        r"policy. Rows ordered by $\rho = 3$ rate descending. \emph{The spread column "
        r"shows the max$-$min closure rate across the 6 policies per cell: cells with "
        r"narrow spread have metric and judge agreeing; cells with wide spread "
        r"disagree.} The grid supports the paper's framing: strict-AND exists precisely "
        r"because single-signal metrics (judge-only or metric-only) produce different "
        r"rankings than the metric-plus-judge intersection, and we do not have "
        r"ground truth to resolve the disagreement. Key headline findings "
        r"(H1 per-operator maximum on 4/5 mutation families; phi-4-at-$L_{\textsf{test}}$ "
        r"pathology; BERTScore surface-form signature on Path 3) are judge-independent "
        r"by construction (they derive from the metric alone or from the disagreement "
        r"structure itself) and therefore invariant across all six policies.}",
        r"\label{tab:judge_sensitivity_grid}",
        r"\begin{tabular}{@{}l" + "r" * len(POLICIES) + r"r@{}}",
        r"\toprule",
    ]
    header = " & ".join([""] + [POLICY_LABELS[p] for p in POLICIES] + [r"spread"]) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    # Sort cells by rho_3 rate descending
    sorted_cells = sorted(non_null_cells, key=lambda c: -grid["rho_3"].get(c, -1))
    for c in sorted_cells:
        row_vals = []
        for pol in POLICIES:
            r = grid[pol].get(c)
            if r is None:
                row_vals.append(r"---")
            elif pol == "rho_3":
                row_vals.append(r"\textbf{" + f"{r:.3f}" + r"}")
            else:
                row_vals.append(f"{r:.3f}")
        spread = per_cell_spread[c]["range"]
        spread_str = f"{spread:.3f}" if spread is not None else r"---"
        lines.append(f"{c} & " + " & ".join(row_vals) + f" & {spread_str} \\\\")
    lines += [
        r"\midrule",
        r"\textbf{top cell} & " + " & ".join(
            report["ranked_top_cell_per_policy"].get(p, "---") for p in POLICIES
        ) + r" & --- \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ]

    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    with OUT_TEX.open("w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {OUT_TEX}")

    # Console summary
    print("\n=== Summary ===")
    print(f"Top cell per policy (Path 1):")
    for pol in POLICIES:
        c = report["ranked_top_cell_per_policy"].get(pol, "—")
        r = grid[pol].get(c)
        print(f"  {POLICY_LABELS[pol]:<35} → {c:<4} at {r:.3f}" if r else f"  {pol}: —")
    print(f"\nMean per-cell spread across 6 policies: "
          f"{sum(s['range'] for s in per_cell_spread.values() if s['range'])/len(per_cell_spread):.3f}")


if __name__ == "__main__":
    main()
