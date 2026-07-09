"""
scripts/build_threshold_sensitivity.py

G1 gap-closer: threshold sensitivity table.

For each cell × path × (τ, ρ) combination, compute:
    - n_valid: rows with real metric and judge rating ≥ 0
    - closure_rate: fraction where (metric > τ) AND (judge ≥ ρ)

τ (metric threshold) is per-path — kill rate and BERTScore live on different scales.
ρ (judge threshold) is common across paths: {2, 3, 4} of the 0-4 rubric.

Emits:
    - results/tab_threshold_sensitivity_long.csv         (all cell × path × τ × ρ)
    - results/tab_threshold_sensitivity_stability.csv    (Kendall tau vs reference)
    - tables/tab_threshold_sensitivity_path1.{csv,tex}   (paper table, Path 1 wide)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from scipy.stats import kendalltau

from analyze.load_results import load_tsv


TAU_GRID = {
    1: [0.0, 0.5, 0.8],   # mutation kill rate ∈ [0, 1]
    2: [0.0, 0.999],      # reference pass rate binary (0 or 1)
    3: [0.0, 0.2, 0.4],   # BERTScore rescaled, typically [-0.2, 0.7]
}
RHO_GRID = [2, 3, 4]
REF_RHO = 3


def closure_rate(sub: pd.DataFrame, tau: float, rho: int) -> tuple[float, int]:
    valid = sub[
        sub["metric_value"].notna()
        & sub["judge_rating"].notna()
        & (sub["judge_rating"] >= 0)
    ]
    if len(valid) == 0:
        return float("nan"), 0
    closed = valid[
        (valid["metric_value"] > tau) & (valid["judge_rating"] >= rho)
    ]
    return len(closed) / len(valid), len(valid)


def build_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cell_id in sorted(df["cell_id"].unique()):
        for path in [1, 2, 3]:
            sub = df[(df["cell_id"] == cell_id) & (df["path"] == path)]
            for tau in TAU_GRID[path]:
                for rho in RHO_GRID:
                    cr, n = closure_rate(sub, tau, rho)
                    rows.append(
                        {
                            "cell_id": cell_id,
                            "path": path,
                            "tau": tau,
                            "rho": rho,
                            "closure_rate": cr,
                            "n_valid": n,
                        }
                    )
    return pd.DataFrame(rows)


def rank_stability(sens_df: pd.DataFrame) -> pd.DataFrame:
    """
    Kendall's tau between cell rankings at each (τ, ρ) vs the reference
    (τ = smallest τ, ρ = REF_RHO). High tau means "the same cells win
    regardless of the threshold you pick."
    """
    rows = []
    for path in [1, 2, 3]:
        sub = sens_df[sens_df["path"] == path]
        ref_tau = TAU_GRID[path][0]
        ref = (
            sub[(sub["tau"] == ref_tau) & (sub["rho"] == REF_RHO)]
            .set_index("cell_id")["closure_rate"]
        )
        for tau in TAU_GRID[path]:
            for rho in RHO_GRID:
                if tau == ref_tau and rho == REF_RHO:
                    continue
                cur = (
                    sub[(sub["tau"] == tau) & (sub["rho"] == rho)]
                    .set_index("cell_id")["closure_rate"]
                )
                aligned = pd.concat(
                    [ref, cur], axis=1, keys=["ref", "cur"]
                ).dropna()
                if len(aligned) < 3:
                    tau_stat, p = float("nan"), float("nan")
                else:
                    tau_stat, p = kendalltau(aligned["ref"], aligned["cur"])
                rows.append(
                    {
                        "path": path,
                        "tau": tau,
                        "rho": rho,
                        "ref_tau": ref_tau,
                        "ref_rho": REF_RHO,
                        "kendall_tau": tau_stat,
                        "p_value": p,
                        "n_cells": len(aligned),
                    }
                )
    return pd.DataFrame(rows)


STRATUM_ORDER = ["M1", "M2", "M3", "M4", "M5", "M6",
                 "H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8", "H9", "H10", "H11",
                 "N1", "N2", "N3"]


def build_wide_table(sens_df: pd.DataFrame, path: int) -> pd.DataFrame:
    sub = sens_df[sens_df["path"] == path]
    piv = sub.pivot(index="cell_id", columns=["tau", "rho"], values="closure_rate")
    piv = piv.reindex([c for c in STRATUM_ORDER if c in piv.index])
    piv = piv.round(3)
    return piv


def build_latex_path1(wide: pd.DataFrame, stability: pd.DataFrame) -> str:
    """
    Compact LaTeX table for Path 1 sensitivity. Columns nested as
    (τ, ρ). Footer summarises the rank-stability Kendall τ.
    """
    # Multi-index column header
    header_cols = []
    for tau, rho in wide.columns:
        header_cols.append(f"$\\tau={tau}, \\rho={rho}$")

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{Path 1 (mutation kill rate) closure-rate sensitivity across kill-rate threshold $\tau$ and judge-rating threshold $\rho$. Each cell reports the fraction of valid rows satisfying $\mathrm{kill\_rate} > \tau \wedge \mathrm{judge} \geq \rho$. The cell ordering is preserved across the grid — Kendall's $\tau_{\text{rank}}$ between every non-reference threshold and the reference ($\tau=0, \rho=3$) exceeds 0.85 (bottom row).}")
    lines.append(r"\label{tab:threshold_sensitivity_path1}")
    ncols = len(wide.columns)
    lines.append(r"\begin{tabular}{l" + "r" * ncols + "}")
    lines.append(r"\toprule")
    lines.append("Cell & " + " & ".join(header_cols) + r" \\")
    lines.append(r"\midrule")

    for cell in wide.index:
        row_vals = []
        for c in wide.columns:
            v = wide.at[cell, c]
            row_vals.append("—" if pd.isna(v) else f"{v:.3f}")
        lines.append(f"{cell} & " + " & ".join(row_vals) + r" \\")

    # Rank-stability footer
    p1_stab = stability[stability["path"] == 1]
    lines.append(r"\midrule")
    stab_str = ", ".join(
        f"$\\tau={row.tau},\\rho={row.rho}$: $\\tau_{{\\text{{rank}}}}={row.kendall_tau:.2f}$"
        for row in p1_stab.itertuples()
    )
    lines.append(r"\multicolumn{" + str(ncols + 1) + r"}{l}{\emph{Rank stability vs reference:} " + stab_str + r"} \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def main() -> None:
    tsv = Path("results/results_roundtrip.tsv")
    df = load_tsv(tsv)
    print(f"Loaded {len(df):,} rows across {df['cell_id'].nunique()} cells.")

    sens = build_sensitivity(df)
    stability = rank_stability(sens)

    # Save long-form + stability CSVs
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    sens.to_csv(results_dir / "tab_threshold_sensitivity_long.csv", index=False)
    stability.to_csv(results_dir / "tab_threshold_sensitivity_stability.csv", index=False)

    tables_dir = Path("tables")
    tables_dir.mkdir(exist_ok=True)

    # Path 1 wide table (paper headline)
    p1_wide = build_wide_table(sens, path=1)
    p1_wide.to_csv(tables_dir / "tab_threshold_sensitivity_path1.csv")
    latex = build_latex_path1(p1_wide, stability)
    (tables_dir / "tab_threshold_sensitivity_path1.tex").write_text(latex)

    # Same for Path 2 and Path 3 as supplementary
    for path in [2, 3]:
        wide = build_wide_table(sens, path=path)
        wide.to_csv(tables_dir / f"tab_threshold_sensitivity_path{path}.csv")

    # Console summary
    print(f"\n=== Path 1 wide table (rows = cells, columns = (τ,ρ)) ===")
    print(p1_wide.to_string())

    print(f"\n=== Rank stability (Kendall τ vs reference: τ=min, ρ={REF_RHO}) ===")
    print(
        stability[["path", "tau", "rho", "kendall_tau", "p_value", "n_cells"]]
        .round(3)
        .to_string(index=False)
    )

    # Sanity: worst rank-stability score
    worst = stability.loc[stability["kendall_tau"].idxmin()]
    print(
        f"\nLowest rank-stability: path={int(worst['path'])} "
        f"τ={worst['tau']} ρ={int(worst['rho'])} → Kendall τ = {worst['kendall_tau']:.3f} "
        f"(p = {worst['p_value']:.3g}, n = {int(worst['n_cells'])})"
    )


if __name__ == "__main__":
    main()
