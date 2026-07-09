"""
scripts/build_per_operator_table.py

G2 gap-closer: paper-ready per-mutation-operator kill rate table.

Reads results/tab_per_operator_summary.csv produced by regen_per_operator.py
and emits:
    - tables/tab_per_operator_kill_rate.csv/tex (cell x operator matrix,
      paper-ready)
    - tables/tab_per_operator_headline.csv/tex (H1 vs best-mono side-by-side,
      the headline finding)

If the summary CSV is not present, falls back to the summary table baked in
below (captured from the Colab regen console output on 2026-07-09).
"""
from __future__ import annotations

import sys
from io import StringIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

# Fallback data (captured from Colab console 2026-07-09).
# Weighted kill rate = sum(killed) / sum(total) per (cell, operator).
FALLBACK_SUMMARY = """cell_id,operator,weighted_kill_rate
H1,arithmetic,0.939
H1,boundary,0.844
H1,comparison,1.000
H1,negate_bool,1.000
H1,return_none,0.970
H10,arithmetic,0.351
H10,boundary,0.449
H10,comparison,0.864
H10,negate_bool,0.875
H10,return_none,0.761
H11,arithmetic,0.772
H11,boundary,0.746
H11,comparison,0.908
H11,negate_bool,0.885
H11,return_none,0.939
H2,arithmetic,0.750
H2,boundary,0.557
H2,comparison,0.815
H2,negate_bool,0.824
H2,return_none,0.891
H3,arithmetic,0.542
H3,boundary,0.591
H3,comparison,0.754
H3,negate_bool,0.750
H3,return_none,0.851
H4,arithmetic,0.448
H4,boundary,0.600
H4,comparison,1.000
H4,negate_bool,1.000
H4,return_none,0.964
H5,arithmetic,0.505
H5,boundary,0.630
H5,comparison,0.857
H5,negate_bool,0.875
H5,return_none,0.881
H6,arithmetic,0.745
H6,boundary,0.740
H6,comparison,0.923
H6,negate_bool,0.941
H6,return_none,0.976
H7,arithmetic,0.769
H7,boundary,0.741
H7,comparison,0.964
H7,negate_bool,0.889
H7,return_none,0.919
H8,arithmetic,0.686
H8,boundary,0.646
H8,comparison,0.930
H8,negate_bool,1.000
H8,return_none,0.954
H9,arithmetic,0.769
H9,boundary,0.741
H9,comparison,0.964
H9,negate_bool,0.889
H9,return_none,0.919
M1,arithmetic,0.319
M1,boundary,0.327
M1,comparison,0.545
M1,negate_bool,0.559
M1,return_none,0.682
M2,arithmetic,0.580
M2,boundary,0.653
M2,comparison,0.743
M2,negate_bool,0.818
M2,return_none,0.806
M3,arithmetic,0.806
M3,boundary,0.748
M3,comparison,0.912
M3,negate_bool,0.917
M3,return_none,0.966
M4,arithmetic,0.729
M4,boundary,0.692
M4,comparison,0.884
M4,negate_bool,0.929
M4,return_none,0.914
M5,arithmetic,0.608
M5,boundary,0.562
M5,comparison,0.797
M5,negate_bool,0.741
M5,return_none,0.833
M6,arithmetic,0.684
M6,boundary,0.663
M6,comparison,0.882
M6,negate_bool,0.944
M6,return_none,0.942
"""

CELL_ORDER = [
    "M1", "M2", "M3", "M4", "M5", "M6",
    "H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8", "H9", "H10", "H11",
]
OPERATOR_ORDER = ["arithmetic", "boundary", "comparison", "negate_bool", "return_none"]
OPERATOR_LABELS = {
    "arithmetic": "Arith.",
    "boundary": "Bound.",
    "comparison": "Comp.",
    "negate_bool": "Negate",
    "return_none": "Ret.",
}


def load_summary() -> pd.DataFrame:
    """Load summary CSV, or fall back to hardcoded values."""
    p = Path("tables/tab_per_operator_summary.csv")
    if p.exists():
        df = pd.read_csv(p)
        if "weighted_kill_rate" not in df.columns:
            df["weighted_kill_rate"] = df["sum_killed"] / df["sum_total"]
        print(f"Loaded {len(df)} rows from {p}")
        return df[["cell_id", "operator", "weighted_kill_rate"]]

    p2 = Path("results/tab_per_operator_summary.csv")
    if p2.exists():
        df = pd.read_csv(p2)
        if "weighted_kill_rate" not in df.columns:
            df["weighted_kill_rate"] = df["sum_killed"] / df["sum_total"]
        print(f"Loaded {len(df)} rows from {p2}")
        return df[["cell_id", "operator", "weighted_kill_rate"]]

    print("Summary CSV not found; using fallback baked-in data.")
    return pd.read_csv(StringIO(FALLBACK_SUMMARY))


def build_wide(summary: pd.DataFrame) -> pd.DataFrame:
    piv = summary.pivot(
        index="cell_id", columns="operator", values="weighted_kill_rate"
    )
    piv = piv.reindex(index=[c for c in CELL_ORDER if c in piv.index])
    piv = piv[[o for o in OPERATOR_ORDER if o in piv.columns]]
    return piv


def build_paper_latex(wide: pd.DataFrame) -> str:
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(
        r"\caption{Per-mutation-operator kill rate (weighted; $\text{killed}/\text{total}$ pooled across samples). Best mono cell per operator (\textbf{bold}); best hetero cell per operator (\underline{underlined}). H1 dominates 4 of 5 operators; every mono ceiling is beaten by at least one hetero cell.}"
    )
    lines.append(r"\label{tab:per_operator_kill_rate}")
    lines.append(r"\begin{tabular}{l" + "r" * len(wide.columns) + "}")
    lines.append(r"\toprule")
    header_cols = " & ".join(OPERATOR_LABELS[c] for c in wide.columns)
    lines.append("Cell & " + header_cols + r" \\")
    lines.append(r"\midrule")

    # Compute best mono and best hetero per operator for bolding/underlining
    monos = [c for c in wide.index if c.startswith("M")]
    heteros = [c for c in wide.index if c.startswith("H")]
    best_mono = {op: wide.loc[monos, op].idxmax() for op in wide.columns}
    best_het = {op: wide.loc[heteros, op].idxmax() for op in wide.columns}

    prev_stratum = None
    for cell in wide.index:
        stratum = cell[0]
        if prev_stratum is not None and stratum != prev_stratum:
            lines.append(r"\midrule")
        prev_stratum = stratum
        vals = []
        for op in wide.columns:
            v = wide.at[cell, op]
            if pd.isna(v):
                vals.append("---")
                continue
            s = f"{v:.3f}"
            if cell == best_mono[op]:
                s = r"\textbf{" + s + "}"
            elif cell == best_het[op]:
                s = r"\underline{" + s + "}"
            vals.append(s)
        lines.append(f"{cell} & " + " & ".join(vals) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def build_headline_latex(wide: pd.DataFrame) -> str:
    """H1 vs best-mono per operator side-by-side."""
    monos = [c for c in wide.index if c.startswith("M")]
    rows = []
    for op in wide.columns:
        best_mono_cell = wide.loc[monos, op].idxmax()
        mono_val = wide.at[best_mono_cell, op]
        het_val = wide.at["H1", op]
        rows.append(
            {
                "Operator": OPERATOR_LABELS[op],
                "Best mono cell": best_mono_cell,
                "Best mono kill rate": f"{mono_val:.3f}",
                "H1 kill rate": f"{het_val:.3f}",
                "$\\Delta$ (H1 $-$ mono)": f"{het_val - mono_val:+.3f}",
            }
        )
    headline = pd.DataFrame(rows)

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(
        r"\caption{Headline per-operator finding: H1 (phi-4 spec + qwen-coder test + qwen-coder code) beats the strongest mono baseline on every mutation operator, with the largest gain on arithmetic operators ($\Delta = +0.133$).}"
    )
    lines.append(r"\label{tab:per_operator_headline}")
    ncols = len(headline.columns)
    lines.append(r"\begin{tabular}{l" + "l" + "r" * (ncols - 2) + "}")
    lines.append(r"\toprule")
    lines.append(" & ".join(headline.columns) + r" \\")
    lines.append(r"\midrule")
    for _, row in headline.iterrows():
        lines.append(" & ".join(str(v) for v in row.tolist()) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines), headline


def main() -> None:
    summary = load_summary()
    wide = build_wide(summary)

    tables_dir = Path("tables")
    tables_dir.mkdir(exist_ok=True)

    # Main paper table
    wide.round(3).to_csv(tables_dir / "tab_per_operator_kill_rate.csv")
    (tables_dir / "tab_per_operator_kill_rate.tex").write_text(build_paper_latex(wide))

    # Headline pull (H1 vs best-mono per operator)
    latex, headline_df = build_headline_latex(wide)
    headline_df.to_csv(tables_dir / "tab_per_operator_headline.csv", index=False)
    (tables_dir / "tab_per_operator_headline.tex").write_text(latex)

    # Console reports
    print("\n=== Full wide table (rows = cells, cols = operators) ===")
    print(wide.round(3).to_string())

    print("\n=== Headline: H1 vs best-mono per operator ===")
    print(headline_df.to_string(index=False))

    # Also compute best-hetero-per-operator for the caption
    heteros = [c for c in wide.index if c.startswith("H")]
    print("\n=== Best hetero cell per operator ===")
    for op in wide.columns:
        best_h = wide.loc[heteros, op].idxmax()
        best_h_val = wide.at[best_h, op]
        best_m = wide.loc[[c for c in wide.index if c.startswith("M")], op].idxmax()
        best_m_val = wide.at[best_m, op]
        print(
            f"  {op:12s}: hetero winner {best_h} = {best_h_val:.3f}, "
            f"mono winner {best_m} = {best_m_val:.3f}, "
            f"Δ = {best_h_val - best_m_val:+.3f}"
        )


if __name__ == "__main__":
    main()
