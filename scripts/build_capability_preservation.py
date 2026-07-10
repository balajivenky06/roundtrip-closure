"""
scripts/build_capability_preservation.py

G4 gap-closer: capability preservation analysis (Epistemic Fortitude "User is
Right" analog).

The question this addresses: does heterogeneous specialization sacrifice mono
strengths? On samples where the strongest mono baseline achieves valid closure,
do hetero cells also achieve closure?

Design:
    - For each path, identify the strongest mono cell (highest per-cell closure
      rate at τ=0, ρ=3).
    - Restrict to samples where that mono achieved valid closure.
    - For every other cell, compute preservation rate = fraction of those
      samples where the cell also achieves valid closure.
    - Report per (cell, path) preservation rate.

A preservation rate ~1.0 means "hetero doesn't hurt on easy cases." A rate
noticeably below the sample size fraction means the cell sacrifices capability.

Emits:
    tables/tab_capability_preservation.csv         — per (cell, path) table
    tables/tab_capability_preservation.tex          — paper-ready summary
    results/tab_capability_preservation_long.csv    — per-sample details
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from analyze.load_results import load_tsv


TAU = {1: 0.0, 2: 0.0, 3: 0.0}
RHO = 3


def is_valid_closure(row) -> bool:
    m = row["metric_value"]
    j = row["judge_rating"]
    if pd.isna(m) or pd.isna(j) or j < 0:
        return False
    return (m > TAU[int(row["path"])]) and (j >= RHO)


def strongest_mono_per_path(df: pd.DataFrame) -> dict[int, str]:
    """Return {path: cell_id} for the mono cell with highest closure rate."""
    monos = ["M1", "M2", "M3", "M4", "M5", "M6"]
    out = {}
    for path in [1, 2, 3]:
        sub = df[(df["path"] == path) & (df["cell_id"].isin(monos))].copy()
        sub["closed"] = sub.apply(is_valid_closure, axis=1)
        rates = sub.groupby("cell_id")["closed"].mean()
        out[path] = rates.idxmax()
    return out


CELL_ORDER = [
    "M1", "M2", "M3", "M4", "M5", "M6",
    "H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8", "H9", "H10", "H11",
    "N1", "N2", "N3",
]


def compute_preservation(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["closed"] = df.apply(is_valid_closure, axis=1)

    ref_cells = strongest_mono_per_path(df)
    print(f"Reference (strongest mono per path): {ref_cells}")

    rows = []
    for path in [1, 2, 3]:
        ref_cell = ref_cells[path]
        # Samples on which the reference achieves valid closure
        ref_sub = df[(df["cell_id"] == ref_cell) & (df["path"] == path)]
        ref_ok = ref_sub[ref_sub["closed"]]["sample_idx"].tolist()
        ref_ok_set = set(ref_ok)

        for cell_id in sorted(df["cell_id"].unique()):
            cell_sub = df[(df["cell_id"] == cell_id) & (df["path"] == path)]
            # Restrict to samples where reference achieved closure
            restricted = cell_sub[cell_sub["sample_idx"].isin(ref_ok_set)]
            n_ref = len(restricted)
            if n_ref == 0:
                rows.append(
                    {
                        "cell_id": cell_id,
                        "path": path,
                        "reference_cell": ref_cell,
                        "n_reference_success": len(ref_ok_set),
                        "n_covered": 0,
                        "n_preserved": 0,
                        "preservation_rate": float("nan"),
                    }
                )
                continue
            n_preserved = int(restricted["closed"].sum())
            rows.append(
                {
                    "cell_id": cell_id,
                    "path": path,
                    "reference_cell": ref_cell,
                    "n_reference_success": len(ref_ok_set),
                    "n_covered": n_ref,
                    "n_preserved": n_preserved,
                    "preservation_rate": n_preserved / n_ref,
                }
            )
    return pd.DataFrame(rows)


def build_wide(pres: pd.DataFrame) -> pd.DataFrame:
    piv = pres.pivot(
        index="cell_id", columns="path", values="preservation_rate"
    )
    piv.columns = [f"Path {p} preservation" for p in piv.columns]
    piv = piv.reindex(index=[c for c in CELL_ORDER if c in piv.index])
    return piv


def build_latex(wide: pd.DataFrame, ref_cells: dict[int, str]) -> str:
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    ref_str = ", ".join(f"Path {p}: {c}" for p, c in ref_cells.items())
    lines.append(
        r"\caption{Capability preservation. For each path, we restrict to samples on which the strongest mono baseline ("
        + ref_str
        + r") achieves valid closure. Each cell reports the fraction of those same samples on which the row-cell also achieves valid closure. Values near 1.0 indicate that the cell does not sacrifice mono strengths.}"
    )
    lines.append(r"\label{tab:capability_preservation}")
    ncols = len(wide.columns)
    lines.append(r"\begin{tabular}{l" + "r" * ncols + "}")
    lines.append(r"\toprule")
    lines.append("Cell & " + " & ".join(wide.columns) + r" \\")
    lines.append(r"\midrule")
    prev_stratum = None
    for cell in wide.index:
        stratum = cell[0]
        if prev_stratum is not None and stratum != prev_stratum:
            lines.append(r"\midrule")
        prev_stratum = stratum
        vals = []
        for col in wide.columns:
            v = wide.at[cell, col]
            if pd.isna(v):
                vals.append("---")
            else:
                vals.append(f"{v:.3f}")
        lines.append(f"{cell} & " + " & ".join(vals) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def main() -> None:
    tsv = Path("results/results_roundtrip.tsv")
    df = load_tsv(tsv)

    pres = compute_preservation(df)
    ref_cells = strongest_mono_per_path(df)

    # Long form
    Path("results").mkdir(exist_ok=True)
    pres.to_csv("results/tab_capability_preservation_long.csv", index=False)

    # Wide + LaTeX
    wide = build_wide(pres)
    wide.round(3).to_csv("tables/tab_capability_preservation.csv")
    Path("tables/tab_capability_preservation.tex").write_text(
        build_latex(wide, ref_cells)
    )

    # Console
    print("\n=== Capability preservation (fraction of ref-success samples "
          "each cell also succeeds on) ===")
    print(wide.round(3).to_string())

    # Headline: mean preservation rate for hetero cells
    heteros = [c for c in wide.index if c.startswith("H")]
    print("\n=== Hetero capability preservation summary ===")
    for path_col in wide.columns:
        vals = wide.loc[heteros, path_col].dropna()
        print(
            f"  {path_col}: mean = {vals.mean():.3f}, "
            f"min = {vals.min():.3f} ({vals.idxmin()}), "
            f"max = {vals.max():.3f} ({vals.idxmax()})"
        )


if __name__ == "__main__":
    main()
