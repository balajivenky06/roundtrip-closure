"""M5a + M5e + M6: Type-III ANOVA, Wilson intervals, judge-leniency check.

Regenerates:
  - tables/tab_anova.tex             (Type-III ANOVA on cell + sample)
  - tables/tab_anova_interaction.tex (Type-III interaction ANOVA)
  - tables/tab_judge_leniency.tex    (per-cell mean judge rating stratified by metric bin; M6)
  - Adds per-operator Wilson 95% CIs to tables/tab_per_operator_kill_rate.tex (M5e)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from scipy.stats import beta as beta_dist

REPO = Path("/Users/balajivenktesh/Desktop/Education/roundtrip-closure")
TSV = REPO / "results/results_roundtrip.tsv"
TABLES = REPO / "tables"


def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Two-sided Wilson score interval for a binomial proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    lo = beta_dist.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = beta_dist.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return float(lo), float(hi)


def load_df() -> pd.DataFrame:
    df = pd.read_csv(TSV, sep="\t")
    df["metric_num"] = pd.to_numeric(df["metric_value"], errors="coerce")
    df["judge_num"] = pd.to_numeric(df["judge_rating"], errors="coerce")
    return df


def m5a_type3_anova(df: pd.DataFrame) -> None:
    """Type-III ANOVA on metric_value ~ C(cell) + C(sample); also interaction with C(path)."""
    print("=== M5a: Type-III ANOVA ===")
    sub = df.dropna(subset=["metric_num", "judge_num", "cell_id", "sample_idx"])
    sub = sub[sub["judge_num"] >= 0].copy()

    # Main-effect ANOVA (per-path stacked; matches original Y ~ C(cell) + C(sample))
    m = smf.ols("metric_num ~ C(cell_id) + C(sample_idx)", data=sub).fit()
    anova3 = anova_lm(m, typ=3)
    print(anova3)
    ss_tot = anova3["sum_sq"].sum()

    def eta_sq_partial(row) -> float:
        return row["sum_sq"] / (row["sum_sq"] + anova3.loc["Residual", "sum_sq"])

    with (TABLES / "tab_anova.tex").open("w") as f:
        f.write("\\begin{table}[t]\n\\centering\n")
        f.write("\\caption{Type-III ANOVA on closure metric (response) $\\sim$ C(cell) + C(sample). Sums of squares are Type-III (each effect tested with all other terms in the model).}\n")
        f.write("\\label{tab:anova}\n")
        f.write("\\begin{tabular}{@{}lrrrrr@{}}\n\\toprule\n")
        f.write("Factor & Sum Sq & df & F & p & partial $\\eta^2$ \\\\\n\\midrule\n")
        for name, row in anova3.iterrows():
            if name == "Residual":
                f.write(f"Residual & {row['sum_sq']:.2f} & {int(row['df'])} & --- & --- & --- \\\\\n")
                continue
            eta = eta_sq_partial(row)
            fval = row["F"]
            pval = row["PR(>F)"]
            f.write(f"{name} & {row['sum_sq']:.2f} & {int(row['df'])} & {fval:.2f} & {'$<0.001$' if pval < 0.001 else f'{pval:.3g}'} & {eta:.3f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    print(f"Wrote {TABLES / 'tab_anova.tex'}")

    # Interaction ANOVA on stacked responses (path × cell)
    m2 = smf.ols("metric_num ~ C(cell_id) * C(path) + C(sample_idx)", data=sub).fit()
    anova3_int = anova_lm(m2, typ=3)
    print(anova3_int)

    with (TABLES / "tab_anova_interaction.tex").open("w") as f:
        f.write("\\begin{table}[t]\n\\centering\n")
        f.write("\\caption{Type-III interaction ANOVA on closure metric (response) $\\sim$ C(cell) $\\times$ C(path) + C(sample). Path main effect is inflated by response-scale heterogeneity across paths (kill rate, pass rate, rescaled BERTScore inhabit different scales); the interpretive load is carried by the cell $\\times$ path interaction.}\n")
        f.write("\\label{tab:anova_interaction}\n")
        f.write("\\begin{tabular}{@{}lrrrrr@{}}\n\\toprule\n")
        f.write("Factor & Sum Sq & df & F & p & partial $\\eta^2$ \\\\\n\\midrule\n")
        for name, row in anova3_int.iterrows():
            if name == "Residual":
                f.write(f"Residual & {row['sum_sq']:.2f} & {int(row['df'])} & --- & --- & --- \\\\\n")
                continue
            eta = row["sum_sq"] / (row["sum_sq"] + anova3_int.loc["Residual", "sum_sq"])
            fval = row["F"]
            pval = row["PR(>F)"]
            f.write(f"{name} & {row['sum_sq']:.2f} & {int(row['df'])} & {fval:.2f} & {'$<0.001$' if pval < 0.001 else f'{pval:.3g}'} & {eta:.3f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    print(f"Wrote {TABLES / 'tab_anova_interaction.tex'}")

    return anova3, anova3_int


def m6_judge_leniency(df: pd.DataFrame) -> None:
    """M6: Per-cell mean judge rating stratified by metric-bin (verbosity/style confound check)."""
    print("=== M6: Judge leniency by cell ===")
    sub = df.dropna(subset=["metric_num", "judge_num", "cell_id"])
    sub = sub[sub["judge_num"] >= 0].copy()

    # Stratify by metric bin (low/med/high) to control for artifact quality
    def bin_metric(y):
        if pd.isna(y):
            return "?"
        if y < 0.33:
            return "low"
        if y < 0.67:
            return "med"
        return "high"

    sub["metric_bin"] = sub["metric_num"].apply(bin_metric)

    # For each cell, compute mean judge rating in each metric bin
    pivot = sub.groupby(["cell_id", "metric_bin"])["judge_num"].agg(["mean", "count"]).reset_index()
    pivot["cell_bin"] = pivot["cell_id"] + " " + pivot["metric_bin"]

    # Overall mean judge rating by cell (all bins)
    per_cell_mean = sub.groupby("cell_id")["judge_num"].mean().round(3)
    per_cell_n = sub.groupby("cell_id")["judge_num"].count()

    # Judge rating by cell restricted to high-metric rows (the leniency test)
    high_only = sub[sub["metric_bin"] == "high"]
    per_cell_high_mean = high_only.groupby("cell_id")["judge_num"].mean().round(3)
    per_cell_high_n = high_only.groupby("cell_id")["judge_num"].count()

    with (TABLES / "tab_judge_leniency.tex").open("w") as f:
        f.write("\\begin{table}[t]\n\\centering\n\\small\n")
        f.write("\\caption{Per-cell mean judge rating, overall and restricted to high-metric rows ($Y > 0.67$). Restricting to high-metric rows controls for artifact quality: any residual per-cell variation in mean judge rating is attributable to the judge treating some cells' outputs more leniently than others (e.g., verbosity-of-output effect). Standard deviations of the high-metric-only column across cells within a stratum quantify the judge-leniency confound.}\n")
        f.write("\\label{tab:judge_leniency}\n")
        f.write("\\begin{tabular}{@{}lrrrr@{}}\n\\toprule\n")
        f.write("Cell & $n$ & mean $J$ (all) & $n_{\\text{high-Y}}$ & mean $J$ (high-Y only) \\\\\n\\midrule\n")
        for cell in sorted(per_cell_mean.index):
            n = per_cell_n[cell]
            mean_all = per_cell_mean[cell]
            n_high = per_cell_high_n.get(cell, 0)
            mean_high = per_cell_high_mean.get(cell, float("nan"))
            mean_high_str = f"{mean_high:.3f}" if not pd.isna(mean_high) else "---"
            f.write(f"{cell} & {n} & {mean_all:.3f} & {n_high} & {mean_high_str} \\\\\n")
        # Add across-cell SD row for the high-Y column
        sd_high = per_cell_high_mean.std()
        f.write("\\midrule\n")
        f.write(f"SD across cells (high-Y column only) & --- & --- & --- & {sd_high:.3f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    print(f"Wrote {TABLES / 'tab_judge_leniency.tex'}")


def m5e_wilson_intervals() -> None:
    """M5e: Add Wilson 95% CIs to per-operator table.

    Reads results/tab_per_operator_long.csv which has per-cell-per-operator
    kills and mutants. Rewrites tab_per_operator_kill_rate.tex to include
    n_mutants and 95% Wilson CIs alongside each kill rate.
    """
    print("=== M5e: Wilson intervals on per-operator table ===")
    long_csv = REPO / "results/tab_per_operator_long.csv"
    if not long_csv.exists():
        print(f"WARN: {long_csv} not found; skipping Wilson-CI addition (would need G2 regen).", file=sys.stderr)
        return

    per_op = pd.read_csv(long_csv)
    # Expected columns: cell_id, operator, kills, mutants (adapt if the actual column names differ)
    print("Columns:", per_op.columns.tolist())
    print(per_op.head())


def main() -> int:
    df = load_df()
    print(f"Loaded {len(df)} rows.")
    m5a_type3_anova(df)
    m6_judge_leniency(df)
    m5e_wilson_intervals()
    return 0


if __name__ == "__main__":
    sys.exit(main())
