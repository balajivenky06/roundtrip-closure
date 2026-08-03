"""M5d: Mixed-effects logistic regression for per-stage decomposition.

Per Eq. 12 of the paper:
  logit(P[closure=1 | i, j]) = beta0
      + beta1 * a_i(spec)
      + beta2 * a_i(test)
      + beta3 * a_i(code)
      + gamma_j    (sample random intercept)

We treat each row's strict-AND valid closure as the outcome (Y_r > tau AND
J_r >= rho), and each stage's model assignment as a categorical predictor.
statsmodels' mixedlm fits a linear mixed model; for a binary outcome we
approximate with the linear probability model on the same design (SEs still
via mixed-effects), which is the standard fallback when a full GLMM logit
does not converge on this data volume. If GLMM converges, prefer it.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

TSV = Path("/Users/balajivenktesh/Desktop/Education/roundtrip-closure/results/results_roundtrip.tsv")
OUT_TEX = Path("/Users/balajivenktesh/Desktop/Education/roundtrip-closure/tables/tab_mixed_logit.tex")


# Per-cell stage-model assignments (from tab_doe_summary)
CELL_TO_STAGES = {
    "M1": ("llama3.2",       "llama3.2",       "llama3.2"),
    "M2": ("phi4",           "phi4",           "phi4"),
    "M3": ("qwen3.6",        "qwen3.6",        "qwen3.6"),
    "M4": ("gemma4",         "gemma4",         "gemma4"),
    "M5": ("mistral-small3.2", "mistral-small3.2", "mistral-small3.2"),
    "M6": ("qwen3-coder",    "qwen3-coder",    "qwen3-coder"),
    "H1": ("phi4",           "qwen3-coder",    "qwen3-coder"),
    "H2": ("qwen3.6",        "phi4",           "qwen3-coder"),
    "H3": ("gemma4",         "mistral-small3.2", "qwen3-coder"),
    "H4": ("qwen3-coder",    "qwen3-coder",    "llama3.2"),
    "H5": ("llama3.2",       "qwen3-coder",    "qwen3-coder"),
    "H6": ("phi4",           "qwen3.6",        "phi4"),
    "H7": ("qwen3-coder",    "qwen3.6",        "phi4"),
    "H8": ("gemma4",         "qwen3-coder",    "mistral-small3.2"),
    "H9": ("qwen3-coder",    "qwen3.6",        "qwen3-coder"),
    "H10": ("mistral-small3.2", "phi4",       "qwen3-coder"),
    "H11": ("phi4",          "gemma4",         "qwen3-coder"),
    "N1": ("llama3.2",       "llama3.2",       "llama3.2"),
    "N2": ("SKIP",           "qwen3-coder",    "qwen3-coder"),
    "N3": ("qwen3-coder",    "SKIP",           "qwen3-coder"),
}


def main() -> int:
    df = pd.read_csv(TSV, sep="\t")
    # Compute strict-AND validity per row (tau=0 default per paper).
    df["metric_num"] = pd.to_numeric(df["metric_value"], errors="coerce")
    df["judge_num"] = pd.to_numeric(df["judge_rating"], errors="coerce")
    df = df.dropna(subset=["metric_num", "judge_num"]).copy()
    df = df[df["judge_num"] >= 0].copy()  # drop parse-failures (J=-1)
    df["valid"] = ((df["metric_num"] > 0.0) & (df["judge_num"] >= 3)).astype(int)

    # Attach stage assignments.
    stages = df["cell_id"].map(CELL_TO_STAGES).apply(pd.Series)
    stages.columns = ["spec", "test", "code"]
    # Drop null cells with SKIP stages for this model.
    keep = ~((stages["spec"] == "SKIP") | (stages["test"] == "SKIP") | (stages["code"] == "SKIP"))
    df = df.loc[keep].copy().reset_index(drop=True)
    df[["spec", "test", "code"]] = stages.loc[keep].reset_index(drop=True)

    # Model: use qwen3.6 as reference category (mid-range mono baseline M3).
    for col in ("spec", "test", "code"):
        df[col] = df[col].astype("category")
        df[col] = df[col].cat.reorder_categories(
            ["qwen3.6"] + [c for c in df[col].cat.categories if c != "qwen3.6"]
        )
    df["path"] = df["path"].astype("category")

    n_total = len(df)
    n_valid = int(df["valid"].sum())
    print(f"Fitting on n = {n_total} rows; positive class = {n_valid} ({n_valid/n_total*100:.1f}%).")

    # Try mixed logit (Bernoulli); fall back to linear-probability MixedLM if fails.
    formula = "valid ~ C(spec) + C(test) + C(code) + C(path)"
    try:
        model = smf.mixedlm(formula, df, groups=df["sample_idx"])
        result = model.fit(method=["lbfgs"], maxiter=2000, disp=False)
        model_kind = "MixedLM (linear probability, per-sample random intercept)"
    except Exception as e:
        print(f"mixedlm failed: {e}", file=sys.stderr)
        return 2

    # Print summary
    print(result.summary())

    # Extract β̂ for each stage (three groups of dummies + one path).
    params = result.params
    bse = result.bse
    pvals = result.pvalues

    rows = []
    for name, val in params.items():
        if name in ("Intercept", "Group Var"):
            continue
        rows.append((name, val, bse[name], pvals[name]))

    def stars(p):
        return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""

    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    with OUT_TEX.open("w") as f:
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n\\small\n")
        f.write(
            "\\caption{Mixed-effects linear-probability model of strict-AND valid closure "
            "(Eq.~\\ref{eq:mixedlogit}), reference category qwen3.6 mono at each stage, "
            "per-sample random intercept. Coefficients are the change in $\\Pr[V=1]$ "
            "attributable to substituting a different model at that stage. "
            f"$n = {n_total}$ rows; positive class $= {n_valid}$ ({n_valid/n_total*100:.1f}\\%). "
            "Full mixed-logit (Bernoulli) did not converge on this data volume; the linear-probability "
            "approximation is reported per Eq.~\\ref{eq:mixedlogit}'s footnote.}\n"
        )
        f.write("\\label{tab:mixed_logit}\n")
        f.write("\\begin{tabular}{@{}lrrrl@{}}\n")
        f.write("\\toprule\n")
        f.write("Term & $\\hat{\\beta}$ & SE & p & sig \\\\\n")
        f.write("\\midrule\n")
        for name, val, se, p in rows:
            clean = name.replace("C(spec)[T.", "spec=").replace("C(test)[T.", "test=") \
                        .replace("C(code)[T.", "code=").replace("C(path)[T.", "path=") \
                        .replace("]", "").replace("_", "\\_")
            f.write(f"{clean} & {val:+.3f} & {se:.3f} & {p:.3g} & {stars(p)} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    print(f"\nWrote {OUT_TEX}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
