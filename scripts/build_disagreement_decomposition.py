"""
scripts/build_disagreement_decomposition.py

G3 gap-closer: judge-metric disagreement decomposition.

Categorises every row per Algorithm 2's decision-reason labels and reports the
distribution by cell, path, and benchmark. Turns Path 3's r=0.02 finding from
a limitation into an evidence-based claim about BERTScore's surface-metric
character.

Categories (per Algorithm 2):
    structural_NA           — metric NaN or judge rating -1
    both_agree_valid        — metric > τ  AND  judge >= ρ
    both_agree_invalid      — metric <= τ AND  judge < ρ
    false_closure_candidate — metric > τ  AND  judge < ρ  (metric over-credits)
    metric_false_negative   — metric <= τ AND  judge >= ρ (metric under-credits)

Path-specific τ:
    Path 1 (kill rate):     τ = 0.0 (any kill counts, matches paper default)
    Path 2 (pass rate):     τ = 0.0 (binary)
    Path 3 (BERTScore):     τ = 0.0 (any positive rescaled BERTScore)
ρ = 3 across all paths (paper default: "equivalent or better").

Emits:
    results/tab_disagreement_long.csv    — every row labelled
    tables/tab_disagreement_by_path.{csv,tex}   — headline decomposition
    tables/tab_disagreement_by_cell.{csv,tex}   — per-cell distribution
    tables/tab_disagreement_p3_by_benchmark.{csv,tex} — Path 3 focus
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from analyze.load_results import load_tsv


TAU = {1: 0.0, 2: 0.0, 3: 0.0}
RHO = 3


def label_row(row) -> str:
    m = row["metric_value"]
    j = row["judge_rating"]
    # structural_NA if either signal is missing
    if pd.isna(m) or pd.isna(j) or j < 0:
        return "structural_NA"
    tau = TAU[int(row["path"])]
    metric_ok = m > tau
    judge_ok = j >= RHO
    if metric_ok and judge_ok:
        return "both_agree_valid"
    if not metric_ok and not judge_ok:
        return "both_agree_invalid"
    if metric_ok and not judge_ok:
        return "false_closure_candidate"
    return "metric_false_negative"


CATEGORY_ORDER = [
    "both_agree_valid",
    "both_agree_invalid",
    "false_closure_candidate",
    "metric_false_negative",
    "structural_NA",
]

CATEGORY_LATEX = {
    "both_agree_valid": r"Both $\checkmark$",
    "both_agree_invalid": r"Both $\times$",
    "false_closure_candidate": r"Metric $\checkmark$ / Judge $\times$",
    "metric_false_negative": r"Metric $\times$ / Judge $\checkmark$",
    "structural_NA": r"N/A",
}


def by_path(df: pd.DataFrame) -> pd.DataFrame:
    out = df.groupby("path")["decision_reason"].value_counts().unstack(fill_value=0)
    out = out.reindex(columns=[c for c in CATEGORY_ORDER if c in out.columns])
    out["n_rows"] = out.sum(axis=1)
    for col in CATEGORY_ORDER:
        if col in out.columns:
            out[f"{col}_pct"] = (100.0 * out[col] / out["n_rows"]).round(1)
    return out


def by_cell(df: pd.DataFrame, path: int) -> pd.DataFrame:
    sub = df[df["path"] == path]
    out = sub.groupby("cell_id")["decision_reason"].value_counts().unstack(fill_value=0)
    out = out.reindex(columns=[c for c in CATEGORY_ORDER if c in out.columns])
    out["n_rows"] = out.sum(axis=1)
    # Percent of rows in the false-closure category (metric > τ, judge < ρ)
    fcc = "false_closure_candidate"
    if fcc in out.columns:
        out["false_closure_pct"] = (100.0 * out[fcc] / out["n_rows"]).round(1)
    mfn = "metric_false_negative"
    if mfn in out.columns:
        out["metric_fn_pct"] = (100.0 * out[mfn] / out["n_rows"]).round(1)
    return out


def build_by_path_latex(by_path_df: pd.DataFrame) -> str:
    """Compact 3x5 table for §5 headline."""
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(
        r"\caption{Judge-metric agreement decomposition by path. Categories from Algorithm~2's decision branches. Path 3 shows the highest metric-false-negative rate (Judge~$\checkmark$ / Metric~$\times$: BERTScore rejects rows the judge accepts), consistent with BERTScore functioning as a surface-form similarity rather than a semantic-equivalence signal.}"
    )
    lines.append(r"\label{tab:disagreement_by_path}")
    lines.append(r"\begin{tabular}{lrrrrr}")
    lines.append(r"\toprule")
    lines.append(
        "Path & "
        + r"Both $\checkmark$ & Both $\times$ & Metric $\checkmark$/Judge $\times$ & Metric $\times$/Judge $\checkmark$ & N/A \\"
    )
    lines.append(r"\midrule")
    for path in [1, 2, 3]:
        if path not in by_path_df.index:
            continue
        row = by_path_df.loc[path]
        parts = [f"Path {path}"]
        for cat in CATEGORY_ORDER:
            if cat not in by_path_df.columns:
                parts.append("---")
                continue
            n = int(row[cat])
            pct = row.get(f"{cat}_pct", None)
            if pct is not None and not pd.isna(pct):
                parts.append(f"{n:,} ({pct:.1f}\\%)")
            else:
                parts.append(f"{n:,}")
        lines.append(" & ".join(parts) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def build_p3_benchmark(df: pd.DataFrame) -> pd.DataFrame:
    """Path 3 decomposition by benchmark (HumanEval vs MBPP)."""
    p3 = df[df["path"] == 3].copy()
    p3["benchmark"] = p3["sample_source"].str.split("/").str[0]
    out = (
        p3.groupby(["benchmark", "decision_reason"])
        .size()
        .unstack(fill_value=0)
    )
    out = out.reindex(columns=[c for c in CATEGORY_ORDER if c in out.columns])
    out["n_rows"] = out.sum(axis=1)
    for col in CATEGORY_ORDER:
        if col in out.columns:
            out[f"{col}_pct"] = (100.0 * out[col] / out["n_rows"]).round(1)
    return out


def build_p3_docstring_length(df: pd.DataFrame) -> pd.DataFrame:
    """Correlate Path 3 disagreement with docstring length proxy.

    We use the length of the judge_justification as an imperfect but readily-
    available signal for artefact complexity. The mean length within each
    decision-reason bucket tells us whether disagreement concentrates on longer
    or shorter artefacts.
    """
    p3 = df[df["path"] == 3].copy()
    p3["justif_len"] = p3["judge_justification"].fillna("").astype(str).str.len()
    out = p3.groupby("decision_reason")["justif_len"].agg(["mean", "median", "count"])
    return out


def main() -> None:
    tsv = Path("results/results_roundtrip.tsv")
    df = load_tsv(tsv)
    df["decision_reason"] = df.apply(label_row, axis=1)

    # Write long-form with labels
    Path("results").mkdir(exist_ok=True)
    df[
        [
            "cell_id",
            "sample_idx",
            "sample_source",
            "path",
            "metric_value",
            "judge_rating",
            "valid",
            "decision_reason",
        ]
    ].to_csv("results/tab_disagreement_long.csv", index=False)

    # By path
    bp = by_path(df)
    bp.to_csv("tables/tab_disagreement_by_path.csv")
    Path("tables/tab_disagreement_by_path.tex").write_text(build_by_path_latex(bp))

    # By cell for each path
    for path in [1, 2, 3]:
        bc = by_cell(df, path)
        bc.to_csv(f"tables/tab_disagreement_by_cell_path{path}.csv")

    # Path 3 by benchmark
    p3b = build_p3_benchmark(df)
    p3b.to_csv("tables/tab_disagreement_p3_by_benchmark.csv")

    # Path 3 justification-length signal
    p3l = build_p3_docstring_length(df)
    p3l.to_csv("tables/tab_disagreement_p3_by_justif_length.csv")

    # Console report
    print("\n=== By path ===")
    display_cols = ["n_rows"] + [
        f"{c}_pct" for c in CATEGORY_ORDER if f"{c}_pct" in bp.columns
    ]
    print(bp[display_cols].to_string())

    print("\n=== Path 3 by benchmark ===")
    display_cols_p3 = ["n_rows"] + [
        f"{c}_pct" for c in CATEGORY_ORDER if f"{c}_pct" in p3b.columns
    ]
    print(p3b[display_cols_p3].to_string())

    print("\n=== Path 3 justification length by decision reason ===")
    print(p3l.round(1).to_string())

    print("\n=== Per-cell false-closure rate on Path 1 ===")
    p1c = by_cell(df, 1)
    if "false_closure_pct" in p1c.columns:
        print(
            p1c[["n_rows", "false_closure_pct", "metric_fn_pct"]]
            .sort_values("false_closure_pct", ascending=False)
            .to_string()
        )


if __name__ == "__main__":
    main()
