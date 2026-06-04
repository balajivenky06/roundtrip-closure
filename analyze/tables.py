"""
analyze/tables.py — LaTeX + CSV table generators for the journal paper.

Every function takes the enriched DataFrame and returns (latex_str, csv_str).
The master runner writes both forms to `tables/<name>.tex` and
`tables/<name>.csv` so you can either paste the LaTeX directly into
`paper_draft.tex` or import the CSV into Excel for spot-checks.

Tables produced (parallel to the SQJ Ch.2 set):
    tab_model_lineup            — Table 1: 7 SLMs + judge with metadata
    tab_doe_summary             — Table 2: the 20-cell DOE with hypotheses
    tab_closure_rate_matrix     — Table 3: cells × paths mean closure
    tab_anova                   — Table 4: Type-III ANOVA
    tab_tukey_significant       — Table 5: significant Tukey HSD pairs only
    tab_per_stage_bottleneck    — Table 6: hetero vs mono comparison
    tab_per_benchmark           — Table 7: HE vs MBPP slice
    tab_judge_correlation       — Table 8: judge ↔ metric Pearson r
    tab_false_closure           — Table 9: false-closure rate by cell
    tab_contamination           — Table 10: HE vs HEM delta
    tab_cache_efficiency        — Table 11: per-cell cache hit rates
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional

import pandas as pd

from config import PIPELINE_MODELS, JUDGE_MODEL
from doe import ALL_CELLS


# ──────────────────────────────────────────────────────────────────────
# Public format
# ──────────────────────────────────────────────────────────────────────
def save(name: str, latex: str, csv: str, out_dir: Path) -> None:
    """Write both forms of a table to disk."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.tex").write_text(latex, encoding="utf-8")
    (out_dir / f"{name}.csv").write_text(csv, encoding="utf-8")


# ──────────────────────────────────────────────────────────────────────
# Table 1 — Model lineup (static; no data needed)
# ──────────────────────────────────────────────────────────────────────
def tab_model_lineup() -> tuple[str, str]:
    """Static table of the 7 SLMs in the lineup."""
    rows = []
    for m in list(PIPELINE_MODELS) + [JUDGE_MODEL]:
        role = "Judge" if m == JUDGE_MODEL else "Pipeline"
        rows.append({
            "Role": role,
            "Ollama tag": m.ollama_tag,
            "Family": m.family,
            "Size (B)": m.size_b,
            "Architecture": m.architecture,
            "Year": m.generation_year,
        })
    df = pd.DataFrame(rows)
    latex = _df_to_latex(df, caption=(
        "Small Language Models in the Chapter-3 pipeline. All models are "
        "open-weight and served via Ollama; total parameter count is "
        "$<30$ B per the §1.1 SLM definition."), label="tab:slm-lineup")
    return latex, df.to_csv(index=False)


# ──────────────────────────────────────────────────────────────────────
# Table 2 — DOE summary
# ──────────────────────────────────────────────────────────────────────
def tab_doe_summary() -> tuple[str, str]:
    """Static table of the 20-cell DOE — one row per cell with hypothesis."""
    rows = []
    for cell in ALL_CELLS:
        rows.append({
            "Cell": cell.cell_id,
            "Stratum": cell.stratum,
            "L\\_spec": cell.L_spec.short_name if cell.L_spec else "(skip)",
            "L\\_test": cell.L_test.short_name if cell.L_test else "(skip)",
            "L\\_code": cell.L_code.short_name if cell.L_code else "(skip)",
            "Hypothesis": cell.hypothesis[:80] + ("…" if len(cell.hypothesis) > 80 else ""),
        })
    df = pd.DataFrame(rows)
    csv = df.to_csv(index=False)
    # CSV uses plain names; LaTeX needs short hypothesis text for layout
    latex = _df_to_latex(df, caption=(
        "Pre-registered $4{\\times}4{\\times}4$ fractional factorial DOE. "
        "20 cells in three strata: 6 mono, 11 hetero, 3 null. Hypothesis "
        "is the rationale recorded at design time."), label="tab:doe")
    return latex, csv


# ──────────────────────────────────────────────────────────────────────
# Table 3 — Closure rate matrix (cell × path)
# ──────────────────────────────────────────────────────────────────────
def tab_closure_rate_matrix(df: pd.DataFrame) -> tuple[str, str]:
    """Mean closure rate per (cell, path) with n_valid in parentheses."""
    from analyze.load_results import filter_valid
    valid = filter_valid(df)
    if valid.empty:
        empty = pd.DataFrame()
        return _df_to_latex(empty, caption="(no data)", label="tab:closure-matrix"), ""

    mean = valid.pivot_table(index="cell_id", columns="path",
                              values="metric_value", aggfunc="mean")
    n = valid.pivot_table(index="cell_id", columns="path",
                           values="metric_value", aggfunc="count")
    out = pd.DataFrame(index=mean.index)
    for p in (1, 2, 3):
        if p in mean.columns:
            out[f"Path {p}"] = mean[p].round(3).astype(str) + " (" + n[p].astype(str) + ")"

    out = out.reset_index().rename(columns={"cell_id": "Cell"})
    latex = _df_to_latex(out, caption=(
        "Mean closure rate per cell $\\times$ path with $n_{\\text{valid}}$ "
        "in parentheses. Path 1: mutation kill rate; Path 2: reference-test "
        "pass rate; Path 3: BERTScore F1."), label="tab:closure-matrix")
    return latex, out.to_csv(index=False)


# ──────────────────────────────────────────────────────────────────────
# Table 4 — ANOVA
# ──────────────────────────────────────────────────────────────────────
def tab_anova(anova_df: pd.DataFrame) -> tuple[str, str]:
    if anova_df.empty:
        return "(no data)", ""
    fmt = anova_df.copy()
    fmt["F"] = fmt["F"].round(2)
    fmt["sum_sq"] = fmt["sum_sq"].round(2)
    fmt["p"] = fmt["p"].apply(_format_p)
    fmt.columns = ["Factor", "Sum Sq", "df", "F", "p", "Sig"]
    latex = _df_to_latex(fmt, caption=(
        "Type-III ANOVA on closure metric (response) $\\sim$ "
        "C(cell) + C(sample)."), label="tab:anova")
    return latex, fmt.to_csv(index=False)


# ──────────────────────────────────────────────────────────────────────
# Table 5 — Significant Tukey HSD pairs
# ──────────────────────────────────────────────────────────────────────
def tab_tukey_significant(tukey_df: pd.DataFrame,
                          alpha: float = 0.05,
                          top_n: int = 25) -> tuple[str, str]:
    if tukey_df.empty:
        return "(no data)", ""
    sig = tukey_df[(tukey_df["p_adj"].notna()) & (tukey_df["p_adj"] < alpha)].copy()
    sig = sig.sort_values("p_adj").head(top_n)
    if sig.empty:
        return "% No Tukey HSD pairs reached $p_{\\text{adj}} < " + str(alpha) + "$.\n", ""

    sig["mean_diff"] = sig["mean_diff"].round(3)
    sig["p_adj"] = sig["p_adj"].apply(_format_p)
    sig = sig[["cell_a", "cell_b", "mean_diff", "p_adj", "sig"]]
    sig.columns = ["Cell A", "Cell B", "$\\Delta$ mean", "$p_{\\text{adj}}$", "Sig"]
    latex = _df_to_latex(sig, caption=(
        f"Tukey HSD post-hoc — significant pairs at $\\alpha = {alpha}$, "
        f"top {top_n} by $p_{{\\text{{adj}}}}$. Bonferroni-corrected across "
        "the full 190-pair family."), label="tab:tukey")
    return latex, sig.to_csv(index=False)


# ──────────────────────────────────────────────────────────────────────
# Table 6 — Per-stage bottleneck
# ──────────────────────────────────────────────────────────────────────
def tab_per_stage_bottleneck(bottleneck_df: pd.DataFrame) -> tuple[str, str]:
    if bottleneck_df.empty:
        return "(no data)", ""
    fmt = bottleneck_df.copy()
    for col in ("hetero_mean", "mono_spec", "mono_test", "mono_code", "bottleneck_delta"):
        if col in fmt.columns:
            fmt[col] = fmt[col].round(3)
    fmt = fmt.rename(columns={
        "hetero_cell": "Hetero",
        "path": "Path",
        "hetero_mean": "Hetero mean",
        "mono_spec": "Mono(L\\_spec)",
        "mono_test": "Mono(L\\_test)",
        "mono_code": "Mono(L\\_code)",
        "bottleneck_stage": "Bottleneck stage",
        "bottleneck_delta": "$\\Delta$ to bottleneck",
    })
    latex = _df_to_latex(fmt, caption=(
        "Per-stage bottleneck decomposition. For each hetero cell + path, "
        "the bottleneck stage is the one whose mono-cell mean is lowest. "
        "$\\Delta$ to bottleneck = hetero\\_mean $-$ mono(bottleneck) — "
        "positive means the hetero pipeline outperformed the bottleneck "
        "stage alone."), label="tab:bottleneck")
    return latex, fmt.to_csv(index=False)


# ──────────────────────────────────────────────────────────────────────
# Table 7 — Per-benchmark slice
# ──────────────────────────────────────────────────────────────────────
def tab_per_benchmark(per_bench: dict) -> tuple[str, str]:
    """Compact summary of ANOVA F-stats per benchmark."""
    rows = []
    for bench, payload in per_bench.items():
        anova = payload.get("anova", pd.DataFrame())
        n = payload.get("n", 0)
        f_cell = ""
        p_cell = ""
        if not anova.empty:
            cell_row = anova[anova["factor"] == "C(cell_id)"]
            if not cell_row.empty:
                f_cell = round(cell_row["F"].iloc[0], 2)
                p_cell = _format_p(cell_row["p"].iloc[0])
        rows.append({
            "Benchmark": bench,
            "n": n,
            "F (cell)": f_cell,
            "p (cell)": p_cell,
        })
    df = pd.DataFrame(rows)
    latex = _df_to_latex(df, caption=(
        "Per-benchmark Type-III ANOVA on closure rate. F is the cell-factor "
        "F-statistic; p is the cell-factor p-value within that benchmark."),
        label="tab:per-benchmark")
    return latex, df.to_csv(index=False)


# ──────────────────────────────────────────────────────────────────────
# Table 8 — Judge ↔ metric correlation
# ──────────────────────────────────────────────────────────────────────
def tab_judge_correlation(judge_corr: dict) -> tuple[str, str]:
    rows = []
    for path, stats in judge_corr.get("per_path", {}).items():
        if stats and stats.get("r") is not None:
            rows.append({
                "Scope": f"Path {path}",
                "n": stats["n"],
                "Pearson r": round(stats["r"], 3),
                "p": _format_p(stats["p"]),
                "Sig": stats.get("sig", ""),
            })
    if judge_corr.get("overall"):
        ov = judge_corr["overall"]
        rows.append({
            "Scope": "Overall",
            "n": ov["n"],
            "Pearson r": round(ov["r"], 3),
            "p": _format_p(ov["p"]),
            "Sig": ov.get("sig", ""),
        })
    df = pd.DataFrame(rows)
    latex = _df_to_latex(df, caption=(
        "Pearson correlation between the automated closure metric and the "
        "external SLM judge's 0--4 equivalence rating. RQ2: do these two "
        "signals agree?"), label="tab:judge-corr")
    return latex, df.to_csv(index=False)


# ──────────────────────────────────────────────────────────────────────
# Table 9 — False-closure rate
# ──────────────────────────────────────────────────────────────────────
def tab_false_closure(false_df: pd.DataFrame, top_n: int = 15) -> tuple[str, str]:
    if false_df.empty:
        return "(no data)", ""
    fmt = false_df.sort_values("false_closure_rate", ascending=False).head(top_n).copy()
    fmt["false_closure_rate"] = fmt["false_closure_rate"].apply(
        lambda x: f"{x:.1%}" if pd.notna(x) else "—"
    )
    fmt = fmt[["cell_id", "path", "n", "n_closure_success",
               "n_false_closure", "false_closure_rate"]]
    fmt.columns = ["Cell", "Path", "n", "n success", "n false",
                   "False-closure rate"]
    latex = _df_to_latex(fmt, caption=(
        "False-closure rate per cell $\\times$ path — fraction of rows where "
        "the automated metric reports SUCCESS but the judge SLM disagrees "
        "(rating $< 3$). Top 15 by rate."), label="tab:false-closure")
    return latex, fmt.to_csv(index=False)


# ──────────────────────────────────────────────────────────────────────
# Table 10 — Contamination delta
# ──────────────────────────────────────────────────────────────────────
def tab_contamination(contam_df: pd.DataFrame) -> tuple[str, str]:
    if contam_df.empty:
        return "(no data)", ""
    fmt = contam_df.copy()
    for col in ("mean_he", "mean_hem", "delta"):
        if col in fmt.columns:
            fmt[col] = fmt[col].round(3)
    fmt["p"] = fmt["p"].apply(_format_p)
    fmt = fmt[["cell_id", "path", "n_he", "n_hem",
               "mean_he", "mean_hem", "delta", "p", "sig"]]
    fmt.columns = ["Cell", "Path", "$n_{HE}$", "$n_{HEM}$",
                   "Mean HE", "Mean HEM", "$\\Delta$ (HE$-$HEM)",
                   "p (Welch)", "Sig"]
    latex = _df_to_latex(fmt, caption=(
        "Contamination sensitivity: closure rate on HumanEval (potentially "
        "contaminated) vs HumanEval-Mutated (decontaminated) per cell + path. "
        "Welch's t-test, $\\Delta > 0$ indicates HE bias."),
        label="tab:contamination")
    return latex, fmt.to_csv(index=False)


# ──────────────────────────────────────────────────────────────────────
# Table 11 — Cache efficiency
# ──────────────────────────────────────────────────────────────────────
def tab_cache_efficiency(cache_dict: dict) -> tuple[str, str]:
    per_cell = cache_dict.get("per_cell", pd.DataFrame())
    if per_cell.empty:
        return "(no data)", ""
    fmt = per_cell.copy()
    fmt["hit_rate"] = fmt["hit_rate"].apply(lambda x: f"{x:.1%}")
    fmt = fmt[["cell_id", "n_calls", "n_hits", "hit_rate"]]
    fmt.columns = ["Cell", "LLM calls", "Cache hits", "Hit rate"]
    latex = _df_to_latex(fmt, caption=(
        "Per-cell cache efficiency. Total cache hit rate across the sweep is "
        f"{cache_dict.get('hit_rate', 0):.1%} "
        f"(${cache_dict.get('cache_hits', 0):,}$ hits / "
        f"${cache_dict.get('total_calls', 0):,}$ calls)."),
        label="tab:cache")
    return latex, fmt.to_csv(index=False)


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────
def _format_p(p) -> str:
    if pd.isna(p):
        return "—"
    p = float(p)
    if p < 0.001:
        return "$<0.001$"
    if p < 0.01:
        return f"${p:.3f}$"
    return f"${p:.3f}$"


def _df_to_latex(df: pd.DataFrame, caption: str, label: str) -> str:
    """Render a DataFrame as a Springer-Nature-compatible LaTeX tabular."""
    if df.empty:
        return f"% (empty table for {label})\n"
    n_cols = len(df.columns)
    col_spec = "@{}l" + "c" * (n_cols - 1) + "@{}"
    header_cells = " & ".join(str(c) for c in df.columns) + " \\\\"
    body_rows = []
    for _, row in df.iterrows():
        body_rows.append(" & ".join(str(v) for v in row.values) + " \\\\")
    body = "\n".join(body_rows)

    return (
        "\\begin{table}[t]\n"
        "\\centering\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        "\\begin{tabular}{" + col_spec + "}\n"
        "\\toprule\n"
        + header_cells + "\n"
        "\\midrule\n"
        + body + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )
