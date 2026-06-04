"""
scripts/run_analysis.py — master analysis runner.

Reads results/results_roundtrip.tsv and produces:
    tables/*.tex  + tables/*.csv     — every paper table
    plots/output/*.png                — every paper figure
    results/analysis_summary.json     — every statistical-test output
    results/paper_ready_summary.md    — copy-paste-ready Markdown summary

Run as:
    python3 scripts/run_analysis.py                       # uses RESULTS_TSV
    python3 scripts/run_analysis.py --tsv path/to.tsv     # custom TSV
    python3 scripts/run_analysis.py --synthetic           # use fake data
    python3 scripts/run_analysis.py --pilot               # use pilot TSV

The synthetic mode is for verifying the analysis chain BEFORE the real
Colab sweep generates data.
"""

from __future__ import annotations
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

# Project root importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from config import (
    RESULTS_TSV, PILOT_RESULTS_TSV, RESULTS_DIR, PROJECT_ROOT,
)
from analyze import load_results, stats as stats_mod, tables, sample_stratify
from plots import figures


TABLES_DIR = PROJECT_ROOT / "tables"
SUMMARY_JSON = RESULTS_DIR / "analysis_summary.json"
PAPER_SUMMARY_MD = RESULTS_DIR / "paper_ready_summary.md"


logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("analysis")


# ──────────────────────────────────────────────────────────────────────
# Entry-point pipeline
# ──────────────────────────────────────────────────────────────────────
def run(df: pd.DataFrame) -> dict:
    """Run every stat + every plot + every table. Returns aggregate summary."""
    if df.empty:
        logger.error("No data — analysis aborted.")
        return {"status": "no_data"}

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Analysing {len(df):,} rows from {df['cell_id'].nunique()} cells.")

    summary: dict = {}

    # ── Statistical tests ────────────────────────────────────────────
    logger.info("Running statistical tests…")
    summary["anova"] = {
        "all_paths": stats_mod.anova_type3(df).to_dict("records"),
        "path_1": stats_mod.anova_type3(df, path=1).to_dict("records"),
        "path_2": stats_mod.anova_type3(df, path=2).to_dict("records"),
        "path_3": stats_mod.anova_type3(df, path=3).to_dict("records"),
    }

    tukey = stats_mod.tukey_hsd(df)
    summary["tukey"] = {
        "n_pairs": len(tukey),
        "n_significant_p_05": int((tukey["p_adj"] < 0.05).sum()) if not tukey.empty else 0,
        "n_significant_p_01": int((tukey["p_adj"] < 0.01).sum()) if not tukey.empty else 0,
    }

    bottleneck = stats_mod.per_stage_bottleneck(df)
    summary["bottleneck"] = {
        "n_rows": len(bottleneck),
        "by_stage": (bottleneck["bottleneck_stage"].value_counts().to_dict()
                     if not bottleneck.empty else {}),
    }

    per_bench = stats_mod.per_benchmark_slice(df)
    summary["per_benchmark_keys"] = list(per_bench.keys())

    judge_corr = stats_mod.judge_metric_correlation(df)
    summary["judge_correlation"] = judge_corr

    false_closure = stats_mod.false_closure_rate(df)
    if not false_closure.empty:
        summary["false_closure"] = {
            "mean": float(false_closure["false_closure_rate"].mean()),
            "max": float(false_closure["false_closure_rate"].max()),
            "max_cell": str(false_closure.loc[false_closure["false_closure_rate"].idxmax(),
                                              "cell_id"]),
        }

    contam = stats_mod.contamination_sensitivity(df)
    summary["contamination_n_rows"] = int(len(contam))

    cache_eff = stats_mod.cache_efficiency(df)
    summary["cache_efficiency"] = {
        "total_calls": cache_eff["total_calls"],
        "cache_hits": cache_eff["cache_hits"],
        "hit_rate": cache_eff["hit_rate"],
    }

    cross_spearman = stats_mod.cross_cell_spearman(df, path=1)
    summary["cross_spearman_n_cells"] = int(cross_spearman.shape[0])

    # ── Tables ───────────────────────────────────────────────────────
    logger.info("Writing tables…")
    _save_table("tab_model_lineup", tables.tab_model_lineup())
    _save_table("tab_doe_summary",  tables.tab_doe_summary())
    _save_table("tab_closure_rate_matrix",  tables.tab_closure_rate_matrix(df))
    if summary["anova"]["all_paths"]:
        _save_table("tab_anova", tables.tab_anova(stats_mod.anova_type3(df)))
    _save_table("tab_tukey_significant", tables.tab_tukey_significant(tukey))
    _save_table("tab_per_stage_bottleneck", tables.tab_per_stage_bottleneck(bottleneck))
    _save_table("tab_per_benchmark", tables.tab_per_benchmark(per_bench))
    _save_table("tab_judge_correlation", tables.tab_judge_correlation(judge_corr))
    _save_table("tab_false_closure", tables.tab_false_closure(false_closure))
    _save_table("tab_contamination", tables.tab_contamination(contam))
    _save_table("tab_cache_efficiency", tables.tab_cache_efficiency(cache_eff))

    # ── Figures ──────────────────────────────────────────────────────
    logger.info("Generating figures…")
    fig_paths = []
    fig_paths.append(figures.make_fig_1_methodology())
    fig_paths.append(figures.make_fig_2_closure_heatmap(df))
    fig_paths.append(figures.make_fig_3_mono_vs_hetero(df))
    fig_paths.append(figures.make_fig_4_stage_bottleneck(bottleneck))
    fig_paths.append(figures.make_fig_5_per_benchmark(df))
    fig_paths.append(figures.make_fig_6_cross_family(df))
    fig_paths.append(figures.make_fig_7_judge_corr(df))
    fig_paths.append(figures.make_fig_8_false_closure(false_closure))
    fig_paths.append(figures.make_fig_9_contamination(contam))
    summary["figures"] = [str(p) for p in fig_paths]

    # ── 60-pair sample for human study (if data allows) ──────────────
    pairs = sample_stratify.stratify_60_pairs(df)
    if not pairs.empty:
        worksheet_path = RESULTS_DIR / "human_eval_worksheet_60.csv"
        pairs_path = RESULTS_DIR / "human_eval_pairs_60.tsv"
        pairs.to_csv(pairs_path, sep="\t", index=False)
        summary["human_eval"] = {
            "n_pairs": len(pairs),
            "buckets": pairs["bucket"].value_counts().to_dict(),
            "pairs_tsv": str(pairs_path),
            "worksheet_template_csv": str(worksheet_path),
        }
        # Skeleton worksheet (artefacts must be populated separately from
        # the cache layer's stored LLM responses)
        skeleton = pairs[["pair_id"]].copy()
        skeleton["original_code"] = ""
        skeleton["reconstructed_code"] = ""
        skeleton["rating"] = ""
        skeleton["justification"] = ""
        skeleton.to_csv(worksheet_path, index=False)

    # ── Write summary JSON ───────────────────────────────────────────
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2, default=str),
                              encoding="utf-8")
    logger.info(f"Wrote {SUMMARY_JSON}")

    # ── Write paper-ready Markdown report ────────────────────────────
    _write_paper_summary(summary, df)
    logger.info(f"Wrote {PAPER_SUMMARY_MD}")

    return summary


def _save_table(name: str, output: tuple[str, str]) -> None:
    latex, csv = output
    if latex:
        tables.save(name, latex, csv, TABLES_DIR)


def _write_paper_summary(summary: dict, df: pd.DataFrame) -> None:
    """Write a Markdown summary with copy-paste-ready paragraphs."""
    parts: list[str] = []
    parts.append("# Paper-ready analysis summary\n")
    parts.append(f"_Generated by `scripts/run_analysis.py`._\n\n")
    parts.append(f"## Data\n")
    parts.append(f"- Total rows in TSV: **{len(df):,}**\n")
    parts.append(f"- Cells with data: **{df['cell_id'].nunique()}**\n")
    parts.append(f"- Paths: **{sorted(df['path'].unique().tolist())}**\n")
    parts.append(f"- Benchmarks: **{sorted(df['benchmark'].dropna().unique().tolist())}**\n\n")

    parts.append("## Key statistical findings\n\n")

    # Tukey HSD
    parts.append(f"### Tukey HSD post-hoc\n")
    parts.append(f"- {summary['tukey']['n_pairs']} cell pairs compared\n")
    parts.append(f"- **{summary['tukey']['n_significant_p_05']}** significant at $p < 0.05$\n")
    parts.append(f"- **{summary['tukey']['n_significant_p_01']}** significant at $p < 0.01$\n\n")

    # Bottleneck
    parts.append(f"### Per-stage bottleneck attribution\n")
    for stage, count in summary["bottleneck"].get("by_stage", {}).items():
        parts.append(f"- `{stage}`: **{count}** (cell, path) combinations attribute failure to this stage\n")
    parts.append("\n")

    # Judge correlation
    parts.append(f"### Judge ↔ automated metric correlation (RQ2)\n")
    for path, stats in summary["judge_correlation"].get("per_path", {}).items():
        if stats and stats.get("r") is not None:
            parts.append(f"- Path {path}: r = **{stats['r']:.3f}** (p = {stats['p']:.3g}, n = {stats['n']}) {stats.get('sig', '')}\n")
    overall = summary["judge_correlation"].get("overall")
    if overall:
        parts.append(f"- **Overall**: r = **{overall['r']:.3f}** (p = {overall['p']:.3g}, n = {overall['n']}) {overall.get('sig', '')}\n")
    parts.append("\n")

    # False closure
    if "false_closure" in summary:
        fc = summary["false_closure"]
        parts.append(f"### False-closure rate (RQ5)\n")
        parts.append(f"- Mean across cells: **{fc['mean']:.1%}**\n")
        parts.append(f"- Maximum: **{fc['max']:.1%}** in cell **{fc['max_cell']}**\n\n")

    # Cache
    ce = summary["cache_efficiency"]
    parts.append(f"### Cache efficiency\n")
    parts.append(f"- Total LLM calls: **{ce['total_calls']:,}**\n")
    parts.append(f"- Cache hits: **{ce['cache_hits']:,}**\n")
    parts.append(f"- Hit rate: **{ce['hit_rate']:.1%}** (Drive-backed, survives Colab disconnects)\n\n")

    # Human-eval status
    if "human_eval" in summary:
        he = summary["human_eval"]
        parts.append(f"## Human-evaluation\n")
        parts.append(f"- 60 pairs stratified for annotation:\n")
        for bucket, n in he["buckets"].items():
            parts.append(f"    - `{bucket}`: {n} pairs\n")
        parts.append(f"- Blinded worksheet template: `{he['worksheet_template_csv']}`\n")
        parts.append(f"- Full stratified-pair info (for de-blinding): `{he['pairs_tsv']}`\n\n")

    # Figures
    parts.append("## Figures (paper-ready)\n")
    for p in summary.get("figures", []):
        parts.append(f"- `{Path(p).relative_to(PROJECT_ROOT)}`\n")
    parts.append("\n")

    parts.append("## Tables (LaTeX + CSV)\n")
    for tex in sorted(TABLES_DIR.glob("*.tex")):
        parts.append(f"- `{tex.relative_to(PROJECT_ROOT)}`\n")
    parts.append("\n")

    PAPER_SUMMARY_MD.write_text("".join(parts), encoding="utf-8")


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsv", default=str(RESULTS_TSV),
                        help="Path to results TSV")
    parser.add_argument("--pilot", action="store_true",
                        help="Use the pilot TSV instead of the full sweep TSV")
    parser.add_argument("--synthetic", action="store_true",
                        help="Synthesize fake data — for verifying the chain "
                             "before real Colab data exists")
    args = parser.parse_args()

    if args.synthetic:
        logger.info("Using synthetic data.")
        df = load_results.synthesize_fake_data(n_samples=30)
    elif args.pilot:
        logger.info(f"Loading pilot TSV: {PILOT_RESULTS_TSV}")
        df = load_results.load_tsv(PILOT_RESULTS_TSV)
    else:
        logger.info(f"Loading sweep TSV: {args.tsv}")
        df = load_results.load_tsv(Path(args.tsv))

    summary = run(df)
    if summary.get("status") == "no_data":
        logger.error("No data found. Either:")
        logger.error("  1. Run the pilot (python3 scripts/run_pilot.py) then re-run with --pilot")
        logger.error("  2. Use --synthetic to verify the chain")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
