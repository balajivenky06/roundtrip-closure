"""
plots/figures.py — all 9 publication figures for the journal paper.

Every figure-maker function:
    - Takes an enriched DataFrame (or pre-computed stats output)
    - Calls apply_publication_style() to set rcParams
    - Writes a PNG to plots/output/<filename>.png
    - Returns the output Path

The master runner (scripts/run_analysis.py) calls them all in sequence.

Figure index (parallel to the Chapter 2 SQJ paper's figure set):

    make_fig_1_methodology      — triangle diagram + 3 paths + 20 cells
    make_fig_2_closure_heatmap  — cells × paths mean-closure heatmap
    make_fig_3_mono_vs_hetero   — boxplot per stratum
    make_fig_4_stage_bottleneck — stacked bars per hetero cell
    make_fig_5_per_benchmark    — HumanEval vs MBPP per-cell
    make_fig_6_cross_family     — 5×5 family-pair heatmap
    make_fig_7_judge_corr       — judge_rating vs metric_value scatter
    make_fig_8_false_closure    — per-cell false-closure rate bars
    make_fig_9_contamination    — HE vs HEM delta heatmap
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Project root importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from plots.style import (
    apply_publication_style,
    COLORS_STRATA, COLORS_FAMILIES, COLORS_PATHS,
    CMAP_CLOSURE, CMAP_DIVERGENT,
)

from analyze.load_results import filter_valid, filter_path


PLOTS_OUTPUT = Path(__file__).resolve().parent / "output"
PLOTS_OUTPUT.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────
# Figure 1 — Methodology diagram
# ──────────────────────────────────────────────────────────────────────
def make_fig_1_methodology(output_path: Path = PLOTS_OUTPUT / "fig1_methodology.png") -> Path:
    """Draw the triangle + 3 closure paths + DOE strata. Static; no data."""
    apply_publication_style()
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    ax.set_xlim(0, 12); ax.set_ylim(0, 9); ax.axis("off")

    # Triangle vertices
    C = (3, 6); D = (6, 8); T = (9, 6)
    for vert, label, color in [
        (C, "C (code)", "#E8EEF7"),
        (D, "D (docstring)", "#FFE9B0"),
        (T, "T (tests)", "#D8E8D8"),
    ]:
        box = FancyBboxPatch((vert[0] - 0.9, vert[1] - 0.4), 1.8, 0.8,
                              boxstyle="round,pad=0.02,rounding_size=0.1",
                              linewidth=1.2, facecolor=color, edgecolor="#333")
        ax.add_patch(box)
        ax.text(vert[0], vert[1], label, ha="center", va="center", fontsize=11,
                fontweight="bold")

    # Triangle edges
    for v1, v2 in [(C, D), (D, T), (C, T)]:
        ax.plot([v1[0], v2[0]], [v1[1], v2[1]], color="#888", linewidth=1.2,
                linestyle="--", zorder=1)

    # 3 closure paths (annotated arrows beside the triangle)
    paths_box_x, paths_box_y = 0.4, 0.5
    ax.add_patch(FancyBboxPatch((paths_box_x, paths_box_y), 11, 4.0,
                                 boxstyle="round,pad=0.05,rounding_size=0.08",
                                 facecolor="#FAFAFA", edgecolor="#888"))
    ax.text(6, 4.2, "Three closure paths (per cell × function)",
            ha="center", fontsize=11, fontweight="bold")
    path_lines = [
        ("Path 1 (RQ1)", "C → D → T", "Mutation kill rate of T against C's mutants"),
        ("Path 2 (RQ1)", "D → T → C", "Pass rate of ORIGINAL C's reference tests against C'"),
        ("Path 3 (RQ2)", "C → T → D", "BERTScore(D, D') + judge SLM equivalence"),
    ]
    for i, (name, arrow, desc) in enumerate(path_lines):
        y = 3.5 - i * 0.8
        ax.text(0.8, y, name, fontsize=10, fontweight="bold",
                color=COLORS_PATHS[i + 1])
        ax.text(2.5, y, arrow, fontsize=10, family="monospace")
        ax.text(5.0, y, desc, fontsize=9, color="#444")

    # DOE strata summary (bottom right)
    ax.text(6, 8.4, "20-cell pre-registered fractional-factorial DOE",
            ha="center", fontsize=10, fontweight="bold")
    strata = [("6 mono", COLORS_STRATA["mono"]),
              ("11 hetero", COLORS_STRATA["hetero"]),
              ("3 null", COLORS_STRATA["null"])]
    x = 4.0
    for label, color in strata:
        ax.add_patch(plt.Circle((x, 8.0), 0.18, facecolor=color, edgecolor="black"))
        ax.text(x + 0.3, 7.97, label, fontsize=9, va="center")
        x += 2.0

    ax.set_title("Multi-SLM closure of the docstring-test-code triangle",
                  fontsize=13, fontweight="bold", pad=20)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


# ──────────────────────────────────────────────────────────────────────
# Figure 2 — Closure rate heatmap (cells × paths)
# ──────────────────────────────────────────────────────────────────────
def make_fig_2_closure_heatmap(df: pd.DataFrame,
                               output_path: Path = PLOTS_OUTPUT / "fig2_closure_heatmap.png") -> Path:
    apply_publication_style()
    valid = filter_valid(df)
    if valid.empty:
        return _save_empty(output_path, "fig2_closure_heatmap")

    pivot = valid.pivot_table(index="cell_id", columns="path",
                                values="metric_value", aggfunc="mean")
    # Stable cell-order: mono → hetero → null
    cell_order = sorted(pivot.index,
                        key=lambda c: ({"M": 0, "H": 1, "N": 2}.get(c[0], 3),
                                       int(c[1:])))
    pivot = pivot.reindex(cell_order)

    fig, ax = plt.subplots(figsize=(7, 8.5))
    im = ax.imshow(pivot.values, cmap=CMAP_CLOSURE, aspect="auto",
                    vmin=0, vmax=1)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"Path {p}" for p in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            v = pivot.values[i, j]
            ax.text(j, i, f"{v:.2f}" if not np.isnan(v) else "—",
                    ha="center", va="center",
                    color="white" if v < 0.5 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, label="Mean closure rate", shrink=0.6)
    ax.set_title("Mean closure rate per cell × path")
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


# ──────────────────────────────────────────────────────────────────────
# Figure 3 — Mono vs Hetero distribution
# ──────────────────────────────────────────────────────────────────────
def make_fig_3_mono_vs_hetero(df: pd.DataFrame,
                              output_path: Path = PLOTS_OUTPUT / "fig3_mono_vs_hetero.png") -> Path:
    apply_publication_style()
    valid = filter_valid(df)
    if valid.empty:
        return _save_empty(output_path, "fig3_mono_vs_hetero")

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4), sharey=True)
    for ax, path in zip(axes, [1, 2, 3]):
        sub = filter_path(valid, path)
        data = [
            sub[sub["cell_stratum"] == "mono"]["metric_value"].dropna().values,
            sub[sub["cell_stratum"] == "hetero"]["metric_value"].dropna().values,
            sub[sub["cell_stratum"] == "null"]["metric_value"].dropna().values,
        ]
        labels = [f"Mono\n(n={len(d)})" for d in data[:1]] + \
                 [f"Hetero\n(n={len(d)})" for d in data[1:2]] + \
                 [f"Null\n(n={len(d)})" for d in data[2:]]
        bp = ax.boxplot(data, labels=labels, patch_artist=True,
                          showmeans=True, widths=0.6)
        for patch, stratum in zip(bp["boxes"], ["mono", "hetero", "null"]):
            patch.set_facecolor(COLORS_STRATA[stratum])
            patch.set_alpha(0.7)
        ax.set_title(f"Path {path}")
        ax.set_ylim(-0.05, 1.05)
    axes[0].set_ylabel("Closure metric")
    fig.suptitle("Closure-rate distribution by stratum × path", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


# ──────────────────────────────────────────────────────────────────────
# Figure 4 — Per-stage bottleneck stacked bars
# ──────────────────────────────────────────────────────────────────────
def make_fig_4_stage_bottleneck(bottleneck_df: pd.DataFrame,
                                output_path: Path = PLOTS_OUTPUT / "fig4_stage_bottleneck.png") -> Path:
    apply_publication_style()
    if bottleneck_df.empty:
        return _save_empty(output_path, "fig4_stage_bottleneck")

    # Aggregate: for each hetero cell, mean Δ across paths grouped by stage
    pivot = bottleneck_df.pivot_table(
        index="hetero_cell", columns="bottleneck_stage",
        values="bottleneck_delta", aggfunc="mean"
    ).fillna(0)
    # Ensure all 3 stages exist as columns
    for stage in ("spec", "test", "code"):
        if stage not in pivot.columns:
            pivot[stage] = 0
    pivot = pivot[["spec", "test", "code"]]
    pivot = pivot.sort_index()

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(pivot.index))
    width = 0.25
    colors = {"spec": "#3274A1", "test": "#E1812C", "code": "#5DAE7C"}
    for i, stage in enumerate(["spec", "test", "code"]):
        ax.bar(x + (i - 1) * width, pivot[stage], width,
                label=f"Bottleneck = {stage}", color=colors[stage])
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=30, ha="right")
    ax.axhline(0, color="#333", linewidth=0.8)
    ax.set_ylabel("Mean Δ (hetero − mono-bottleneck)")
    ax.set_title("Per-stage bottleneck contribution by hetero cell")
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


# ──────────────────────────────────────────────────────────────────────
# Figure 5 — Per-benchmark closure (HE vs MBPP)
# ──────────────────────────────────────────────────────────────────────
def make_fig_5_per_benchmark(df: pd.DataFrame,
                             output_path: Path = PLOTS_OUTPUT / "fig5_per_benchmark.png") -> Path:
    apply_publication_style()
    valid = filter_valid(df)
    if valid.empty:
        return _save_empty(output_path, "fig5_per_benchmark")

    pivot = valid.pivot_table(
        index="cell_id", columns="benchmark", values="metric_value",
        aggfunc="mean",
    )
    cell_order = sorted(pivot.index,
                        key=lambda c: ({"M": 0, "H": 1, "N": 2}.get(c[0], 3),
                                       int(c[1:])))
    pivot = pivot.reindex(cell_order)

    fig, ax = plt.subplots(figsize=(10, 6))
    bench_cols = [c for c in ["humaneval", "mbpp", "livecodebench",
                                "humaneval_mutated"] if c in pivot.columns]
    x = np.arange(len(pivot.index))
    width = 0.8 / max(len(bench_cols), 1)
    palette = ["#3274A1", "#E1812C", "#5DAE7C", "#9C5DA1"]
    for i, b in enumerate(bench_cols):
        ax.bar(x + i * width, pivot[b].values, width, label=b,
                color=palette[i % len(palette)])
    ax.set_xticks(x + width * (len(bench_cols) - 1) / 2)
    ax.set_xticklabels(pivot.index, rotation=45, ha="right")
    ax.set_ylabel("Mean closure rate (averaged over paths)")
    ax.set_title("Per-cell closure rate by benchmark")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


# ──────────────────────────────────────────────────────────────────────
# Figure 6 — Cross-family heatmap (L_spec family × L_code family)
# ──────────────────────────────────────────────────────────────────────
def make_fig_6_cross_family(df: pd.DataFrame,
                            output_path: Path = PLOTS_OUTPUT / "fig6_cross_family.png") -> Path:
    apply_publication_style()
    valid = filter_valid(df)
    if valid.empty:
        return _save_empty(output_path, "fig6_cross_family")

    pivot = valid.pivot_table(
        index="l_spec_family", columns="l_code_family",
        values="metric_value", aggfunc="mean",
    )
    family_order = ["Meta", "Microsoft", "Alibaba", "Alibaba-coder",
                      "Google", "Mistral"]
    pivot = pivot.reindex(index=family_order, columns=family_order)

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(pivot.values, cmap=CMAP_CLOSURE, aspect="auto",
                    vmin=0, vmax=1)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("L_code family")
    ax.set_ylabel("L_spec family")
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v < 0.5 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, label="Mean closure rate", shrink=0.6)
    ax.set_title("Cross-family closure: L_spec family × L_code family")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


# ──────────────────────────────────────────────────────────────────────
# Figure 7 — Judge ↔ metric scatter
# ──────────────────────────────────────────────────────────────────────
def make_fig_7_judge_corr(df: pd.DataFrame,
                          output_path: Path = PLOTS_OUTPUT / "fig7_judge_corr.png") -> Path:
    apply_publication_style()
    valid = filter_valid(df)
    valid = valid[(valid["judge_rating"].notna()) & (valid["judge_rating"] >= 0)]
    if valid.empty:
        return _save_empty(output_path, "fig7_judge_corr")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, path in zip(axes, (1, 2, 3)):
        sub = filter_path(valid, path)
        if sub.empty:
            ax.set_title(f"Path {path} — no data")
            continue
        ax.scatter(sub["metric_value"], sub["judge_rating"],
                    alpha=0.4, s=14, color=COLORS_PATHS[path])
        try:
            from scipy import stats as scs
            r, p = scs.pearsonr(sub["metric_value"].astype(float),
                                  sub["judge_rating"].astype(float))
            ax.set_title(f"Path {path}\nPearson r = {r:.2f}, p = {p:.3g}")
        except Exception:
            ax.set_title(f"Path {path}")
        ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.5, 4.5)
        ax.set_xlabel("Automated metric")
    axes[0].set_ylabel("Judge SLM rating (0–4)")
    fig.suptitle("Judge SLM rating vs automated closure metric (RQ2)", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


# ──────────────────────────────────────────────────────────────────────
# Figure 8 — False-closure rate bars
# ──────────────────────────────────────────────────────────────────────
def make_fig_8_false_closure(false_df: pd.DataFrame,
                             output_path: Path = PLOTS_OUTPUT / "fig8_false_closure.png") -> Path:
    apply_publication_style()
    if false_df.empty:
        return _save_empty(output_path, "fig8_false_closure")
    pivot = false_df.pivot_table(
        index="cell_id", columns="path", values="false_closure_rate",
        aggfunc="mean",
    )
    cell_order = sorted(pivot.index,
                        key=lambda c: ({"M": 0, "H": 1, "N": 2}.get(c[0], 3),
                                       int(c[1:])))
    pivot = pivot.reindex(cell_order)

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(pivot.index))
    width = 0.27
    for i, p in enumerate(sorted(pivot.columns)):
        ax.bar(x + (i - 1) * width, pivot[p].fillna(0), width,
                label=f"Path {p}", color=COLORS_PATHS.get(int(p), "#888"))
    ax.set_xticks(x); ax.set_xticklabels(pivot.index, rotation=45, ha="right")
    ax.set_ylabel("False-closure rate")
    ax.set_title("False-closure rate per cell × path (RQ5)")
    ax.legend(loc="upper right")
    ax.set_ylim(0, max(0.6, pivot.fillna(0).values.max() * 1.1))
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


# ──────────────────────────────────────────────────────────────────────
# Figure 9 — Contamination delta heatmap
# ──────────────────────────────────────────────────────────────────────
def make_fig_9_contamination(contam_df: pd.DataFrame,
                             output_path: Path = PLOTS_OUTPUT / "fig9_contamination.png") -> Path:
    apply_publication_style()
    if contam_df.empty:
        return _save_empty(output_path, "fig9_contamination")
    pivot = contam_df.pivot_table(
        index="cell_id", columns="path", values="delta", aggfunc="mean",
    )
    cell_order = sorted(pivot.index,
                        key=lambda c: ({"M": 0, "H": 1, "N": 2}.get(c[0], 3),
                                       int(c[1:])))
    pivot = pivot.reindex(cell_order)

    vmax = max(abs(pivot.min().min()), abs(pivot.max().max()), 0.05)
    fig, ax = plt.subplots(figsize=(6, 8))
    im = ax.imshow(pivot.values, cmap=CMAP_DIVERGENT, aspect="auto",
                    vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"Path {p}" for p in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                        color="white" if abs(v) > vmax * 0.7 else "black",
                        fontsize=8)
    fig.colorbar(im, ax=ax, label="HE − HEM Δ", shrink=0.6)
    ax.set_title("Contamination sensitivity: HE − HEM per cell × path")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────
def _save_empty(path: Path, label: str) -> Path:
    apply_publication_style()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.text(0.5, 0.5, f"(no data)\n{label}",
            ha="center", va="center", fontsize=12,
            transform=ax.transAxes, color="#888")
    ax.axis("off")
    fig.savefig(path)
    plt.close(fig)
    return path


# ──────────────────────────────────────────────────────────────────────
# Convenience: run all on synthetic data
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from analyze.load_results import synthesize_fake_data
    from analyze import stats as stats_mod

    print("Generating all 9 figures on synthetic data…")
    df = synthesize_fake_data(n_samples=30)
    paths = []
    paths.append(make_fig_1_methodology())
    paths.append(make_fig_2_closure_heatmap(df))
    paths.append(make_fig_3_mono_vs_hetero(df))
    paths.append(make_fig_4_stage_bottleneck(stats_mod.per_stage_bottleneck(df)))
    paths.append(make_fig_5_per_benchmark(df))
    paths.append(make_fig_6_cross_family(df))
    paths.append(make_fig_7_judge_corr(df))
    paths.append(make_fig_8_false_closure(stats_mod.false_closure_rate(df)))
    # Contamination figure needs both HE and HEM data — synthetic data only has
    # humaneval/mbpp, so we skip it cleanly
    try:
        contam = stats_mod.contamination_sensitivity(df)
        paths.append(make_fig_9_contamination(contam))
    except Exception as e:
        print(f"  fig9 skipped: {e}")
    for p in paths:
        print(f"  {p}")
    print("\n✓ All figures generated.")
