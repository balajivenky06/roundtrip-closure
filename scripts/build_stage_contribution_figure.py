"""
scripts/build_stage_contribution_figure.py

G6 gap-closer: stacked-bar stage-contribution figure (Epistemic Fortitude
Figure 3 analog).

For each hetero cell, decomposes its Path 1 closure rate into:
    - Baseline floor: closure rate of the WORST mono baseline whose model
      appears at any stage in this cell (i.e., "if we couldn't specialize
      and had to pick the worst").
    - Specialization gain: hetero closure rate MINUS baseline floor.
      Segmented by which stage attribution (spec / test / code) the gain
      comes from, using the per_stage_bottleneck.csv attribution.

Also overlays:
    - The strongest mono ceiling (M3 = 0.900 on Path 1) as a horizontal line.
    - N2 (spec ablated) and N3 (test ablated) values as reference markers.

Emits:
    plots/output/fig10_stage_contribution.png (paper-ready)
    plots/output/fig10_stage_contribution.pdf
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze.load_results import load_tsv
from doe import CELLS_BY_ID

TAU = {1: 0.0, 2: 0.0, 3: 0.0}
RHO = 3


def valid_closure(m: float, j: float, path: int) -> bool:
    if pd.isna(m) or pd.isna(j) or j < 0:
        return False
    return (m > TAU[path]) and (j >= RHO)


def closure_rate_per_cell_path(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["closed"] = df.apply(
        lambda r: valid_closure(r["metric_value"], r["judge_rating"], int(r["path"])),
        axis=1,
    )
    return (
        df.groupby(["cell_id", "path"])["closed"]
        .mean()
        .reset_index(name="closure_rate")
    )


HETERO_CELLS = [f"H{i}" for i in range(1, 12)]
MONO_CELLS = [f"M{i}" for i in range(1, 7)]


def cell_model_short_names(cell_id: str) -> tuple[str, str, str]:
    cell = CELLS_BY_ID[cell_id]
    return (
        cell.L_spec.short_name if cell.L_spec else "—",
        cell.L_test.short_name if cell.L_test else "—",
        cell.L_code.short_name if cell.L_code else "—",
    )


def mono_lookup(short_name: str) -> str | None:
    """Return the mono cell ID whose model has the given short_name."""
    for mid in MONO_CELLS:
        cell = CELLS_BY_ID[mid]
        if cell.L_spec and cell.L_spec.short_name == short_name:
            return mid
    return None


def draw_figure(closure: pd.DataFrame, output: Path) -> None:
    path = 1  # focus on Path 1 (headline metric)
    p1 = closure[closure["path"] == path].set_index("cell_id")["closure_rate"]

    fig, ax = plt.subplots(figsize=(12, 6.5))

    x = np.arange(len(HETERO_CELLS))
    width = 0.6

    baselines = []
    hetero_vals = []
    spec_monos = []
    test_monos = []
    code_monos = []

    for cell_id in HETERO_CELLS:
        if cell_id not in p1.index:
            baselines.append(0.0)
            hetero_vals.append(0.0)
            spec_monos.append(0.0)
            test_monos.append(0.0)
            code_monos.append(0.0)
            continue

        spec_n, test_n, code_n = cell_model_short_names(cell_id)
        spec_m = p1.get(mono_lookup(spec_n), np.nan)
        test_m = p1.get(mono_lookup(test_n), np.nan)
        code_m = p1.get(mono_lookup(code_n), np.nan)
        stage_monos = [v for v in (spec_m, test_m, code_m) if not np.isnan(v)]

        if not stage_monos:
            baselines.append(0.0)
        else:
            baselines.append(min(stage_monos))
        hetero_vals.append(p1[cell_id])
        spec_monos.append(spec_m)
        test_monos.append(test_m)
        code_monos.append(code_m)

    baselines = np.array(baselines)
    hetero_vals = np.array(hetero_vals)
    raw_gain = hetero_vals - baselines
    positive_gain = np.where(raw_gain > 0, raw_gain, 0)
    negative_gain = np.where(raw_gain < 0, -raw_gain, 0)  # absolute value

    # Base + positive-gain stacked bars
    ax.bar(x, baselines, width, color="#7fb3d5", label="Worst-stage mono baseline")
    ax.bar(
        x, positive_gain, width, bottom=baselines,
        color="#f4a261",
        label="Heterogeneity gain (hetero > mono baseline)",
    )
    # Negative gains are shown as a red segment TAKEN OUT of the baseline
    # (i.e. hetero < baseline). We draw a red bar from hetero_val UP to
    # baseline for cells where gain is negative.
    for xi, hv, bl, ng in zip(x, hetero_vals, baselines, negative_gain):
        if ng > 0:
            ax.bar(
                [xi], [ng], width, bottom=[hv],
                color="#e63946", alpha=0.85,
                label="_nolegend_",
            )
    # Add legend entry for negative gain if any exist
    if (negative_gain > 0).any():
        ax.bar([], [], color="#e63946", alpha=0.85,
               label="Heterogeneity loss (hetero < mono baseline)")

    # Overlay: strongest mono ceiling
    ceiling = p1.reindex(MONO_CELLS).max()
    ceiling_cell = p1.reindex(MONO_CELLS).idxmax()
    ax.axhline(
        ceiling, color="#2a9d8f", linestyle="--", linewidth=1.5,
        label=f"Strongest mono ceiling ({ceiling_cell} = {ceiling:.3f})",
    )

    # Overlay: N2 (spec ablated) — only exists on Path 2, so skip for Path 1

    # Annotate hetero values on top of bars
    for xi, hv in zip(x, hetero_vals):
        ax.text(
            xi, hv + 0.01, f"{hv:.2f}",
            ha="center", va="bottom", fontsize=9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(HETERO_CELLS)
    ax.set_ylabel("Closure rate (metric $> 0$ $\\wedge$ judge $\\geq 3$)")
    ax.set_title(
        "Path 1 stage-contribution decomposition: baseline (worst-stage mono) + heterogeneity gain"
    )
    ax.set_ylim(0, max(hetero_vals.max(), ceiling) * 1.15)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    tsv = Path("results/results_roundtrip.tsv")
    df = load_tsv(tsv)
    closure = closure_rate_per_cell_path(df)

    output = Path("plots/output/fig10_stage_contribution.png")
    draw_figure(closure, output)
    print(f"Wrote {output} and {output.with_suffix('.pdf')}")

    # Summary preview
    p1 = closure[closure["path"] == 1].set_index("cell_id")["closure_rate"]
    print("\n=== Path 1 closure rates (τ=0, ρ=3) ===")
    print(p1.round(3).sort_values(ascending=False).to_string())

    print("\n=== Hetero decomposition summary ===")
    for cell_id in HETERO_CELLS:
        spec_n, test_n, code_n = cell_model_short_names(cell_id)
        stage_monos = {
            "spec": p1.get(mono_lookup(spec_n), np.nan),
            "test": p1.get(mono_lookup(test_n), np.nan),
            "code": p1.get(mono_lookup(code_n), np.nan),
        }
        het = p1.get(cell_id, np.nan)
        min_mono = min(v for v in stage_monos.values() if not np.isnan(v))
        gain = het - min_mono
        print(
            f"  {cell_id:4s} ({spec_n:>16s}/{test_n:>14s}/{code_n:>14s}): "
            f"baseline(min mono)={min_mono:.3f}, hetero={het:.3f}, "
            f"gain={gain:+.3f}"
        )


if __name__ == "__main__":
    main()
