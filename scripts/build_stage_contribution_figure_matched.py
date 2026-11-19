"""
Stage-contribution figure (Fig. 4), sample-matched and active-stage version.

Differences from build_stage_contribution_figure.py:
  * all cells are evaluated on the samples every non-null cell was run on
    (sample_idx 0-74), so hetero bars and mono baselines are comparable;
  * the floor is the minimum over the two stages Path 1 actually invokes
    (L_spec, L_test); the L_code model is not used on Path 1.

V = strict-AND valid-closure fraction: (metric > 0) and (judge >= 3),
denominator = all Path-1 rows of the cell on the shared samples.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STAGES = {  # (spec, test) mono cell of the model used at each active stage
    "H1": ("M2", "M6"), "H2": ("M3", "M2"), "H3": ("M4", "M5"), "H4": ("M6", "M6"),
    "H5": ("M1", "M6"), "H6": ("M2", "M3"), "H7": ("M6", "M3"), "H8": ("M4", "M6"),
    "H9": ("M6", "M3"), "H10": ("M5", "M2"), "H11": ("M2", "M4"),
}

def main(out: str) -> None:
    d = pd.read_csv("results/results_roundtrip.tsv", sep="\t")
    d = d[(d.path == 1) & (d.sample_idx < 75)].copy()
    d["v"] = (d.metric_value > 0) & (d.judge_rating >= 3)
    V = d.groupby("cell_id").v.mean()
    n = d.groupby("cell_id").size()
    assert (n.loc[list(STAGES) + [f"M{i}" for i in range(1, 7)]] == 75).all(), n
    cells = list(STAGES)
    het = np.array([V[c] for c in cells])
    floor = np.array([min(V[a], V[b]) for a, b in STAGES.values()])
    gain = np.where(het > floor, het - floor, 0.0)
    loss = np.where(het < floor, floor - het, 0.0)
    monos = V.loc[[f"M{i}" for i in range(1, 7)]]
    x = np.arange(len(cells)); w = 0.6
    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.bar(x, np.minimum(floor, het), w, color="#7fb3d5", label="Worst active-stage mono baseline (floor)")
    ax.bar(x, gain, w, bottom=floor, color="#f4a261", label="Heterogeneity gain (hetero > floor)")
    ax.bar(x, loss, w, bottom=het, color="#e63946", alpha=0.85, label="Heterogeneity loss (hetero < floor)")
    ax.axhline(monos.max(), color="#2a9d8f", ls="--", lw=1.5,
               label=f"Strongest mono on the same samples ({monos.idxmax()} = {monos.max():.3f})")
    for xi, hv, fl in zip(x, het, floor):
        ax.text(xi, max(hv, fl) + 0.01, f"{hv:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(cells)
    ax.set_ylabel("Strict-AND valid-closure fraction $V$ (Path 1, 75 shared samples)")
    ax.set_ylim(0, 1.12); ax.legend(loc="upper center", ncol=2, frameon=True); ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout(); Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight"); plt.close(fig)
    for c, hv, fl in zip(cells, het, floor):
        print(f"{c:4s} V={hv:.3f} floor={fl:.3f} diff={hv-fl:+.3f}")
    print("monos:", monos.round(3).to_dict())

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "plots/output/fig10_stage_contribution.png")
