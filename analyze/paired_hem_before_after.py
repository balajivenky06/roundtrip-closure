"""
R2.7 — paired before/after comparison on the same 25 functions.

Inputs
    results_rerun/results_hem_original_25.tsv          (original functions)
    results_rerun/results_holdout_humaneval_mutated_v2.tsv  (transformed)
Both are indexed by sample_idx 0..24, aligned 1:1 by construction
(data/humaneval_original_25.jsonl was built from the first 25 HEM items).

For each cell (H1, M3) and path (1, 2, 3) it reports:
    * canonical V  = strict-AND closures / 25        (before, after)
    * V'           = strict-AND closures / metric-computable rows
    * all_dropped  = Path-1 rows whose tests all fail on the reference
    * the 2x2 paired table (closed before? x closed after?) and an exact
      two-sided McNemar test on the discordant pairs (binomial, p = 0.5)

Strict-AND: metric_value > 0 and judge_rating >= 3 (paper default tau=0, rho=3).

Outputs
    results/paired_hem_before_after.json
    tables/tab_paired_hem_before_after.tex
"""
from __future__ import annotations

import json
from math import comb
from pathlib import Path

import pandas as pd

ORIG = Path("results_rerun/results_hem_original_25.tsv")
HEM = Path("results_rerun/results_holdout_humaneval_mutated_v2.tsv")


def load(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path, sep="\t")
    d["v"] = (d["metric_value"] > 0) & (d["judge_rating"] >= 3)
    d["ok"] = d["metric_value"].notna()
    d["ad"] = d["notes"].astype(str).str.contains("dropped all")
    return d


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value on discordant counts b, c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def main() -> None:
    o, h = load(ORIG), load(HEM)
    out, tex_rows = [], []
    for cell in ("H1", "M3"):
        for path in (1, 2, 3):
            a = o[(o.cell_id == cell) & (o.path == path)].set_index("sample_idx")
            b = h[(h.cell_id == cell) & (h.path == path)].set_index("sample_idx")
            idx = sorted(set(a.index) & set(b.index))
            a, b = a.loc[idx], b.loc[idx]
            n = len(idx)
            both = int((a.v & b.v).sum())
            lost = int((a.v & ~b.v).sum())      # closed before, not after
            gained = int((~a.v & b.v).sum())    # closed after, not before
            neither = n - both - lost - gained
            rec = {
                "cell": cell, "path": path, "n_paired": n,
                "V_before": a.v.mean(), "V_after": b.v.mean(),
                "Vp_before": a.v.sum() / max(1, a.ok.sum()),
                "Vp_after": b.v.sum() / max(1, b.ok.sum()),
                "all_dropped_before": int(a.ad.sum()) if path == 1 else None,
                "all_dropped_after": int(b.ad.sum()) if path == 1 else None,
                "both": both, "lost": lost, "gained": gained, "neither": neither,
                "mcnemar_p": mcnemar_exact(lost, gained),
            }
            out.append(rec)
            ad = (f"{rec['all_dropped_before']}\\,/\\,{rec['all_dropped_after']}"
                  if path == 1 else "---")
            tex_rows.append(
                f"    {cell} & {path} & ${a.v.sum()}/{n}$ & ${b.v.sum()}/{n}$ & "
                f"${100*(rec['V_after']-rec['V_before']):+.1f}$~pp & {ad} & "
                f"{lost} / {gained} & ${rec['mcnemar_p']:.2f}$ \\\\"
            )
            print(f"{cell} P{path}: n={n} V {rec['V_before']:.3f}->{rec['V_after']:.3f} "
                  f"lost={lost} gained={gained} McNemar p={rec['mcnemar_p']:.3f}")
    Path("results").mkdir(exist_ok=True)
    Path("results/paired_hem_before_after.json").write_text(json.dumps(out, indent=1, default=float))
    tex = r"""\begin{table}[!htbp]
  \centering
  \small
  \caption{Paired before/after comparison on the same 25 HumanEval functions:
    original (before) versus HumanEval-Mutated (after; function rename,
    docstring paraphrase, parameter permutation). Entries are strict-AND
    closures out of 25 ($\tau = 0$, $\rho = 3$). ``Lost / gained'' counts
    functions that close before but not after, and vice versa; $p$ is the
    exact two-sided McNemar test on those discordant pairs. ``All-dropped''
    gives Path~1 rows in which every generated test fails on the reference
    code (before / after).}
  \label{tab:paired_hem_before_after}
  \begin{tabular}{llrrrcrr}
    \toprule
    Cell & Path & Before & After & $\Delta V$ & All-dropped & Lost / gained & $p$ \\
    \midrule
""" + "\n".join(tex_rows[:3]) + "\n    \\midrule\n" + "\n".join(tex_rows[3:]) + r"""
    \bottomrule
  \end{tabular}
\end{table}
"""
    Path("tables").mkdir(exist_ok=True)
    Path("tables/tab_paired_hem_before_after.tex").write_text(tex)
    print("Wrote results/paired_hem_before_after.json and tables/tab_paired_hem_before_after.tex")


if __name__ == "__main__":
    main()
