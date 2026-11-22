"""
Table 10: per-mutation-operator kill rate with 95% Clopper-Pearson (exact
binomial) intervals and the number of mutants.

Input:  results/tab_per_operator_long.csv  (one row per cell x sample x operator,
        with `total` mutants and `killed` mutants on the cache-recoverable
        Path 1 rows)
Output: tables/tab_per_operator_kill_rate.tex  (tabular body as in the paper)

Kill rate = sum(killed) / sum(total) over a cell's samples. The interval is the
Clopper-Pearson interval, i.e. beta quantiles
    lower = Beta(0.025; k, n - k + 1),  upper = Beta(0.975; k + 1, n - k),
with lower = 0 when k = 0 and upper = 1 when k = n. Bold marks the highest
point estimate in each column (all cells tied at the maximum are bolded).

Run:  python3 scripts/build_per_operator_ci.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from scipy.stats import beta

ROOT = Path(__file__).resolve().parent.parent
OPS = ["arithmetic", "boundary", "comparison", "negate_bool", "return_none"]
CELLS = ["M1", "M2", "M3", "M4", "M5", "M6", "H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8", "H9", "H10", "H11"]


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    lo = 0.0 if k == 0 else beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - alpha / 2, k + 1, n - k)
    return float(lo), float(hi)


def main() -> pd.DataFrame:
    long = pd.read_csv(ROOT / "results/tab_per_operator_long.csv")
    agg = long.groupby(["cell_id", "operator"])[["killed", "total"]].sum()
    rows = []
    for c in CELLS:
        for op in OPS:
            k, n = (int(v) for v in agg.loc[(c, op)])
            lo, hi = clopper_pearson(k, n)
            rows.append({"cell": c, "operator": op, "killed": k, "mutants": n,
                         "kill_rate": round(k / n, 3), "lo": lo, "hi": hi})
    t = pd.DataFrame(rows)
    best = t.groupby("operator").kill_rate.max()
    lines = []
    for c in CELLS:
        cells = []
        for op in OPS:
            r = t[(t.cell == c) & (t.operator == op)].iloc[0]
            val = f"{r.kill_rate:.3f}"
            val = f"\\mathbf{{{val}}}" if r.kill_rate == best[op] else val
            cells.append(f"${val}$ [{r.lo:.2f},{r.hi:.2f}] ({r.mutants})")
        lines.append(f"{c} & " + " & ".join(cells) + r" \\")
    body = "\n".join(lines)
    (ROOT / "tables").mkdir(exist_ok=True)
    (ROOT / "tables/tab_per_operator_kill_rate_body.tex").write_text(body + "\n")
    print(body)
    return t


if __name__ == "__main__":
    main()
