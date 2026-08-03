"""Verify the per-path duplicate-cell disclosure in section_6_discussion.tex.

Recomputes empirical identity rates for every claimed duplicate group against
results/results_roundtrip.tsv. This is the check whose omission produced the
round-3 error where {M1, N1} was mistakenly listed as a duplicate group.

Usage:
    python3 scripts/verify_duplicate_groups.py

Expected output matches the numbers in §6 "Cache reuse across cells and
per-path duplicate cells" paragraph.
"""

import sys
from pathlib import Path
import pandas as pd

TSV = Path("/Users/balajivenktesh/Desktop/Education/roundtrip-closure/results/results_roundtrip.tsv")

CLAIMED_GROUPS = [
    (("M6", "H4"),     1, "genuine"),
    (("M6", "H4"),     3, "genuine"),
    (("M6", "H1", "H5"), 2, "genuine"),
    (("H7", "H9"),     1, "partial"),
    (("H2", "H10"),    2, "partial"),
    (("H6", "H7"),     2, "partial"),
    (("H7", "H9"),     3, "not-duplicate"),
    (("M1", "N1"),     1, "not-duplicate"),
    (("M1", "N1"),     2, "not-duplicate"),
    (("M1", "N1"),     3, "not-duplicate"),
]


def group_identity(df: pd.DataFrame, cells: list[str], path: int) -> tuple[int, int, float]:
    sub = df[(df["cell_id"].isin(cells)) & (df["path"] == path)]
    piv_m = sub.pivot_table(index="sample_idx", columns="cell_id", values="m", aggfunc="first")
    piv_j = sub.pivot_table(index="sample_idx", columns="cell_id", values="j", aggfunc="first")
    has_all = piv_m.notna().all(axis=1) & piv_j.notna().all(axis=1)
    piv_m = piv_m.loc[has_all, cells]
    piv_j = piv_j.loc[has_all, cells]
    n = len(piv_m)
    if n == 0:
        return 0, 0, 0.0
    identical = int(((piv_m.nunique(axis=1) == 1) & (piv_j.nunique(axis=1) == 1)).sum())
    return identical, n, identical / n * 100


def verdict(pct: float) -> str:
    # Paper text thresholds: >=98% genuine; ~50% partial (mid-sweep NUM_CTX era
    # split); <=20% not-duplicate (structurally distinct, cache reuse coincidence).
    if pct >= 98:
        return "genuine"
    if pct > 20:
        return "partial"
    return "not-duplicate"


def main() -> int:
    df = pd.read_csv(TSV, sep="\t")
    df["m"] = pd.to_numeric(df["metric_value"], errors="coerce")
    df["j"] = pd.to_numeric(df["judge_rating"], errors="coerce")

    print(f"{'Group':<20} {'Path':>4} {'Same':>5}/{'Total':>5} {'Rate':>7}   {'Claim':<14} {'Match?':<7}")
    print("-" * 76)
    n_ok = 0
    n_total = 0
    for cells, path, claim in CLAIMED_GROUPS:
        same, total, pct = group_identity(df, list(cells), path)
        got = verdict(pct)
        ok = got == claim
        n_total += 1
        n_ok += int(ok)
        print(
            f"{'/'.join(cells):<20} {path:>4} {same:>5}/{total:>5} {pct:>6.1f}%   "
            f"{claim:<14} {'OK' if ok else 'MISMATCH'}"
        )

    print(f"\n{n_ok}/{n_total} claims match the released TSV.")
    return 0 if n_ok == n_total else 1


if __name__ == "__main__":
    sys.exit(main())
