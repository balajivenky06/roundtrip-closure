"""
analyze/paired_bootstrap_h1_vs_mono.py

Reviewer 2 Comment 4 (second half) — paired bootstrap CIs on
per-operator H1-vs-mono kill-rate differences, accounting for
clustering by sample (shared functions) and multiple mutants per
function.

The marginal Clopper-Pearson intervals in Table 9 are per-cell
confidence intervals on the kill-rate at each operator; they do NOT
support claims of H1 \emph{dominance} over mono baselines because
dominance is a claim about the paired difference, not the two
marginals. For a paired-difference inference we need per-sample
pairing between H1 and each mono baseline.

Approach:
    For each (operator, mono_baseline) pair:
        1. Identify the shared set of samples where both H1 and the mono
           cell have Path 1 kill-rate observations on that operator.
        2. Compute delta_i = h1_kr_i - mono_kr_i per shared sample i.
        3. Bootstrap-resample samples (with replacement) B=10,000 times,
           re-computing mean(delta) each time.
        4. Report mean delta and 2.5/97.5 percentile of the bootstrap
           distribution.

This treats sample as the clustering unit (correctly, because mutants
on the same function are not independent) and the kill-rate aggregate
over mutants-per-sample as the quantity of interest.

Reads:
    results/tab_per_operator_long.csv

Emits:
    results/paired_bootstrap_h1_vs_mono.json
    tables/tab_paired_bootstrap_h1_vs_mono.tex

Run: python3 analyze/paired_bootstrap_h1_vs_mono.py
"""
from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
LONG_CSV = ROOT / "results" / "tab_per_operator_long.csv"
OUT_JSON = ROOT / "results" / "paired_bootstrap_h1_vs_mono.json"
OUT_TEX = ROOT / "tables" / "tab_paired_bootstrap_h1_vs_mono.tex"

OPERATORS = ("arithmetic", "boundary", "comparison", "negate_bool", "return_none")
MONOS = ("M1", "M2", "M3", "M4", "M5", "M6")
N_BOOT = 10_000
RNG = random.Random(42)   # fixed seed for reproducibility


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def bootstrap_ci(deltas: list[float], n_boot: int = N_BOOT,
                 lo_pct: float = 2.5, hi_pct: float = 97.5
                 ) -> tuple[float, float, float]:
    """Return (mean_delta, ci_lo, ci_hi) via case bootstrap over samples."""
    if not deltas:
        return (float("nan"), float("nan"), float("nan"))
    n = len(deltas)
    boot_means = []
    for _ in range(n_boot):
        resample = [deltas[RNG.randrange(n)] for _ in range(n)]
        boot_means.append(mean(resample))
    boot_means.sort()
    lo = boot_means[int(lo_pct / 100 * n_boot)]
    hi = boot_means[int(hi_pct / 100 * n_boot)]
    return (mean(deltas), lo, hi)


def main() -> None:
    # Load long CSV: (cell_id, sample_idx, operator) -> kill_rate
    index: dict[tuple[str, int, str], float] = {}
    with LONG_CSV.open() as f:
        for r in csv.DictReader(f):
            if r["cell_id"] not in ("H1", *MONOS):
                continue
            if r["operator"] not in OPERATORS:
                continue
            try:
                kr = float(r["kill_rate"])
            except (ValueError, TypeError):
                continue
            index[(r["cell_id"], int(r["sample_idx"]), r["operator"])] = kr

    # For each (operator, mono) pair, build paired deltas
    report: dict = {
        "n_bootstrap": N_BOOT,
        "seed": 42,
        "operator_mono_pairs": [],
    }

    for operator in OPERATORS:
        for mono in MONOS:
            # Shared samples: those where both H1 and mono have an entry
            # for this operator
            shared = sorted(
                i for i in {k[1] for k in index if k[0] in ("H1", mono)}
                if (("H1", i, operator) in index and (mono, i, operator) in index)
            )
            if len(shared) < 3:
                continue
            deltas = [index[("H1", i, operator)] - index[(mono, i, operator)]
                      for i in shared]
            mean_d, lo, hi = bootstrap_ci(deltas)
            entry = {
                "operator": operator,
                "mono_baseline": mono,
                "n_shared_samples": len(shared),
                "mean_delta": mean_d,
                "ci_lo": lo,
                "ci_hi": hi,
                "dominance": "dominant" if lo > 0 else ("tied" if hi >= 0 else "dominated"),
            }
            report["operator_mono_pairs"].append(entry)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSON.open("w") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote {OUT_JSON}")

    # Build LaTeX table — one row per operator, showing H1 vs best-mono delta
    # (best-mono = the mono with the highest kill rate on that operator)
    best_mono_by_operator: dict[str, str] = {}
    # Compute best-mono by averaging kill_rate across all samples
    for op in OPERATORS:
        best_mono = None
        best_mean = -1.0
        for m in MONOS:
            krs = [v for (c, _, o), v in index.items() if c == m and o == op]
            if krs:
                mean_kr = mean(krs)
                if mean_kr > best_mean:
                    best_mean = mean_kr
                    best_mono = m
        best_mono_by_operator[op] = best_mono or "?"

    # Build detailed table
    lines = [
        r"\begin{table*}[!htbp]",
        r"\centering",
        r"\small",
        r"\caption{Paired bootstrap 95\% confidence intervals on the per-"
        r"operator H1-vs-mono kill-rate \emph{difference} (R2 Comment 4). "
        r"Each row reports $\Delta = \mathrm{kill\_rate}(H1) - "
        r"\mathrm{kill\_rate}(\mathrm{mono})$ averaged over the shared set "
        r"of samples where both cells have an observation on the operator. "
        r"The bootstrap resamples $n_{\text{shared}}$ samples with "
        r"replacement $B = 10{,}000$ times (seed 42), respecting the "
        r"clustering of mutants within a sample. Positive CIs excluding "
        r"zero establish H1 dominance over that mono baseline on that "
        r"operator in a paired sense, which is stronger than the marginal "
        r"Clopper-Pearson intervals of Table~\ref{tab:per_operator_kill_rate}. "
        r"$^\dagger$ marks the best-mono comparator (highest mean kill-rate on the "
        r"operator); $\mathbf{bold}$ marks CIs that exclude zero from below "
        r"(H1 statistically dominant).}",
        r"\label{tab:paired_bootstrap_h1_vs_mono}",
        r"\begin{tabular}{@{}llrrrr@{}}",
        r"\toprule",
        r"Operator & Mono & $n_{\text{shared}}$ & $\Delta$ & $\text{CI}_{\text{lo}}$ & $\text{CI}_{\text{hi}}$ \\",
        r"\midrule",
    ]
    for operator in OPERATORS:
        rows_for_op = [e for e in report["operator_mono_pairs"]
                       if e["operator"] == operator]
        rows_for_op.sort(key=lambda e: -e["mean_delta"])
        for entry in rows_for_op:
            mono = entry["mono_baseline"]
            star = r"$^\dagger$" if mono == best_mono_by_operator[operator] else ""
            delta_s = f"{entry['mean_delta']:+.3f}"
            lo_s = f"{entry['ci_lo']:+.3f}"
            hi_s = f"{entry['ci_hi']:+.3f}"
            if entry["dominance"] == "dominant":
                delta_s = r"$\mathbf{" + delta_s + r"}$"
                lo_s = r"$\mathbf{" + lo_s + r"}$"
                hi_s = r"$\mathbf{" + hi_s + r"}$"
            lines.append(
                f"{operator} & {mono}{star} & {entry['n_shared_samples']} & "
                f"{delta_s} & {lo_s} & {hi_s} \\\\"
            )
        lines.append(r"\midrule")
    if lines[-1].strip() == r"\midrule":
        lines.pop()
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]

    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    with OUT_TEX.open("w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {OUT_TEX}")

    # Console summary
    print("\n=== H1 dominance summary (vs best-mono per operator) ===")
    for operator in OPERATORS:
        best = best_mono_by_operator[operator]
        entry = next((e for e in report["operator_mono_pairs"]
                      if e["operator"] == operator and e["mono_baseline"] == best), None)
        if entry:
            status = ("DOMINANT" if entry["dominance"] == "dominant"
                      else "tied" if entry["dominance"] == "tied" else "dominated")
            print(f"  {operator:12s} vs {best}: Δ = {entry['mean_delta']:+.3f} "
                  f"[{entry['ci_lo']:+.3f}, {entry['ci_hi']:+.3f}] "
                  f"n_shared={entry['n_shared_samples']:<4}  → {status}")


if __name__ == "__main__":
    main()
